"""Bounded model execution and durable, identity-checked subtask recovery.

Operational metadata lives here, outside scientific record fingerprints.
Only one process owns a state directory. SQLite stays on the shared POSIX
filesystem with rollback journaling (not WAL, which is unsuitable for NFS).
"""
from __future__ import annotations

import contextlib
import copy
import fcntl
import json
import os
import sqlite3
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import requests

from frontier_common import VLLMClient, json_fingerprint, parse_chat_completion_data


class Stopped(RuntimeError):
    pass


class BoundedPool:
    def __init__(self, workers, backlog=None):
        if workers < 1:
            raise ValueError("workers must be positive")
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.slots = threading.BoundedSemaphore(backlog or workers * 2)

    def submit(self, fn, *args, **kwargs):
        self.slots.acquire()
        try:
            f = self.pool.submit(fn, *args, **kwargs)
        except BaseException:
            self.slots.release()
            raise
        f.add_done_callback(lambda _: self.slots.release())
        return f

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)


def immediate(fn, *args, **kwargs):
    f = Future()
    try:
        f.set_result(fn(*args, **kwargs))
    except Exception as exc:
        f.set_exception(exc)
    return f


class Journal:
    def __init__(self, directory, contract):
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.owner = open(self.root / "owner.lock", "a+")
        try:
            fcntl.flock(self.owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.owner.close()
            raise RuntimeError("state directory already has an active executor")
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.root / "progress.sqlite", check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=DELETE")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS results (key TEXT PRIMARY KEY, value TEXT, digest TEXT)")
        encoded = json.dumps(contract, sort_keys=True)
        old = self.db.execute("SELECT value FROM meta WHERE key='contract'").fetchone()
        if old and old[0] != encoded:
            self.db.close()
            self.owner.close()
            raise RuntimeError("checkpoint contract mismatch; use a new state directory")
        self.db.execute("INSERT OR IGNORE INTO meta VALUES ('contract', ?)", (encoded,))
        self.db.commit()
        self.telemetry = open(self.root / "requests.jsonl", "a", buffering=1)

    def get(self, identity):
        key = json_fingerprint(identity)
        with self.lock:
            row = self.db.execute("SELECT value,digest FROM results WHERE key=?", (key,)).fetchone()
        if row is None:
            return None
        value = json.loads(row[0])
        if json_fingerprint(value) != row[1]:
            raise RuntimeError("checkpoint digest mismatch")
        return value

    def put(self, identity, result):
        key = json_fingerprint(identity)
        value = json.dumps(result, ensure_ascii=False, sort_keys=True)
        digest = json_fingerprint(result)
        with self.lock:
            old = self.db.execute("SELECT digest FROM results WHERE key=?", (key,)).fetchone()
            if old and old[0] != digest:
                raise RuntimeError("conflicting scientific result for identical work identity")
            self.db.execute("INSERT OR IGNORE INTO results VALUES (?, ?, ?)", (key, value, digest))
            self.db.commit()

    def event(self, **values):
        with self.lock:
            self.telemetry.write(json.dumps({"time": time.time(), **values}) + "\n")

    def close(self):
        self.telemetry.close()
        self.db.close()
        self.owner.close()


class Servers:
    """Per-request caps plus soft chain affinity, shared by all record workers."""
    def __init__(self, model, urls, per_server, stop, journal):
        if not urls or per_server < 1:
            raise ValueError("need endpoints and a positive request limit")
        self.model, self.urls = model, [u.rstrip('/') for u in urls]
        self.limit, self.stop, self.journal = per_server, stop, journal
        self.busy = [0] * len(urls)
        self.cv = threading.Condition()
        self.local = threading.local()
        self.sessions = []
        self.session_lock = threading.Lock()

    def check(self):
        for url in self.urls:
            if not VLLMClient(self.model, url).health_check():
                raise RuntimeError(f"server unavailable: {url}")
            response = requests.get(url + '/v1/models', timeout=10,
                headers={'Authorization': 'Bearer ' + os.environ.get('VLLM_API_KEY', 'EMPTY')})
            response.raise_for_status()
            if self.model not in {x['id'] for x in response.json().get('data', [])}:
                raise RuntimeError(f"server model mismatch: {url}")

    @contextlib.contextmanager
    def chain(self):
        previous = getattr(self.local, 'preferred', None)
        self.local.preferred = None
        try:
            yield
        finally:
            self.local.preferred = previous

    def request(self, messages, **kwargs):
        if self.stop.is_set():
            raise Stopped("execution stopped; successful subtasks remain checkpointed")
        started = time.monotonic()
        preferred = getattr(self.local, 'preferred', None)
        with self.cv:
            while True:
                if self.stop.is_set():
                    raise Stopped("execution stopped")
                candidates = [i for i, n in enumerate(self.busy) if n < self.limit]
                if candidates:
                    # Affinity when equally loaded; otherwise take free capacity.
                    i = min(candidates, key=lambda x: (self.busy[x], x != preferred, x))
                    self.busy[i] += 1
                    break
                self.cv.wait(timeout=0.25)
        self.local.preferred = i
        session = getattr(self.local, 'session', None)
        if session is None:
            session = self.local.session = requests.Session()
            with self.session_lock:
                self.sessions.append(session)
        queued = time.monotonic() - started
        url = self.urls[i]
        payload = dict(model=self.model, messages=copy.deepcopy(messages),
                       temperature=float(kwargs.get('temperature', 0)), top_p=1.0,
                       max_tokens=int(kwargs['max_tokens']), seed=int(kwargs['seed']))
        event = dict(model=self.model, server=url, queue_seconds=queued,
                     request_fingerprint=json_fingerprint(payload))
        try:
            response = session.post(url + '/v1/chat/completions', json=payload,
                                    headers={'Authorization': 'Bearer ' + os.environ.get('VLLM_API_KEY', 'EMPTY')},
                                    timeout=(10, 600))
            response.raise_for_status()
            data = response.json()
            event.update(usage=data.get('usage'), finish_reason=(data.get('choices') or [{}])[0].get('finish_reason'))
            return parse_chat_completion_data(data, require_stop=kwargs.get('require_stop', True),
                                               require_usage=kwargs.get('require_usage', False))
        except requests.exceptions.RequestException as exc:
            # A timed-out server request may still be running. Stop admission so
            # retries cannot form an uncontrolled duplicate-request feedback loop.
            self.stop.set()
            event['error'] = repr(exc)
            raise
        except Exception as exc:
            event['error'] = repr(exc)
            raise
        finally:
            event['elapsed_seconds'] = time.monotonic() - started
            try:
                self.journal.event(**event)
            finally:
                with self.cv:
                    self.busy[i] -= 1
                    self.cv.notify_all()

    def close(self):
        for session in self.sessions:
            session.close()


class Target:
    def __init__(self, servers, journal, scope):
        self.servers, self.journal, self.scope = servers, journal, scope
        self.model = servers.model

    def chat_result(self, messages, **kwargs):
        identity = dict(scope=self.scope, kind='target', messages=messages, kwargs=kwargs, model=self.model)
        cached = self.journal.get(identity)
        if cached is not None:
            return cached
        result = self.servers.request(messages, **kwargs)
        tokens = result.get('completion_tokens')
        if result.get('finish_reason') == 'stop' and type(tokens) is int and 0 < tokens <= kwargs['max_tokens']:
            self.journal.put(identity, result)
        return result

    def chat(self, messages, **kwargs):
        return self.chat_result(messages, require_stop=True, require_usage=False, **kwargs)['content']

    def chain(self):
        return self.servers.chain()

    def scoped(self, purpose):
        return Target(self.servers, self.journal, [self.scope, purpose])


class Judge:
    def __init__(self, servers, journal, scope, pool):
        self.servers, self.journal, self.scope, self.pool = servers, journal, scope, pool
        self.model = servers.model

    def chat(self, messages, **kwargs):
        return self.servers.request(messages, require_stop=True, require_usage=False, **kwargs)['content']

    def run_pass(self, fn, prefix, **kwargs):
        identity = dict(scope=self.scope, kind='validated_judge_pass', prefix=prefix,
                        kwargs=kwargs, model=self.model)
        cached = self.journal.get(identity)
        if cached is not None:
            return cached
        value = fn(self, prefix, **kwargs)
        self.journal.put(identity, value)
        return value

    def submit(self, fn, *args, **kwargs):
        return self.pool.submit(fn, *args, **kwargs)


def atomic_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
