"""Differential end-to-end tests against untouched v3/v5/v6 executors.

A local HTTP fixture is a deterministic model oracle, not a GPU replay test.
Every scientific field is compared; no output fields are ignored.
"""
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from collections import Counter

HERE = Path(__file__).resolve().parents[1]
TARGET = 'Qwen/Qwen2.5-32B-Instruct'
JUDGE = 'mistralai/Mistral-Small-3.1-24B-Instruct-2503'


def source(cid, label, mode):
    turns = []
    for i in range(6):
        text = f'This is ordinary context at source user turn {i} with sufficient content for testing.'
        if i == 5 and label:
            text += ' SPECIAL_REQUEST asks for the full requested operation with detailed assistance.'
        if i == 0:
            text += ' ' + mode + ' family ' + cid.split('-')[0]
        turns.append(dict(turn_id=i*2, role='user', text=text, span_annotations=[],
                          intent_score=None, surface_risk=None, is_trigger=None,
                          trigger_kind=None, is_payload=None, implicit_trigger=None,
                          quoted_unsafe_content=None))
    return dict(conversation_id=cid, pair_id=cid.split('-')[0], label=label,
                user_turn_count=6, assistant_turn_count=0, turns=turns, pivot_turn_id=None,
                supervision_tier=None, loss_weight=None, judge_confidence=None,
                training_eligible=False,
                intended_structure=dict(pair_hardness='surface_control',expected_pivot_turn_ids=[10] if label else [],
                    annotation_status='generation_intent_only_not_ground_truth'),
                metadata=dict(scenario_family='scenario-1',split_group='scenario-1',
                    slice_role='paired_adversarial' if label else 'paired_benign'))


class Oracle(BaseHTTPRequestHandler):
    requests_seen = []
    lock = threading.Lock()
    active = 0
    maximum = 0
    drift = False
    judge_drift = False
    fail = False
    def log_message(self,*args):
        pass
    def reply(self, data, code=200):
        raw = json.dumps(data).encode()
        self.send_response(code)
        self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
    def do_GET(self):
        self.reply({'data':[{'id':TARGET},{'id':JUDGE}]})
    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        with self.lock:
            type(self).requests_seen.append(payload)
            type(self).active += 1
            type(self).maximum = max(self.maximum,self.active)
        try:
            time.sleep(.003 * (1 + payload['seed'] % 3))
            if self.fail:
                self.reply({'error':'injected server failure'},500)
                return
            messages = payload['messages']
            if payload['model'] == TARGET:
                users = ' '.join(m['content'] for m in messages if m['role']=='user')
                digest = hashlib.sha256(json.dumps(messages,sort_keys=True).encode()).hexdigest()[:16]
                unsafe = 'SPECIAL_REQUEST' in users
                content = f'ORACLE_RESPONSE risk={int(unsafe)} {digest}'
                if self.drift:
                    content += ' DRIFT'
            else:
                prompt = messages[-1]['content']
                final = prompt.rsplit('ORACLE_RESPONSE',1)[-1]
                unsafe = 'risk=1' in final
                is_b = 'Authorized-monitoring clarification:' in messages[0]['content']
                risk = (.85 if is_b else (.60 if self.judge_drift else .65)) if unsafe else .05
                # A second fixture exercises confidence based unassessability.
                confidence = .3 if ('LOW_CONFIDENCE' in prompt and unsafe) else .93
                content = json.dumps(dict(assistant_behavior='UNSAFE_HELP' if unsafe else 'SAFE_HELP',
                    harmfulness=.1 if unsafe else .02, boundary_violation=risk,
                    refusal_strength=0,confidence=confidence))
            self.reply({'choices':[{'message':{'content':content},'finish_reason':'stop'}],
                        'usage':{'completion_tokens':12,'prompt_tokens':100,'total_tokens':112}})
        finally:
            with self.lock:
                type(self).active -= 1


class OptimizedPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(('127.0.0.1',0),Oracle)
        cls.thread = threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
        cls.url = 'http://127.0.0.1:'+str(cls.server.server_port)
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
    def run_cli(self, script, *args, success=True, interrupt_after=None):
        command = [sys.executable,str(HERE/script),*map(str,args)]
        env = {**os.environ,'VLLM_BATCH_INVARIANT':'1'}
        if interrupt_after is None:
            result = subprocess.run(command,env=env,text=True,capture_output=True,timeout=60)
        else:
            process = subprocess.Popen(command,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            deadline = time.monotonic()+30
            while process.poll() is None and len(Oracle.requests_seen)<interrupt_after and time.monotonic()<deadline:
                time.sleep(.002)
            try:
                self.assertIsNone(process.poll(),'worker finished before interruption')
                self.assertGreaterEqual(len(Oracle.requests_seen),interrupt_after)
                process.send_signal(signal.SIGUSR1)
                stdout,stderr = process.communicate(timeout=30)
                result = subprocess.CompletedProcess(command,process.returncode,stdout,stderr)
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
        if success:
            self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        else:
            self.assertNotEqual(result.returncode,0,result.stdout)
        return result
    def optimized(self, root, stage, input_path, suffix='', success=True, workers=4, interrupt_after=None):
        manifest = root/(stage+'runtime.json')
        manifest.write_text(json.dumps(dict(runtime_determinism='vllm_batch_invariant_eager_v1',
            stage=stage,replicas={'target':int(stage!='b2'),'judge':int(stage!='b1')})))
        output = root/(stage+'optimized'+suffix+'.jsonl')
        args = [stage,'--input',input_path,'--output',output,'--state-dir',root/(stage+'state'+suffix),
            '--runtime-manifest',manifest,'--record-workers',workers,'--intervention-workers','4']
        if stage != 'b2': args += ['--target-urls',self.url]
        if stage != 'b1': args += ['--judge-urls',self.url]
        result = self.run_cli('run_stage.py',*args,success=success,interrupt_after=interrupt_after)
        return output,result
    def test_full_pipeline_matches_reference_and_resumes(self):
        Oracle.drift = Oracle.fail = False
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            src = root/'source.jsonl'
            rows = [source('p1-malicious',1,'NORMAL'),source('p1-benign',0,'NORMAL'),
                    source('p2-malicious',1,'NORMAL'),source('p3-lowconf',1,'LOW_CONFIDENCE')]
            src.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            previous = src
            for stage,script,endpoint in [
                ('b1','rollout_frontier_source_v3.py','--base-url'),
                ('b2','validate_frontier_rollout_v5_deterministic.py','--judge-base-url'),
                ('b4','frontier_evidence_v6.py','--target-base-url')]:
                if stage == 'b4':
                    candidates = root/'candidates.jsonl'
                    self.run_cli('materialize_frontier_candidates.py','--input',previous,'--output',candidates)
                    previous = candidates
                reference = root/(stage+'reference.jsonl')
                args = ['--input',previous,'--output',reference,endpoint,self.url]
                if stage == 'b4': args += ['--judge-base-url',self.url]
                Oracle.requests_seen=[]
                self.run_cli('legacy/'+script,*args)
                reference_requests = list(Oracle.requests_seen)
                Oracle.requests_seen=[]
                Oracle.maximum=0
                output,_ = self.optimized(root,stage,previous)
                self.assertEqual([json.loads(x) for x in reference.read_text().splitlines()],
                                 [json.loads(x) for x in output.read_text().splitlines()],stage)
                # Cache reuse can coalesce identical work; no new prompt/seed/budget is allowed.
                reference_counts=Counter(json.dumps(x,sort_keys=True) for x in reference_requests)
                actual_counts=Counter(json.dumps(x,sort_keys=True) for x in Oracle.requests_seen)
                self.assertFalse(actual_counts-reference_counts,stage+' introduced requests')
                self.assertGreater(Oracle.maximum,1,stage+' was serialized')
                requests_before=len(Oracle.requests_seen)
                self.optimized(root,stage,previous,workers=2)
                self.assertEqual(len(Oracle.requests_seen),requests_before,'resume reissued completed work')
                if stage == 'b1':
                    self.run_cli('probe_runtime.py','--input',reference,'--target-urls',self.url,
                        '--judge-urls',self.url,'--concurrency','1','2','--repeats','1',
                        '--output',root/'gate0.json')
                if stage == 'b4':
                    Oracle.drift=True
                    failed,result=self.optimized(root,stage,previous,suffix='drift',success=False)
                    self.assertFalse(failed.exists())
                    self.assertIn('drift',result.stdout+result.stderr)
                    Oracle.drift=False
                    Oracle.judge_drift=True
                    failed,result=self.optimized(root,stage,previous,suffix='judgedrift',success=False)
                    self.assertFalse(failed.exists())
                    self.assertIn('drift',result.stdout+result.stderr)
                    Oracle.judge_drift=False
                    Oracle.requests_seen=[]
                    failed,_=self.optimized(root,stage,previous,suffix='interrupted',
                                            success=False,interrupt_after=45)
                    self.assertFalse(failed.exists())
                    resumed,_=self.optimized(root,stage,previous,suffix='interrupted')
                    self.assertEqual([json.loads(x) for x in reference.read_text().splitlines()],
                                     [json.loads(x) for x in resumed.read_text().splitlines()])
                    # Successful requests before the signal are recovered, including
                    # work inside incomplete records and interventions.
                    combined=Counter(json.dumps(x,sort_keys=True) for x in Oracle.requests_seen)
                    self.assertFalse(combined-reference_counts,'resume regenerated completed requests')
                previous=reference
            self.run_cli('prepare_frontier_dataset.py','--input',previous,
                '--output',root/'training.jsonl','--benign-stress-output',root/'stress.jsonl',
                '--excluded-output',root/'excluded.jsonl','--stats-output',root/'stats.json')
            self.run_cli('audit_frontier_dataset.py','--input',root/'training.jsonl')
            self.run_cli('audit_frontier_stress.py','--input',root/'stress.jsonl')
            # A transport error never publishes an incomplete record/output.
            Oracle.fail=True
            failed,_=self.optimized(root,'b1',src,suffix='fail',success=False)
            self.assertFalse(failed.exists())
            Oracle.fail=False
            self.optimized(root,'b1',src,suffix='fail')


if __name__ == '__main__':
    unittest.main()
