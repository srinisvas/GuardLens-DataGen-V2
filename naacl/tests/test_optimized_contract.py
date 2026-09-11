"""Active-module contract and recovery tests, isolated from archived imports."""
import ast
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

HERE=Path(__file__).resolve().parents[1]


def run_active(code):
    result=subprocess.run([sys.executable,'-c','import sys;sys.path.insert(0,'+repr(str(HERE))+');\n'+textwrap.dedent(code)],
                          text=True,capture_output=True,timeout=30)
    if result.returncode:
        raise AssertionError(result.stdout+'\n'+result.stderr)


class OptimizedContractTests(unittest.TestCase):
    def test_frozen_rubrics_parser_aggregation_and_shared_helpers(self):
        for name in ['frontier_common.py','frontier_seed_policy.py','frontier_runtime_determinism.py',
                     'materialize_frontier_candidates.py','split_consolidated.py']:
            self.assertEqual((HERE/name).read_bytes(),(HERE/'legacy'/name).read_bytes(),name)
        old=ast.parse((HERE/'legacy/frontier_judge_v5.py').read_text())
        new=ast.parse((HERE/'frontier_judge.py').read_text())
        def named(tree):
            out={}
            for node in tree.body:
                if isinstance(node,ast.FunctionDef):out[node.name]=node
                if isinstance(node,ast.Assign):
                    for target in node.targets:
                        if isinstance(target,ast.Name):out[target.id]=node
            return out
        old,new=named(old),named(new)
        for name,node in old.items():
            if name=='judge_assistant_response_v5':continue
            current=new['_judge_with_prompt_uncached' if name=='_judge_with_prompt' else name]
            if isinstance(current,ast.FunctionDef):current.name=name
            self.assertEqual(ast.dump(node),ast.dump(current),name)
    def test_no_active_legacy_imports_or_global_patching(self):
        for path in HERE.glob('*.py'):
            tree=ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom):
                    self.assertNotIn('legacy',node.module or '',str(path))
                if isinstance(node,ast.Import):
                    for alias in node.names:self.assertNotIn('legacy',alias.name,str(path))
                self.assertNotIsInstance(node,ast.Global,str(path))
    def test_journal_ownership_contract_digest_and_conflicts(self):
        run_active('''
            import tempfile
            from execution import Journal
            from frontier_common import json_fingerprint
            def rejects(fn):
                try:fn()
                except RuntimeError:return
                raise AssertionError('accepted incompatible/corrupt state')
            with tempfile.TemporaryDirectory() as d:
                journal=Journal(d,{'source':'a','runtime':'frozen'})
                rejects(lambda:Journal(d,{'source':'a','runtime':'frozen'}))
                journal.put({'record':1},{'score':.8})
                rejects(lambda:journal.put({'record':1},{'score':.1}))
                journal.close()
                rejects(lambda:Journal(d,{'source':'b','runtime':'frozen'}))
                journal=Journal(d,{'source':'a','runtime':'frozen'})
                assert journal.get({'record':1})=={'score':.8}
                journal.db.execute('UPDATE results SET value=?', ('{"score":0.2}',))
                journal.db.commit()
                rejects(lambda:journal.get({'record':1}))
                journal.close()
        ''')
    def test_unfinished_target_and_invalid_judge_not_cached(self):
        run_active('''
            import tempfile
            from execution import Journal,Target,Judge,BoundedPool
            from frontier_judge import judge_assistant_response_v5
            class Servers:
                model='fixture'
                calls=0
                def request(self,*args,**kwargs):
                    self.calls+=1
                    return dict(content='invalid JSON',finish_reason='length',completion_tokens=2048)
            with tempfile.TemporaryDirectory() as d:
                journal=Journal(d,{'source':'fixture'})
                servers=Servers()
                target=Target(servers,journal,'record')
                for _ in range(2):target.chat_result([],seed=42,max_tokens=2048)
                assert servers.calls==2
                pool=BoundedPool(2)
                judge=Judge(servers,journal,'record',pool)
                for _ in range(2):
                    try:judge_assistant_response_v5(judge,[{'role':'assistant','content':'fixture'}],seed=42,max_context_chars=100000)
                    except RuntimeError:pass
                    else:raise AssertionError('invalid judge accepted')
                pool.close()
                assert servers.calls==14,servers.calls
                assert journal.db.execute('SELECT count(*) FROM results').fetchone()[0]==0
                journal.close()
        ''')
    def test_concurrency_limits_and_immutable_judge_prefixes(self):
        run_active('''
            import threading,time
            from execution import BoundedPool
            from frontier_judge import submit_judgment
            class Judge:
                def __init__(self):
                    self.pool=BoundedPool(2)
                    self.active=0
                    self.maximum=0
                    self.lock=threading.Lock()
                def submit(self,fn,*args,**kwargs):return self.pool.submit(fn,*args,**kwargs)
                def chat(self,messages,**kwargs):
                    with self.lock:
                        self.active+=1
                        self.maximum=max(self.maximum,self.active)
                    time.sleep(.02)
                    assert 'MUTATED' not in messages[-1]['content']
                    with self.lock:self.active-=1
                    return '{"assistant_behavior":"SAFE_HELP","harmfulness":0,"boundary_violation":0,"refusal_strength":0,"confidence":0.9}'
            judge=Judge()
            prefix=[{'role':'user','content':'original'},{'role':'assistant','content':'response'}]
            future=submit_judgment(judge,prefix,seed=42,max_context_chars=100000)
            prefix[0]['content']='MUTATED'
            assert future.result()['unsafe_score']==0
            judge.pool.close()
            assert judge.maximum==2
        ''')


if __name__=='__main__':unittest.main()
