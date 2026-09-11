"""Regressions for the six pre-production diff review findings."""
import unittest
from test_optimized_contract import run_active


class ReviewRegressionTests(unittest.TestCase):
    def test_intervention_identity_survives_annotation_updates(self):
        run_active('''
            import threading,tempfile,copy
            from execution import Journal,BoundedPool
            from frontier_evidence import EvidenceValidator
            class Client:model='fixture'
            with tempfile.TemporaryDirectory() as d:
                journal=Journal(d,{'test':'immutable-key'})
                client=Client();client.journal=journal;client.scope=['b4','record']
                validator=EvidenceValidator(client,Client())
                validator.intervention_pool=BoundedPool(1)
                turns=[{'turn_id':0,'role':'user','text':'original','span_annotations':[{'label':'EVIDENCE_CANDIDATE'}]}]
                original=copy.deepcopy(turns);baseline={'trajectory':[]}
                entered=threading.Event();release=threading.Event()
                def task(*args,**kwargs):
                    entered.set()
                    assert release.wait(2)
                    return {'status':'supported_strong'}
                future=validator.submit_intervention(task,validator,turns,baseline,turn_id=0)
                assert entered.wait(2)
                turns[0]['span_annotations'][0]['evidence_status']='supported_strong'
                release.set();future.result()
                key={'kind':'intervention','scope':client.scope,'args':(original,baseline),'kwargs':{'turn_id':0}}
                assert journal.get(key)=={'status':'supported_strong'}
                def unexpected(*args,**kwargs):raise AssertionError('completed intervention reran')
                assert validator.submit_intervention(unexpected,validator,original,baseline,turn_id=0).result()=={'status':'supported_strong'}
                validator.intervention_pool.close();journal.close()
        ''')

    def test_destination_lock_and_inherited_ownership(self):
        run_active('''
            import tempfile,subprocess,sys,json
            from pathlib import Path
            from execution import Publication,OutputLease
            with tempfile.TemporaryDirectory() as d:
                output=Path(d)/'output.jsonl'
                with Publication(output,'parent') as owner:
                    owner.start(stage='b1')
                    try:OutputLease(output)
                    except RuntimeError:pass
                    else:raise AssertionError('second writer acquired the destination')
                    code='from execution import Publication; import sys; p=Publication(sys.argv[1],"parent",int(sys.argv[2])); p.complete([{"conversation_id":"one"}]); p.owner.close()'
                    result=subprocess.run([sys.executable,'-c','import sys;sys.path.insert(0,sys.argv[3]);'+code,
                        str(output),str(owner.owner.fileno()),sys.path[0]],pass_fds=(owner.owner.fileno(),),capture_output=True,text=True)
                    assert result.returncode==0,result.stderr
                    try:OutputLease(output)
                    except RuntimeError:pass
                    else:raise AssertionError('child released the parent lock')
                assert json.loads(Path(str(output)+'.completion.json').read_text())['status']=='complete'
                with OutputLease(output):pass
        ''')

    def test_atomic_writers_have_distinct_temporary_files(self):
        run_active('''
            import tempfile,threading,json,os
            from pathlib import Path
            from unittest.mock import patch
            from execution import atomic_jsonl
            with tempfile.TemporaryDirectory() as d:
                output=Path(d)/'output.jsonl'
                barrier=threading.Barrier(2);original=os.replace;names=[];errors=[]
                def replace(a,b):
                    names.append(a)
                    barrier.wait(timeout=2)
                    return original(a,b)
                def write(label):
                    try:atomic_jsonl(output,[{'conversation_id':label}])
                    except BaseException as exc:errors.append(exc)
                with patch('execution.os.replace',replace):
                    threads=[threading.Thread(target=write,args=(label,)) for label in ['a','b']]
                    for t in threads:t.start()
                    for t in threads:t.join()
                assert not errors,errors
                assert len(set(names))==2
                assert json.loads(output.read_text())['conversation_id'] in {'a','b'}
                assert not list(Path(d).glob('*.tmp'))
        ''')

    def test_worker_defaults_cover_configured_request_capacity(self):
        run_active('''
            import os
            from unittest.mock import patch
            from launch_job import concurrency_settings
            with patch.dict(os.environ,{'TARGET_INFLIGHT':'8','JUDGE_INFLIGHT':'4'},clear=True):
                b1=concurrency_settings(4,0)
                b4=concurrency_settings(3,1)
                assert b1['records']>=32
                assert b4['interventions']>=24
                assert b4['records']>=24
            with patch.dict(os.environ,{'TARGET_INFLIGHT':'8','RECORD_WORKERS':'2','INTERVENTION_WORKERS':'3'},clear=True):
                custom=concurrency_settings(3,1)
                assert custom['records']==2 and custom['interventions']==3
        ''')


if __name__=='__main__':unittest.main()
