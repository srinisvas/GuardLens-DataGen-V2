"""Regressions for the six pre-production diff review findings."""
import unittest
from test_optimized_contract import run_active


class ReviewRegressionTests(unittest.TestCase):
    def test_launcher_preserves_receipt_until_admission(self):
        run_active('''
            import os,sys,json,tempfile,subprocess
            from pathlib import Path
            from types import SimpleNamespace
            from unittest.mock import patch
            from execution import Publication
            from frontier_common import json_fingerprint
            from compare_outputs import compare
            import launch_job
            for failure in ['cache','preflight','capture','mismatch','startup']:
                with tempfile.TemporaryDirectory() as d:
                    root=Path(d);output=root/'output.jsonl';state=root/'state'
                    contract=dict(stage='b1',input_fingerprint='fixture',scientific_config={},runtime={},code={})
                    with Publication(output,'previous') as publication:
                        publication.start()
                        publication.complete([{'conversation_id':'one'}],contract=contract,
                            contract_fingerprint=json_fingerprint(contract),input_fingerprint='fixture')
                    receipt=Path(str(output)+'.completion.json')
                    previous_receipt=receipt.read_bytes();previous_output=output.read_bytes()
                    reference=root/'reference.jsonl';reference.write_bytes(previous_output)
                    env=dict(SLURM_JOB_ID='new',CUDA_VISIBLE_DEVICES='0',INPUT_FILE=str(root/'input.jsonl'),
                        OUTPUT_FILE=str(output),STATE_DIR=str(state),RUN_MODE='smoke')
                    if failure!='cache':env['MODEL_CACHE']=str(root/'cache')
                    if failure=='mismatch':
                        state.mkdir();(state/'runtime.json').write_text('{}')
                    runtime={'server_flags':{'target':[]}}
                    with patch.dict(os.environ,env,clear=True), patch.object(sys,'argv',['launch_job.py','b1']), \
                         patch('launch_job.subprocess.run',side_effect=subprocess.CalledProcessError(1,'preflight') if failure=='preflight' else None), \
                         patch('launch_job.capture_runtime',side_effect=RuntimeError('capture failed') if failure=='capture' else None,return_value=runtime), \
                         patch('launch_job.signal.signal'), patch('launch_job.socket.socket'), \
                         patch('launch_job.subprocess.Popen',side_effect=RuntimeError('startup failed')) as spawn:
                        try:launch_job.main()
                        except (KeyError,RuntimeError,subprocess.CalledProcessError):pass
                        else:raise AssertionError('expected launcher rejection')
                    assert output.read_bytes()==previous_output
                    if failure=='startup':
                        spawn.assert_called_once()
                        current=json.loads(receipt.read_text())
                        assert current['trial_id']=='new' and current['status']=='failed'
                    else:
                        spawn.assert_not_called()
                        assert receipt.read_bytes()==previous_receipt,failure
                        compare(SimpleNamespace(reference=str(reference),optimized=str(output),trial_id='previous'))
                        try:compare(SimpleNamespace(reference=str(reference),optimized=str(output),trial_id='new'))
                        except RuntimeError:pass
                        else:raise AssertionError('rejected trial certified old output')
        ''')

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
