"""Length recovery preserves complete trajectories and fails closed at its ceiling."""
import json
from pathlib import Path
import tempfile
import unittest
import test_optimized_pipeline as pipeline
from test_optimized_contract import run_active

ADAPTIVE = 'length_retry_2048_4096_8192_v1'


class BudgetPipelineTests(unittest.TestCase):
    setUpClass = classmethod(pipeline.OptimizedPipelineTests.setUpClass.__func__)
    tearDownClass = classmethod(pipeline.OptimizedPipelineTests.tearDownClass.__func__)
    tearDown = pipeline.OptimizedPipelineTests.tearDown
    run_cli = pipeline.OptimizedPipelineTests.run_cli
    optimized = pipeline.OptimizedPipelineTests.optimized

    def test_full_adaptive_chain_and_exact_budget_replay(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root/'source.jsonl'
            source.write_text(''.join(json.dumps(pipeline.source(cid,label,mode))+'\n' for cid,label,mode in
                [('p1-m',1,'NEEDS4096'),('p1-b',0,'NEEDS4096'),('p2-m',1,'NEEDS8192'),('p2-b',0,'NEEDS8192')]))
            pipeline.Oracle.requests_seen = []
            b1,_ = self.optimized(root,'b1',source,budget_policy=ADAPTIVE)
            records = [json.loads(x) for x in b1.read_text().splitlines()]
            self.assertEqual(records[0]['rollout_provenance']['protocol'],'frontier_fixed_user_rollout_v4')
            for row, budget in zip(records,[4096,4096,8192,8192]):
                for turn in row['turns'][1::2]:
                    self.assertEqual(turn['generation_provenance']['max_tokens'],budget)
            before = len(pipeline.Oracle.requests_seen)
            self.optimized(root,'b1',source,budget_policy=ADAPTIVE)
            self.assertEqual(len(pipeline.Oracle.requests_seen),before)
            b2,_ = self.optimized(root,'b2',b1,budget_policy=ADAPTIVE)
            b3 = root/'b3.jsonl'
            self.run_cli('materialize_frontier_candidates.py','--input',b2,'--output',b3,
                         '--max-turn-candidates','4','--spans-per-turn','2','--controls','2')
            b4,_ = self.optimized(root,'b4',b3,budget_policy=ADAPTIVE)
            evidence = [json.loads(x) for x in b4.read_text().splitlines()]
            self.assertEqual(evidence[0]['frontier_evidence_analysis']['protocol'],'frontier_context_paired_counterfactual_v7')
            for row in evidence:
                for item in row['frontier_evidence_analysis'].get('baseline',{}).get('trajectory',[]):
                    original = row['turns'][item['user_turn_id']+1]['generation_provenance']
                    self.assertEqual(item['target_generation'],original)
            self.run_cli('prepare_frontier_dataset.py','--input',b4,'--output',root/'training.jsonl',
                         '--benign-stress-output',root/'stress.jsonl','--excluded-output',root/'excluded.jsonl',
                         '--stats-output',root/'stats.json')
            self.run_cli('audit_frontier_dataset.py','--input',root/'training.jsonl')
            # A changed truncated-attempt fingerprint must fail factual replay
            # even when the final response and all judge scores are unchanged.
            tampered=[json.loads(x) for x in b3.read_text().splitlines()]
            tampered[0]['turns'][1]['generation_provenance']['budget_attempts'][0]['response_fingerprint']='0'*64
            drift_input=root/'budget-drift.jsonl'
            drift_input.write_text(''.join(json.dumps(x)+'\n' for x in tampered))
            failed,result=self.optimized(root,'b4',drift_input,suffix='budget-drift',budget_policy=ADAPTIVE,success=False)
            self.assertFalse(failed.exists())
            self.assertIn('budget/usage provenance drift',result.stdout+result.stderr)
            # Reordered or fabricated attempt history must fail the public audit.
            evidence[0]['frontier_evidence_analysis']['baseline']['trajectory'][0]['target_generation']['budget_attempts'][0]['finish_reason']='stop'
            corrupt = root/'corrupt.jsonl'
            corrupt.write_text(''.join(json.dumps(x)+'\n' for x in evidence))
            self.run_cli('audit_frontier_evidence.py','--input',corrupt,success=False)

    def test_terminal_record_failure_does_not_block_other_records_or_publish(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d);source=root/'source.jsonl'
            source.write_text(''.join(json.dumps(pipeline.source(cid,1,mode))+'\n' for cid,mode in
                [('p0-m','ALWAYS_LENGTH'),('p1-m','NORMAL'),('p2-m','NEEDS4096')]))
            pipeline.Oracle.requests_seen=[]
            output,result=self.optimized(root,'b1',source,workers=1,budget_policy=ADAPTIVE,success=False)
            self.assertFalse(output.exists())
            self.assertIn('2/3 records complete',result.stderr)
            failures=[json.loads(x) for x in (root/'b1state/failed-records.jsonl').read_text().splitlines()]
            self.assertEqual([x['conversation_id'] for x in failures],['p0-m'])
            self.assertEqual([x['max_tokens'] for x in failures[0]['details']['attempts']],[2048,4096,8192])
            self.assertEqual(json.loads(Path(str(output)+'.completion.json').read_text())['status'],'failed')
            before=len(pipeline.Oracle.requests_seen)
            self.optimized(root,'b1',source,workers=1,budget_policy=ADAPTIVE,success=False)
            self.assertEqual(len(pipeline.Oracle.requests_seen),before,'resume repeated terminal failure or completed work')

    def test_context_limit_failure_is_isolated_without_truncation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source.jsonl'
            source.write_text(''.join(json.dumps(pipeline.source(cid,1,mode))+'\n' for cid,mode in
                [('p0-m','CONTEXT_LIMIT'),('p1-m','NORMAL')]))
            output,result=self.optimized(root,'b1',source,workers=1,budget_policy=ADAPTIVE,success=False)
            self.assertFalse(output.exists())
            self.assertIn('1/2 records complete',result.stderr)
            failure=json.loads((root/'b1state/failed-records.jsonl').read_text())
            self.assertIn('context envelope',failure['error'])

    def test_completed_adaptive_attempt_recovered_after_interruption(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'source.jsonl'
            source.write_text(json.dumps(pipeline.source('p0-m',1,'NEEDS8192'))+'\n')
            pipeline.Oracle.requests_seen=[]
            self.optimized(root,'b1',source,budget_policy=ADAPTIVE,success=False,interrupt_after=2)
            self.optimized(root,'b1',source,budget_policy=ADAPTIVE)
            requests=[json.dumps(x,sort_keys=True) for x in pipeline.Oracle.requests_seen]
            self.assertEqual(len(requests),len(set(requests)),'resume repeated a stored budget attempt')


class BudgetMigrationTests(unittest.TestCase):
    def test_policy_no_escalation_for_invalid_finish_usage_or_transport(self):
        run_active('''
            from completion_policy import target_completion,ADAPTIVE,RecordCompletionError
            import requests
            for result in [dict(content='x',finish_reason='stop',completion_tokens=None),
                           dict(content='x',finish_reason='content_filter',completion_tokens=12),
                           dict(content='x',finish_reason='length',completion_tokens=20),
                           requests.Timeout('fixture')]:
                class Client:
                    calls=0
                    def chat_result(self,*args,**kwargs):
                        self.calls+=1
                        if isinstance(result,Exception):raise result
                        return result
                client=Client()
                try:target_completion(client,[],seed=42,max_tokens=2048,policy=ADAPTIVE)
                except (RecordCompletionError,requests.Timeout):pass
                else:raise AssertionError('invalid completion accepted')
                assert client.calls==1
        ''')

    def test_larger_budgets_have_sufficient_transport_timeout(self):
        run_active('''
            import tempfile,threading
            from unittest.mock import patch,Mock
            from execution import Journal,Servers
            with tempfile.TemporaryDirectory() as d:
                journal=Journal(d,{'test':'timeout'})
                response=Mock(status_code=200)
                response.json.return_value={'choices':[{'message':{'content':'done'},'finish_reason':'stop'}],
                                            'usage':{'completion_tokens':12}}
                session=Mock();session.post.return_value=response
                with patch('execution.requests.Session',return_value=session):
                    server=Servers('fixture',['http://fixture'],1,threading.Event(),journal)
                    for budget,timeout in [(2048,600),(4096,1200),(8192,2400)]:
                        server.request([],seed=42,max_tokens=budget)
                        assert session.post.call_args.kwargs['timeout']==(10,timeout)
                    server.close()
                journal.close()
        ''')

    def test_checked_migration_preserves_full_and_partial_request_cache(self):
        run_active('''
            import json,tempfile,sys,copy
            from pathlib import Path
            sys.path.insert(0,str(Path(sys.path[0])/'tests'))
            from test_optimized_pipeline import source
            from run_stage import HERE,TARGET,B1,execution_contract,stage_config
            from execution import Journal,Target
            from frontier_rollout import rollout_record
            from frontier_common import json_fingerprint
            from migrate_b1_state import migrate
            from completion_policy import ADAPTIVE
            with tempfile.TemporaryDirectory() as d:
                root=Path(d);old=root/'old';old.mkdir();(old/'allocation.lock').touch()
                records=[source('p1-m',1,'NORMAL'),source('p2-m',1,'NORMAL')]
                runtime=dict(stage='b1',runtime_determinism='vllm_batch_invariant_eager_v1',replicas={'target':1,'judge':0})
                contract=execution_contract('b1',records,runtime)
                contract['code']=json.loads((HERE/'migrations/b1_v3_c9ce931.json').read_text())['code']
                journal=Journal(old,contract)
                class Server:
                    model=TARGET
                    calls=0
                    stop_after=100
                    def request(self,*args,**kwargs):
                        if self.calls>=self.stop_after:raise RuntimeError('interrupted')
                        self.calls+=1
                        return dict(content='complete fixture',finish_reason='stop',completion_tokens=12)
                server=Server()
                for index,record in enumerate(records):
                    scope=['b1',record['conversation_id'],json_fingerprint(record)]
                    client=Target(server,journal,scope)
                    if index:server.stop_after=server.calls+2
                    try:out=rollout_record(record,client,**B1)
                    except RuntimeError:continue
                    journal.put(dict(kind='record',id=record['conversation_id'],input=json_fingerprint(record)),out)
                journal.close()
                (old/'runtime.json').write_text(json.dumps(runtime))
                original=(old/'progress.sqlite').read_bytes()
                report=migrate(old,root/'new',records,runtime)
                assert report['recovered_complete_records']==1
                assert report['recovered_target_requests']==8
                assert (old/'progress.sqlite').read_bytes()==original
                new=Journal(root/'new',execution_contract('b1',records,runtime,ADAPTIVE))
                server=Server()
                for record in records:
                    target=Target(server,new,['b1',record['conversation_id'],json_fingerprint(record)])
                    out=rollout_record(record,target,budget_policy=ADAPTIVE,**B1)
                    assert out['rollout_status']=='complete'
                assert server.calls==4,server.calls
                new.close()
                for bad_records,bad_runtime in [(records,{**runtime,'driver':['different']}),
                    ([{**records[0],'conversation_id':'changed'},records[1]],runtime)]:
                    try:migrate(old,root/'rejected',bad_records,bad_runtime)
                    except RuntimeError:pass
                    else:raise AssertionError('accepted incompatible migration')
                    assert not (root/'rejected').exists()
                assert (old/'progress.sqlite').read_bytes()==original
                import sqlite3
                db=sqlite3.connect(old/'progress.sqlite')
                bad=copy.deepcopy(contract);bad['code']['run_stage.py']='wrong-code'
                db.execute("UPDATE meta SET value=? WHERE key='contract'",(json.dumps(bad),));db.commit()
                try:migrate(old,root/'bad-code',records,runtime)
                except RuntimeError:pass
                else:raise AssertionError('accepted unapproved source code')
                db.execute("UPDATE meta SET value=? WHERE key='contract'",(json.dumps(contract),))
                db.execute("UPDATE results SET digest='corrupt'");db.commit();db.close()
                try:migrate(old,root/'bad-digest',records,runtime)
                except RuntimeError:pass
                else:raise AssertionError('accepted corrupt checkpoint digest')
        ''')
