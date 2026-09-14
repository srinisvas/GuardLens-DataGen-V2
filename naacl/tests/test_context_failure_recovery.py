"""Regression checks for target overflow and pending judge failures."""
import unittest

from test_optimized_contract import run_active


class ContextFailureRecoveryTests(unittest.TestCase):
    def test_both_replay_paths_propagate_pending_judge_error(self):
        run_active('''
            from types import SimpleNamespace
            from unittest.mock import Mock, patch
            from completion_policy import ADAPTIVE, RecordCompletionError
            from frontier_evidence import EvidenceValidator

            turns=[dict(role='user',turn_id=0,text='one'),
                   dict(role='user',turn_id=2,text='two'),
                   dict(role='user',turn_id=4,text='three')]
            result=dict(content='done',finish_reason='stop',completion_tokens=1,
                        max_tokens=2048,budget_attempts=[])
            for suffix in (False,True):
                for fail_judge in (False,True):
                    validator=EvidenceValidator(SimpleNamespace(model='target'),
                        SimpleNamespace(model='judge'),budget_policy=ADAPTIVE,
                        target_max_model_len=32768)
                    first=Mock(); second=Mock()
                    first.drain.side_effect=RuntimeError('judge parsing failed') if fail_judge else None
                    error=RecordCompletionError('target request exceeds the frozen context envelope')
                    with patch('frontier_evidence.target_completion',side_effect=[result,result,error]), \
                         patch('frontier_evidence.submit_judgment',side_effect=[first,second]):
                        try:
                            if suffix:
                                validator._replay_suffix_v5(turns,baseline={'trajectory':[]},
                                    seed=42,intervention_turn_id=0,replacement_text='replacement')
                            else:
                                validator._full_replay_v5(turns,seed=42,replacements={0:'replacement'})
                        except RuntimeError as exc:
                            if fail_judge:
                                assert str(exc)=='judge parsing failed',repr(exc)
                            else:
                                assert isinstance(exc,RecordCompletionError)
                        else:raise AssertionError('replay failure swallowed')
                    first.drain.assert_called_once()
                    second.drain.assert_called_once()
        ''')

    def test_drain_observes_both_judge_passes(self):
        run_active('''
            from unittest.mock import Mock
            from frontier_judge import JudgmentFuture
            a=Mock(); b=Mock()
            a.result.side_effect=RuntimeError('pass A failed')
            future=JudgmentFuture(a,b)
            try:future.drain()
            except RuntimeError as exc:assert str(exc)=='pass A failed'
            else:raise AssertionError('judge error swallowed')
            a.result.assert_called_once()
            b.result.assert_called_once()
        ''')

    def test_client_and_audit_share_context_classifier(self):
        run_active('''
            from completion_policy import is_context_envelope_error
            import execution
            import audit_frontier_evidence as audit
            assert execution.is_context_envelope_error is is_context_envelope_error
            assert audit.is_context_envelope_error is is_context_envelope_error
            details=dict(request_fingerprint='a'*64,max_tokens=4096,user_turn_id=2,
                         intervention_turn_id=0,replay_kind='counterfactual_suffix')
            value=dict(delta=None,counterfactual_failure=dict(
                error='target request exceeds the frozen context envelope',details=details))
            for message in ('This model maximum context length is 32768 tokens.',
                            'max_tokens is too large',
                            "'max_tokens' or 'max_completion_tokens' is too large"):
                details['server_message']=message
                assert is_context_envelope_error(message)
                audit._audit_context_unassessable('fixture',value,[],'turn',
                    intervention_turn_id=0,delta_field='delta')
            for message in ('invalid model','unauthorized','invalid request',None):
                assert not is_context_envelope_error(message)
                details['server_message']=message
                try:
                    audit._audit_context_unassessable('fixture',value,[],'turn',
                        intervention_turn_id=0,delta_field='delta')
                except RuntimeError:pass
                else:raise AssertionError('unrelated error accepted')
        ''')
