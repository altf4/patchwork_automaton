"""File-only synthetic recovery fixtures; never run games or neural models."""
from copy import deepcopy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

MODULE = Path(__file__).resolve().parents[1]/'examples/run_rustmage_140k_evaluation_recovery.py'
spec = importlib.util.spec_from_file_location('evaluation_recovery_fixture_module',MODULE)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class EvaluationRecoveryTests(unittest.TestCase):
    def fixture(self, root):
        runtime,original,output = root/'runtime',root/'original',root/'recovery'
        (runtime/'examples').mkdir(parents=True)
        for source,name in [(Path(study.phase.__file__),'run_rustmage_continuation_140k.py'),
                            (Path(study.safe.__file__),'run_rustmage_continuation.py'),
                            (MODULE,'run_rustmage_140k_evaluation_recovery.py')]:
            shutil.copyfile(source,runtime/'examples'/name)
        for n in range(149):(runtime/f'fixture-{n}.txt').write_text(str(n))
        runtime_pins={str(p.relative_to(runtime)):study.sha(p) for p in runtime.rglob('*') if p.is_file()}
        training=original/'training'
        rng=np.random.default_rng(123).bit_generator.state
        for n in range(141):
            cp=training/'checkpoints'/f'round-{n:04d}';cp.mkdir(parents=True)
            with zipfile.ZipFile(cp/'policy.keras','w') as z:z.writestr('synthetic-weights',str(n))
            (cp/'policy.manifest.json').write_text('{"synthetic":true}\n')
            actor,critic=study.SOURCE_COUNTERS.get(n,(n,n))
            np.savez(cp/'optimizer.npz',v0=np.array(actor),v1=np.array([1.,2.]))
            np.savez(cp/'critic-optimizer.npz',v0=np.array(critic),v1=np.array([3.,4.]))
            study.publish(cp/'state.json',dict(round=n,config=study.phase.LEARNING_CONFIG,
                optimizer_steps=actor,critic_optimizer_steps=critic,trainer_rng_state=rng))
        # Deliberately invalid JSON: a safe recovery must only stream its hash.
        (training/'report.json').write_bytes(b'large report placeholder: must never be decoded\n')
        study.publish(training/'checkpoints/latest.json',{'checkpoint':'round-0140'})
        old_owner=dict(pid=900001,pgid=900001,sid=900001,start_ticks=1,uid=os.getuid())
        study.publish(original/'study.json',dict(run_id=study.PARENT_RUN_ID,status='error',
            last_error={'message':'Child failed with exit -9: synthetic.log'}))
        old_cleanup=dict(coordinator_pid=900001,status='error',shared_lock_released=True,
            owned_children_remaining=0,finished_utc='2026-10-27T01:11:22Z')
        study.publish(original/'cleanup.json',old_cleanup)
        old_manifest=root/'original-study-manifest.json';old_launch=root/'original-launch.json';old_receipt=root/'original-terminal.json'
        old_runtime=dict(runtime_pins);old_runtime.pop('examples/run_rustmage_140k_evaluation_recovery.py')
        original_native=dict(windows_pid=77,creation_utc='2026-10-26T23:00:00Z')
        study.publish(old_manifest,dict(run_id=study.PARENT_RUN_ID,recovery_epoch=1,
            recovery_coordinator_sha256=study.PHASE_SHA256,source_sha256=old_runtime,
            output=str(original),target_round=140,target_games=140000,
            native_manifest_sha256='a'*64,expected_native_owner=original_native))
        study.publish(old_launch,{'owner':old_owner})
        study.publish(old_receipt,dict(run_id=study.PARENT_RUN_ID,recovery_epoch=1,status='error',
            coordinator=old_owner,recovery_coordinator_sha256=study.PHASE_SHA256,
            native_manifest_sha256='a'*64,native_owner=original_native,
            cleanup_sha256=study.sha(original/'cleanup.json'),cleanup=dict(old_cleanup,cleanup_verified=True)))
        imitation=root/'imitation.keras';imitation.write_bytes(b'synthetic imitation')
        imitation.with_suffix('.manifest.json').write_text('{}')
        source_pins={str(p):study.sha(p) for p in original.rglob('*') if p.is_file()}
        for p in [old_manifest,old_launch,old_receipt,imitation,imitation.with_suffix('.manifest.json')]:source_pins[str(p)]=study.sha(p)
        manifest=dict(run_id=study.RUN_ID,parent_run_id=study.PARENT_RUN_ID,recovery_epoch=2,
            source_round=100,target_round=140,target_games=140000,additional_training_games=0,training=False,
            monitoring_seed=320261025,monitoring_games=2000,confirmation_seed=520261026,confirmation_games=10000,
            head_to_head_seed=530261026,head_to_head_games=5000,head_to_head_opponents=list(study.HEAD_TO_HEAD_NAMES),
            remaining_evaluation_attempts=57000,owned_child_plan=10,python=sys.executable,runtime=str(runtime),
            output=str(output),source_training=str(training),original_output=str(original),shared_lock=str(root/'shared.lock'),
            original_study_manifest=str(old_manifest),original_launch_record=str(old_launch),original_terminal_receipt=str(old_receipt),
            native_liveness=str(root/'live.json'),native_terminal_receipt=str(root/'terminal.json'),
            imitation_checkpoint=str(imitation),hard_deadline_utc=None,
            recovery_coordinator_sha256=study.sha(MODULE),source_sha256=runtime_pins,source_artifacts_sha256=source_pins,
            expected_native_owner=dict(windows_pid=123,creation_utc='2026-10-27T02:30:00Z'),
            native_manifest_sha256='b'*64,source140_rng_sha256=study.canonical(rng))
        return manifest,training,output

    def test_strict_manifest_has_no_training_smoke_passive_or_extended_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            m,_,_=self.fixture(Path(directory));self.assertIs(study.validate_manifest(m),m)
            for key,value in [('run_id',study.PARENT_RUN_ID),('recovery_epoch',1),('additional_training_games',40000),
                ('training',True),('owned_child_plan',11),('remaining_evaluation_attempts',57001),
                ('monitoring_seed',1),('confirmation_seed',450261026),('head_to_head_seed',460261026),
                ('head_to_head_opponents',['round-100']),('hard_deadline_utc','tomorrow'),
                ('recovery_coordinator_sha256',study.PHASE_SHA256)]:
                changed=deepcopy(m);changed[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):study.validate_manifest(changed)
            changed=deepcopy(m);changed['source_artifacts_sha256'].pop(str(Path(m['source_training'])/'checkpoints/round-0000/state.json'))
            with self.assertRaisesRegex(ValueError,'All705'):study.validate_manifest(changed)

    def test_all_source_pins_and_checkpoint_adam_rng_preservation_without_report_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            m,training,_=self.fixture(Path(directory));study.validate_manifest(m)
            original_read=study.read
            def small_only(path):
                self.assertNotEqual(Path(path),training/'report.json')
                return original_read(path)
            with patch.object(study,'read',side_effect=small_only):
                study.verify_sources(m);proof=study.verify_original_stopped(m);snapshot=study.source_snapshot(m)
            self.assertEqual(proof['original_status'],'error')
            self.assertFalse(snapshot['training_report_decoded'])
            self.assertEqual(snapshot['checkpoints']['100']['optimizer_steps'],32822)
            self.assertEqual(snapshot['checkpoints']['140']['critic_optimizer_steps'],22137)
            self.assertEqual(snapshot['checkpoints']['140']['trainer_rng_state_sha256'],m['source140_rng_sha256'])
            (training/'checkpoints/round-0140/optimizer.npz').write_bytes(b'mutation')
            with self.assertRaisesRegex(ValueError,'Immutable original artifact'):study.verify_sources(m)

    def test_original_live_owner_bad_cleanup_receipt_or_early_final_admission_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            m,training,_=self.fixture(Path(directory));old=study.read(m['original_launch_record'])['owner']
            with patch.object(study,'process_identity',return_value=old),self.assertRaisesRegex(ValueError,'remains live'):
                study.verify_original_stopped(m)
            original=Path(m['original_output']);cp=original/'cleanup.json';saved=cp.read_bytes()
            data=study.read(cp);data['shared_lock_released']=False;study.publish(cp,data)
            with self.assertRaisesRegex(ValueError,'cleanup'):study.verify_original_stopped(m)
            cp.write_bytes(saved)
            receipt=Path(m['original_terminal_receipt']);saved=receipt.read_bytes();data=study.read(receipt)
            data['recovery_epoch']=2;study.publish(receipt,data)
            with self.assertRaisesRegex(ValueError,'terminal receipt'):study.verify_original_stopped(m)
            receipt.write_bytes(saved)
            admission=original/'confirmation-round-140-previous.process.json'
            child=dict(pid=900002,pgid=900002,sid=900002,start_ticks=2,uid=os.getuid())
            study.publish(admission,dict(owner=child))
            study.publish(original/'confirmation-round-140-previous.cleanup.json',
                dict(owner=child,remaining=[],verified_utc='2026-10-27T01:11:22Z'))
            with self.assertRaisesRegex(ValueError,'already admitted'):study.verify_original_stopped(m)

    def test_wrong_source140_rng_adam_config_and_latest_pointer_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            m,training,_=self.fixture(Path(directory))
            changed=dict(m,source140_rng_sha256='f'*64)
            with self.assertRaisesRegex(ValueError,'PCG64'):study.source_snapshot(changed)
            statefile=training/'checkpoints/round-0140/state.json';saved=statefile.read_bytes();state=study.read(statefile)
            state['config']['train_batch']=1024;study.publish(statefile,state)
            with self.assertRaisesRegex(ValueError,'learning recipe'):study.source_snapshot(m)
            statefile.write_bytes(saved)
            state=study.read(statefile);state['optimizer_steps']-=1;study.publish(statefile,state)
            with self.assertRaisesRegex(ValueError,'counter'):study.source_snapshot(m)
            statefile.write_bytes(saved)
            study.publish(training/'checkpoints/latest.json',{'checkpoint':'round-0139'})
            with self.assertRaisesRegex(ValueError,'committed140k'):study.source_snapshot(m)

    def test_streaming_sha_never_uses_whole_file_read_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'large';p.write_bytes(b'x'*(3*1024**2+17))
            with patch.object(Path,'read_bytes',side_effect=AssertionError('whole-file allocation')):
                self.assertEqual(len(study.sha(p)),64)

    def mock_evaluation(self,events):
        def evaluate(manifest,checkpoint,filename,seed,games,*,opponent=None):
            events.append((filename,seed,games,None if opponent is None else str(opponent)))
            result=dict(summary={'attempted_games':games,'terminal_games':games},games=[],
                checkpoint_sha256=study.sha(checkpoint),opponent_checkpoint_sha256='c'*64)
            return result,dict(path=str(Path(manifest['output'])/filename),sha256='d'*64,seed=seed,attempts=games,
                checkpoint_sha256=result['checkpoint_sha256'])
        return evaluate

    def test_finite_ten_frozen_children_exact_cohorts_no_source_mutation_or_report_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            m,training,output=self.fixture(Path(directory));events=[];before=Path.cwd()
            original=dict(m['source_artifacts_sha256'])
            try:
                with patch.object(study,'evaluate',side_effect=self.mock_evaluation(events)),\
                     patch('patchwork_automaton.neural.rustmage_evaluation.paired_match_improvement',return_value={'gain':0}):
                    study.run(m)
            finally:os.chdir(before)
            self.assertEqual(len(events),10);self.assertEqual(sum(row[2]for row in events),57000)
            self.assertEqual(events[0][:3],('monitor-0140.json',320261025,2000))
            self.assertEqual(events[1][:3],('confirmation-round-100.json',520261026,10000))
            self.assertEqual(events[2][:3],('confirmation-round-140.json',520261026,10000))
            self.assertEqual([row[0]for row in events[3:]],[f'head-to-head-{name}.json'for name in study.HEAD_TO_HEAD_NAMES])
            self.assertTrue(all(row[1:3]==(530261026,5000) for row in events[3:]))
            self.assertEqual(original,{name:study.sha(name)for name in original})
            self.assertFalse((output/'training').exists())
            report=study.read(output/'report.json');self.assertEqual(report['status'],'complete')
            self.assertFalse(report['training']);self.assertTrue(report['no_training_or_passive_replay'])
            self.assertEqual(len(report['completed_evaluations']),10)
            self.assertTrue(study.read(output/'cleanup.json')['shared_lock_released'])
            with Path(m['shared_lock']).open('a+') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_precondition_failure_never_opens_final_evaluations(self):
        with tempfile.TemporaryDirectory() as directory:
            m,_,_=self.fixture(Path(directory))
            path=Path(m['original_output'])/'cleanup.json';bad=study.read(path);bad['owned_children_remaining']=1
            study.publish(path,bad);m['source_artifacts_sha256'][str(path)]=study.sha(path)
            with patch.object(study,'evaluate') as evaluate,self.assertRaisesRegex(ValueError,'cleanup'):
                study.run(m)
            evaluate.assert_not_called()

    def test_evaluation_failure_preserved_and_no_complete_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            m,_,output=self.fixture(Path(directory));before=Path.cwd()
            try:
                with patch.object(study,'evaluate',side_effect=RuntimeError('fixture evaluator failed')):
                    with self.assertRaisesRegex(RuntimeError,'fixture evaluator failed'):study.run(m)
            finally:os.chdir(before)
            report=study.read(output/'report.json');self.assertEqual(report['status'],'error')
            self.assertEqual(report['completed_evaluations'],{})
            self.assertTrue(study.read(output/'cleanup.json')['shared_lock_released'])
            self.assertFalse(Path(m['native_terminal_receipt']).exists())

    def test_cleanup_failure_stays_visible_and_lock_descriptor_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            m,_,output=self.fixture(Path(directory));before=Path.cwd();calls=0
            def sessions(*args):
                nonlocal calls
                calls+=1
                if calls>1:raise RuntimeError('owned worker remains')
                return []
            try:
                with patch.object(study,'verify_terminal_owned_sessions',side_effect=sessions),\
                     patch.object(study,'evaluate',side_effect=RuntimeError('fixture failure')):
                    with self.assertRaisesRegex(RuntimeError,'owned worker remains'):study.run(m)
            finally:os.chdir(before)
            self.assertFalse(study.read(output/'cleanup.json')['shared_lock_released'])
            self.assertFalse(Path(m['native_terminal_receipt']).exists())
            with Path(m['shared_lock']).open('a+') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_new_terminal_context_binds_own_sha_epoch2_and_zero_training_native(self):
        with tempfile.TemporaryDirectory() as directory:
            m,_,_=self.fixture(Path(directory));owner=dict(pid=12345,pgid=12345,sid=12345,start_ticks=7,uid=os.getuid())
            class Guard:
                last_probe={'expected_manifest_sha256':'b'*64}
                def require_fresh(self):m['_native_owner']=(123,'2026-10-27T02:30:00Z')
            live=dict(run_id=study.RUN_ID,recovery_epoch=2,recovery_coordinator_sha256=study.sha(MODULE),
                finite_game_budget=140000,starting_game_count=140000,additional_game_budget=0)
            study.publish(m['native_liveness'],live)
            with patch.object(study,'process_identity',return_value=owner),patch.object(study.safe,'native_guard_monitor',return_value=Guard()):
                context=study.terminal_receipt_context(m)
                self.assertEqual(context['recovery_epoch'],2)
                self.assertEqual(context['recovery_coordinator_sha256'],study.sha(MODULE))
                self.assertNotEqual(context['recovery_coordinator_sha256'],study.PHASE_SHA256)
                live['additional_game_budget']=40000;study.publish(m['native_liveness'],live)
                with self.assertRaisesRegex(ValueError,'evaluation-only scope'):study.terminal_receipt_context(m)

    def test_wrapper_does_not_replace_original_run_failure_with_receipt_failure(self):
        m={'run_id':study.RUN_ID};context={'coordinator':{'pid':123}}
        with patch.object(study,'validate_manifest'),patch.object(study,'terminal_receipt_context',return_value=context),\
             patch.object(study,'run',side_effect=RuntimeError('original fixture failure')),\
             patch.object(study,'publish_native_terminal_receipt',side_effect=ValueError('cleanup not proven')) as receipt,\
             patch('sys.stderr'):
            with self.assertRaisesRegex(RuntimeError,'original fixture failure'):study.run_with_terminal_receipt(m)
        self.assertEqual(receipt.call_args.args[1],context)
        self.assertEqual(receipt.call_args.args[2]['message'],'original fixture failure')


if __name__=='__main__':unittest.main(verbosity=2)
