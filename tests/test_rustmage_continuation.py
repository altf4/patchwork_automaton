"""A continuation changes host topology without resetting learned state."""
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from examples import run_rustmage_continuation as study


class RustMageContinuationTests(unittest.TestCase):
    def fixture(self, root):
        source, output, runtime = root/'source', root/'continuation', root/'runtime'
        runtime.mkdir()
        (runtime/'frozen.py').write_text('# frozen runtime\n')
        config = dict(algorithm='ppo', architecture='separate', train_mode='versus_baseline',
            games_per_round=1000, seed=300261025, train_batch=512, ppo_epochs=2,
            validation_fraction=.2, workers=48, inference_processes=4, collector='processes',
            learning_rate=.0001, value_learning_rate=.0003, value_max_steps=157,
            target_kl=.02, max_state_kl=.5, entropy_weight=.01, critic_unroll=1,
            critic_update_scope='all', advantage_baseline='critic', learner_device='gpu',
            inference_device='cpu', validation_device='cpu', critic_warmup_games=1000,
            critic_warmup_max_steps=157, train_horizon=60, eval_every_games=0,
            baseline_eval_games=0, concurrent_updates=True, prefetch_batches=2,
            measurement_workers=4, rollout_cache_mib=512, rollout_storage='pickle',
            eval_games=200, eval_horizon=25, inference_wait_ms=0., guard_audit=False,
            validation_batch=32, baseline_eval_seed=48261024, observation_pipeline='python',
            advantage_scale_floor=1., ppo_clip=.2, pooled_collector_config='/old/pool.json')
        rng = np.random.default_rng(79).bit_generator.state
        for number in range(41):
            checkpoint = source/'checkpoints'/f'round-{number:04d}'
            checkpoint.mkdir(parents=True)
            with zipfile.ZipFile(checkpoint/'policy.keras', 'w') as archive:
                archive.writestr('synthetic-weights', bytes([number]))
            (checkpoint/'policy.manifest.json').write_text('{"backend":"synthetic"}\n')
            actor, critic = (13289, 6437) if number == 40 else (number, 157+number)
            np.savez(checkpoint/'optimizer.npz', v0=np.array(actor), v1=np.array([.25, .5]))
            np.savez(checkpoint/'critic-optimizer.npz', v0=np.array(critic), v1=np.array([.125]))
            study.publish(checkpoint/'state.json', dict(round=number, config=config,
                optimizer_steps=actor, critic_optimizer_steps=critic, trainer_rng_state=rng,
                unrelated_provenance={'keep':'exactly'}))
        study.publish(source/'checkpoints/latest.json', {'checkpoint':'round-0040'})
        study.publish(source/'report.json', dict(status='complete', config=config,
            rounds=[dict(round=i) for i in range(1,41)], invocations=[{'historical':True}],
            critic_warmup={'status':'complete', 'requested_games':1000},
            training_games_completed=40000, initial_training_games=0,
            historical_paths={'rollouts':'/old/source/rollouts'}))
        imitation = root/'imitation.keras'
        imitation.write_bytes((source/'checkpoints/round-0000/policy.keras').read_bytes())
        imitation.with_suffix('.manifest.json').write_text('{}\n')
        pins = {str(p):study.sha(p) for p in source.rglob('*') if p.is_file()}
        pins.update({str(imitation):study.sha(imitation),
                     str(imitation.with_suffix('.manifest.json')):study.sha(imitation.with_suffix('.manifest.json'))})
        manifest = dict(run_id='test-continuation', output=str(output), runtime=str(runtime),
            source_training=str(source), python=sys.executable, shared_lock=str(root/'shared.lock'),
            native_liveness=str(root/'native-live.json'), native_terminal_receipt=str(root/'terminal.json'),
            native_manifest_sha256='a'*64, expected_native_owner={'windows_pid':123,'creation_utc':'2026-10-26T12:00:00Z'},
            recovery_epoch=1, recovery_coordinator_sha256=study.sha(study.__file__),
            source_sha256={'frozen.py':study.sha(runtime/'frozen.py')}, source_artifacts_sha256=pins,
            imitation_checkpoint=str(imitation), source_round=40, target_round=100,
            target_games=100000, training_seed=300261025, milestones=list(study.MILESTONES),
            monitoring_rounds=list(study.MONITORING_ROUNDS), monitoring_seed=320261025, monitoring_games=2000,
            confirmation_seed=450261026, confirmation_games=10000,
            head_to_head_seed=460261026, head_to_head_games=5000,
            preflight_head_to_head_seed=470261026, preflight_head_to_head_games=8,
            head_to_head_opponents=list(study.HEAD_TO_HEAD_NAMES),
            runtime_options=dict(learner_gru_backend='cudnn-memory', update_pipeline='gpu',
                                 policy_cache_mib=8192, actor_epoch_cache_mib=8192))
        return manifest, source, output, config

    def test_fork_changes_only_copied_topology_and_keeps_every_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, source, output, old_config = self.fixture(Path(temporary))
            study.validate_manifest(manifest)
            before = {str(p):study.sha(p) for p in source.rglob('*') if p.is_file()}
            training, config, migration = study.prepare(manifest)
            self.assertIsNone(config['pooled_collector_config'])
            self.assertEqual({k:v for k,v in config.items() if k != 'pooled_collector_config'},
                             {k:v for k,v in old_config.items() if k != 'pooled_collector_config'})
            self.assertEqual(study.read(training/'checkpoints/latest.json'), {'checkpoint':'round-0040'})
            self.assertEqual(study.read(training/'report.json')['rounds'], [{'round':i} for i in range(1,41)])
            self.assertEqual(len(migration['state_changes']), 41)
            self.assertEqual(migration['changed_config_fields'], ['pooled_collector_config'])
            for number in range(41):
                left = source/'checkpoints'/f'round-{number:04d}'
                right = training/'checkpoints'/f'round-{number:04d}'
                for name in study.FILES[:-1]:
                    self.assertEqual((left/name).read_bytes(), (right/name).read_bytes())
                original, copied = study.read(left/'state.json'), study.read(right/'state.json')
                original['config']['pooled_collector_config'] = None
                self.assertEqual(original, copied)
            self.assertEqual(before, {str(p):study.sha(p) for p in source.rglob('*') if p.is_file()})
            again = study.prepare(manifest)
            self.assertEqual(again[0], training)
            self.assertEqual(again[2], migration)

    def test_recipe_change_partial_copy_and_source_mutation_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, source, output, config = self.fixture(Path(temporary))
            changed = deepcopy(config); changed['learning_rate'] *= 2
            with self.assertRaisesRegex(ValueError, 'recipe'):
                study.migration_config(changed)
            (output/'training.prepare').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'Partial preparation'):
                study.prepare(manifest)
            (source/'checkpoints/round-0040/policy.keras').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'source changed'):
                study.verify_sources(manifest)

    def test_training_command_resumes_finite_target_without_pool_or_reinitialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _, _, config = self.fixture(Path(temporary))
            migrated = study.migration_config(config)
            command = study.train_command(sys.executable, '/new/training', migrated, 100,
                                          manifest['runtime_options'])
            self.assertIn('--resume', command)
            self.assertNotIn('--initialize-from', command)
            self.assertNotIn('--pooled-collector-config', command)
            for flag, value in [('--seed','300261025'), ('--train-batch','512'),
                                ('--critic-warmup-games','1000'), ('--rounds','100'),
                                ('--workers','48'), ('--inference-processes','4')]:
                self.assertEqual(command[command.index(flag)+1], value)
            with self.assertRaisesRegex(ValueError, 'Unregistered'):
                study.train_command(sys.executable, '/new', migrated, 101, manifest['runtime_options'])

    def test_fixed_design_and_source_pins_cannot_be_relaxed(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _, _, _ = self.fixture(Path(temporary))
            for key, value in [('target_games',101000), ('source_round',0), ('training_seed',1),
                               ('confirmation_seed',330261025), ('head_to_head_opponents',['round-40'])]:
                candidate = deepcopy(manifest); candidate[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    study.validate_manifest(candidate)
            candidate = deepcopy(manifest)
            candidate['source_artifacts_sha256'].pop(str(Path(manifest['source_training'])/'report.json'))
            with self.assertRaisesRegex(ValueError, 'pin all'):
                study.validate_manifest(candidate)

    @staticmethod
    def round_record(number, actor, critic):
        first = (number-1)*1000
        games = [dict(index=i, seed=300261025+1000+i, seat=i%2,
                      starting_player=f'P{(i//2)%2}', mode='versus_baseline',
                      status='win', trajectories=[]) for i in range(first, first+1000)]
        return dict(round=number, games=games,
            split=dict(training_indices=list(range(first,first+800)),
                       validation_indices=list(range(first+800,first+1000)),
                       training={'terminal_games':800}, validation={'attempted_games':200}),
            update=dict(accepted_steps=3, critic_steps=2,
                        optimizer_steps=actor, critic_optimizer_steps=critic))

    def prepared_prefix(self, root):
        manifest, source, output, _ = self.fixture(root)
        training, config, _ = study.prepare(manifest)
        report = study.read(training/'report.json')
        report['rounds'] = [self.round_record(i, i, 157+i) for i in range(1,41)]
        report['rounds'].append(self.round_record(41, 13292, 6439))
        study.publish(training/'report.json', report)
        old = training/'checkpoints/round-0040'
        new = training/'checkpoints/round-0041'; new.mkdir()
        for name in ('policy.keras','policy.manifest.json'):
            (new/name).write_bytes((old/name).read_bytes())
        state = study.read(old/'state.json')
        state.update(round=41, optimizer_steps=13292, critic_optimizer_steps=6439)
        study.publish(new/'state.json', state)
        np.savez(new/'optimizer.npz', v0=np.array(13292), v1=np.array([.25]))
        np.savez(new/'critic-optimizer.npz', v0=np.array(6439), v1=np.array([.125]))
        return manifest, training, report

    def test_committed_round41_counts_seeds_holdout_and_adam_continuity(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, training, report = self.prepared_prefix(Path(temporary))
            result = study.validate_training(training, 41)
            self.assertEqual(result['games'], 41000)
            self.assertEqual(result['added_games'], 1000)
            self.assertEqual(result['continuation_raytracer_games'], 1000)
            self.assertEqual(result['continuation_laptop_games'], 0)
            rows = report['rounds'][-1]['games']
            self.assertEqual((rows[0]['index'], rows[0]['seed']), (40000,300302025))
            self.assertEqual((rows[-1]['index'], rows[-1]['seed']), (40999,300303024))
            self.assertEqual((99000,300261025+1000+99000), (99000,300361025))
            self.assertEqual(300261025+1000+99999,300362024)
            for mutation in ('seed','duplicate','holdout','laptop','fallback','adam'):
                changed = deepcopy(report)
                last = changed['rounds'][-1]
                if mutation == 'seed': last['games'][0]['seed'] -= 1
                elif mutation == 'duplicate': last['games'][-1]['index'] = last['games'][0]['index']
                elif mutation == 'holdout': last['split']['validation_indices'][0] = last['split']['training_indices'][0]
                elif mutation == 'laptop': last['games'][0]['collection_host'] = 'laptop'
                elif mutation == 'fallback': last['games'][0]['pool_fallback'] = True
                else: last['update']['optimizer_steps'] -= 1
                study.publish(training/'report.json', changed)
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    study.validate_training(training, 41)

    def evaluation_result(self, checkpoint, *, opponent=None, seed=450261026, count=4):
        mode = 'versus_neural' if opponent is not None else 'versus_baseline'
        rows = [dict(index=i, seed=seed+i, seat=i%2, starting_player=f'P{(i//2)%2}',
                     mode=mode, status=status, trajectories=[])
                for i,status in enumerate(('win','loss','cutoff','error'))]
        result = dict(status='complete', checkpoint_sha256=study.sha(checkpoint),
            config={'cohort':{'seed_base':seed, 'seed_last':seed+count-1,
                             'attempted_games':count, 'index_base':0, 'training_trajectories':False},
                    'collection':{'greedy':False, 'max_own_turns':60,
                                  'max_decisions':6000, 'repeat_limit':20}}, games=rows,
            summary=dict(attempted_games=count,wins=1,losses=1,draws=0,cutoffs=1,errors=1))
        if opponent is not None:
            result['opponent_checkpoint_sha256'] = study.sha(opponent)
        return result

    def test_baseline_and_actual_neural_evaluation_validate_every_attempt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint, opponent = root/'candidate.keras', root/'opponent.keras'
            checkpoint.write_bytes(b'candidate'); opponent.write_bytes(b'opponent')
            for old in (None, opponent):
                result = self.evaluation_result(checkpoint, opponent=old)
                self.assertIs(study.validate_evaluation(result, checkpoint=checkpoint, seed=450261026,
                                                       games=4, opponent=old), result)
                for mutation in ('seed','policy','count','mode','trajectories','denominator'):
                    changed = deepcopy(result)
                    if mutation == 'seed': changed['games'][0]['seed'] += 1
                    elif mutation == 'policy': changed['checkpoint_sha256'] = '0'*64
                    elif mutation == 'count': changed['games'].pop()
                    elif mutation == 'mode': changed['games'][0]['mode'] = 'selfplay'
                    elif mutation == 'trajectories': changed['games'][0]['trajectories'] = [{'path':'bad'}]
                    else: changed['summary']['cutoffs'] = 0
                    with self.subTest(opponent=old, mutation=mutation), self.assertRaises(ValueError):
                        study.validate_evaluation(changed, checkpoint=checkpoint, seed=450261026,
                                                  games=4, opponent=old)

    def test_native_owner_mismatch_is_immediate_and_read_blips_have_monotonic_grace(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _, _, _ = self.fixture(Path(temporary))
            now = datetime.now(timezone.utc).isoformat()
            live = dict(manifest_sha256='a'*64, utc=now, **manifest['expected_native_owner'])
            study.publish(manifest['native_liveness'], live)
            self.assertTrue(study.native_guard_probe(manifest)['valid'])
            live['windows_pid'] += 1
            study.publish(manifest['native_liveness'], live)
            observation = study.native_guard_probe(manifest)
            self.assertTrue(observation['hard_failure'])
            self.assertEqual(observation['reason'], 'pinned_owner_identity_mismatch')
            clock = iter((0.,14.9,15.))
            monitor = study.NativeGuardMonitor(manifest, monotonic=lambda:next(clock), observer=lambda _:None)
            failure = dict(valid=False, reason='file_read_error', hard_failure=False)
            with patch.object(study, 'native_guard_probe', return_value=failure):
                self.assertFalse(monitor.check()); self.assertFalse(monitor.check())
                with self.assertRaises(study.NativeGuardFailure): monitor.check()

    def test_registered_smoke_is_small_terminal_balanced_and_uses_both_cpu_policies(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _, _, _ = self.fixture(Path(temporary))
            training, _, _ = study.prepare(manifest)
            checkpoint = training/'checkpoints/round-0040/policy.keras'
            opponent = training/'checkpoints/round-0010/policy.keras'
            rows = [dict(index=i, seed=470261026+i, seat=i%2, starting_player=f'P{(i//2)%2}',
                         mode='versus_neural', status='win' if i%2 else 'loss', trajectories=[])
                    for i in range(8)]
            result = dict(status='complete', checkpoint_sha256=study.sha(checkpoint),
                opponent_checkpoint_sha256=study.sha(opponent), games=rows,
                summary=dict(terminal_games=8, attempted_games=8,wins=4,losses=4,draws=0,cutoffs=0,errors=0,
                    seat_start_balanced=True, seat_start_strata={
                        f'seat_{seat}_start_{starter}':{'attempted_games':2}
                        for seat in (0,1) for starter in ('P0','P1')}),
                source_files_sha256={str(path.resolve()):study.sha(path)
                    for policy in (checkpoint,opponent) for path in (policy,policy.with_suffix('.manifest.json'))},
                game_engines=4, replicas_per_policy=1, inference_device='cpu', training=False,
                batching={role:{'requests':12, 'device':'/CPU:0', 'replicas':[{'visible_gpus':[]}]} for role in ('focal','opponent')})
            with patch.object(study, 'evaluate', return_value=result) as evaluate:
                observed = study.preflight_head_to_head(manifest, training)
                self.assertEqual(observed['status'], 'passed')
                self.assertEqual(evaluate.call_args.args[3:5], (470261026,8))
                self.assertEqual(evaluate.call_args.kwargs, {'opponent':opponent,'workers':4,'replicas_per_policy':1})
            for mutation in ('role','cutoff','balance','hash'):
                changed = deepcopy(result)
                if mutation == 'role': changed['batching']['opponent']['requests'] = 0
                elif mutation == 'cutoff': changed['summary']['cutoffs'] = 1
                elif mutation == 'balance': changed['summary']['seat_start_balanced'] = False
                else: changed['source_files_sha256'] = {}
                with self.subTest(mutation=mutation), patch.object(study,'evaluate',return_value=changed), \
                        self.assertRaises(ValueError):
                    study.preflight_head_to_head(manifest, training)

    def test_finite_orchestration_smoke_precedes_training_and_sealed_evaluations_follow_100k(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _, output, _ = self.fixture(Path(temporary))
            events = []
            def child(command, log, manifest):
                self.assertIn('--resume', command)
                self.assertNotIn('--pooled-collector-config', command)
                target = int(command[command.index('--rounds')+1])
                events.append(('train',target))
                study.publish(output/'training/checkpoints/latest.json', {'checkpoint':f'round-{target:04d}'})
            def frozen(manifest, checkpoint, output, seed, games, **kwargs):
                events.append(('evaluation',seed,games,kwargs.get('opponent')))
                return dict(summary={'attempted_games':games}, games=[], checkpoint_sha256='a',
                            opponent_checkpoint_sha256='b')
            def smoke(*_):
                events.append(('smoke',))
                return {'status':'passed'}
            previous = Path.cwd()
            try:
                with patch.object(study,'run_child',side_effect=child), \
                        patch.object(study,'preflight_head_to_head',side_effect=smoke), \
                        patch.object(study,'validate_training',side_effect=lambda _,n:{'games':n*1000}), \
                        patch.object(study,'evaluate',side_effect=frozen), \
                        patch('patchwork_automaton.neural.rustmage_evaluation.paired_match_improvement',return_value={'gain':0.}):
                    study.run(manifest)
            finally:
                os.chdir(previous)
            self.assertEqual(events[0], ('smoke',))
            self.assertEqual([event[1] for event in events if event[0]=='train'], list(study.MILESTONES))
            final_training_position = events.index(('train',100))
            sealed = [i for i,event in enumerate(events)
                      if event[0]=='evaluation' and event[1] in (450261026,460261026)]
            self.assertTrue(all(i>final_training_position for i in sealed))
            self.assertEqual(sum(event[:3]==('evaluation',450261026,10000) for event in events),2)
            self.assertEqual(sum(event[:3]==('evaluation',460261026,5000) for event in events),6)
            self.assertEqual(study.read(output/'study.json')['status'], 'complete')
            self.assertTrue(study.read(output/'cleanup.json')['shared_lock_released'])
            with Path(manifest['shared_lock']).open('a+') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                fcntl.flock(lock,fcntl.LOCK_UN)

    def test_terminal_receipt_requires_new_cleanup_lock_release_and_exact_native_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _, output, _ = self.fixture(Path(temporary))
            output.mkdir()
            owner = dict(pid=os.getpid(), pgid=os.getpid(), sid=os.getpid(),
                         start_ticks=123, uid=os.getuid())
            context = dict(receipt=manifest['native_terminal_receipt'], coordinator=owner,
                recovery_epoch=1, recovery_coordinator_sha256=manifest['recovery_coordinator_sha256'],
                native_owner=manifest['expected_native_owner'], native_manifest_sha256='a'*64,
                prior_cleanup_sha256=None, started_utc=study.utc())
            manifest['_native_owner'] = tuple(manifest['expected_native_owner'].values())
            cleanup = dict(coordinator_pid=os.getpid(), finished_utc=study.utc(),
                           status='complete', shared_lock_released=False)
            study.publish(output/'cleanup.json', cleanup)
            with patch.object(study, 'process_identity', return_value=owner), \
                    patch.object(study, 'verify_terminal_owned_sessions', return_value=[]):
                with self.assertRaisesRegex(ValueError, 'shared-lock release'):
                    study.publish_native_terminal_receipt(manifest, context, None)
                self.assertFalse(Path(manifest['native_terminal_receipt']).exists())
                cleanup['shared_lock_released'] = True
                study.publish(output/'cleanup.json', cleanup)
                manifest['_native_owner'] = (999,'2026-10-26T12:00:00Z')
                with self.assertRaisesRegex(ValueError, 'Native owner changed'):
                    study.publish_native_terminal_receipt(manifest, context, None)
                manifest['_native_owner'] = tuple(manifest['expected_native_owner'].values())
                receipt = study.publish_native_terminal_receipt(manifest, context, None)
                self.assertEqual(receipt['native_owner'], manifest['expected_native_owner'])
                self.assertTrue(receipt['cleanup']['shared_lock_released'])
                self.assertTrue(receipt['cleanup']['cleanup_verified'])
                self.assertEqual(receipt['cleanup']['owned_children_remaining'],0)
                with self.assertRaisesRegex(ValueError, 'already exists'):
                    study.publish_native_terminal_receipt(manifest, context, None)


if __name__ == '__main__':
    unittest.main()
