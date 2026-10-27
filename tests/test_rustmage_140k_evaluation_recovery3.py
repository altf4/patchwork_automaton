"""Synthetic epoch3 adoption/replay fixtures only; no models,games or network."""
from copy import deepcopy
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

BASE=Path(__file__).resolve().parents[1]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


previous=load('examples.run_rustmage_140k_evaluation_recovery',BASE/'examples/run_rustmage_140k_evaluation_recovery.py')
old_tests=load('epoch2_fixture_tests',BASE/'tests/test_rustmage_140k_evaluation_recovery.py')
study=load('epoch3_fixture_module',BASE/'examples/run_rustmage_140k_evaluation_recovery3.py')
from patchwork_automaton.neural.rustmage_evaluation import baseline_evaluation_config,match_summary


def result(checkpoint,seed,games,opponent=None):
    mode='versus_neural'if opponent else'versus_baseline'
    config=baseline_evaluation_config(seed_base=seed,games=games,max_own_turns=60,greedy=False)
    if opponent:
        collection=dict(config['collection']);collection.pop('mode');collection.pop('output')
        config=dict(collection=collection,cohort=config['cohort'],mode=mode,policy='sampled temperature 1',
            focal_seat='index % 2',starting_player='P((index // 2) % 2)',decks=["foundations_deck('red')","foundations_deck('red')"])
    rows=[dict(index=i,seed=seed+i,seat=i%2,starting_player=f'P{(i//2)%2}',mode=mode,status='win'if i%2 else'loss',
        winner=i%2 if i%2 else 1-i%2,trajectories=[])for i in range(games)]
    summary=match_summary([dict(r,mode='versus_baseline')for r in rows],expected_games=games)
    if opponent:summary.update(mode=mode,metric='focal_recorded_wins_per_scheduled_attempt')
    d=dict(status='complete',checkpoint=str(checkpoint),checkpoint_sha256=study.sha(checkpoint),config=config,
        games=rows,summary=summary,training=False)
    if opponent:d.update(opponent_checkpoint=str(opponent),opponent_checkpoint_sha256=study.sha(opponent))
    return d


class Epoch3Tests(unittest.TestCase):
    @contextmanager
    def fixture(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);m2,training,prior=old_tests.EvaluationRecoveryTests().fixture(root/'chain')
            prior.mkdir();runtime=root/'runtime3';shutil.copytree(m2['runtime'],runtime)
            shutil.copyfile(study.__file__,runtime/'examples/run_rustmage_140k_evaluation_recovery3.py')
            m=deepcopy(m2);m.update(run_id=study.RUN_ID,parent_run_id=study.PARENT_RUN_ID,prior_recovery_run_id=study.PRIOR_RUN_ID,
                recovery_epoch=3,runtime=str(runtime),output=str(root/'output3'),prior_recovery_output=str(prior),
                registered_evaluation_attempts=57000,adopted_evaluation_attempts=27000,remaining_evaluation_attempts=30000,
                owned_child_plan=6,recovery_coordinator_sha256=study.sha(study.__file__),
                native_terminal_receipt=str(root/'terminal3.json'),native_liveness=str(root/'live3.json'))
            m['source_sha256']={str(p.relative_to(runtime)):study.sha(p)for p in runtime.rglob('*')if p.is_file()}
            adopted={}
            for stem in study.ADOPTED:
                cp,seed,games,opponent=study.evaluation_specs(m)[stem];path=prior/(stem+'.json');d=result(cp,seed,games,opponent)
                study.publish(path,d);adopted[stem]=dict(path=str(path),sha256=study.sha(path),seed=seed,attempts=games,
                    checkpoint_sha256=d['checkpoint_sha256'])
                if opponent:adopted[stem]['opponent_checkpoint_sha256']=d['opponent_checkpoint_sha256']
            m['adopted_results']=adopted
            study.publish(prior/'report.json',dict(run_id=study.PRIOR_RUN_ID,recovery_epoch=2,status='interrupted',
                source_snapshot=study.source_snapshot(m),completed_evaluations=adopted))
            owner=dict(pid=900003,pgid=900003,sid=900003,start_ticks=3,uid=os.getuid())
            cleanup=dict(coordinator_pid=owner['pid'],status='interrupted',shared_lock_released=True,
                owned_children_remaining=0,owned_process_records_checked=5,finished_utc='2026-10-27T03:30:51Z')
            study.publish(prior/'cleanup.json',cleanup)
            for i in range(5):
                child=dict(pid=900010+i,pgid=900010+i,sid=900010+i,start_ticks=4+i,uid=os.getuid())
                study.publish(prior/f'old-{i}.process.json',dict(owner=child))
                study.publish(prior/f'old-{i}.cleanup.json',dict(owner=child,remaining=[],child_exit=-2 if i==4 else 0,
                    verified_utc='2026-10-27T03:30:51Z'))
            pm=root/'epoch2-manifest.json';pl=root/'epoch2-launch.json';pr=root/'epoch2-terminal.json'
            study.publish(pm,m2);study.publish(pl,dict(owner=owner))
            study.publish(pr,dict(run_id=study.PRIOR_RUN_ID,recovery_epoch=2,status='interrupted',coordinator=owner,
                recovery_coordinator_sha256=study.PREVIOUS_SHA256,native_owner=m2['expected_native_owner'],
                native_manifest_sha256=m2['native_manifest_sha256'],cleanup_sha256=study.sha(prior/'cleanup.json'),
                cleanup=dict(cleanup,cleanup_verified=True)))
            m.update(prior_recovery_manifest=str(pm),prior_recovery_launch_record=str(pl),prior_recovery_terminal_receipt=str(pr))
            log=prior/'head-to-head-round-10-interrupted.log';log.write_text('toy5000complete; interrupted before durable write\n')
            m['replay_provenance']=dict(result_stem='head-to-head-round-10',seed=530261026,index_base=0,attempts=5000,
                completed_physical_attempts=5000,durable_result_present=False,same_ids_replayed=True,
                counts_as_independent_extra_sample=False,log_path=str(log),log_sha256=study.sha(log))
            m['source_artifacts_sha256'].update({str(p):study.sha(p)for p in prior.rglob('*')if p.is_file()})
            m['source_artifacts_sha256'].update({str(p):study.sha(p)for p in [pm,pl,pr]})
            # Production pins the actual reviewed interrupted log. Only this
            # toy fixture substitutes its own known immutable log digest.
            with patch.object(study,'REPLAY_LOG_SHA256',study.sha(log)):yield m,training,prior,Path(m['output'])

    def test_strict_scope_exact_four_adoptions_and_replay_semantics(self):
        self.assertEqual(study.REPLAY_LOG_SHA256,'23e41164329ae66c5e52efce72743d7481dcb1b1fa05062928765a7ad2f0f8c6')
        with self.fixture()as(m,_,_,_):
            study.validate_manifest(m);study.verify_sources(m)
            for key,value in [('recovery_epoch',2),('training',True),('registered_evaluation_attempts',62000),
                ('remaining_evaluation_attempts',35000),('owned_child_plan',7),('adopted_evaluation_attempts',32000)]:
                changed=deepcopy(m);changed[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):study.validate_manifest(changed)
            changed=deepcopy(m);changed['adopted_results'].pop('monitor-0140')
            with self.assertRaisesRegex(ValueError,'four durable'):study.validate_manifest(changed)
            changed=deepcopy(m);changed['replay_provenance']['counts_as_independent_extra_sample']=True
            with self.assertRaisesRegex(ValueError,'replay provenance'):study.validate_manifest(changed)

    def test_paused_owner_receipt_and_any_durable_or_partial_remaining_result_block(self):
        with self.fixture()as(m,_,prior,_):
            proof=study.verify_prior_stopped(m);self.assertEqual(proof['prior_owned_records_checked'],5)
            owner=study.read(m['prior_recovery_launch_record'])['owner']
            with patch.object(study,'process_identity',return_value=owner),self.assertRaisesRegex(ValueError,'remains live'):
                study.verify_prior_stopped(m)
            for suffix in ['.json','.json.tmp']:
                path=prior/('head-to-head-round-10'+suffix);path.write_text('{}')
                with self.subTest(suffix=suffix),self.assertRaisesRegex(ValueError,'durable/partial'):study.verify_prior_stopped(m)
                path.unlink()

    def test_adoption_is_validated_byte_identical_and_sources_unchanged(self):
        with self.fixture()as(m,_,prior,output):
            output.mkdir();before={p:study.sha(p)for p in m['source_artifacts_sha256']}
            proof=study.adopt_results(m);self.assertEqual(proof['adopted_evaluation_attempts'],27000)
            for stem in study.ADOPTED:self.assertEqual(study.sha(output/(stem+'.json')),m['adopted_results'][stem]['sha256'])
            self.assertEqual(before,{p:study.sha(p)for p in before});self.assertEqual(study.adopt_results(m),proof)
            (output/'monitor-0140.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'adoption bytes'):study.adopt_results(m)

    def test_partial_adoption_and_changed_source_are_preserved_fail_closed(self):
        with self.fixture()as(m,_,prior,output):
            output.mkdir();staging=output/'adoption.prepare';staging.mkdir();(staging/'partial').write_bytes(b'preserve')
            with self.assertRaisesRegex(ValueError,'Partial adoption'):study.adopt_results(m)
            self.assertEqual((staging/'partial').read_bytes(),b'preserve')
            source=prior/'confirmation-round-100.json';source.write_text('{}')
            with self.assertRaisesRegex(ValueError,'Immutable original artifact'):study.verify_sources(m)

    def test_only_six_new_duels_admitted_no_durable_training_smoke_or_passive_replay(self):
        with self.fixture()as(m,_,prior,output):
            commands=[];before=Path.cwd();pins=dict(m['source_artifacts_sha256'])
            def child(command,log,manifest):
                commands.append(command);self.assertEqual(command[command.index('-m')+1],'examples.evaluate_rustmage_head_to_head')
                arg=lambda flag:command[command.index(flag)+1]
                self.assertEqual(arg('--seed'),'530261026');self.assertEqual(arg('--games'),'5000')
                self.assertIn(Path(arg('--output')).stem.removeprefix('head-to-head-'),study.NEW_OPPONENTS)
                cp=Path(arg('--checkpoint'));op=Path(arg('--opponent-checkpoint'));study.publish(arg('--output'),result(cp,530261026,5000,op))
                i=len(commands);pid=900100+i;owner=dict(pid=pid,pgid=pid,sid=pid,start_ticks=study.process_identity(os.getpid())['start_ticks']+i,uid=os.getuid())
                study.publish(Path(log).with_suffix('.process.json'),dict(owner=owner,command=command))
                study.publish(Path(log).with_suffix('.cleanup.json'),dict(owner=owner,remaining=[],child_exit=0,verified_utc='now'))
            try:
                with patch.object(study.phase,'run_child',side_effect=child),\
                     patch('patchwork_automaton.neural.rustmage_evaluation.paired_match_improvement',
                         return_value={'attempted_pairs':10000,'fixture':'existing pure statistics covered separately'}):
                    study.run(m)
            finally:os.chdir(before)
            self.assertEqual(len(commands),6);self.assertEqual(pins,{p:study.sha(p)for p in pins})
            report=study.read(output/'report.json');self.assertEqual(report['status'],'complete')
            self.assertEqual(report['completed_evaluation_attempts'],57000);self.assertEqual(report['completed_new_evaluation_attempts'],30000)
            self.assertEqual(report['physical_recovery_evaluation_attempts_including_interrupted_replay'],62000)
            self.assertEqual(report['continuation_100k_to140k_and_evaluations_physical_executions'],109008)
            self.assertEqual(len(report['completed_evaluations']),10)
            self.assertEqual(study.read(output/'cleanup.json')['owned_process_records_checked'],6)
            self.assertFalse((output/'training').exists())

    def test_new_terminal_context_is_epoch3_own_hash_and_native_six_duel_budget(self):
        with self.fixture()as(m,_,_,_):
            owner=dict(pid=123,pgid=123,sid=123,start_ticks=4,uid=os.getuid())
            class Guard:
                last_probe={'expected_manifest_sha256':'b'*64}
                def require_fresh(self):m['_native_owner']=(123,'2026-10-27T04:00:00Z')
            live=dict(run_id=study.RUN_ID,recovery_epoch=3,recovery_coordinator_sha256=study.sha(study.__file__),
                finite_game_budget=140000,starting_game_count=140000,additional_game_budget=0,remaining_evaluation_attempts=30000)
            study.publish(m['native_liveness'],live)
            with patch.object(study,'process_identity',return_value=owner),patch.object(study.safe,'native_guard_monitor',return_value=Guard()):
                context=study.terminal_receipt_context(m);self.assertEqual(context['recovery_epoch'],3)
                self.assertEqual(context['recovery_coordinator_sha256'],study.sha(study.__file__))
                live['remaining_evaluation_attempts']=57000;study.publish(m['native_liveness'],live)
                with self.assertRaisesRegex(ValueError,'six-duel'):study.terminal_receipt_context(m)


if __name__=='__main__':unittest.main(verbosity=2)
