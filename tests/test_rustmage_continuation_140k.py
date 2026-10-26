"""Strict finite continuation fixtures; no games, neural models or network."""
from copy import deepcopy
import fcntl
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

from examples import run_rustmage_continuation_140k as study
from patchwork_automaton.neural.rustmage_evaluation import baseline_evaluation_config,match_summary


class Continuation140kTests(unittest.TestCase):
    def fixture(self,root):
        source,output,runtime=root/'source',root/'new-study',root/'runtime'
        (runtime/'examples').mkdir(parents=True)
        shutil.copyfile(study.safe.__file__,runtime/'examples/run_rustmage_continuation.py')
        shutil.copyfile(study.__file__,runtime/'examples/run_rustmage_continuation_140k.py')
        config=deepcopy(study.LEARNING_CONFIG);rng=np.random.default_rng(123).bit_generator.state
        for n in range(101):
            cp=source/'checkpoints'/f'round-{n:04d}';cp.mkdir(parents=True)
            with zipfile.ZipFile(cp/'policy.keras','w') as archive:archive.writestr('synthetic-weights',str(n).encode())
            (cp/'policy.manifest.json').write_text('{"synthetic":true}\n')
            actor,critic=(32822,15857) if n==100 else (n,157+n)
            np.savez(cp/'optimizer.npz',v0=np.array(actor),v1=np.array([.1,.2]))
            np.savez(cp/'critic-optimizer.npz',v0=np.array(critic),v1=np.array([.3,.4]))
            study.publish(cp/'state.json',dict(round=n,config=config,optimizer_steps=actor,
                critic_optimizer_steps=critic,trainer_rng_state=rng,preserved_history={'arbitrary':[n,'bytes']}))
        study.publish(source/'checkpoints/latest.json',{'checkpoint':'round-0100'})
        study.publish(source/'report.json',dict(status='complete',config=config,
            rounds=[dict(round=n) for n in range(1,101)],critic_warmup={'status':'complete','requested_games':1000},
            invocations=[{'source':'preserve'}],training_games_completed=100000))
        imitation=root/'imitation.keras';imitation.write_bytes(b'imitation');imitation.with_suffix('.manifest.json').write_text('{}')
        pins={str(p):study.sha(p) for p in source.rglob('*') if p.is_file()}
        pins.update({str(p):study.sha(p) for p in [imitation,imitation.with_suffix('.manifest.json')]})
        manifest=dict(run_id='synthetic-140k',python=sys.executable,runtime=str(runtime),output=str(output),
            source_training=str(source),shared_lock=str(root/'training.lock'),native_liveness=str(root/'native.json'),
            native_terminal_receipt=str(root/'terminal.json'),imitation_checkpoint=str(imitation),
            expected_native_owner=dict(windows_pid=123,creation_utc='2026-10-26T23:00:00Z'),
            native_manifest_sha256='a'*64,recovery_epoch=1,recovery_coordinator_sha256=study.sha(study.__file__),
            source_round=100,target_round=140,target_games=140000,additional_training_games=40000,
            training_seed=300261025,milestones=list(study.MILESTONES),monitoring_rounds=list(study.MONITORING_ROUNDS),
            monitoring_seed=320261025,monitoring_games=2000,confirmation_seed=520261026,confirmation_games=10000,
            head_to_head_seed=530261026,head_to_head_games=5000,head_to_head_opponents=list(study.HEAD_TO_HEAD_NAMES),
            preflight_head_to_head_seed=540261026,preflight_head_to_head_games=8,
            runtime_options=deepcopy(study.RUNTIME_OPTIONS),hard_deadline_utc=None,
            source_sha256={str(p.relative_to(runtime)):study.sha(p) for p in runtime.rglob('*') if p.is_file()},
            source_artifacts_sha256=pins)
        return manifest,source,output

    def test_all101_checkpoint_files_and_report_copied_byte_identically(self):
        with tempfile.TemporaryDirectory() as temporary:
            m,source,output=self.fixture(Path(temporary));study.validate_manifest(m)
            before={str(p.relative_to(source)):study.sha(p) for p in source.rglob('*') if p.is_file()}
            training,config,fork=study.prepare(m)
            self.assertEqual(config,study.LEARNING_CONFIG)
            self.assertEqual(fork['changed_config_fields'],[])
            for rel,digest in before.items():self.assertEqual(study.sha(training/rel),digest)
            self.assertEqual(before,{str(p.relative_to(source)):study.sha(p) for p in source.rglob('*') if p.is_file()})
            self.assertEqual(study.read(training/'checkpoints/round-0100/state.json')['optimizer_steps'],32822)
            self.assertEqual(study.read(training/'checkpoints/round-0100/state.json')['critic_optimizer_steps'],15857)
            self.assertTrue(fork['preservation']['all101_checkpoint_files_byte_identical'])
            self.assertTrue(fork['preservation']['no_critic_rewarm'])
            self.assertEqual(study.prepare(m),(training,config,fork))

    def test_strict_recipe_plan_and_new_source_hash_pins(self):
        with tempfile.TemporaryDirectory() as temporary:
            m,_,_=self.fixture(Path(temporary))
            for key,value in [('source_round',40),('target_round',141),('target_games',140001),
                ('additional_training_games',True),('training_seed',1),('confirmation_seed',450261026),
                ('head_to_head_seed',460261026),('preflight_head_to_head_seed',470261026),
                ('head_to_head_opponents',['round-100']),('hard_deadline_utc','tomorrow'),
                ('recovery_coordinator_sha256',study.SAFE_HELPER_SHA256)]:
                changed=deepcopy(m);changed[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):study.validate_manifest(changed)
            for key,value in [('pooled_collector_config','/pool.json'),('train_batch',1024),('seed',1),
                ('value_learning_rate',.001),('critic_warmup_games',0)]:
                changed=deepcopy(study.LEARNING_CONFIG);changed[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):study.validate_config(changed)
            changed=deepcopy(m);changed['source_artifacts_sha256'].pop(str(Path(m['source_training'])/'checkpoints/round-0055/state.json'))
            with self.assertRaisesRegex(ValueError,'All101'):study.validate_manifest(changed)

    def test_partial_preparation_or_existing_tampering_fail_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            m,source,output=self.fixture(Path(temporary));staging=output/'training.prepare';staging.mkdir(parents=True)
            (staging/'preserve').write_bytes(b'partial')
            with self.assertRaisesRegex(ValueError,'Partial preparation'):study.prepare(m)
            self.assertEqual((staging/'preserve').read_bytes(),b'partial')
            shutil.rmtree(staging)
            training,_,_=study.prepare(m);p=training/'checkpoints/round-0007/state.json';p.write_text('{}')
            with self.assertRaisesRegex(ValueError,'Historical checkpoint bytes differ'):study.prepare(m)
            self.assertEqual(p.read_text(),'{}')

    def test_source_mutation_and_wrong_boundary_counters_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            m,source,_=self.fixture(Path(temporary));path=source/'checkpoints/round-0100/state.json'
            state=study.read(path);state['optimizer_steps']=32821;study.publish(path,state)
            with self.assertRaisesRegex(ValueError,'source changed'):study.verify_sources(m)
            m['source_artifacts_sha256'][str(path)]=study.sha(path)
            with self.assertRaisesRegex(ValueError,'boundary counters'):study.prepare(m)

    def test_training_command_preserves_recipe_and_has_no_reinitialization_or_pool(self):
        command=study.train_command(sys.executable,'/new/training',study.LEARNING_CONFIG,140,study.RUNTIME_OPTIONS)
        self.assertIn('--resume',command)
        self.assertNotIn('--initialize-from',command);self.assertNotIn('--pooled-collector-config',command)
        for flag,value in [('--rounds','140'),('--seed','300261025'),('--train-batch','512'),
            ('--critic-warmup-games','1000'),('--workers','48'),('--inference-processes','4'),
            ('--update-pipeline','gpu'),('--policy-cache-mib','8192'),('--eval-games','200')]:
            self.assertEqual(command[command.index(flag)+1],value)
        for invalid in [100,102,139,141,True]:
            with self.subTest(invalid=invalid),self.assertRaises(ValueError):
                study.train_command(sys.executable,'/new',study.LEARNING_CONFIG,invalid,study.RUNTIME_OPTIONS)

    @staticmethod
    def historical_round(n):
        start=(n-1)*1000
        games=[dict(index=i,seed=300261025+1000+i,seat=i%2,starting_player=f'P{(i//2)%2}',
            mode='versus_baseline',status='win') for i in range(start,start+1000)]
        return dict(round=n,games=games,split=dict(training_indices=list(range(start,start+800)),
            validation_indices=list(range(start+800,start+1000)),training={'terminal_games':800},validation={'attempted_games':200}))

    def add_round101(self,m,training,*,report_lag=False):
        report=study.read(training/'report.json');report['rounds']=[self.historical_round(n) for n in range(1,101)]
        before=study.read(training/'checkpoints/round-0100/state.json');state=deepcopy(before)
        row=self.historical_round(101);rng=np.random.default_rng();rng.bit_generator.state=before['trainer_rng_state']
        for _ in range(4):rng.permutation(800)
        row.update(update=dict(samples=800,accepted_steps=4,critic_steps=2,optimizer_steps=32826,
            critic_optimizer_steps=15859,early_stop=False),collection_resources={})
        state.update(round=101,optimizer_steps=32826,critic_optimizer_steps=15859,
            trainer_rng_state=rng.bit_generator.state,round_result=row)
        cp=training/'checkpoints/round-0101';cp.mkdir()
        for fn in ['policy.keras','policy.manifest.json']:shutil.copyfile(training/'checkpoints/round-0100'/fn,cp/fn)
        np.savez(cp/'optimizer.npz',v0=np.array(32826),v1=np.array([.1,.2]))
        np.savez(cp/'critic-optimizer.npz',v0=np.array(15859),v1=np.array([.3,.4]))
        study.publish(cp/'state.json',state);study.publish(training/'checkpoints/latest.json',{'checkpoint':'round-0101'})
        if not report_lag:report['rounds'].append(row)
        study.publish(training/'report.json',report)
        return report,row,state

    def test_round101_exact_schedule_holdout_rng_and_adam_continuity(self):
        with tempfile.TemporaryDirectory() as temporary:
            m,_,_=self.fixture(Path(temporary));training,_,_=study.prepare(m);report,row,state=self.add_round101(m,training)
            result=study.validate_training(training,101)
            self.assertEqual(result['games'],101000);self.assertEqual(result['added_games'],1000)
            self.assertEqual(result['continuation_raytracer_games'],1000);self.assertEqual(result['continuation_laptop_games'],0)
            self.assertEqual(row['games'][0]['index'],100000);self.assertEqual(row['games'][0]['seed'],300362025)
            self.assertEqual(row['games'][-1]['seed'],300363024)
            self.assertEqual(300261025+1000+139999,300402024)
            self.assertEqual(result['actor_optimizer_steps'],32826);self.assertEqual(result['critic_optimizer_steps'],15859)
            for mutation in ['seed','duplicate','holdout','laptop','fallback','adam','rng']:
                changed=deepcopy(report);cpstate=deepcopy(state);last=changed['rounds'][-1]
                if mutation=='seed':last['games'][0]['seed']+=1
                elif mutation=='duplicate':last['games'][-1]['index']=last['games'][0]['index']
                elif mutation=='holdout':last['split']['validation_indices'][0]=last['split']['training_indices'][0]
                elif mutation=='laptop':last['games'][0]['collection_host']='laptop'
                elif mutation=='fallback':last['games'][0]['pool_fallback']=True
                elif mutation=='adam':last['update']['optimizer_steps']-=1
                else:cpstate['trainer_rng_state']=np.random.default_rng(999).bit_generator.state
                study.publish(training/'report.json',changed);study.publish(training/'checkpoints/round-0101/state.json',cpstate)
                with self.subTest(mutation=mutation),self.assertRaises(ValueError):study.validate_training(training,101)
            study.publish(training/'report.json',report);study.publish(training/'checkpoints/round-0101/state.json',state)

    def test_checkpoint_recoverable_report_lag_and_uncommitted_rollout_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            m,_,_=self.fixture(Path(temporary));training,_,_=study.prepare(m);self.add_round101(m,training,report_lag=True)
            self.assertEqual(study.validate_training(training,101)['games'],101000)
            rollout=training/'rollouts/game-101000-p0.pkl.gz';rollout.parent.mkdir();rollout.write_bytes(b'preserve')
            with self.assertRaisesRegex(ValueError,'Uncommitted rollouts'):study._reject_uncommitted_rollouts(training,101)
            self.assertEqual(rollout.read_bytes(),b'preserve')

    def eval_fixture(self,checkpoint,*,opponent=None,seed=520261026):
        config=baseline_evaluation_config(seed_base=seed,games=4,max_own_turns=60,greedy=False)
        mode='versus_neural' if opponent else 'versus_baseline'
        rows=[dict(index=i,seed=seed+i,seat=i%2,starting_player=f'P{(i//2)%2}',mode=mode,
            status=status,trajectories=[],winner=(i%2 if status=='win' else 1-i%2) if status in ['win','loss'] else None)
            for i,status in enumerate(['win','loss','cutoff','error'])]
        summary=match_summary([dict(row,mode='versus_baseline') for row in rows],expected_games=4)
        if opponent:
            collection=dict(config['collection']);collection.pop('mode');collection.pop('output')
            config=dict(collection=collection,cohort=config['cohort'],mode='versus_neural',
                policy='sampled temperature 1',focal_seat='index % 2',starting_player='P((index // 2) % 2)',
                decks=["foundations_deck('red')","foundations_deck('red')"])
            summary.update(mode=mode,metric='focal_recorded_wins_per_scheduled_attempt')
        result=dict(status='complete',config=config,summary=summary,games=rows,
            checkpoint_sha256=study.sha(checkpoint),training=False)
        if opponent:result['opponent_checkpoint_sha256']=study.sha(opponent)
        return result

    def test_full_outcome_denominators_actual_duel_mode_and_winner_semantics(self):
        with tempfile.TemporaryDirectory() as temporary:
            cp,op=Path(temporary)/'cp.keras',Path(temporary)/'old.keras';cp.write_bytes(b'new');op.write_bytes(b'old')
            for opponent in [None,op]:
                result=self.eval_fixture(cp,opponent=opponent)
                self.assertIs(study.validate_evaluation(result,checkpoint=cp,seed=520261026,games=4,opponent=opponent),result)
                for mutation in ['winner','cohort','outcomes','mode','trajectory','opponent']:
                    changed=deepcopy(result)
                    if mutation=='winner':changed['games'][0]['winner']=1
                    elif mutation=='cohort':changed['config']['cohort']['replace_failed_attempts']=True
                    elif mutation=='outcomes':changed['summary']['cutoffs']=0
                    elif mutation=='mode':changed['games'][0]['mode']='goldfish'
                    elif mutation=='trajectory':changed['games'][0]['trajectories']=[{'reward':1}]
                    elif opponent:changed['opponent_checkpoint_sha256']='f'*64
                    else:changed['checkpoint_sha256']='f'*64
                    with self.subTest(opponent=opponent,mutation=mutation),self.assertRaises(ValueError):
                        study.validate_evaluation(changed,checkpoint=cp,seed=520261026,games=4,opponent=opponent)

    def test_registered_smoke_requires_both_cpu_roles_and_unchanged_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            m,_,_=self.fixture(Path(temporary));training,_,_=study.prepare(m)
            cp=training/'checkpoints/round-0040/policy.keras';op=training/'checkpoints/round-0010/policy.keras'
            result=dict(summary={'terminal_games':8,'errors':0,'cutoffs':0},
                game_engines=4,replicas_per_policy=1,inference_device='cpu',training=False,
                source_files_sha256={str(p.resolve()):study.sha(p) for policy in [cp,op]
                    for p in [policy,policy.with_suffix('.manifest.json')]},
                batching={role:dict(requests=8,device='/CPU:0',replicas=[{'visible_gpus':[]}])
                    for role in ['focal','opponent']})
            with patch.object(study,'evaluate',return_value=result) as evaluate:
                self.assertEqual(study.preflight(m,training)['status'],'passed')
                self.assertEqual(evaluate.call_args.args[3:5],(540261026,8))
                self.assertEqual(evaluate.call_args.kwargs,dict(opponent=op,workers=4,replicas_per_policy=1))
            for kind in ['role','gpu','cutoff','hash']:
                changed=deepcopy(result)
                if kind=='role':changed['batching']['opponent']['requests']=0
                elif kind=='gpu':changed['batching']['focal']['replicas'][0]['visible_gpus']=['GPU:0']
                elif kind=='cutoff':changed['summary']['cutoffs']=1
                else:changed['source_files_sha256']={}
                with self.subTest(kind=kind),patch.object(study,'evaluate',return_value=changed),self.assertRaises(ValueError):
                    study.preflight(m,training)

    def test_new_wrapper_preserves_run_failure_and_reports_receipt_failure(self):
        manifest={'run_id':'fixture'};context={'coordinator':{'pid':123}}
        with patch.object(study,'validate_manifest'),patch.object(study,'terminal_receipt_context',return_value=context),\
             patch.object(study,'run',side_effect=RuntimeError('original failure')),\
             patch.object(study,'publish_native_terminal_receipt',side_effect=ValueError('no verified cleanup')) as receipt,\
             patch('sys.stderr'):
            with self.assertRaisesRegex(RuntimeError,'original failure'):study.run_with_terminal_receipt(manifest)
        self.assertEqual(receipt.call_args.args[1],context)
        self.assertEqual(receipt.call_args.args[2],{'type':'RuntimeError','message':'original failure'})
        self.assertEqual(study.safe.SOURCE_ROUND,40);self.assertEqual(study.safe.TARGET_ROUND,100)
        self.assertEqual(study.sha(study.safe.__file__),study.SAFE_HELPER_SHA256)

    def test_new_terminal_context_pins_new_module_not_old_helper(self):
        with tempfile.TemporaryDirectory() as temporary:
            m,_,_=self.fixture(Path(temporary));owner=dict(pid=os.getpid(),pgid=os.getpid(),sid=os.getpid(),start_ticks=5,uid=os.getuid())
            class Guard:
                last_probe={'expected_manifest_sha256':'a'*64}
                def require_fresh(self):m['_native_owner']=(123,'2026-10-26T23:00:00Z')
            with patch.object(study,'process_identity',return_value=owner),patch.object(study,'native_guard_monitor',return_value=Guard()):
                context=study.terminal_receipt_context(m)
                self.assertEqual(context['recovery_coordinator_sha256'],study.sha(study.__file__))
                self.assertNotEqual(context['recovery_coordinator_sha256'],study.SAFE_HELPER_SHA256)
                m['recovery_coordinator_sha256']=study.SAFE_HELPER_SHA256
                with self.assertRaisesRegex(ValueError,'New coordinator'):study.terminal_receipt_context(m)

    def test_finite19_child_plan_and_final_cohorts_only_after140k(self):
        with tempfile.TemporaryDirectory() as temporary:
            m,_,output=self.fixture(Path(temporary));events=[]
            def child(command,log,manifest):
                target=int(command[command.index('--rounds')+1]);events.append(('train',target))
                self.assertIn('--resume',command);self.assertNotIn('--initialize-from',command)
                self.assertNotIn('--pooled-collector-config',command)
                training=output/'training';study.publish(training/'checkpoints/latest.json',{'checkpoint':f'round-{target:04d}'})
                report=study.read(training/'report.json');report['rounds']=[dict(round=n) for n in range(1,target+1)]
                study.publish(training/'report.json',report)
            def frozen(manifest,checkpoint,path,seed,games,**kwargs):
                events.append(('evaluation',seed,games,str(kwargs.get('opponent'))))
                return dict(summary={'attempted_games':games},games=[],checkpoint_sha256='a',opponent_checkpoint_sha256='b')
            def smoke(*args):events.append(('smoke',540261026,8));return {'status':'passed'}
            before=Path.cwd()
            try:
                with patch.object(study,'run_child',side_effect=child),patch.object(study,'preflight',side_effect=smoke),\
                     patch.object(study,'validate_training',side_effect=lambda _,n:{'games':n*1000}),\
                     patch.object(study,'evaluate',side_effect=frozen),\
                     patch('patchwork_automaton.neural.rustmage_evaluation.paired_match_improvement',return_value={'gain':0}):
                    study.run(m)
            finally:os.chdir(before)
            self.assertEqual(events[0],('smoke',540261026,8))
            self.assertEqual([e[1] for e in events if e[0]=='train'],list(study.MILESTONES))
            last_train=events.index(('train',140))
            self.assertTrue(all(i>last_train for i,e in enumerate(events) if e[0]=='evaluation' and e[1] in [520261026,530261026]))
            self.assertEqual(sum(e[:3]==('evaluation',320261025,2000) for e in events),4)
            self.assertEqual(sum(e[:3]==('evaluation',520261026,10000) for e in events),2)
            self.assertEqual(sum(e[:3]==('evaluation',530261026,5000) for e in events),7)
            self.assertEqual(len(events),19)
            self.assertEqual(40000+sum(e[2] for e in events if e[0] in ['evaluation','smoke'])+5*200,104008)
            self.assertEqual(study.read(output/'study.json')['status'],'complete')
            self.assertTrue(study.read(output/'cleanup.json')['shared_lock_released'])
            with Path(m['shared_lock']).open('a+') as lock:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_failed_owned_cleanup_keeps_visible_failure_and_never_publishes_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            m,_,output=self.fixture(Path(temporary));before=Path.cwd()
            try:
                with patch.object(study,'preflight',side_effect=RuntimeError('smoke failed')),\
                     patch.object(study,'verify_terminal_owned_sessions',side_effect=RuntimeError('owned child remains')):
                    with self.assertRaisesRegex(RuntimeError,'owned child remains'):study.run(m)
            finally:os.chdir(before)
            cleanup=study.read(output/'cleanup.json');self.assertFalse(cleanup['shared_lock_released'])
            self.assertIn('owned child remains',cleanup['cleanup_error'])
            self.assertFalse(Path(m['native_terminal_receipt']).exists())


if __name__=='__main__':unittest.main(verbosity=2)
