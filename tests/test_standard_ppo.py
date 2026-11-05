"""Standard PPO schedule, player labels, frozen recording and retention safety."""
from collections import Counter
from pathlib import Path
import pickle
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from patchwork_automaton.rl.standard_ppo import schedule,normalization,validate_collection,cleanup_rollouts,play_ppo_game
from patchwork_automaton.rl.standard_imitation import timed_standard_game
from patchwork_automaton.neural.rollout_records import Sample,TrajectoryWriter,read_trajectory


class StandardPPOTests(unittest.TestCase):
    def test_exact_ten_round_budget_stratified_split_and_learner_roles(self):
        ids=set();seeds=set()
        for number in range(1,11):
            rows=schedule(10080,round_id=number)
            self.assertEqual(len(rows),10080)
            self.assertEqual(Counter(r['partition'] for r in rows),dict(training=8064,validation=2016))
            self.assertEqual(set(Counter(r['learner_deck'] for r in rows).values()),{3360})
            self.assertEqual(set(Counter(tuple(r['pairing']) for r in rows).values()),{1680})
            strata=Counter((tuple(r['pairing']),r['learner_deck'],r['learner_seat'],r['starting_player'],r['partition']) for r in rows)
            for key,count in strata.items():
                mirror=key[0][0]==key[0][1]
                self.assertEqual(count,(336 if mirror else 168) if key[-1]=='training' else (84 if mirror else 42))
            self.assertFalse(ids & {r['index'] for r in rows});self.assertFalse(seeds & {r['seed'] for r in rows})
            ids.update(r['index'] for r in rows);seeds.update(r['seed'] for r in rows)
        self.assertEqual(len(ids),100800)

    def test_hundred_thousand_games_use_rotating_balanced_thousand_game_rounds(self):
        ids=set();seeds=set();pairs=Counter();strata=Counter()
        for number in range(1,101):
            rows=schedule(round_id=number)
            self.assertEqual(len(rows),1000)
            self.assertEqual({g['index'] for g in rows},set(range((number-1)*1000,number*1000)))
            self.assertEqual(Counter(g['partition'] for g in rows),dict(training=800,validation=200))
            counts=Counter(tuple(g['pairing']) for g in rows)
            self.assertEqual(set(counts.values()),{166,167})
            pairs.update(counts)
            self.assertLessEqual(max(pairs.values())-min(pairs.values()),1)
            self.assertFalse(ids & {g['index'] for g in rows})
            self.assertFalse(seeds & {g['seed'] for g in rows})
            ids.update(g['index'] for g in rows);seeds.update(g['seed'] for g in rows)
            for g in rows:
                self.assertEqual(g['decks'][g['learner_seat']],g['learner_deck'])
                strata[(tuple(g['pairing']),g['partition'],g['learner_deck'],g['learner_seat'],g['starting_player'])]+=1
            for pair in counts:
                mirror=pair[0]==pair[1]
                for partition in ('training','validation'):
                    values=[n for (p,t,*_),n in strata.items() if p==pair and t==partition]
                    self.assertEqual(len(values),4 if mirror else 8)
                    self.assertLessEqual(max(values)-min(values),1)
        self.assertEqual(len(ids),100000)
        self.assertEqual(len(seeds),100000)
        self.assertEqual(set(pairs.values()),{16666,16667})

    def test_bad_sizes_rejected(self):
        for count in (0,120,10000):
            with self.assertRaises(ValueError):schedule(count)

    def test_driver_defaults_match_hundred_thousand_game_request(self):
        from examples.train_standard_ppo import parser
        args=parser().parse_args(['--output','unused','--source','unused.keras'])
        self.assertEqual((args.rounds,args.games_per_round),(100,1000))
        self.assertEqual(args.milestones,list(range(10,101,10)))

    def test_labels_version_and_learner_seat_are_validated(self):
        planned=schedule(240)
        rows=[]
        for g in planned:
            rows.append(dict(g,status='win',winner_seat=0,behavior_version=0,
                trajectories=[dict(player=g['learner_seat'],deck=g['learner_deck'],
                    reward=1. if g['learner_seat']==0 else -1.)]))
        validate_collection(rows,planned,0)
        rows[0]['trajectories'][0]['reward']*=-1
        with self.assertRaises(ValueError):validate_collection(rows,planned,0)

    def test_frozen_opponent_metadata_is_required_and_validated(self):
        planned=schedule(240)
        planned[0].update(opponent_kind='frozen_neural',
            opponent_checkpoint_sha256='a'*64,opponent_version=200)
        rows=[dict(g,status='win',winner_seat=g['learner_seat'],behavior_version=200,
            trajectories=[dict(player=g['learner_seat'],deck=g['learner_deck'],reward=1.)])
            for g in planned]
        validate_collection(rows,planned,200)
        rows[0]['opponent_checkpoint_sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'schedule'):
            validate_collection(rows,planned,200)
        rows[0]['opponent_checkpoint_sha256']='a'*64
        planned[0]['opponent_checkpoint_sha256']='not-a-hash'
        with self.assertRaisesRegex(ValueError,'SHA256'):
            validate_collection(rows,planned,200)

    def test_frozen_opponent_game_resets_both_inference_pipes(self):
        class Remote:
            broken = False
            def __init__(self): self.resets=[]
            def reset(self,seed): self.resets.append(seed)
        game=schedule(240)[0]
        digest='a'*64
        game.update(opponent_kind='frozen_neural',opponent_checkpoint_sha256=digest,
            opponent_version=200)
        learner=Remote()
        opponent=Remote()
        with patch('patchwork_automaton.rl.standard_imitation.timed_standard_game',
                   return_value={'status':'win'}) as run:
            result=play_ppo_game(seed=game['seed'],index=game['index'],game=game,
                model=object(),batcher=learner,version=200,
                opponent_model=object(),opponent_batcher=opponent,
                opponent_version=200,opponent_sha256=digest)
        self.assertEqual(result['status'],'win')
        self.assertEqual(learner.resets,[game['seed']])
        self.assertEqual(opponent.resets,[game['seed']])
        self.assertEqual(run.call_args.kwargs['opponent_sha256'],digest)
        with patch('patchwork_automaton.rl.standard_imitation.timed_standard_game',
                   side_effect=RuntimeError('game failure')):
            with self.assertRaisesRegex(RuntimeError,'game failure'):
                play_ppo_game(seed=game['seed'],index=game['index'],game=game,
                    model=object(),batcher=learner,version=200,
                    opponent_model=object(),opponent_batcher=opponent,
                    opponent_version=200,opponent_sha256=digest)
        self.assertEqual(learner.resets,[game['seed']]*2)
        self.assertEqual(opponent.resets,[game['seed']]*2)
        with self.assertRaisesRegex(ValueError,'hash mismatch'):
            play_ppo_game(seed=game['seed'],index=game['index'],game=game,
                model=object(),batcher=learner,version=200,
                opponent_model=object(),opponent_batcher=opponent,
                opponent_version=200,opponent_sha256='b'*64)

    def test_normalization_uses_only_supplied_training_outcomes(self):
        games=[dict(trajectories=[dict(samples=2,reward=1.,value_sum=0.,value_square_sum=.5)])]
        result=normalization(games)
        self.assertAlmostEqual(result['center'],1.)
        self.assertEqual(result['training_choices'],2)
        self.assertEqual(result['scale'],1.)

    def test_cleanup_preserves_current_previous_and_uncommitted_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for n in range(1,6):
                p=root/f'round-{n:04d}';p.mkdir();(p/'trajectory').write_text('keep')
                if n!=2:(p/'committed.json').write_text('{}')
            (root/'unrelated').mkdir()
            self.assertEqual(cleanup_rollouts(root,4),[1])
            self.assertTrue(all((root/f'round-{n:04d}').exists() for n in (2,3,4,5)))

    def test_numeric_records_roundtrip_and_no_tensorflow_in_worker(self):
        import sys
        self.assertNotIn('tensorflow',sys.modules)
        with tempfile.TemporaryDirectory() as directory:
            writer=TrajectoryWriter(Path(directory)/'sample.pkl.gz')
            sample=Sample(dict(candidate_mask=np.ones(2,bool)),np.zeros(96,np.float32),True,1,.25,np.log([.3,.7]).astype(np.float32))
            writer.write(sample);writer.close()
            restored=list(read_trajectory(writer.path))[0]
            self.assertEqual(restored.action,1)
            np.testing.assert_array_equal(restored.memory,sample.memory)
            self.assertEqual(writer.value_sum,.25)
            self.assertNotIn('tensorflow',sys.modules)

    def test_timeout_recovers_learner_prefix_and_assigns_draw_once(self):
        import sys
        if 'tensorflow' in sys.modules:self.skipTest('Supervisor requires a TensorFlow-free test process')
        with tempfile.TemporaryDirectory() as directory:
            game=schedule(240)[0];seat=game['learner_seat']
            def blocked(game,**options):
                writer=TrajectoryWriter(Path(options['output'])/f"game-{game['index']:06d}-p{seat}.pkl.gz")
                writer.write(Sample(dict(candidate_mask=np.ones(2,bool)),np.zeros(96,np.float32),True,0,.2,np.log([.5,.5]).astype(np.float32)))
                writer.file.flush();time.sleep(30)
            with patch('patchwork_automaton.rl.standard_imitation.play_standard_game',side_effect=blocked):
                row=timed_standard_game(game,diagnostics=directory,output=Path(directory)/'rollouts',
                    record_ppo=True,learner_seat=seat,game_timeout=.15)
            self.assertEqual(row['status'],'draw');self.assertTrue(row['adjudicated'])
            self.assertEqual(len(row['trajectories']),1)
            t=row['trajectories'][0]
            self.assertEqual((t['reward'],t['samples'],t['player']),(0.,1,seat))
            self.assertAlmostEqual(t['value_sum'],.2)
            self.assertEqual(len(list(read_trajectory(t['path']))),1)


if __name__=='__main__':unittest.main()
