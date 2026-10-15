"""Payment mechanics, truthful cancellation recovery and two legal Blaze targets."""
import unittest
from test_neural import AVAILABLE
from patchwork_automaton.backends.xmage_choices import choices,encode_response


class PoolProtocolTests(unittest.TestCase):
    def test_pool_choices_bind_only_own_available_real_colors(self):
        from uuid import UUID
        me,other=map(lambda i:str(UUID(int=i)),(1,2))
        view=dict(myPlayerId=me,players=[dict(playerId=me,manaPool=dict(red=2,white=1,blue=0,
            green=-1,colorless=3,generic=4)),dict(playerId=other,manaPool=dict(black=10))])
        options,bindings=choices('GAME_PLAY_MANA',{},view)
        self.assertEqual({o.id for o in options},{'mana:RED','mana:WHITE','mana:COLORLESS','no'})
        self.assertEqual(encode_response(bindings['mana:RED'],None),
            ('sendPlayerManaType',dict(playerId=me,manaType='RED')))
        for method in ('GAME_SELECT','GAME_TARGET','GAME_PLAY_XMANA'):
            self.assertFalse(any(o.id.startswith('mana:') for o in choices(method,{},view)[0]))


@unittest.skipUnless(AVAILABLE,'Install neural dependencies with uv')
class PaymentTests(unittest.TestCase):
    def test_teachers_execute_all_payment_branches_and_encode_each_step(self):
        import numpy as np
        from patchwork_automaton.neural.curriculum import PAYMENT_TASKS,make_episode
        from patchwork_automaton.neural.features import Encoder
        from patchwork_automaton.core import Response
        branches=set()
        for heldout in (False,True):
            for task in PAYMENT_TASKS:
                for seed in range(30):
                    episode=make_episode(task,np.random.default_rng(seed),heldout=heldout)
                    branches.add(episode.branch)
                    if task=='spend_pool':
                        # Printed Charm costs two: a one-pip debt means another
                        # source already paid a pip in addition to floating mana.
                        paid=int(episode.spell_name=='Boros Charm')
                        self.assertEqual(sum(episode.pool.values())+paid,len(episode.board))
                    while not episode.terminal:
                        self.assertLess(episode.steps,16)
                        obs=episode.observation();Encoder().encode(obs)
                        winners=episode.oracle_options()
                        self.assertTrue(winners)
                        self.assertFalse(winners-{o.id for o in obs.decision.options})
                        reward=episode.step(Response(sorted(winners)[0]))
                        self.assertEqual(reward,1. if episode.terminal else None)
                    self.assertEqual(episode.reward,1.,(task,seed))
        self.assertIn('own_creature_prowess',branches)
        self.assertIn('kill_blocker',branches)
        self.assertIn('unpayable:payment',branches)

    def test_tapping_does_not_skip_the_manual_pool_step(self):
        import numpy as np
        from patchwork_automaton.neural.payment_scenarios import PaymentEpisode
        from patchwork_automaton.core import Response
        for seed in range(15):
            episode=PaymentEpisode('tap_then_spend',np.random.default_rng(seed))
            while episode.stage!='payment' or not any(episode.pool.values()):
                episode.step(Response(sorted(episode.oracle_options())[0]))
            self.assertFalse(episode.terminal)
            self.assertTrue(all(o.startswith('mana:') for o in episode.oracle_options()))
            self.assertEqual(episode.step(Response(sorted(episode.oracle_options())[0])),1.)

    def test_wrong_pool_color_makes_no_progress_but_can_recover(self):
        import numpy as np
        from patchwork_automaton.neural.payment_scenarios import PaymentEpisode
        from patchwork_automaton.core import Response
        episode=next(e for seed in range(30) if (e:=PaymentEpisode('spend_pool',np.random.default_rng(seed))).spell_name=='Boros Charm')
        debt=dict(episode.debt);pool=dict(episode.pool)
        wrong='WHITE' if episode.wanted=='red' else 'RED'
        for _ in range(3):
            self.assertIsNone(episode.step(Response('mana:'+wrong)))
            self.assertEqual(episode.debt,debt)
            self.assertEqual(episode.pool,pool)
        self.assertEqual(episode.step(Response('mana:'+episode.wanted.upper())),1.)

    def test_cancel_returns_the_card_and_pass_ends_retry_sequence(self):
        import numpy as np
        from patchwork_automaton.neural.payment_scenarios import PaymentEpisode
        from patchwork_automaton.core import Response
        episode=next(e for seed in range(30) if not (e:=PaymentEpisode('cancel_retry',np.random.default_rng(seed))).payable)
        if episode.stage=='payment':episode.step(Response('no'))
        for _ in range(3):
            episode.step(Response('object:'+episode.bolt))
            episode.step(Response('object:'+episode.other))
            self.assertIsNone(episode.step(Response('no')))
            self.assertEqual(episode.stage,'priority')
            self.assertIn(episode.bolt,episode.hand)
            self.assertFalse(episode.stack)
        self.assertEqual(episode.step(Response('no')),1.)

    def test_blaze_cancel_is_recoverable_and_wrong_creature_can_lose(self):
        import numpy as np
        from patchwork_automaton.neural.payment_scenarios import PaymentEpisode
        from patchwork_automaton.core import Response
        episode=next(e for seed in range(30) if (e:=PaymentEpisode('blaze_targets',np.random.default_rng(seed))).blocker)
        episode.step(Response('object:'+episode.bolt));episode.step(Response('object:'+episode.other))
        self.assertIsNone(episode.step(Response('no')))
        self.assertEqual(episode.board[episode.swift]['power'],'1')
        episode.step(Response('object:'+episode.bolt));episode.step(Response('object:'+episode.other))
        episode.step(Response('object:'+episode.swift))
        self.assertEqual(episode.step(Response('mana:RED')),-1.)

    def test_nonlethal_blaze_finishes_the_task_without_zeroing_life_or_killing_a_two_toughness_creature(self):
        import numpy as np
        from patchwork_automaton.neural.payment_scenarios import PaymentEpisode
        from patchwork_automaton.core import Response
        episode=next(e for seed in range(30) if (e:=PaymentEpisode('blaze_targets',np.random.default_rng(seed))).completion)
        life=episode.enemy_life;count=len(episode.nuisance)
        while not episode.terminal:episode.step(Response(sorted(episode.oracle_options())[0]))
        self.assertEqual(episode.reward,1.)
        self.assertEqual(episode.enemy_life,life-1)
        self.assertEqual(len(episode.nuisance),count)
        self.assertEqual(sum(c['damage'] for c in episode.nuisance.values()),1)

    def test_baseline_understands_the_new_pool_response(self):
        import numpy as np
        from patchwork_automaton.neural.payment_scenarios import PaymentEpisode
        from patchwork_automaton.baselines.modern_burn import BurnBaseline
        from patchwork_automaton.core import Response
        for seed in range(12):
            episode=PaymentEpisode('spend_pool',np.random.default_rng(seed))
            baseline=BurnBaseline()
            while not episode.terminal:
                self.assertLess(episode.steps,5)
                episode.step(baseline.choose(episode.observation()))
            self.assertEqual(episode.reward,1.)

    def test_pool_candidate_colors_are_numeric_and_reference_own_player(self):
        import numpy as np
        from patchwork_automaton.neural.payment_scenarios import PaymentEpisode
        from patchwork_automaton.neural.features import Encoder,ABILITY_ID_SLICE
        episode=next(e for seed in range(30) if (e:=PaymentEpisode('spend_pool',np.random.default_rng(seed))).spell_name=='Boros Charm')
        encoded=Encoder().encode(episode.observation())
        positions={o.id:i for i,o in enumerate(encoded.options)}
        red,white=[positions['mana:'+color] for color in ('RED','WHITE')]
        self.assertFalse(np.array_equal(encoded.arrays['candidates'][red],encoded.arrays['candidates'][white]))
        for index in (red,white):
            self.assertEqual(encoded.arrays['candidate_refs'][index,0],1.)
            self.assertEqual(encoded.arrays['candidate_refs'][index].sum(),1.)
            # Protocol actions reuse unknown ability ID, not a fabricated card ID.
            self.assertEqual(encoded.arrays['candidates'][index,ABILITY_ID_SLICE][0],1.)
