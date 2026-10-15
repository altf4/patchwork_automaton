"""Scripted multi-prompt goals have no intermediate rewards or oracle inputs."""
import unittest
from test_neural import AVAILABLE


@unittest.skipUnless(AVAILABLE,'Install neural dependencies with uv')
class EpisodeTests(unittest.TestCase):
    def test_charm_counterfactual_changes_debt_not_card_or_board_identity(self):
        import numpy as np
        from patchwork_automaton.neural.episodes import CastBoltEpisode
        from patchwork_automaton.neural.features import Encoder
        left=CastBoltEpisode(np.random.default_rng(12),payment_only=True,mana_color='red',charm=True)
        right=CastBoltEpisode(np.random.default_rng(12),payment_only=True,mana_color='white',charm=True)
        a=Encoder().encode(left.observation()).arrays;b=Encoder().encode(right.observation()).arrays
        for key in a.keys()-{'context'}:np.testing.assert_array_equal(a[key],b[key])
        self.assertEqual(np.count_nonzero(a['context']!=b['context']),2)
    def test_charm_uses_the_actual_remaining_white_debt(self):
        import numpy as np
        from patchwork_automaton.neural.episodes import CastBoltEpisode
        from patchwork_automaton.backends.xmage_prompts import prompt_metadata
        episode=CastBoltEpisode(np.random.default_rng(3),payment_only=True,mana_color='white')
        self.assertEqual(prompt_metadata(episode.method,episode.data)['mana_cost']['white'],1)
        while not episode.terminal:episode.step(self.oracle_response(episode))
        self.assertEqual(episode.reward,1.)
        self.assertEqual(episode.enemy_life,8)
    def oracle_response(self,episode,*,target_self=False):
        from patchwork_automaton.core import Response
        stage=episode.stage
        if stage=='priority':
            option='no' if episode.unpayable else 'object:'+(episode.land if episode.land in episode.hand else episode.bolt)
        elif stage=='shock':option='yes'
        elif stage=='target':option='object:'+(episode.me if target_self else episode.other)
        elif stage=='payment':option='object:'+episode.land
        else:option='object:'+(episode.white if episode.wanted=='white' else episode.red)
        return Response(option)

    def test_full_casts_and_payments_are_terminal_only_and_protocol_encodable(self):
        import numpy as np
        from patchwork_automaton.neural.episodes import CastBoltEpisode
        from patchwork_automaton.neural.features import Encoder
        from patchwork_automaton.backends.xmage_prompts import prompt_metadata
        seen=set()
        for heldout in (False,True):
            for payment_only in (False,True):
                for seed in range(15):
                    episode=CastBoltEpisode(np.random.default_rng(seed),heldout=heldout,payment_only=payment_only)
                    self.assertEqual(episode.observation(),episode.observation())
                    generations=[]
                    while not episode.terminal:
                        obs=episode.observation();generations.append(obs.decision.token.generation)
                        Encoder().encode(obs);seen.add(episode.stage)
                        if episode.stage=='ability':
                            self.assertEqual(prompt_metadata(obs.decision.kind,obs.decision.native)['mana_cost']['red'],1)
                        reward=episode.step(self.oracle_response(episode))
                        self.assertEqual(reward,1. if episode.terminal else None)
                    self.assertEqual(episode.enemy_life,6 if heldout else 9)
                    self.assertEqual(len(episode.graveyard),1)
                    self.assertTrue(episode.board[episode.land]['tapped'])
                    self.assertEqual(generations,list(range(1,len(generations)+1)))
        self.assertEqual(seen,{'priority','shock','target','payment','ability'})

    def test_unpayable_pass_and_wrong_target_goals(self):
        import numpy as np
        from patchwork_automaton.core import Response
        from patchwork_automaton.neural.episodes import CastBoltEpisode
        for action,expected in [('pass',1.),('cast',-1.)]:
            episode=CastBoltEpisode(np.random.default_rng(3),unpayable=True)
            self.assertEqual(episode.step(Response('no' if action=='pass' else 'object:'+episode.bolt)),expected)
        episode=CastBoltEpisode(np.random.default_rng(4))
        while not episode.terminal:episode.step(self.oracle_response(episode,target_self=True))
        self.assertEqual(episode.reward,-1.)
        self.assertEqual(episode.enemy_life,12)

    def test_wrong_color_cannot_resolve_and_unoffered_actions_are_rejected(self):
        import numpy as np
        from patchwork_automaton.core import Response
        from patchwork_automaton.neural.episodes import CastBoltEpisode
        episode=CastBoltEpisode(np.random.default_rng(5),heldout=True,payment_only=True)
        with self.assertRaises(ValueError):episode.step(Response('invented'))
        self.assertIsNone(episode.step(Response('object:'+episode.land)))
        self.assertEqual(episode.stage,'ability')
        self.assertIsNone(episode.step(Response('object:'+episode.white)))
        self.assertEqual({o.id for o in episode.observation().decision.options},{'no','mana:WHITE'})
        self.assertEqual(episode.step(Response('no')),-1.)
