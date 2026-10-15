"""Typed live prompts and factual summaries, without strategy labels."""
from dataclasses import replace
import json
from pathlib import Path
import unittest
from patchwork_automaton.backends.xmage_prompts import prompt_metadata,prompt_source
from test_neural import AVAILABLE


class GameplayPromptTests(unittest.TestCase):
    def test_unpaid_cost_survives_source_click_but_not_next_priority(self):
        from types import SimpleNamespace
        from patchwork_automaton.backends.xmage import XMageGame
        source='00000000-0000-0000-0000-000000000001'
        game=XMageGame(SimpleNamespace(session_id='test'),'game')
        game._view=dict(myPlayerId='me',players=[dict(playerId='me',battlefield={source:dict(id=source,name='Sacred Foundry')})])
        game._offer('GAME_PLAY_MANA',dict(message='Pay {W}<div>spell</div>'))
        game._choice_source=source
        game._offer('GAME_CHOOSE_ABILITY',dict(choices={'00000000-0000-0000-0000-000000000002':'{T}: Add {W}.'}))
        metadata=prompt_metadata(game._decision.kind,game._decision.native)
        self.assertEqual(metadata['mana_cost']['white'],1)
        game._offer('GAME_SELECT',{})
        self.assertIsNone(game._payment_context)
    def test_captured_prompts_have_distinct_purposes(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/xmage_gameplay_prompts.json').read_text())
        purposes={prompt_metadata(p['method'],p['data'])['purpose'] for p in fixture['prompts']}
        self.assertEqual(purposes,{'priority','ability','attackers','shock_land','starting_player','mana_payment','mulligan','target'})
        shock=next(p for p in fixture['prompts'] if 'Sacred Foundry becomes tapped' in p['data']['message'])
        self.assertEqual(prompt_metadata(shock['method'],shock['data']),dict(purpose='shock_land',life_payment=2))
        self.assertIsNotNone(prompt_source(shock['data']))

    def test_payment_uses_only_unpaid_prefix_and_unknown_is_not_zero(self):
        debt=prompt_metadata('GAME_PLAY_MANA',dict(message="Pay {2}{R}{R}{W}<div>Card called {U}</div>"))['mana_cost']
        self.assertEqual(debt,dict(white=1,blue=0,black=0,red=2,green=0,colorless=0,generic=2))
        for text in ('Pay {X}','Pay {2/W}','Card mentions {R}'):
            self.assertEqual(prompt_metadata('GAME_PLAY_MANA',dict(message=text)),{'purpose':'mana_payment'})

    def test_source_ids_are_validated_and_ambiguous_sources_stay_unknown(self):
        self.assertIsNone(prompt_source(dict(message="<font object_id='not-a-uuid'>x</font>")))
        self.assertIsNone(prompt_source(dict(message="<font object_id='00000000-0000-0000-0000-000000000001'>a</font><font object_id='00000000-0000-0000-0000-000000000002'>b</font>")))


@unittest.skipUnless(AVAILABLE,'Install neural dependencies with uv')
class FactualFeatureTests(unittest.TestCase):
    def test_fallback_source_is_mana_without_guessing_which_dual_ability(self):
        import numpy as np
        from patchwork_automaton.neural.episodes import CastBoltEpisode
        from patchwork_automaton.neural.features import Encoder,OPTION_KINDS,ROLES
        episode=CastBoltEpisode(np.random.default_rng(5),payment_only=True,heldout=True)
        encoded=Encoder().encode(episode.observation())
        index=next(i for i,o in enumerate(encoded.options) if o.id=='object:'+episode.land)
        self.assertEqual(encoded.arrays['candidates'][index,len(OPTION_KINDS)+ROLES.index('mana')],1)
    def test_hand_counts_are_identity_independent_and_do_not_expose_hidden_data(self):
        import numpy as np
        from patchwork_automaton.neural.scenarios import sample
        from patchwork_automaton.neural.features import hand_summary,numeric,TYPES
        obs=sample('mulligan',np.random.default_rng(42)).observation
        hand=list(obs.objects.values())
        expected=sum((numeric(sum(t in o.properties['cardTypes'] for o in hand)) for t in TYPES),[])
        expected+=sum((numeric(sum(o.properties['manaValue']==v for o in hand)) for v in range(9)),[])
        np.testing.assert_array_equal(hand_summary(obs),expected)
        renamed=replace(obs,objects={h:replace(o,name='Other printing') for h,o in obs.objects.items()})
        np.testing.assert_array_equal(hand_summary(obs),hand_summary(renamed))
        h=next(iter(obs.objects))
        hidden=replace(obs,objects={**obs.objects,h:replace(obs.objects[h],properties=dict(obs.objects[h].properties,faceDown=True))})
        np.testing.assert_array_equal(hand_summary(hidden),0)

    def test_empty_history_preserves_nonzero_decision_memory(self):
        import numpy as np
        from patchwork_automaton.neural.batching import example_batch
        from patchwork_automaton.neural.policy import untrained_policy
        policy=untrained_policy(seed=30,width=16,memory_size=16,blocks=1)
        a=example_batch(16);a['advance_memory'][:]=0
        a['memory_in'][:]=.4
        out=policy.model(a)
        np.testing.assert_allclose(np.asarray(out['memory_out']),a['memory_in'])
        # With a real decision, differing memories must not both be reset by
        # the entirely masked history GRU.
        a['advance_memory'][:]=1
        first=np.asarray(policy.model(a)['memory_out'])
        a['memory_in'][:]=0
        second=np.asarray(policy.model(a)['memory_out'])
        self.assertGreater(float(np.max(np.abs(first-second))),.01)
