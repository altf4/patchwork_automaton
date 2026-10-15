"""Live prompt bindings and limited synthetic Burn opening objectives."""
import json
from pathlib import Path
import unittest
from dataclasses import replace
from patchwork_automaton.backends.xmage_prompts import prompt_metadata


class PromptTests(unittest.TestCase):
    def test_user_heuristic_boundaries(self):
        from patchwork_automaton.baselines.burn_mulligan import keep_hand
        land=dict(cardTypes=['LAND'],manaValue=0)
        cheap=dict(cardTypes=['INSTANT'],manaValue=1)
        two=dict(cardTypes=['INSTANT'],manaValue=2)
        for n in range(8):
            self.assertEqual(keep_hand([land]*n+[cheap]*(7-n)),1<=n<=3)
        self.assertTrue(keep_hand([land]+[two]*2+[cheap]*4))
        self.assertFalse(keep_hand([land]+[two]*3+[cheap]*3))
        for n in (2,3):self.assertTrue(keep_hand([land]*n+[two]*(7-n)))
        # At four or below keep anything, including zero or all lands.
        for size in range(5):
            self.assertTrue(keep_hand([cheap]*size))
            self.assertTrue(keep_hand([land]*size))
            self.assertTrue(keep_hand([two]*size))
        # Exactly MV 2 is counted.
        self.assertTrue(keep_hand([land,dict(cardTypes=['SORCERY'],manaValue=3)]))

    def test_captured_xmage_sequence_has_correct_sizes_and_purposes(self):
        fixture=json.loads((Path(__file__).parent/'fixtures/xmage_mulligan_prompts.json').read_text())
        values=[prompt_metadata(p['method'],p['data']) for p in fixture['prompts']]
        self.assertEqual(values,[dict(purpose='mulligan',next_hand_size=6),
            dict(purpose='bottom_hand',bottom_remaining=1),dict(purpose='mulligan',next_hand_size=5),
            dict(purpose='bottom_hand',bottom_remaining=2),dict(purpose='bottom_hand',bottom_remaining=1),
            dict(purpose='mulligan',next_hand_size=4)])

    def test_unrecognized_prompts_remain_unknown(self):
        buttons={'UI.left.btn.text':'Mulligan','UI.right.btn.text':'Keep'}
        for method,data in [('GAME_ASK',dict(message='Pay 2 life?',options=buttons)),
                            ('GAME_ASK',dict(message='Mulligan down to 6 cards?',options={}))]:
            self.assertEqual(prompt_metadata(method,data),{'purpose':'unknown'})
        self.assertEqual(prompt_metadata('GAME_TARGET',dict(message='Select a card (1 more) to put on the bottom of your library',options={'targetZone':'GRAVEYARD'})),{'purpose':'target'})
        free=dict(message='Mulligan <font color=#00ff00>for free</font>, draw another 7 cards?',options=buttons)
        self.assertEqual(prompt_metadata('GAME_ASK',free),dict(purpose='mulligan',next_hand_size=7))


from test_neural import AVAILABLE

@unittest.skipUnless(AVAILABLE,'Install neural dependencies with uv')
class OpeningScenarioTests(unittest.TestCase):
    def test_native_layout_contains_visible_sideboard_and_matches_runtime_history(self):
        import numpy as np
        from patchwork_automaton.neural.scenarios import sample
        from patchwork_automaton.neural.history import History
        from patchwork_automaton.neural.features import Encoder
        from examples.train_scenarios import batch_for
        rng=np.random.default_rng(516);s=sample('mulligan',rng,distribution='native')
        obs=s.observation
        self.assertEqual(sum(o.zone=='sideboard' for o in obs.objects.values()),15)
        self.assertFalse(any(o.zone=='command' for o in obs.objects.values()))
        self.assertEqual(obs.turn,1)
        for o in obs.objects.values():
            if o.zone=='hand':
                self.assertIn('color',o.properties);self.assertIn('canAttack',o.properties)
        encoder=Encoder();expected=History(encoder.registry);expected.observe(obs)
        events=expected.take(obs)
        batch=batch_for([s],encoder,16,runtime_history=True)
        for key,value in events.items():np.testing.assert_array_equal(batch[key][0],value)
    def test_diverse_hands_use_only_the_actual_main_deck_with_copy_limits(self):
        from collections import Counter
        import numpy as np
        from patchwork_automaton.baselines.modern_burn import DECK_METADATA
        from patchwork_automaton.baselines.burn_mulligan import keep_hand
        from patchwork_automaton.neural.scenarios import sample
        quantities=Counter({c['name']:c['quantity'] for c in DECK_METADATA['mainboard']})
        seen=set()
        for heldout in (False,True):
            rng=np.random.default_rng(126)
            for task in ('mulligan','mulligan_bottom'):
                for _ in range(200):
                    s=sample(task,rng,heldout=heldout,distribution='diverse')
                    cards=list(s.observation.objects.values());counts=Counter(c.name for c in cards);seen.update(counts)
                    self.assertFalse(counts-quantities)
                    if task=='mulligan':
                        self.assertEqual(s.winning_options,frozenset(['no' if keep_hand(c.properties for c in cards) else 'yes']))
        self.assertEqual(seen,set(quantities))
    def test_opening_oracles_depend_on_visible_hand_and_allow_equivalent_cards(self):
        import numpy as np
        from patchwork_automaton.neural.scenarios import sample
        seen=set()
        for heldout in (False,True):
            rng=np.random.default_rng(9)
            for task in ('mulligan','mulligan_bottom'):
                for _ in range(100):
                    s=sample(task,rng,heldout=heldout);obs=s.observation;seen.add(s.variant)
                    hand=list(obs.objects.values());lands=[c for c in hand if 'LAND' in c.properties['cardTypes']]
                    self.assertTrue(s.winning_options)
                    self.assertEqual(len(hand),next(p.hand_count for p in obs.players if p.id==obs.actor))
                    if task=='mulligan':
                        two=sum(c.properties['manaValue']==2 and 'LAND' not in c.properties['cardTypes'] for c in hand)
                        keep=len(hand)<=4 or len(lands) in (2,3) or (len(lands)==1 and two<=2)
                        self.assertEqual(s.winning_options,frozenset(['no' if keep else 'yes']))
                    else:
                        for option in obs.decision.options:
                            card=next(c for c in hand if 'object:'+c.id.value==option.id)
                            expected='LAND' in card.properties['cardTypes'] if len(lands)>=3 else card.properties['manaValue']==2
                            self.assertEqual(s.reward(option.id)>0,expected)
        self.assertEqual(len(seen),9)

    def test_bottom_sequence_removes_each_selected_card_and_reaches_terminal(self):
        import numpy as np
        from patchwork_automaton.neural.scenarios import sample,advance_bottom
        rng=np.random.default_rng(15);long_sequence=False
        for _ in range(100):
            scenario=sample('mulligan_bottom',rng)
            count=prompt_metadata(scenario.observation.decision.kind,scenario.observation.decision.native)['bottom_remaining']
            long_sequence |= count>1
            steps=0
            while scenario is not None:
                before=scenario.observation
                chosen=next(iter(scenario.winning_options));scenario=advance_bottom(scenario,chosen);steps+=1
                if scenario:
                    after=scenario.observation
                    self.assertEqual(len(after.objects),len(before.objects)-1)
                    self.assertNotIn(chosen,{o.id for o in after.decision.options})
                    self.assertEqual(after.decision.token.generation,before.decision.token.generation+1)
                    self.assertEqual(after.revision,before.revision+1)
            self.assertEqual(steps,count)
        self.assertTrue(long_sequence)

    def test_prompt_meaning_and_counts_are_numeric_inputs_not_labels(self):
        import numpy as np
        from patchwork_automaton.neural.scenarios import sample
        from patchwork_automaton.neural.features import Encoder
        obs=sample('mulligan',np.random.default_rng(3)).observation
        original=Encoder().encode(obs).arrays
        renamed=replace(obs,decision=replace(obs.decision,options=tuple(replace(o,label='display only') for o in obs.decision.options)))
        for k,v in original.items():np.testing.assert_array_equal(v,Encoder().encode(renamed).arrays[k])
        changed=replace(obs,decision=replace(obs.decision,native=dict(obs.decision.native,message='Pay 2 life?')))
        self.assertFalse(np.array_equal(original['context'],Encoder().encode(changed).arrays['context']))
        changed=replace(obs,decision=replace(obs.decision,native=dict(obs.decision.native,message='Mulligan down to 1 card?')))
        self.assertFalse(np.array_equal(original['context'],Encoder().encode(changed).arrays['context']))
