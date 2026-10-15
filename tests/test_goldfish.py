"""Goldfish metric counts own turns and retains failures alongside win speed."""
from dataclasses import replace
import unittest

from test_burn_baseline import obs,option,card,ME,OTHER
from patchwork_automaton.core import Option
from patchwork_automaton.baselines.passive import PassivePolicy,passive_deck
from patchwork_automaton.errors import UnsupportedInteraction
from patchwork_automaton.rl.goldfish import OwnTurnCounter,RepeatGuard,decision_state_key,summarize


class GoldfishTests(unittest.TestCase):
    def test_repeat_guard_ignores_revisions_and_map_order_but_preserves_progress(self):
        state=dict(revision=1,turn=3,phase='precombat_main',players=[dict(id='p',life=20)],
            objects=[dict(id='b',zone='hand'),dict(id='a',zone='stack')])
        options=[dict(id='object:b',kind='uuid'),dict(id='no',kind='boolean')]
        key=decision_state_key(state,'GAME_SELECT',options,dict(purpose='priority'))
        reordered=dict(state,revision=99,objects=list(reversed(state['objects'])))
        self.assertEqual(key,decision_state_key(reordered,'GAME_SELECT',list(reversed(options)),dict(purpose='priority')))
        self.assertNotEqual(key,decision_state_key(dict(state,players=[dict(id='p',life=19)]),'GAME_SELECT',options,dict(purpose='priority')))
        self.assertNotEqual(key,decision_state_key(dict(state,turn=4),'GAME_SELECT',options,dict(purpose='priority')))
        self.assertNotEqual(key,decision_state_key(state,'GAME_PLAY_MANA',options,dict(purpose='mana_payment')))
        stack=dict(state,objects=[dict(id='a',zone='stack'),dict(id='b',zone='stack')])
        self.assertNotEqual(decision_state_key(stack,'GAME_SELECT',options,{}),
            decision_state_key(dict(stack,objects=list(reversed(stack['objects']))),'GAME_SELECT',options,{}))
        guard=RepeatGuard(2)
        self.assertFalse(guard.observe(key));self.assertFalse(guard.observe(key));self.assertTrue(guard.observe(key))
        self.assertFalse(guard.observe(b'different state'))
        disabled=RepeatGuard(0)
        self.assertFalse(any(disabled.observe(key) for _ in range(100)))

    def test_counts_actual_own_turns_not_priority_prompts_global_turn_or_seat(self):
        for actor in (ME,OTHER):
            counter=OwnTurnCounter(actor)
            counter.observe(replace(obs(turn=1),actor=actor,active_player=actor,phase=None))
            self.assertEqual(counter.turns,0)  # Pregame starting-player prompt.
            for turn in (2,2,4,4,5):
                # Turn five represents an extra consecutive turn; parity is
                # deliberately insufficient for this metric.
                state=replace(obs(turn=turn),actor=actor,active_player=actor)
                counter.observe(state)
                counter.observe(replace(state,actor=ME if actor==OTHER else OTHER))
            counter.observe(replace(state,turn=6,active_player=ME if actor==OTHER else OTHER))
            self.assertEqual(counter.turns,3)
            self.assertEqual(counter.turn_ids,{2,4,5})

    def test_nonwins_do_not_disappear_or_receive_fictitious_win_turns(self):
        rows=[dict(status='win',own_turns_to_win=4),dict(status='win',own_turns_to_win=6),
            dict(status='cutoff',own_turns_observed=20),dict(status='loss'),dict(status='draw'),dict(status='error')]
        result=summarize(rows)
        self.assertEqual(result['mean_own_turns_to_win'],5)
        self.assertEqual(result['median_own_turns_to_win'],5)
        self.assertEqual(result['p90_own_turns_to_win'],6)
        self.assertEqual(result['win_rate'],2/6)
        self.assertEqual(result['failed_attempts'],4)
        self.assertEqual(result['failure_rate'],4/6)
        self.assertEqual(result['wins_by_own_turn']['4'],1/6)
        self.assertEqual(result['wins_by_own_turn']['6'],2/6)
        self.assertEqual(result['win_turn_histogram'],{'4':1,'6':1})
        self.assertEqual(result['cutoff'],1)
        self.assertIsNone(summarize([dict(status='cutoff')])['mean_own_turns_to_win'])
        self.assertIsNone(summarize([])['win_rate'])

    def test_passive_never_plays_even_an_available_land_or_creature(self):
        policy=PassivePolicy()
        state=obs(objects=(card('Mountain','land'),card('Goblin Guide','guide')),
            options=(option('land'),option('guide'),Option('no','boolean','Pass')))
        self.assertEqual(policy.choose(state).option,'no')
        with self.assertRaises(UnsupportedInteraction):
            policy.choose(replace(state,decision=replace(state.decision,options=(option('land'),))))
        self.assertEqual(sum(c.quantity for c in passive_deck().cards),60)

    def test_passive_keeps_and_performs_required_cleanup_discard(self):
        policy=PassivePolicy()
        keep=obs('GAME_ASK',options=(Option('yes','boolean','Mulligan'),Option('no','boolean','Keep')))
        self.assertEqual(policy.choose(keep).option,'no')
        discard=obs('GAME_TARGET',options=(option('z'),option('a'),Option('no','boolean','Cancel')))
        self.assertEqual(policy.choose(discard).option,'object:a')
