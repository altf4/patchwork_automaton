"""Shared aggro intent translated into the two engines' different action menus."""
from dataclasses import replace
import unittest

from patchwork_automaton.baselines.foundations import FoundationsAggro
from patchwork_automaton.core import (Decision, DecisionToken, Handle, ObjectView,
                                    Observation, Option, PlayerView)
from test_burn_baseline import obs, card, option, ME, OTHER


class FoundationsPolicyTests(unittest.TestCase):
    def test_benchmark_throughput_retains_failed_attempt_time(self):
        from examples.compare_engines import summary
        common = dict(setup_seconds=1, play_seconds=8, cleanup_seconds=1, decisions=100)
        rows = [dict(common, status='completed', seconds=10, turn=12, winner_color='red'),
                dict(common, status='cutoff', seconds=20)]
        result = summary(rows)
        self.assertEqual((result['attempts'], result['completed'], result['failures']), (2, 1, 1))
        self.assertEqual(result['games_per_minute'], 2)  # One completion / all 30 seconds.
        self.assertEqual(result['mean_global_turns'], 12)

    def test_main_phase_affordability_land_first_and_card_order(self):
        p = FoundationsAggro()
        objects = (card('Mountain', 'land'), card('Mountain', 'mana', zone='battlefield'),
                   card('Fanatical Firebrand', 'one'), card('Firespitter Whelp', 'three'),
                   card('Burst Lightning', 'burst'))
        choices = tuple(option(k) for k in ('land', 'mana', 'one', 'three', 'burst')) + (Option('no', 'boolean', 'Pass'),)
        state = obs(objects=objects, options=choices)
        self.assertEqual(p.choose(state).option, 'object:land')
        state = replace(state, decision=replace(state.decision, options=choices[1:]))
        self.assertEqual(p.choose(state).option, 'object:burst')  # Same one-mana/name ordering as Rust.
        self.assertEqual(p.choose(replace(state, active_player=OTHER)).option, 'no')
        self.assertEqual(p.choose(replace(state, phase='end_combat')).option, 'no')
        stack = card('Burst Lightning', 'stack', zone='stack')
        self.assertEqual(p.choose(replace(state, objects={**state.objects, stack.id: stack})).option, 'no')
        state = replace(state, decision=replace(state.decision, options=(option('three'), Option('no', 'boolean', 'Pass'))))
        self.assertEqual(p.choose(state).option, 'no')

    def test_rust_translation_selects_the_same_affordable_card_and_does_not_float_mana(self):
        actor = Handle('rustmage', 'player', 'P0'); game = Handle('rustmage', 'game', 'game')
        token = DecisionToken('rustmage', 'session', game, actor, 1)
        cards = {str(i): ObjectView(Handle('rustmage', 'object', str(i)), n, z, actor, {'tapped': False})
                 for i, n, z in ((1, 'Mountain', 'battlefield'), (2, 'Burst Lightning', 'hand'),
                                  (3, 'Fanatical Firebrand', 'hand'), (4, 'Firespitter Whelp', 'hand'))}
        actions = [{'kind': 'TapMana', 'card': 1}, {'kind': 'AnnounceCast', 'card': 3},
                   {'kind': 'AnnounceCast', 'card': 2}, {'kind': 'AnnounceCast', 'card': 4}, {'kind': 'Pass'}]
        options = tuple(Option(str(i), 'finite', a['kind'], {'action': a}) for i, a in enumerate(actions))
        state = Observation('rustmage', game, actor, 1, (PlayerView(actor, 'P0', 20, native={'mana': [0]*6}),),
                            {o.id: o for o in cards.values()}, Decision(token, 'priority', '', options),
                            phase='precombat_main', active_player=actor)
        self.assertEqual(FoundationsAggro().choose(state).option, '2')
        self.assertEqual(FoundationsAggro().choose(replace(state, phase='upkeep')).option, '4')

    def test_keep_no_kicker_target_opponent_never_block_and_attack_all(self):
        p = FoundationsAggro()
        keep = obs('GAME_ASK', options=(Option('yes', 'boolean', 'Mulligan'), Option('no', 'boolean', 'Keep')))
        self.assertEqual(p.choose(keep).option, 'no')
        optional = obs('GAME_ASK', options=(Option('yes', 'boolean', 'Yes'), Option('no', 'boolean', 'No')))
        self.assertEqual(p.choose(optional).option, 'no')
        target = obs('GAME_TARGET', options=(option('me'), option('other'), option('creature')))
        self.assertEqual(p.choose(target).option, 'object:other')
        blocking = obs(native={'options': {'possibleBlockers': ['creature']}},
                       options=(option('creature'), Option('no', 'boolean', 'Done')))
        self.assertEqual(p.choose(blocking).option, 'no')
        attacking = obs(native={'options': {'possibleAttackers': ['creature']}},
                        options=(option('creature'), Option('special', 'string', 'All attack'), Option('no', 'boolean', 'Done')))
        self.assertEqual(p.choose(attacking).option, 'special')
        done = replace(attacking, decision=replace(attacking.decision, token=replace(attacking.decision.token, generation=2)))
        self.assertEqual(p.choose(done).option, 'no')
