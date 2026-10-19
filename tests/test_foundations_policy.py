"""Shared aggro intent translated into the two engines' different action menus."""
from dataclasses import replace
import unittest

from patchwork_automaton.baselines.foundations import FoundationsAggro
from patchwork_automaton.core import (Decision, DecisionToken, Handle, ObjectView,
                                    Observation, Option, PlayerView)
from test_burn_baseline import obs, card, option, ME, OTHER


class FoundationsPolicyTests(unittest.TestCase):
    def test_native_combat_damage_assignment_uses_complete_domain(self):
        from patchwork_automaton.backends.rustmage import action_options, domain_action
        actor = Handle('rustmage', 'player', 'P0')
        game = Handle('rustmage', 'game', 'game')
        actions = [{'kind': 'Concede'}, *({'kind': 'AssignDamage', 'target': {'Card': 7}, 'amount': n}
                                        for n in (0, 1, 3, 2))]
        for maximum in (None, 3, 2**80):
            raw = dict(actions=actions)
            if maximum is not None:
                raw['damage_assignment'] = dict(target={'Card': 7}, minimum=0, maximum=maximum)
            options = action_options(raw)
            state = Observation('rustmage', game, actor, 1,
                (PlayerView(actor, 'P0', 20, native={'mana': [0]*6}),), {},
                Decision(None, 'damage', '', options))
            response = FoundationsAggro().choose(state)
            chosen = next(o for o in options if o.id == response.option)
            action = chosen.constraints['action'] if chosen.kind == 'finite' else domain_action(chosen, response.value)
            self.assertEqual(action, dict(kind='AssignDamage', target={'Card': 7},
                                          amount=3 if maximum is None else maximum))

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

    def test_new_instants_require_suitable_targets_and_buff_only_before_combat(self):
        for spell, target_name, controller in (
                ('Kindled Fury', 'Firebrand Archer', ME),
                ('Sure Strike', 'Firebrand Archer', ME),
                ('Scorching Dragonfire', 'Savannah Lions', OTHER),
                ('Fiery Annihilation', 'Savannah Lions', OTHER)):
            with self.subTest(spell=spell):
                p = FoundationsAggro()
                objects = (card(spell, 'spell'), *(card('Mountain', str(i), zone='battlefield') for i in range(3)))
                state = obs(objects=objects, options=(option('spell'), Option('no', 'boolean', 'Pass')))
                self.assertEqual(p.choose(state).option, 'no')
                target = card(target_name, 'target', zone='battlefield', controller=controller)
                state = replace(state, objects={**state.objects, target.id: target})
                self.assertEqual(p.choose(state).option, 'object:spell')
                decoy = card('Fanatical Firebrand', 'decoy', zone='battlefield',
                             controller=OTHER if controller == ME else ME)
                targeting = replace(state, objects={**state.objects, decoy.id: decoy},
                    decision=replace(state.decision, kind='GAME_TARGET', options=(option('decoy'), option('target'))))
                self.assertEqual(p.choose(targeting).option, 'object:target')
                expected = 'no' if controller == ME else 'object:spell'
                self.assertEqual(p.choose(replace(state, phase='postcombat_main')).option, expected)

    def test_native_buff_and_removal_target_binding(self):
        actor = Handle('rustmage', 'player', 'P0'); enemy = Handle('rustmage', 'player', 'P1')
        game = Handle('rustmage', 'game', 'game')
        token = DecisionToken('rustmage', 'session', game, actor, 1)
        objects = [ObjectView(Handle('rustmage', 'object', str(i)), 'Fanatical Firebrand',
                              'battlefield', owner, {'tapped': False})
                   for i, owner in ((1, actor), (2, enemy))]
        options = tuple(Option(str(i), 'finite', '', {'action': {'kind': 'ChooseTarget', 'target': {'Card': i}}})
                        for i in (1, 2))
        for spell, expected in (('Kindled Fury', '1'), ('Sure Strike', '1'),
                                ('Scorching Dragonfire', '2'), ('Fiery Annihilation', '2')):
            state = Observation('rustmage', game, actor, 1,
                (PlayerView(actor, 'P0', 20, native={'mana': [0]*6}), PlayerView(enemy, 'P1', 20)),
                {o.id:o for o in objects}, Decision(token, 'targets', '', options),
                native={'announcing': {'source_card': spell}})
            self.assertEqual(FoundationsAggro().choose(state).option, expected)
