"""Deterministic policy decisions and the deliberately weak strategy contract."""
from dataclasses import replace
import unittest

from patchwork_automaton.baselines.modern_burn import BurnBaseline, modern_burn_deck
from patchwork_automaton.core import (Handle, DecisionToken, Decision, Option, Observation,
                                    ObjectView, PlayerView)

ME = Handle('xmage', 'player', 'me')
OTHER = Handle('xmage', 'player', 'other')
GAME = Handle('xmage', 'game', 'game')


def obs(kind='GAME_SELECT', *, turn=1, generation=1, active=ME,
        options=None, objects=(), message='', native=None, view=None, phase='precombat_main'):
    token = DecisionToken('xmage', 'session', GAME, ME, generation)
    decision = Decision(token, kind, message, options or (Option('no', 'boolean', 'Pass'),), native or {})
    return Observation('xmage', GAME, ME, generation,
        (PlayerView(ME, 'me', 20), PlayerView(OTHER, 'other', 20)),
        {o.id: o for o in objects}, decision, native=view or {}, turn=turn,
        phase=phase, active_player=active, priority_player=ME)


def card(name, value, *, zone='hand', controller=ME, tapped=False):
    handle = Handle('xmage', 'object', value)
    return ObjectView(handle, name, zone, controller, {'tapped': tapped})


def option(value, name='Object'):
    return Option('object:' + value, 'uuid', name)


class BaselineTests(unittest.TestCase):
    def test_published_deck_has_exactly_sixty_main_fifteen_side(self):
        deck = modern_burn_deck()
        self.assertEqual(sum(c.quantity for c in deck.cards if not c.sideboard), 60)
        self.assertEqual(sum(c.quantity for c in deck.cards if c.sideboard), 15)
        self.assertIn('Wear // Tear', [c.name for c in deck.cards])

    def test_never_casts_discretionary_instants_on_opponents_turn(self):
        bolt = card('Lightning Bolt', 'bolt')
        state = obs(active=OTHER, objects=(bolt,), options=(option('bolt'), Option('no', 'boolean', 'Pass')))
        self.assertEqual(BurnBaseline().choose(state).option, 'no')

    def test_main_phase_plays_land_before_creature_before_burn(self):
        land, creature, bolt = card('Mountain', 'land'), card('Goblin Guide', 'guide'), card('Lightning Bolt', 'bolt')
        choices = (option('land'), option('guide'), option('bolt'), Option('no', 'boolean', 'Pass'))
        state = obs(objects=(land, creature, bolt), options=choices,
                    view={'canPlayObjects': {'objects': {'land': {'basicPlayAbilities': [1]}}}})
        self.assertEqual(BurnBaseline().choose(state).option, 'object:land')
        state = replace(state, native={})
        self.assertEqual(BurnBaseline().choose(state).option, 'object:guide')

    def test_skips_fourth_own_turn_for_both_seats(self):
        for active_turns in ((1, 3, 5, 7), (2, 4, 6, 8)):
            policy = BurnBaseline()
            for index, turn in enumerate(active_turns, 1):
                state = obs(turn=turn, generation=index, objects=(card('Mountain', 'land'),),
                    options=(option('land'), Option('no', 'boolean', 'Pass')),
                    view={'canPlayObjects': {'objects': {'land': {'basicPlayAbilities': [1]}}}})
                self.assertEqual(policy.choose(state).option, 'no' if index == 4 else 'object:land')

    def test_same_observation_is_idempotent_even_for_attack_all(self):
        state = obs(options=(Option('special', 'string', 'All attack'), Option('no', 'boolean', 'Pass')),
                    native={'options': {'possibleAttackers': ['guide']}})
        policy = BurnBaseline()
        self.assertEqual(policy.choose(state).option, 'special')
        self.assertEqual(policy.choose(state).option, 'special')
        next_state = replace(state, decision=replace(state.decision, token=replace(state.decision.token, generation=2)))
        self.assertEqual(policy.choose(next_state).option, 'no')

    def test_deliberately_never_blocks(self):
        state = obs(options=(option('guide'), Option('no', 'boolean', 'Pass')),
                    native={'options': {'possibleBlockers': ['guide']}})
        self.assertEqual(BurnBaseline().choose(state).option, 'no')

    def test_burn_targets_opponent_before_any_creature(self):
        state = obs('GAME_TARGET', options=(option('creature'), option('other'), option('me')),
                    objects=(card('Eidolon of the Great Revel', 'creature', zone='battlefield', controller=OTHER),))
        self.assertEqual(BurnBaseline().choose(state).option, 'object:other')

    def test_does_not_loop_casting_searing_blaze_without_a_creature(self):
        state = obs(objects=(card('Searing Blaze', 'blaze'),),
                    options=(option('blaze'), Option('no', 'boolean', 'Pass')))
        self.assertEqual(BurnBaseline().choose(state).option, 'no')
        enemy = card('Goblin Guide', 'guide', zone='battlefield', controller=OTHER)
        enemy = replace(enemy, properties={'cardTypes': ['CREATURE']})
        state = replace(state, objects={**state.objects, enemy.id: enemy})
        self.assertEqual(BurnBaseline().choose(state).option, 'object:blaze')

    def test_mulligans_keep_even_zero_land_hand_and_shocks_pay_life(self):
        state = obs('GAME_ASK', options=(Option('yes', 'boolean', 'Mulligan'), Option('no', 'boolean', 'Keep')))
        self.assertEqual(BurnBaseline().choose(state).option, 'no')
        state = obs('GAME_ASK', message='Pay 2 life?', options=(Option('yes', 'boolean', 'Yes'), Option('no', 'boolean', 'No')))
        self.assertEqual(BurnBaseline().choose(state).option, 'yes')

    def test_payment_chooses_needed_color_and_avoids_tapped_sources(self):
        state = obs('GAME_PLAY_MANA', message='Pay {W}<div>Lightning Helix</div>',
            objects=(card('Mountain', 'mountain', zone='battlefield'),
                     card('Sacred Foundry', 'foundry', zone='battlefield'),
                     card('Inspiring Vantage', 'vantage', zone='battlefield', tapped=True)),
            options=(option('mountain'), option('foundry'), option('vantage'), Option('no', 'boolean', 'Cancel')))
        self.assertEqual(BurnBaseline().choose(state).option, 'object:foundry')

    def test_fetch_target_from_selector_view_and_charm_mode(self):
        state = obs('GAME_TARGET', options=(option('foundry'), option('mountain')),
            native={'cardsView1': {'foundry': {'id': 'foundry', 'name': 'Sacred Foundry'},
                                   'mountain': {'id': 'mountain', 'name': 'Mountain'}}})
        self.assertEqual(BurnBaseline().choose(state).option, 'object:foundry')
        state = obs('GAME_CHOOSE_CHOICE', options=(Option('choice:0', 'string', 'Permanents gain indestructible'),
            Option('choice:1', 'string', 'Boros Charm deals 4 damage to target player or planeswalker')))
        self.assertEqual(BurnBaseline().choose(state).option, 'choice:1')

    def test_required_trigger_targets_are_answered_on_opponents_turn(self):
        state = obs('GAME_TARGET', active=OTHER, phase='upkeep', options=(option('me'), option('other')))
        self.assertEqual(BurnBaseline().choose(state).option, 'object:other')

    def test_damage_distribution_is_feasible_and_repeatable(self):
        state = obs('GAME_GET_MULTI_AMOUNT', options=(Option('amounts', 'distribution', 'Damage',
            {'minimum': 4, 'maximum': 4, 'entries': [{'min': 0, 'max': 3}, {'min': 1, 'max': 4}]}),))
        self.assertEqual(BurnBaseline().choose(state).value, (3, 1))
