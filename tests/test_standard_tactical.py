"""The held-out Standard opponent makes distinct, legal tactical choices."""

import unittest

from patchwork_automaton.backends.rustmage import action_options
from patchwork_automaton.baselines.standard import StandardAggro
from patchwork_automaton.baselines.standard_tactical import StandardTactical
from patchwork_automaton.core import Decision, DecisionToken, Handle, ObjectView, Observation, PlayerView


ME = Handle('rustmage', 'player', 'P0')
OTHER = Handle('rustmage', 'player', 'P1')
GAME = Handle('rustmage', 'game', 'tactical-baseline-test')


def card(identity, name, *, controller=ME, zone='battlefield', power=1, toughness=1, kinds=1):
    handle = Handle('rustmage', 'object', str(identity))
    properties = dict(card=name, zone=zone.title(), controller=controller.value,
                      kinds=kinds, power=power, toughness=toughness,
                      tapped=False, summoning_sick=False)
    return ObjectView(handle, name, zone, controller, properties)


def observation(actions, *, objects=(), kind='priority', active=ME, life=20, phase='precombat_main'):
    raw = dict(actions=actions, announcing=None, resolving=None, stack=[], waiting_triggers=[],
               turn=3, active=active.value, viewer=ME.value, actor=ME.value)
    players = (PlayerView(ME, 'P0', life), PlayerView(OTHER, 'P1', 20))
    decision = Decision(DecisionToken('rustmage', 'session', GAME, ME, 1), kind, '', action_options(raw))
    return Observation('rustmage', GAME, ME, 1, players, {obj.id: obj for obj in objects},
                       decision, native=raw, turn=3, phase=phase, active_player=active)


def selected(policy, obs):
    response = policy.choose(obs)
    return next(o.constraints['action'] for o in obs.decision.options if o.id == response.option)


class StandardTacticalTests(unittest.TestCase):
    def setUp(self):
        self.tactical = StandardTactical('mono-green-landfall')
        self.aggro = StandardAggro('mono-green-landfall')

    def test_blocks_a_favorable_attack_where_aggro_does_not(self):
        actions = [dict(kind='Block', blocker=1, attacker=2), dict(kind='FinishBlockers')]
        obs = observation(actions, kind='blockers', phase='declare_blockers', active=OTHER,
                          objects=[card(1, 'Mossborn Hydra', power=4, toughness=4),
                                   card(2, 'Llanowar Elves', controller=OTHER, power=1, toughness=1)])
        self.assertEqual(selected(self.tactical, obs), actions[0])
        self.assertEqual(selected(self.aggro, obs), actions[1])

    def test_trades_a_small_creature_for_an_attacker(self):
        actions = [dict(kind='Block', blocker=1, attacker=2), dict(kind='FinishBlockers')]
        obs = observation(actions, kind='blockers', phase='declare_blockers', active=OTHER,
                          objects=[card(1, 'Llanowar Elves', power=2, toughness=2),
                                   card(2, 'Drake Hatcher', controller=OTHER, power=2, toughness=2)])
        self.assertEqual(selected(self.tactical, obs), actions[0])

    def test_declines_an_obviously_bad_block_with_plenty_of_life(self):
        actions = [dict(kind='Block', blocker=1, attacker=2), dict(kind='FinishBlockers')]
        obs = observation(actions, kind='blockers', phase='declare_blockers', active=OTHER,
                          objects=[card(1, 'Llanowar Elves', power=1, toughness=1),
                                   card(2, 'Mossborn Hydra', controller=OTHER, power=4, toughness=4)])
        self.assertEqual(selected(self.tactical, obs), actions[1])

    def test_declines_attack_into_an_obviously_superior_blocker(self):
        actions = [dict(kind='Attack', card=1), dict(kind='FinishAttackers')]
        obs = observation(actions, kind='attackers', phase='declare_attackers',
                          objects=[card(1, 'Llanowar Elves', power=1, toughness=1),
                                   card(2, 'Mossborn Hydra', controller=OTHER, power=4, toughness=4)])
        self.assertEqual(selected(self.tactical, obs), actions[1])
        self.assertEqual(selected(self.aggro, obs), actions[0])

    def test_holds_removal_for_a_larger_threat(self):
        actions = [dict(kind='AnnounceCast', card=1), dict(kind='Pass')]
        hand = card(1, 'Bitter Triumph', zone='hand', kinds=0)
        small = card(2, 'Llanowar Elves', controller=OTHER, power=1, toughness=1)
        large = card(2, 'Mossborn Hydra', controller=OTHER, power=4, toughness=4)
        self.assertEqual(selected(self.tactical, observation(actions, objects=[hand, small])), actions[1])
        self.assertEqual(selected(self.aggro, observation(actions, objects=[hand, small])), actions[0])
        self.assertEqual(selected(self.tactical, observation(actions, objects=[hand, large])), actions[0])


if __name__ == '__main__':
    unittest.main()
