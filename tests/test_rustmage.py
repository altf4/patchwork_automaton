"""Real-library contract tests; native tests skip explicitly without a local build."""
import asyncio
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from patchwork_automaton.backends.rustmage import (
    RustMageMatch, action_options, domain_action, _plain)
from patchwork_automaton.baselines.foundations import DATA, FACTS, FoundationsAggro, foundations_deck
from patchwork_automaton.core import CardSpec, DeckSpec, GameHandle, Response
from patchwork_automaton.errors import Disconnected, StaleDecision, UnsupportedInteraction
from patchwork_automaton.rl import EpisodeEnd, RustMageSelfPlay

ROOT = Path(os.environ.get('RUSTMAGE_ROOT', '.local/rustmage'))
NATIVE = (ROOT/'target/release/librustmage.so').is_file()


class DomainTests(unittest.TestCase):
    def test_full_numeric_domains_not_only_enumerated_representatives(self):
        raw = dict(actions=[{'kind': 'Pass'}],
            mana_payment=dict(available=[100]*6, required=[1, 0, 0, 0, 0, 0], generic=100),
            damage_assignment=dict(source={'card': 1, 'generation': 1}, target={'Player': 'P1'}, minimum=0, maximum=2**80),
            x_choice=dict(card={'card': 1, 'generation': 1}, intervals=[dict(minimum=0, maximum=2), dict(minimum=10, maximum=30)]),
            mana_lot_spend=dict(minimum=1, maximum=4, values=[1, 4]))
        options = {o.id: o for o in action_options(raw)}
        payment = options['domain:mana_payment']
        self.assertEqual(domain_action(payment, [1, 20, 20, 20, 20, 20])['payment'], [1, 20, 20, 20, 20, 20])
        for v in ([1]*6, [101, 0, 0, 0, 0, 0], [True, 20, 20, 20, 20, 20]):
            with self.assertRaises(ValueError): domain_action(payment, v)
        damage = options['domain:damage_assignment']
        self.assertEqual(domain_action(damage, 2**70)['amount'], 2**70)
        with self.assertRaises(ValueError): domain_action(damage, -1)
        x = options['domain:x_choice']
        self.assertEqual(domain_action(x, 17), {'kind': 'ChooseX', 'value': 17})
        with self.assertRaises(ValueError): domain_action(x, 5)
        lot = options['domain:mana_lot_spend']
        self.assertEqual(domain_action(lot, 4)['amount'], 4)
        with self.assertRaises(ValueError): domain_action(lot, 2)

    def test_decks_exactly_sixty_four_copy_and_foundations_printings(self):
        for color in ('red', 'white'):
            deck = foundations_deck(color)
            self.assertEqual(sum(c.quantity for c in deck.cards), 60)
            counts = Counter()
            for c in deck.cards:
                kind = FACTS[c.name]['type_line'].split(' — ')[0].removeprefix('Basic ')
                counts[kind] += c.quantity
            expected = dict(Land=22, Creature=19, Instant=19) if color == 'red' else dict(Land=24, Creature=36)
            self.assertEqual(counts, expected)
            self.assertTrue(all(c.set_code == 'FDN' and c.collector_number and not c.sideboard for c in deck.cards))
            self.assertTrue(all(c.quantity <= 4 or c.name in ('Mountain', 'Plains') for c in deck.cards))


@unittest.skipUnless(NATIVE, 'Build pinned RustMage with uv run python scripts/rustmage-local.py')
class NativeTests(unittest.IsolatedAsyncioTestCase):
    def match(self, **kwargs):
        match = RustMageMatch(foundations_deck('red'), foundations_deck('white'), checkout=ROOT, **kwargs)
        self.addCleanup(match.close)
        return match

    async def test_action_response_reused_only_for_its_private_viewer(self):
        m = self.match(seed=12)
        first = await m.games['player_0'].snapshot()
        await m.games['player_1'].snapshot()
        with patch.object(m.native_game, 'observe', wraps=m.native_game.observe) as observe:
            await m.games['player_0'].submit(first.decision.token, FoundationsAggro().choose(first))
            updated = await m.games['player_0'].snapshot()
            self.assertEqual(observe.call_count, 0)
            opponent = await m.games['player_1'].snapshot()
            observe.assert_called_once_with('P1')
        self.assertGreater(updated.revision, first.revision)
        self.assertEqual(_plain(updated.native), m.native_game.observe('P0'))
        self.assertEqual(_plain(opponent.native), m.native_game.observe('P1'))
        self.assertTrue(all(o.controller == opponent.actor for o in opponent.objects.values() if o.zone == 'hand'))
        with self.assertRaises(TypeError): updated.native['players'][0]['life'] = 0
        raw = m._raw('P0')
        with self.assertRaises(UnsupportedInteraction):
            m._after_action(dict(raw, viewer='P1'), 'P0', raw['revision']-1)
        self.assertFalse(m._cache)
        with self.assertRaises(StaleDecision):
            m._after_action(raw, 'P0', raw['revision'])
        self.assertFalse(m._cache)

    async def test_private_views_freshness_foreign_actions_and_close(self):
        m = self.match(seed=12); other = self.match(seed=12)
        g0, g1 = m.games.values()
        self.assertIsInstance(g0, GameHandle)
        first, opponent = await g0.snapshot(), await g1.snapshot()
        self.assertEqual(len([o for o in first.objects.values() if o.zone == 'hand']), 7)
        for obs in (first, opponent):
            self.assertTrue(all(o.controller == obs.actor for o in obs.objects.values() if o.zone == 'hand'))
            self.assertFalse(any(o.zone == 'library' for o in obs.objects.values()))
            self.assertTrue(all(o['owner'] == obs.actor.value for o in obs.native['objects'] if o['zone'] == 'Hand'))
        self.assertIsNone(opponent.decision)
        self.assertEqual(opponent.native['actions'], ())
        foreign = await other.games['player_0'].snapshot()
        policy = FoundationsAggro(); response = policy.choose(first)
        with self.assertRaises(StaleDecision): await g0.submit(foreign.decision.token, response)
        with self.assertRaises(UnsupportedInteraction): await g0.submit(first.decision.token, Response('action:999999'))
        self.assertEqual(await g0.snapshot(), first)  # Rejection is transactional.
        with self.assertRaises(ValueError): await g0.submit(first.decision.token, Response(response.option, 1))
        await g0.submit(first.decision.token, response)
        with self.assertRaises(StaleDecision): await g0.submit(first.decision.token, response)
        m.close()
        with self.assertRaises(Disconnected): await g0.snapshot()

    async def test_owned_fork_restore_replay_midcast_and_foreign_checkpoint(self):
        m = self.match(seed=42); policy = FoundationsAggro()
        for _ in range(200):
            actor = m._raw('P0')['actor']; g = m.games['player_' + actor[1]]
            obs = await g.snapshot()
            if obs.decision.kind == 'pay_costs': break
            await g.submit(obs.decision.token, policy.choose(obs))
        else: self.fail('No paid spell reached')
        before = _plain(obs.native); action = policy.choose(obs); point = m.checkpoint()
        with m.fork() as fork:
            fg = fork.games['player_' + actor[1]]; fo = await fg.snapshot()
            self.assertEqual(_plain(fo.native), before)
            with self.assertRaises(StaleDecision): await fg.submit(obs.decision.token, action)
            with self.assertRaises(ValueError): fork.restore(point)
            await fg.submit(fo.decision.token, action)
            expected = _plain((await fg.snapshot()).native)
            self.assertEqual(_plain((await g.snapshot()).native), before)
        await g.submit(obs.decision.token, action)
        self.assertEqual(_plain((await g.snapshot()).native), expected)
        m.restore(point)
        restored = await g.snapshot()
        self.assertGreater(restored.revision, obs.revision)
        now = _plain(restored.native); now['revision'] = before['revision']; self.assertEqual(now, before)
        with self.assertRaises(StaleDecision): await g.submit(obs.decision.token, action)
        await g.submit(restored.decision.token, action)
        now = _plain((await g.snapshot()).native); now['revision'] = expected['revision']; self.assertEqual(now, expected)
        m.drop_checkpoint(point)
        with self.assertRaises(ValueError): m.restore(point)

    async def test_concede_from_nonactive_seat_and_waiter_wakes_on_close(self):
        m = self.match(); g0, g1 = m.games.values()
        waiter = asyncio.create_task(g1.wait_for_decision())
        await asyncio.sleep(0)
        await g1.concede()
        result = await asyncio.wait_for(waiter, 1)
        self.assertTrue(result.terminal)
        self.assertFalse(result.outcome.won)
        self.assertTrue((await g0.wait_for_result(timeout=1)).outcome.won)
        other = self.match()
        waiter = asyncio.create_task(other.games['player_1'].wait_for_decision())
        await asyncio.sleep(0); other.close()
        with self.assertRaises(Disconnected): await waiter

    async def test_same_seed_full_replay_natural_games_and_runner_limit(self):
        async def play(seed, start):
            f = RustMageSelfPlay(foundations_deck('red'), foundations_deck('white'), checkout=ROOT,
                                starting_player=start, max_decisions=2000)
            p = FoundationsAggro(); actions = []; kinds = Counter()
            try:
                r = await f.reset(seed=seed)
                while True:
                    turn = await r.next()
                    if isinstance(turn, EpisodeEnd):
                        self.assertFalse(turn.truncated)
                        self.assertEqual(sum(o.outcome.won for o in turn.observations.values()), 1)
                        self.assertLessEqual(min(p.life for p in turn.observations['player_0'].players), 0)
                        return actions, kinds
                    o = turn.observation; a = p.choose(o)
                    native = next(v.constraints['action'] for v in o.decision.options if v.id == a.option)
                    self.assertNotEqual(native['kind'], 'Concede')
                    actions.append((turn.actor, _plain(native))); kinds[native['kind']] += 1
                    await r.submit(a)
            finally: await f.close()
        first, kinds = await play(42, 'P0')
        self.assertEqual(first, (await play(42, 'P0'))[0])
        self.assertTrue(all(kinds[k] > 0 for k in ('PlayLand', 'AnnounceCast', 'TapMana', 'PayCosts', 'Attack', 'ChooseTarget')))
        self.assertEqual(kinds['Block'], 0)
        self.assertGreater((await play(45, 'P0'))[1]['PutTrigger'], 0)
        await play(43, 'P1')
        f = RustMageSelfPlay(foundations_deck('red'), checkout=ROOT, max_decisions=1)
        try:
            r = await f.reset(); turn = await r.next()
            await r.submit(FoundationsAggro().choose(turn.observation))
            end = await r.next(); self.assertTrue(end.truncated)
            self.assertEqual(end.reason, 'decision_limit')
            self.assertTrue(all(o.outcome is None for o in end.observations.values()))
        finally: await f.close()

    async def test_deck_provenance_and_rejected_decks(self):
        fdn = {c['name']: c for c in json.loads((ROOT/'data/sets/fdn.json').read_text())['cards']}
        for color in ('red', 'white'):
            for c in foundations_deck(color).cards:
                self.assertIn(c.collector_number, fdn[c.name]['printings'])
                definition = json.loads((ROOT/DATA['cards'][c.name]['definition_path']).read_text())
                self.assertTrue(definition['complete'])
                self.assertEqual(hashlib.sha256((ROOT/DATA['cards'][c.name]['definition_path']).read_bytes()).hexdigest(),
                                 DATA['cards'][c.name]['definition_sha256'])
        with self.assertRaises(UnsupportedInteraction):
            RustMageMatch(DeckSpec((CardSpec('Mountain', 60), CardSpec('Mountain', 1, sideboard=True))), checkout=ROOT)
        with self.assertRaisesRegex(RuntimeError, 'constructed needs 60'):
            RustMageMatch(DeckSpec((CardSpec('Mountain', 7),)), checkout=ROOT)
        with self.assertRaises(RuntimeError):
            RustMageMatch(DeckSpec((CardSpec('Not a real card', 60),)), checkout=ROOT)

    async def test_unseeded_factory_uses_fresh_entropy_and_records_seed(self):
        f = RustMageSelfPlay(foundations_deck('red'), checkout=ROOT)
        try:
            with patch('patchwork_automaton.rl.rustmage.secrets.randbits', side_effect=[123, 456]) as entropy:
                await f.reset(); self.assertEqual(f.seed, 123)
                await f.reset(); self.assertEqual(f.seed, 456)
                await f.reset(seed=789); self.assertEqual(f.seed, 789)
                self.assertEqual(entropy.call_count, 2)
        finally: await f.close()


if __name__ == '__main__': unittest.main()
