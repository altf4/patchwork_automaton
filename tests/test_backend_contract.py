"""Common API integration against the existing synthetic MTGO framed backend."""
import unittest

from patchwork_automaton.backends.mtgo import MTGOBackend
from patchwork_automaton.core import Response, Handle, GameHandle
from patchwork_automaton.errors import StaleDecision
import test_interactions as fixtures


class MTGOContractTests(unittest.IsolatedAsyncioTestCase):
    make_game=fixtures.InteractionAPITests.make_game
    last_message=fixtures.InteractionAPITests.last_message

    async def test_common_prompt_submission_and_namespace(self):
        game,backend=await self.make_game(fixtures.prompt())
        common=MTGOBackend(game.client).game(game)
        obs=await common.wait_for_decision(timeout=1)
        self.assertIsInstance(common,GameHandle)
        self.assertEqual(obs.backend,'mtgo')
        self.assertEqual(obs.actor,Handle('mtgo','player','0'))
        self.assertEqual(obs.phase,'precombat_main')
        await common.submit(obs.decision.token,Response('button:1'))
        self.assertEqual(self.last_message(game,backend)[0],4638)
        with self.assertRaises(StaleDecision):await common.submit(obs.decision.token,Response('button:1'))

    async def test_common_structured_targets_use_original_native_validation(self):
        game,backend=await self.make_game(fixtures.prompt(),fixtures.thing(123,action=fixtures.targeted()),fixtures.thing(200),fixtures.thing(201))
        common=MTGOBackend(game.client).game(game)
        obs=await common.snapshot()
        option=next(o for o in obs.decision.options if o.kind=='card')
        self.assertEqual(option.constraints['targets'][0]['candidates'],('200','201'))
        await common.submit(obs.decision.token,Response(option.id,{'targets':[['201']]}))
        self.assertEqual(self.last_message(game,backend)[0],4645)

    async def test_local_partial_selection_refreshes_common_token(self):
        local={'ACTION_NAME_STRING':'Attack','ALTACTION_DONT_RECALCULATE':1}
        game,_=await self.make_game(fixtures.prompt(),fixtures.thing(123,action=local),fixtures.thing(124,action=local))
        common=MTGOBackend(game.client).game(game)
        old=await common.snapshot()
        option=next(o for o in old.decision.options if o.kind=='card')
        await common.submit(old.decision.token,Response(option.id))
        new=await common.snapshot()
        self.assertNotEqual(old.decision.token,new.decision.token)
        self.assertNotIn(option.id,[o.id for o in new.decision.options])
        with self.assertRaises(StaleDecision):await common.submit(old.decision.token,Response('button:1'))
        await common.submit(new.decision.token,Response('button:1'))

    async def test_other_actor_prompt_is_not_an_action_for_this_account(self):
        game,_=await self.make_game(fixtures.prompt())
        await game._set_player_order((123,42))
        common=MTGOBackend(game.client).game(game)
        self.assertIsNone((await common.snapshot()).decision)

    async def test_legacy_game_events_and_common_adapter_coexist(self):
        game,_=await self.make_game(fixtures.prompt())
        common=MTGOBackend(game.client).game(game)
        self.assertIs(common.native_game,game)
        self.assertFalse(common.capabilities.supports('seed'))
        with self.assertRaises(ValueError):MTGOBackend().game(game)
