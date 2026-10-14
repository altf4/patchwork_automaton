"""Exercise scheduling, terminal results, reset and synchronous RL adapters."""
import asyncio
import unittest

from patchwork_automaton.core import Decision,DecisionToken,Handle,Observation,Option,Outcome,Response
from patchwork_automaton.rl import DecisionRunner,EpisodeEnd


class Episode:
    def __init__(self,actors=('player_0','player_0','player_1','player_0')):
        self.actors=actors;self.index=0;self.changed=asyncio.Condition()


class Game:
    def __init__(self,episode,actor):self.episode,self.actor=episode,actor

    async def snapshot(self):
        ep=self.episode;terminal=ep.index==len(ep.actors)
        game=Handle('test','game','episode');actor=Handle('test','player',self.actor)
        token=DecisionToken('test','session',game,actor,ep.index)
        pending=not terminal and ep.actors[ep.index]==self.actor
        decision=Decision(token,'ask','Choose',(Option('yes','boolean','Yes'),)) if pending else None
        return Observation('test',game,actor,ep.index,(),{},decision,terminal,
                           Outcome(won=self.actor=='player_0') if terminal else None)

    async def wait_for_decision(self):
        async with self.episode.changed:
            while True:
                obs=await self.snapshot()
                if obs.terminal or obs.decision:return obs
                await self.episode.changed.wait()

    async def submit(self,token,response):
        obs=await self.snapshot()
        if not obs.decision or token!=obs.decision.token:raise ValueError('stale')
        if response.option!='yes':raise ValueError('bad response')
        async with self.episode.changed:
            self.episode.index+=1
            self.episode.changed.notify_all()


class Factory:
    def __init__(self,**limits):self.runner=None;self.resets=0;self.limits=limits;self.closed=False
    async def reset(self,seed=None):
        await self.close();self.closed=False;self.resets+=1
        episode=Episode()
        self.runner=DecisionRunner({a:Game(episode,a) for a in ('player_0','player_1')},**self.limits)
        return self.runner
    async def close(self):
        if self.runner:await self.runner.close()
        self.closed=True


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_actual_actor_and_result(self):
        factory=Factory();runner=await factory.reset();actors=[]
        try:
            while True:
                turn=await runner.next()
                if isinstance(turn,EpisodeEnd):break
                actors.append(turn.actor)
                with self.assertRaises(RuntimeError):await runner.next()
                await runner.submit(Response('yes'))
            self.assertEqual(actors,['player_0','player_0','player_1','player_0'])
            self.assertFalse(turn.truncated)
            self.assertTrue(turn.observations['player_0'].outcome.won)
            self.assertFalse(turn.observations['player_1'].outcome.won)
        finally:await factory.close()
        self.assertFalse(runner._tasks)

    async def test_decision_limit_is_truncation_and_reset_cleans_tasks(self):
        factory=Factory(max_decisions=1);runner=await factory.reset()
        await runner.next();await runner.submit(Response('yes'))
        end=await runner.next()
        self.assertTrue(end.truncated);self.assertEqual(end.reason,'decision_limit')
        replacement=await factory.reset()
        self.assertTrue(runner.closed);self.assertFalse(runner._tasks)
        self.assertEqual((await replacement.next()).actor,'player_0')
        await factory.close()

    async def test_timeout_does_not_invent_loss(self):
        class Waiting(Game):
            async def wait_for_decision(self):await asyncio.Future()
        runner=DecisionRunner({'player_0':Waiting(Episode(),'player_0')},timeout=.01)
        try:
            end=await runner.next()
            self.assertTrue(end.truncated);self.assertEqual(end.reason,'decision_timeout')
            self.assertIsNone(end.observations['player_0'].outcome)
        finally:await runner.close()


try:
    import gymnasium
    from patchwork_automaton.rl.envs import DecisionEnv,SelfPlayEnv
except ImportError:
    gymnasium=None


@unittest.skipUnless(gymnasium,'optional RL dependencies not installed')
class EnvironmentTests(unittest.TestCase):
    def encoder(self):
        class Encoder:
            observation_space=gymnasium.spaces.Discrete(10)
            action_space=gymnasium.spaces.Discrete(1)
            def encode(self,obs):return obs.revision
            def decode(self,action,obs):return Response('yes')
        return Encoder()

    def test_gym_opponent_steps_and_reset(self):
        factory=Factory();env=DecisionEnv(factory,self.encoder(),lambda obs:Response('yes'))
        try:
            obs,info=env.reset();self.assertEqual(obs,0)
            self.assertEqual(env.step(0)[:4],(1,0.,False,False))
            self.assertEqual(env.step(0)[:4],(3,0.,False,False))
            self.assertEqual(env.step(0)[:4],(4,1.,True,False))
            self.assertEqual(env.reset()[0],0);self.assertEqual(factory.resets,2)
        finally:env.close()
        self.assertFalse(env._worker.thread.is_alive());self.assertTrue(factory.closed)

    def test_aec_actor_order_rewards_and_dead_steps(self):
        env=SelfPlayEnv(Factory(),self.encoder());actors=[]
        try:
            env.reset()
            for actor in env.agent_iter(max_iter=10):
                obs,reward,terminated,truncated,info=env.last()
                if terminated or truncated:
                    self.assertFalse(truncated)
                    self.assertEqual(reward,1. if actor=='player_0' else -1.)
                    env.step(None)
                else:
                    actors.append(actor);env.step(0)
            self.assertEqual(actors,['player_0','player_0','player_1','player_0'])
            self.assertEqual(env.agents,[])
        finally:env.close()
        self.assertFalse(env._worker.thread.is_alive())
