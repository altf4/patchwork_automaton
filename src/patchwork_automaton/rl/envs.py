"""Optional Gymnasium/PettingZoo wrappers with caller-supplied spaces/encoders.

Encoder contract: observation_space, action_space, encode(Observation), and
 decode(action, Observation) -> core.Response. No finite-card/action limit or
feature representation is silently imposed by the library.
"""
import asyncio
import inspect
import threading

try:
    import gymnasium as gym
    from pettingzoo import AECEnv
except ImportError as error:
    raise ImportError('Install patchwork-automaton[rl,xmage] for environment wrappers') from error

from .runner import EpisodeEnd


class _Loop:
    """Keep transport readers alive while a synchronous policy is thinking."""
    def __init__(self):
        self.loop=asyncio.new_event_loop()
        self.thread=threading.Thread(target=self._run,name='patchwork-rl',daemon=True)
        self.thread.start()
        self.closed=False

    def _run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
        self.loop.run_until_complete(self.loop.shutdown_asyncgens())
        self.loop.run_until_complete(self.loop.shutdown_default_executor())
        self.loop.close()

    def call(self,coro):
        if self.closed:
            coro.close()
            raise RuntimeError('Environment loop is closed')
        return asyncio.run_coroutine_threadsafe(coro,self.loop).result()

    def close(self):
        if self.closed:return
        self.closed=True
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=10)
        if self.thread.is_alive():raise RuntimeError('Environment worker did not stop')


def _results(end,actors):
    # Infrastructure failure / unknown result never silently becomes a loss.
    truncated=end.truncated
    rewards={a:0.0 for a in actors}
    for actor in actors:
        obs=end.observations.get(actor)
        result=obs.outcome if obs else None
        if result and result.draw is True:rewards[actor]=0.0
        elif result and result.won is not None:rewards[actor]=1.0 if result.won else -1.0
        else:truncated=True
    if truncated:rewards={a:0.0 for a in actors}
    return rewards,truncated


class DecisionEnv(gym.Env):
    """Single learner; opponent(Observation) -> Response is explicitly supplied."""
    metadata={'render_modes':[]}

    def __init__(self,factory,encoder,opponent,*,learner='player_0'):
        self.factory,self.encoder,self.opponent,self.learner=factory,encoder,opponent,learner
        self.observation_space,self.action_space=encoder.observation_space,encoder.action_space
        self._worker=_Loop();self.runner=None;self.turn=None;self.closed=False

    async def _advance(self):
        while True:
            turn=await self.runner.next()
            if isinstance(turn,EpisodeEnd) or turn.actor==self.learner:return turn
            answer=await asyncio.to_thread(self.opponent,turn.observation)
            if inspect.isawaitable(answer):answer=await answer
            await self.runner.submit(answer)

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        if options:raise ValueError('Configure the episode factory explicitly; unknown reset options are not ignored')
        self.runner=self._worker.call(self.factory.reset(seed=seed))
        self.turn=self._worker.call(self._advance())
        obs=self.turn.observations[self.learner] if isinstance(self.turn,EpisodeEnd) else self.turn.observation
        return self.encoder.encode(obs),{'backend':obs.backend,'observation':obs}

    def step(self,action):
        if self.turn is None or isinstance(self.turn,EpisodeEnd):raise RuntimeError('Reset before stepping a finished environment')
        response=self.encoder.decode(action,self.turn.observation)
        self._worker.call(self.runner.submit(response))
        self.turn=self._worker.call(self._advance())
        if isinstance(self.turn,EpisodeEnd):
            obs=self.turn.observations.get(self.learner)
            if obs is None:obs=self._worker.call(self.runner.games[self.learner].snapshot())
            rewards,truncated=_results(self.turn,self.runner.games)
            return self.encoder.encode(obs),rewards[self.learner],not truncated,truncated,{'observation':obs,'reason':self.turn.reason}
        obs=self.turn.observation
        return self.encoder.encode(obs),0.0,False,False,{'observation':obs}

    def close(self):
        if self.closed:return
        try:self._worker.call(self.factory.close())
        finally:self._worker.close();self.closed=True


class SelfPlayEnv(AECEnv):
    """Actual prompt order drives agent_selection, including repeated actors."""
    metadata={'render_modes':[],'name':'patchwork_selfplay_v0','is_parallelizable':False}

    def __init__(self,factory,encoder):
        super().__init__()
        self.factory,self.encoder=factory,encoder
        self.possible_agents=['player_0','player_1']
        self._worker=_Loop();self.runner=None;self.turn=None;self._observations={};self.closed=False

    def observation_space(self,agent):return self.encoder.observation_space
    def action_space(self,agent):return self.encoder.action_space

    def _set_turn(self,turn):
        self.turn=turn
        if isinstance(turn,EpisodeEnd):
            self._observations.update(turn.observations)
            rewards,truncated=_results(turn,self.possible_agents)
            self.rewards.update(rewards)
            self.terminations={a:not truncated for a in self.agents}
            self.truncations={a:truncated for a in self.agents}
            self.infos={a:{'reason':turn.reason} for a in self.agents}
            self.agent_selection=self.agents[0]
        else:
            self._observations[turn.actor]=turn.observation
            self.agent_selection=turn.actor

    def reset(self,seed=None,options=None):
        if options:raise ValueError('Configure the episode factory explicitly')
        self.runner=self._worker.call(self.factory.reset(seed=seed))
        self.agents=self.possible_agents[:]
        self.rewards={a:0.0 for a in self.agents}
        self._cumulative_rewards={a:0.0 for a in self.agents}
        self.terminations={a:False for a in self.agents}
        self.truncations={a:False for a in self.agents}
        self.infos={a:{} for a in self.agents}
        self._observations={a:self._worker.call(g.snapshot()) for a,g in self.runner.games.items()}
        self._set_turn(self._worker.call(self.runner.next()))

    def observe(self,agent):
        obs=self._observations[agent]
        if self.runner and not self.terminations.get(agent) and not self.truncations.get(agent) and agent!=self.agent_selection:
            obs=self._worker.call(self.runner.games[agent].snapshot())
        return self.encoder.encode(obs)

    def step(self,action):
        actor=self.agent_selection
        if self.terminations[actor] or self.truncations[actor]:
            return self._was_dead_step(action)
        self._cumulative_rewards[actor]=0
        self._clear_rewards()
        response=self.encoder.decode(action,self.turn.observation)
        self._worker.call(self.runner.submit(response))
        self._set_turn(self._worker.call(self.runner.next()))
        self._accumulate_rewards()

    def close(self):
        if self.closed:return
        try:self._worker.call(self.factory.close())
        finally:self._worker.close();self.closed=True
