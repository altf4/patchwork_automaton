"""Schedule actual prompted actors, including concurrent mulligan prompts."""
import asyncio
from dataclasses import dataclass
from collections.abc import Mapping
from collections import deque

from ..core import Observation, Response
from ..properties import freeze


@dataclass(frozen=True)
class Turn:
    actor: str
    observation: Observation


@dataclass(frozen=True)
class EpisodeEnd:
    observations: Mapping
    truncated: bool = False
    reason: str | None = None


class DecisionRunner:
    def __init__(self,games,*,timeout=30,max_decisions=10000,trace_limit=0):
        if not games or timeout<=0 or max_decisions<1:
            raise ValueError('Games, positive timeout and positive decision limit are required')
        self.games=dict(games)
        self.timeout,self.max_decisions=timeout,max_decisions
        self._tasks={}
        self._finished={}
        self.current=None
        self.steps=0
        self.closed=False
        self.trace=deque(maxlen=trace_limit)

    async def next(self):
        if self.closed:raise RuntimeError('Runner is closed')
        if self.current:raise RuntimeError('Submit the current decision before requesting another')
        if self.steps>=self.max_decisions:
            return EpisodeEnd(freeze({a:await g.snapshot() for a,g in self.games.items()}),True,'decision_limit')
        while True:
            for actor,game in self.games.items():
                if actor not in self._tasks and actor not in self._finished:
                    self._tasks[actor]=asyncio.create_task(game.wait_for_decision(),name='decision:'+actor)
            if not self._tasks:
                return EpisodeEnd(freeze(self._finished))
            done,_=await asyncio.wait(self._tasks.values(),timeout=self.timeout,return_when=asyncio.FIRST_COMPLETED)
            if not done:
                return EpisodeEnd(freeze({a:await g.snapshot() for a,g in self.games.items()}),True,'decision_timeout')
            # Stable seat order for prompts already available concurrently.
            actor=next(a for a,t in self._tasks.items() if t in done)
            observation=self._tasks.pop(actor).result()
            latest=await self.games[actor].snapshot()
            if latest.terminal:
                observation=latest
            elif (observation.decision is None or latest.decision is None
                  or observation.decision.token!=latest.decision.token):
                continue
            if observation.terminal:
                game=self.games[actor]
                if observation.outcome is None and hasattr(game,'wait_for_result'):
                    try:observation=await game.wait_for_result(timeout=self.timeout)
                    except TimeoutError:
                        return EpisodeEnd(freeze({**self._finished,actor:observation}),True,'result_timeout')
                self._finished[actor]=observation
                continue
            self.current=Turn(actor,observation)
            return self.current

    async def submit(self,response: Response):
        if self.current is None:raise RuntimeError('No decision has been delivered')
        turn=self.current
        try:
            result=await self.games[turn.actor].submit(turn.observation.decision.token,response)
        except Exception:
            # The caller sees the rejection; do not retry or pick another move.
            self.current=None
            raise
        self.trace.append((turn.actor,turn.observation,response))
        self.steps+=1
        self.current=None
        return result

    async def close(self):
        self.closed=True
        for task in self._tasks.values():task.cancel()
        await asyncio.gather(*self._tasks.values(),return_exceptions=True)
        self._tasks.clear()
