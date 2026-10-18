"""Seedable in-memory self-play with the same DecisionRunner as XMage."""
import secrets

from ..backends.rustmage import RustMageMatch
from .runner import DecisionRunner


class RustMageSelfPlay:
    def __init__(self, first_deck, second_deck=None, *, timeout=30,
                 max_decisions=10000, **match_options):
        self.decks = first_deck, second_deck
        self.timeout, self.max_decisions = timeout, max_decisions
        self.match_options = match_options
        self.match = self.runner = None

    async def reset(self, *, seed=None):
        await self.close()
        # Unseeded training resets should explore new shuffles, not replay seed 0.
        self.seed = secrets.randbits(64) if seed is None else seed
        self.match = RustMageMatch(*self.decks, seed=self.seed, **self.match_options)
        self.runner = DecisionRunner(self.match.games, timeout=self.timeout, max_decisions=self.max_decisions)
        return self.runner

    async def close(self):
        if self.runner: await self.runner.close()
        if self.match: self.match.close()
        self.match = self.runner = None
