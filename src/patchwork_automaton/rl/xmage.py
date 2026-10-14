"""Local XMage self-play episode lifecycle, independent of any learner policy."""
import asyncio
import uuid

from ..backends.xmage import XMageClient
from ..errors import UnsupportedInteraction
from .runner import DecisionRunner


class XMageSelfPlay:
    def __init__(self,first_deck,second_deck=None,*,deck_type='Constructed - Modern',
                 client_options=None,timeout=30,max_decisions=10000):
        self.decks=(first_deck,second_deck or first_deck)
        self.deck_type=deck_type
        self.client_options=dict(client_options or {})
        self.timeout,self.max_decisions=timeout,max_decisions
        self.clients=[]
        self.runner=None
        self.table_id=None

    async def reset(self,*,seed=None):
        if seed is not None:
            raise UnsupportedInteraction('Stock XMage has no verified per-episode seed API; seed was not applied')
        await self.close()
        prefix='pw'+uuid.uuid4().hex[:8]
        self.clients=[XMageClient(**self.client_options) for _ in range(2)]
        try:
            for seat,client in enumerate(self.clients):await client.login(prefix+str(seat))
            first,second=self.clients
            table=await first.create_table(deck_type=self.deck_type)
            self.table_id=table['tableId']
            for seat,client in enumerate(self.clients):
                await client.join_table(self.table_id,self.decks[seat],player_name=prefix+str(seat))
            await first.start_match(self.table_id)
            games=await asyncio.gather(*(c.wait_for_game(self.table_id,timeout=self.timeout) for c in self.clients))
            self.runner=DecisionRunner(dict(zip(('player_0','player_1'),games)),timeout=self.timeout,max_decisions=self.max_decisions)
            return self.runner
        except BaseException:
            await self.close()
            raise

    async def close(self):
        if self.runner:await self.runner.close()
        if self.table_id and self.clients and self.clients[0].connected:
            try:await self.clients[0].remove_table(self.table_id)
            except Exception:pass
        for client in self.clients:await client.close()
        self.clients=[];self.runner=None;self.table_id=None
