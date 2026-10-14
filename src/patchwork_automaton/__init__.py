"""Clientless MTGO server interactions. No strategy or local rules engine."""
from .actions import CardChoice, PromptChoice
from .client import Game, MTGOClient
from .errors import (AuthenticationError, Disconnected, EventOverflow,
                     RequestRejected, StaleDecision, UnsupportedInteraction)
from .game import DeckItem
from .models import (CardAction, DecisionID, Deck, DeckList, GameMessage, Match,
                     Observation, Submission)
from .protocol import ProtocolError

__all__ = ['MTGOClient', 'Game', 'Deck', 'DeckItem', 'DeckList', 'Match',
           'Observation', 'DecisionID', 'CardAction', 'CardChoice', 'PromptChoice',
           'GameMessage', 'Submission', 'AuthenticationError', 'Disconnected',
           'EventOverflow', 'RequestRejected', 'StaleDecision',
           'UnsupportedInteraction', 'ProtocolError']
