"""Clientless MTGO server interactions. No strategy or local rules engine."""
from .actions import (CardChoice, CardSelection, DamageChoice, ManaChoice,
                      OrderingChoice, PromptChoice, SelectorChoice)
from .choice_catalog import ChoiceCatalog
from .interactions import (DamageAssignment, DamageTarget, GameObject, ManaEntry,
    ModeOption, ModeSpec, NamedOption, Ordering, PromptCode, SelectionSource, Selector, TargetGroup)
from .client import Game, MTGOClient
from .errors import (AuthenticationError, Disconnected, EventOverflow,
                     RequestRejected, StaleDecision, UnsupportedInteraction)
from .game import DeckItem
from .models import (CardAction, CollectionItem, DecisionID, Deck, DeckList, GameMessage, Match,
                     Observation, Submission)
from .protocol import ProtocolError

__version__ = '0.2.0'

__all__ = ['MTGOClient', 'Game', 'Deck', 'DeckItem', 'DeckList', 'Match', 'CollectionItem',
           'Observation', 'DecisionID', 'CardAction', 'CardChoice', 'PromptChoice',
           'GameMessage', 'Submission', 'AuthenticationError', 'Disconnected',
           'EventOverflow', 'RequestRejected', 'StaleDecision',
           'UnsupportedInteraction', 'ProtocolError', 'CardSelection', 'DamageChoice',
           'ManaChoice', 'OrderingChoice', 'SelectorChoice', 'ChoiceCatalog', 'DamageAssignment',
           'DamageTarget', 'GameObject', 'ManaEntry', 'ModeOption', 'ModeSpec', 'NamedOption',
           'Ordering', 'PromptCode', 'SelectionSource', 'Selector', 'TargetGroup']
