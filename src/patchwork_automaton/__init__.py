"""Clientless MTGO server interactions. No strategy or local rules engine."""
from .actions import (CardChoice, CardSelection, DamageChoice, ManaChoice,
                      OrderingChoice, PromptChoice, SelectorChoice)
from .choice_catalog import ChoiceCatalog
from .interactions import (AutoYield, Commander, Dungeon, DamageAssignment, DamageTarget, GameObject, ManaEntry,
    ModeOption, ModeSpec, NamedOption, Ordering, PromptCode, SelectionSource, Selector, TargetGroup)
from .client import Game, MTGOClient
from .controls import ControlReceipt, PlayerStops, Step, YieldControl, YieldPreference
from .errors import (AuthenticationError, Disconnected, EventOverflow, GameEnded,
                     RequestRejected, StaleDecision, UnsupportedInteraction)
from .game import DeckItem
from .models import (CardAction, CollectionItem, DecisionID, Deck, DeckList, GameMessage, Match,
                     Observation, GameResult, PlayerResult, SideboardResult, ValidationError, Submission)
from .protocol import ProtocolError

__version__ = '0.4.0'

__all__ = ['GameEnded', 'GameResult', 'PlayerResult', 'AutoYield', 'Commander', 'Dungeon', 'ControlReceipt',
           'PlayerStops', 'Step',
           'YieldControl', 'YieldPreference', 'SideboardResult', 'ValidationError', 'MTGOClient',
           'Game', 'Deck', 'DeckItem', 'DeckList', 'Match', 'CollectionItem',
           'Observation', 'DecisionID', 'CardAction', 'CardChoice', 'PromptChoice',
           'GameMessage', 'Submission', 'AuthenticationError', 'Disconnected',
           'EventOverflow', 'RequestRejected', 'StaleDecision',
           'UnsupportedInteraction', 'ProtocolError', 'CardSelection', 'DamageChoice',
           'ManaChoice', 'OrderingChoice', 'SelectorChoice', 'ChoiceCatalog', 'DamageAssignment',
           'DamageTarget', 'GameObject', 'ManaEntry', 'ModeOption', 'ModeSpec', 'NamedOption',
           'Ordering', 'PromptCode', 'SelectionSource', 'Selector', 'TargetGroup']
