"""Finite, reviewed prompt bindings; prose never enters numerical model inputs."""
import re
from uuid import UUID
from collections.abc import Mapping

PROMPT_RULE_VERSION = 2
PURPOSES = ('unknown', 'mulligan', 'bottom_hand', 'priority', 'attackers', 'blockers',
            'target', 'ability', 'color', 'starting_player', 'shock_land', 'mana_payment')
MANA_COLORS = ('white','blue','black','red','green','colorless','generic')
TARGET_ZONES = ('unknown','ALL','HAND','LIBRARY','BATTLEFIELD','GRAVEYARD','EXILE','STACK','COMMAND')


def prompt_source(data):
    values = set(re.findall(r"object_id=['\"]([^'\"]+)['\"]",
                 (data.get('message') or '') + ((data.get('options') or {}).get('secondMessage') or '')))
    if len(values) != 1: return None
    try: return str(UUID(values.pop()))
    except ValueError: return None


def prompt_metadata(method, data):
    options = data.get('options') or {}
    message = re.sub(r'<[^>]*>', '', data.get('message') or '').strip()
    if (method == 'GAME_ASK' and options.get('UI.left.btn.text') == 'Mulligan'
            and options.get('UI.right.btn.text') == 'Keep'):
        match = re.fullmatch(r'Mulligan down to (\d+) cards?\?', message)
        if match:
            return dict(purpose='mulligan', next_hand_size=int(match[1]))
        match = re.fullmatch(r'Mulligan for free, draw another (\d+) cards?\?', message)
        if match:
            return dict(purpose='mulligan', next_hand_size=int(match[1]))
    if method == 'GAME_TARGET' and options.get('targetZone') == 'HAND':
        match = re.fullmatch(r'Select a card \((\d+) more\) to put on the bottom of your library', message)
        if match:
            return dict(purpose='bottom_hand', bottom_remaining=int(match[1]))
    if method == 'GAME_PLAY_MANA':
        prefix = (data.get('message') or '').split('<',1)[0].strip()
        match = re.fullmatch(r'Pay ((?:\{(?:\d+|[WUBRGC])\})+)', prefix)
        if match:
            debt = dict.fromkeys(MANA_COLORS, 0)
            names = dict(zip('WUBRGC', MANA_COLORS))
            for symbol in re.findall(r'\{([^}]+)\}', match[1]):
                if symbol.isdigit(): debt['generic'] += int(symbol)
                else: debt[names[symbol]] += 1
            return dict(purpose='mana_payment',mana_cost=debt)
        return dict(purpose='mana_payment')
    if method == 'GAME_SELECT':
        if 'possibleAttackers' in options: return dict(purpose='attackers')
        if 'possibleBlockers' in options: return dict(purpose='blockers')
        return dict(purpose='priority')
    if method == 'GAME_TARGET':
        return dict(purpose='starting_player' if message == 'Select a starting player' else 'target')
    if method == 'GAME_CHOOSE_ABILITY':
        debt = data.get('_unpaid_mana')
        if isinstance(debt,Mapping):
            if set(debt)==set(MANA_COLORS) and all(type(v) is int and v>=0 for v in debt.values()):
                return dict(purpose='ability',mana_cost=dict(debt))
        return dict(purpose='ability')
    if method == 'GAME_CHOOSE_CHOICE':
        choice = data.get('choice') or {}
        labels = list((choice.get('keyChoices') or {}).values()) or choice.get('choices') or ()
        if labels and all(str(v).lower() in MANA_COLORS[:-1] for v in labels):
            return dict(purpose='color')
    if method == 'GAME_ASK' and message == 'pay 2 life? (otherwise Sacred Foundry becomes tapped)':
        return dict(purpose='shock_land',life_payment=2)
    return dict(purpose='unknown')
