"""User-specified synthetic Burn mulligan oracle; never a neural input."""


def keep_hand(cards):
    cards = tuple(cards)
    if len(cards) <= 4: return True
    lands = sum('LAND' in card.get('cardTypes', ()) for card in cards)
    two_mana_spells = sum('LAND' not in card.get('cardTypes', ()) and card.get('manaValue') == 2
                         for card in cards)
    return lands in (2, 3) or (lands == 1 and two_mana_spells <= 2)
