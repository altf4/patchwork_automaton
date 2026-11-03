"""Numerical action support for the pinned Standard catalogue.

Shares the catalogue trait encoder with Foundations. This explicit, separately
fingerprinted action schema also describes mana colors, library destinations,
entry choices and referenced objects. No card-name embedding or policy fallback.
"""
import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace

from ..backends.rustmage import domain_action
from ..core import Response
from ..errors import UnsupportedInteraction
from .features import numeric, onehot
from . import card_traits
from .rustmage_features import (ACTION_FIELDS, DECISIONS, FEATURE_SPEC,
                               RustMageEncoder, canonical, entity_row)

BASICS = ('Plains', 'Island', 'Swamp', 'Mountain', 'Forest')
DESTINATIONS = ('Top', 'Bottom', 'Graveyard', 'Shuffle', 'SecondFromTop')
FACES = ('Normal', 'Adventure', 'Omen', 'Back')
FIELDS = {
    'AssignLibraryPile': ('object', 'pile'),
    'ChooseLibraryPile': ('pile',),
    'KeepPermanentType': ('object',),
    'ChooseZoneReplacement': ('effect',),
    'FinishStackTargets': (),
    'FinishCraftMaterials': (),
    'AllocateTargetAmount': ('amount',),
    'RestoreStackTarget': ('index',),
    'SetStackTarget': ('index', 'target'),
    'ChooseCollectEvidence': ('collect',),
    'ChooseEvidence': ('object',),
    'FinishEvidence': (),
    'DeclineEvidence': (),
    'ChooseReplacement': ('source', 'ability'),
    'ChooseDrawReplacement': ('source',),
    'ChooseSelfCounterReplacement': ('effect',),
    'ChooseDamagePrevention': ('source', 'prevention'),
    'ChooseDamageEvent': ('index',),
    'ChooseCreatedToken': ('object',),
    'ChooseEndure': ('put_counters',),
    'BeginConvoke': (),
    'ChooseHarmonizer': ('object',),
    'BeginImprovise': (),
    'BeginWaterbend': (),
    'BeginOrderedCosts': (),
    'ChooseImproviser': ('object',),
    'ChooseWaterbender': ('object',),
    'ChooseConvoker': ('object', 'contribution'),
    'ChooseMobilizeDefender': ('defender',),
    'ChooseToken': ('token',),
    'ChooseKeyword': ('index',),
    'PutHandCreature': ('object',),
    'PutHandPermanent': ('object',),
    'PutHandLand': ('object',),
    'SelectLand': ('object',),
    'SelectGraveyardReturn': ('object',),
    'FinishGraveyardReturn': (),
    'AnnounceSourceGrant': ('source', 'ability'),
    'SelectResolutionExile': ('object',),
    'ProliferatePermanent': ('object',),
    'ProliferatePlayer': ('player',),
    'FinishProliferate': (),
    'ChooseArmy': ('object',),
    'ChooseJace': ('object',),
    'FinishLandSelection': (),
    'DeclineHandEntry': (),
    'ChooseHandDefender': ('defender',),
    'ChooseBlight': ('pay',),
    'PayBlight': ('object',),
    'ChooseTeamwork': ('pay',),
    'ChooseTeammate': ('object',),
    'FinishTeamwork': (),
    'ChooseSaddler': ('object',),
    'ChooseCrewer': ('object',),
    'FinishCrew': (),
    'FinishSaddle': (),
    'ChooseLibraryDestination': ('object', 'destination'),
    'ChooseManaValueParity': ('parity',),
    'ChooseLinkedExile': ('object',),
    'ChooseForageRoute': ('route',),
    'ChooseForage': ('object',),
    'DeclineForage': (),
    'RevealMiracle': (),
    'DeclineMiracle': (),
    'AnnounceCounterGraveyard': ('card',),
    'ChooseCreatureCounterCost': ('object', 'counter'),
    'AnnounceAttachedAbility': ('grantor', 'ability'),
    'AnnouncePlaneswalkerGrant': ('source', 'grantor', 'ability'),
    'AnnounceSelfGrantedAbility': ('source', 'grant_timestamp'),
    'ChooseAdditionalCost': ('object',),
    'ChooseAdditionalLife': (),
    'ChooseExplore': ('object', 'graveyard'),
    'ChooseManifest': ('object',),
    'ChooseCasualty': ('pay',),
    'ChooseCastCost': ('clause',),
    'PayCasualty': ('object',),
    'ChooseBargain': ('sacrifice',),
    'ChooseX': ('value',),
    'ChooseSourceCounterAmount': ('amount',),
    'ChooseKicker': ('pay',),
    'ChooseGift': ('beneficiary',),
    'ChooseOffspring': ('pay',),
    'ChooseSpellCost': ('cost',),
    'ChooseActivationCost': ('cost',),
    'ChooseConniver': ('object',),
    'PayResolutionMana': ('payment',),
    'DeclineResolutionMana': (),
    'ChooseHandCard': ('object',),
    'ChooseRevealedDiscard': ('object',),
    'SelectTargetCounter': ('counter',),
    'FinishTargetCounterRemoval': (),
    'RecoverMilled': ('object',),
    'SelectMilled': ('object',),
    'FinishMilledSelection': (),
    'ChooseEntryTimestamp': ('object',),
    'ChooseEntryReplacement': ('source', 'intrinsic'),
    'ChooseEntryBehold': ('object',),
    'ChooseEntryValue': ('value',),
    'ChooseCreatureType': ('creature_type',),
    'ChooseEntryLife': ('pay',),
    'ChooseAuraAttachment': ('object',),
    'Keep': (),
    'BeginOnBattlefield': ('object',),
    'FinishOpeningHand': (),
    'CastExiledDuringResolution': ('object', 'face'),
    'FinishResolutionCasts': (),
    'FinishGraveyardSymbolCost': (),
    'Mulligan': (),
    'Bottom': ('card',),
    'Pass': (),
    'PlayLand': ('card',),
    'TapMana': ('card',),
    'TapGrantedMana': ('card', 'grantor', 'production'),
    'AnnounceCast': ('card',),
    'PlayGraveyardPermanent': ('card', 'permission', 'permanent_type'),
    'AnnounceFree': ('card', 'face', 'flash_surcharge'),
    'AnnounceFreeCounterGraveyard': ('card', 'face', 'flash_surcharge'),
    'AnnounceFreeGraveyardPermanent': ('card', 'face', 'flash_surcharge', 'permission', 'permanent_type'),
    'AnnounceExileFree': ('card', 'face', 'flash_surcharge'),
    'AnnounceFreeFromPermission': ('card', 'face', 'flash_surcharge', 'permission'),
    'AnnounceGrantedFlashback': ('card', 'flash_surcharge'),
    'PayResolutionLife': (),
    'DeclineResolutionLife': (),
    'AnnounceExile': ('card',),
    'AnnounceExileLife': ('card', 'face', 'flash_surcharge'),
    'AnnounceAirbend': ('card', 'face', 'flash_surcharge'),
    'AnnounceFlashSurcharge': ('card',),
    'AnnounceWarp': ('card',),
    'AnnounceDisguise': ('card',),
    'AnnounceDisguiseGraveyard': ('card', 'permission', 'permanent_type'),
    'AnnounceSneak': ('card', 'permission'),
    'AnnounceAdventure': ('card',),
    'AnnounceOmen': ('card',),
    'AnnounceBack': ('card',),
    'AnnouncePlot': ('card',),
    'PayPlot': ('payment',),
    'BeginFaceUp': ('object', 'cost'),
    'PayFaceUp': ('payment',),
    'AbortFaceUpPayment': (),
    'AnnouncePlotted': ('card',),
    'AnnounceFlashback': ('card',),
    'AnnounceHarmonize': ('card',),
    'AnnounceMayhem': ('card',),
    'AnnounceGraveyardBorrowedAbility': ('card', 'graveyard', 'ability', 'counter_payment'),
    'AnnounceGraveyardBorrowedMana': ('card', 'graveyard', 'ability', 'counter_payment', 'production'),
    'ActivateGraveyardBorrowedMana': ('card', 'graveyard', 'ability', 'counter_payment', 'payment', 'production'),
    'AnnounceBorrowedAbility': ('card', 'exiled', 'ability', 'counter_payment'),
    'AnnounceBorrowedMana': ('card', 'exiled', 'ability', 'counter_payment', 'production'),
    'ActivateBorrowedMana': ('card', 'exiled', 'ability', 'counter_payment', 'payment', 'production'),
    'AnnounceActivate': ('card', 'ability', 'counter_payment'),
    'AnnounceMana': ('card', 'ability', 'counter_payment', 'production'),
    'PayCosts': ('payment',),
    'ChooseManaRollback': ('keep',),
    'Cast': ('card', 'target', 'payment'),
    'BeginCast': ('card', 'payment'),
    'Activate': ('card', 'ability', 'target', 'payment', 'counter_payment'),
    'ActivateMana': ('card', 'ability', 'payment', 'counter_payment', 'production'),
    'ChooseManaLotSpend': ('amount',),
    'ChooseMana': ('production',),
    'BeginActivate': ('card', 'ability', 'payment', 'counter_payment'),
    'ChooseMode': ('mode',),
    'ChooseTarget': ('target',),
    'ChooseTargetCardType': ('target', 'type_mask'),
    'FinishTargets': (),
    'Attack': ('card',),
    'AttackPlaneswalker': ('card', 'defender'),
    'AttackBattle': ('card', 'defender'),
    'FinishAttackers': (),
    'CancelAttackDeclaration': (),
    'PayAttackCost': ('payment',),
    'AbortAttackPayment': (),
    'Block': ('blocker', 'attacker'),
    'FinishBlockers': (),
    'AssignDamage': ('target', 'amount'),
    'Discard': ('card',),
    'PutTrigger': ('id',),
    'ChooseOutsideCard': ('index',),
    'ChooseOutsideBasic': ('card',),
    'CancelOutsideCard': (),
    'DeclineOutsideCards': (),
    'ChooseOutsideRevealed': ('index',),
    'ChooseOutsideEntryReplacement': ('source', 'ability'),
    'ChooseOutsideSelfReplacement': ('effect',),
    'ChooseSearch': ('card',),
    'ChooseSearchZones': ('library', 'graveyard'),
    'ChooseHandLibrarySearchZones': ('hand', 'library'),
    'FinishSearch': (),
    'ChooseSearchBattlefield': ('card',),
    'ChooseSearchDestination': ('destination',),
    'SelectLibrary': ('card',),
    'FinishLibrarySelection': (),
    'ChooseEquipmentAttachment': ('equipment',),
    'OrderLibraryBottom': ('card',),
    'ArrangeLibrary': ('card', 'destination'),
    'ChooseOptional': ('accept',),
    'Sacrifice': ('object',),
    'DeclineEffectPayment': (),
    'KeepRole': ('object',),
    'KeepLegend': ('object',),
    'Concede': (),
}
STANDARD_DECISIONS = ('attack_cost', 'draw_replacement', 'miracle_reveal', 'resolution_casts', 'mana_lot_spend', 'zone_replacement', 'replacement', 'created_token', 'resolution_mana', 'resolution_life', 'connive_order', 'opening_hand', 'mulligan', 'bottom', 'priority', 'plot_payment', 'face_up_payment', 'role_rule', 'legend_rule', 'retarget', 'target_amount', 'targets', 'offspring', 'casualty', 'mana_rollback', 'cast_costs', 'casualty_payment', 'bargain', 'kicker', 'gift', 'endure', 'harmonize', 'convoke', 'improvise', 'waterbend', 'mobilize', 'blight_intent', 'blight_payment', 'teamwork_intent', 'teamwork_payment', 'saddle', 'crew', 'explore', 'manifest', 'resolution_x', 'source_counter_amount', 'target_counter_removal', 'x', 'evidence_intent', 'evidence', 'spell_cost', 'activation_cost', 'spell_mode', 'additional_cost', 'pay_costs', 'order_triggers', 'optional_effect', 'target_library', 'mana_value_parity', 'linked_exile', 'forage_method', 'forage', 'mana_choice', 'library_choice', 'equipment_attachment', 'library_selection', 'hand_card_choice', 'revealed_discard', 'recover_milled', 'milled_selection', 'creature_type_choice', 'hand_library_search', 'library_graveyard_search', 'library_search_destination', 'library_search', 'effect_discard', 'outside_game_choice', 'outside_entry_replacement', 'hand_defender', 'token_choice', 'keyword_choice', 'proliferate', 'effect_choice', 'entry_timestamps', 'linked_choice', 'entry_value', 'land_card_name', 'entry_life', 'entry_behold', 'entry_replacement', 'aura_entry', 'attackers', 'blockers', 'damage', 'discard')
BOOL_FIELDS = ('keep', 'graveyard', 'intrinsic', 'hand', 'library', 'flash_surcharge', 'collect', 'put_counters')
NUMBER_FIELDS = ('mode', 'cost', 'clause', 'index', 'effect', 'type_mask', 'destination_count', 'replacement_ability', 'grant_timestamp')
PERMANENT_TYPES = ('Creature', 'Artifact', 'Enchantment', 'Planeswalker', 'Land', 'Battle')
CATEGORIES = {
    'pile': ('FaceDown', 'FaceUp'),
    'effect_kind': ('RevealShuffle', 'Prevent', 'Exile', 'ExileReflexive', 'Battlefield'),
    'prevention': ('ShieldCounter', 'CombatBySource', 'CombatToRecipient', 'ProtectionEverything', 'ProtectionInstantSorcery', 'SourceLinkedDamage'),
    'contribution': ('Generic', 'White', 'Blue', 'Black', 'Red', 'Green'),
    'parity': ('Odd', 'Even'),
    'route': ('ExileThree', 'Food', 'Black'),
    'counter': ('Lore', 'PlusOne', 'MinusOne', 'Stun', 'Loyalty', 'Incubation', 'Bait', 'Page', 'Divinity', 'Revival', 'Soul', 'Fellowship', 'Wish', 'Stash', 'Charge', 'Growth', 'Plan', 'Invasion', 'Haste', 'Flying', 'FirstStrike', 'DoubleStrike', 'Deathtouch', 'Indestructible', 'Lifelink', 'Menace', 'Reach', 'Trample', 'Vigilance', 'Finality', 'Shield', 'Hone', 'Quest', 'Defense', 'Hexproof', 'Blight', 'Bloodstain', 'Bore', 'Chorus', 'Coin', 'Conqueror', 'Decayed', 'Dread', 'Dream', 'Film', 'Fire', 'Flood', 'Impostor', 'Ingenuity', 'Landmark', 'Loot', 'Nest', 'Net', 'Point', 'Possession', 'Rev', 'Skewer', 'Supply', 'Time', 'Unlock'),
    'creature_type': ('Advisor', 'Aetherborn', 'Alien', 'Ally', 'Angel', 'Antelope', 'Ape', 'Archer', 'Archon', 'Armadillo', 'Army', 'Artificer', 'Assassin', 'Assembly-Worker', 'Astartes', 'Atog', 'Aurochs', 'Avatar', 'Azra', 'Badger', 'Balloon', 'Barbarian', 'Bard', 'Basilisk', 'Bat', 'Bear', 'Beast', 'Beaver', 'Beeble', 'Beholder', 'Berserker', 'Bird', 'Bison', 'Blinkmoth', 'Boar', 'Bringer', 'Brushwagg', "C'tan", 'Camarid', 'Camel', 'Capybara', 'Caribou', 'Carrier', 'Cat', 'Centaur', 'Child', 'Chimera', 'Citizen', 'Cleric', 'Clown', 'Cockatrice', 'Construct', 'Coward', 'Coyote', 'Crab', 'Crocodile', 'Custodes', 'Cyberman', 'Cyclops', 'Dalek', 'Dauthi', 'Demigod', 'Demon', 'Deserter', 'Detective', 'Devil', 'Dinosaur', 'Djinn', 'Doctor', 'Dog', 'Dragon', 'Drake', 'Dreadnought', 'Drix', 'Drone', 'Druid', 'Dryad', 'Dwarf', 'Echidna', 'Efreet', 'Egg', 'Elder', 'Eldrazi', 'Elemental', 'Elephant', 'Elf', 'Elk', 'Employee', 'Eternal', 'Eye', 'Faerie', 'Ferret', 'Fish', 'Flagbearer', 'Fox', 'Fractal', 'Frog', 'Fungus', 'Gamer', 'Gamma', 'Gargoyle', 'Germ', 'Giant', 'Giraffe', 'Gith', 'Glimmer', 'Gnoll', 'Gnome', 'Goat', 'Goblin', 'God', 'Golem', 'Gorgon', 'Graveborn', 'Gremlin', 'Griffin', 'Guest', 'Hag', 'Halfling', 'Hamster', 'Harpy', 'Hedgehog', 'Hellion', 'Hero', 'Hippo', 'Hippogriff', 'Homarid', 'Homunculus', 'Horror', 'Horse', 'Human', 'Hydra', 'Hyena', 'Illusion', 'Imp', 'Incarnation', 'Inhuman', 'Inkling', 'Inquisitor', 'Insect', 'Jackal', 'Jellyfish', 'Juggernaut', 'Kangaroo', 'Kavu', 'Kirin', 'Kithkin', 'Knight', 'Kobold', 'Kor', 'Kraken', 'Kree', 'Lamia', 'Lammasu', 'Leech', 'Lemur', 'Leviathan', 'Lhurgoyf', 'Licid', 'Lizard', 'Llama', 'Lobster', 'Manticore', 'Masticore', 'Mercenary', 'Merfolk', 'Metathran', 'Minion', 'Minotaur', 'Mite', 'Mole', 'Monger', 'Mongoose', 'Monk', 'Monkey', 'Moogle', 'Moonfolk', 'Mount', 'Mouse', 'Mutant', 'Myr', 'Mystic', 'Nautilus', 'Necron', 'Nephilim', 'Nightmare', 'Nightstalker', 'Ninja', 'Noble', 'Noggle', 'Nomad', 'Nymph', 'Octopus', 'Ogre', 'Ooze', 'Orb', 'Orc', 'Orgg', 'Otter', 'Ouphe', 'Ox', 'Oyster', 'Pangolin', 'Peasant', 'Pegasus', 'Pentavite', 'Performer', 'Pest', 'Phelddagrif', 'Phoenix', 'Phyrexian', 'Pilot', 'Pincher', 'Pirate', 'Plant', 'Platypus', 'Porcupine', 'Possum', 'Praetor', 'Primarch', 'Prism', 'Processor', 'Qu', 'Rabbit', 'Raccoon', 'Ranger', 'Rat', 'Rebel', 'Reflection', 'Rhino', 'Rigger', 'Robot', 'Rogue', 'Sable', 'Salamander', 'Samurai', 'Sand', 'Saproling', 'Satyr', 'Scarecrow', 'Scientist', 'Scion', 'Scorpion', 'Scout', 'Sculpture', 'Seal', 'Serf', 'Serpent', 'Servo', 'Shade', 'Shaman', 'Shapeshifter', 'Shark', 'Sheep', "Shi'ar", 'Siren', 'Skeleton', 'Skrull', 'Skunk', 'Slith', 'Sliver', 'Sloth', 'Slug', 'Snail', 'Snake', 'Soldier', 'Soltari', 'Sorcerer', 'Spawn', 'Specter', 'Spellshaper', 'Sphinx', 'Spider', 'Spike', 'Spirit', 'Splinter', 'Sponge', 'Spy', 'Squid', 'Squirrel', 'Starfish', 'Surrakar', 'Survivor', 'Symbiote', 'Synth', 'Tentacle', 'Tetravite', 'Thalakos', 'Thopter', 'Thrull', 'Tiefling', 'Time Lord', 'Toy', 'Treefolk', 'Trilobite', 'Triskelavite', 'Troll', 'Turtle', 'Tyranid', 'Unicorn', 'Utrom', 'Vampire', 'Varmint', 'Vedalken', 'Villain', 'Volver', 'Wall', 'Walrus', 'Warlock', 'Warrior', 'Weasel', 'Weird', 'Werewolf', 'Whale', 'Wizard', 'Wolf', 'Wolverine', 'Wombat', 'Worm', 'Wraith', 'Wurm', 'Yeti', 'Zombie', 'Zubera'),
    'defender_kind': ('Player', 'Planeswalker', 'Battle'),
    'entry_kind': ('BasicLandType', 'Color', 'CreatureType', 'ManaValueParity', 'CardName'),
    'entry_color': ('White', 'Blue', 'Black', 'Red', 'Green'),
    'entry_parity': ('Odd', 'Even'),
    'destination_kind': ('Top', 'Bottom', 'Graveyard', 'Shuffle', 'SecondFromTop', 'HandOrGraveyard', 'LinkedExile', 'ExileFreeThisTurn', 'Hand', 'Battlefield', 'BattlefieldThenHand'),
}
NAMED_COUNTERS = tuple(k for k in CATEGORIES['counter']
                       if k not in ('Lore', 'PlusOne', 'MinusOne', 'Stun', 'Loyalty'))
STANDARD_PLAYER_STATS = ('speed', 'speed_triggered_this_turn')


def named_counter_features(counters):
    values = dict(counters.get('named') or ())
    if any(kind not in NAMED_COUNTERS for kind in values):
        raise UnsupportedInteraction('Unsupported named counter kind')
    return sum((numeric(values.get(kind, 0)) for kind in NAMED_COUNTERS), [])


EXTRA_WIDTH = (12 + 2 * len(NUMBER_FIELDS) + 2 + len(BASICS) + len(DESTINATIONS)
               + 2 * len(BOOL_FIELDS) + len(FACES) + 14 + len(PERMANENT_TYPES)
               + sum(map(len, CATEGORIES.values())) + 1 + card_traits.TRAIT_WIDTH + 2 * len(NAMED_COUNTERS))


STANDARD_CANDIDATE_LIMIT = 256


def expand_standard_options(decision, *, allow_concede=False, limit=STANDARD_CANDIDATE_LIMIT):
    """Retain a deterministic legal prefix; stop generating once the cap is full."""
    if type(limit) is not int or limit < 1:
        raise ValueError("Candidate limit must be a positive integer")
    choices = {}

    def add(response, action):
        choices.setdefault(canonical(action), (response, action))
        return len(choices) >= limit

    def payments(required, available, generic, prefix=()):
        index = len(prefix)
        if index == 6:
            if generic == 0:
                yield prefix
            return
        capacity = sum(hi - lo for lo, hi in zip(required[index + 1:], available[index + 1:]))
        low = max(0, generic - capacity)
        high = min(generic, available[index] - required[index])
        for extra in range(low, high + 1):
            yield from payments(required, available, generic - extra, prefix + (required[index] + extra,))

    for option in decision.options:
        if option.kind == 'finite':
            action = option.constraints['action']
            if action['kind'] != 'Concede' or allow_concede:
                if add(Response(option.id), action):
                    return tuple(choices.values())
            continue
        c = option.constraints
        if option.id == 'domain:mana_payment':
            values = payments(c['required'], c['available'], c['generic'])
        elif option.id == 'domain:x_choice':
            values = (v for interval in c['intervals'] for v in range(interval['minimum'], interval['maximum'] + 1))
        elif option.id == 'domain:source_counter_choice':
            values = range(c['maximum'] + 1)
        elif option.id in ('domain:damage_assignment', 'domain:target_amount', 'domain:mana_lot_spend'):
            values = c['values'] if c.get('values') is not None else range(c['minimum'], c['maximum'] + 1)
        else:
            raise UnsupportedInteraction('Unsupported Standard domain: ' + option.id)
        for value in values:
            if add(Response(option.id, value), domain_action(option, value)):
                return tuple(choices.values())
    if not choices:
        raise UnsupportedInteraction('No playable Standard action')
    return tuple(choices.values())


class StandardEncoder(RustMageEncoder):
    action_fields, actions, decisions = FIELDS, tuple(FIELDS), STANDARD_DECISIONS
    decision_aliases = {'optional_search': 'optional_effect', 'attack_payment': 'attack_cost'}
    feature_spec = dict(FEATURE_SPEC,
        entities=FEATURE_SPEC['entities'] + len(BASICS) + 2 * (len(NAMED_COUNTERS) + len(STANDARD_PLAYER_STATS)),
        candidates=FEATURE_SPEC['candidates'] - len(ACTION_FIELDS) + len(FIELDS) + 3 * (len(BASICS) + 2 * (len(NAMED_COUNTERS) + len(STANDARD_PLAYER_STATS))) + EXTRA_WIDTH,
        context=FEATURE_SPEC['context'] - len(DECISIONS) + len(STANDARD_DECISIONS))

    def expand_options(self, decision):
        return expand_standard_options(decision, allow_concede=self.allow_concede)

    def encode(self, obs, *, choices=None):
        if choices is None:
            choices = self.expand_options(obs.decision)
        # These cards are revealed only to this actor by the current legal
        # search/scry/selection. Unchosen or concealed pool entries stay absent.
        from .standard_choice_features import _choice_objects, _with_visible_objects
        obs = _with_visible_objects(obs, _choice_objects(obs, choices))
        # Serde omits an absent optional Sneak permission; the numerical schema
        # records its absence explicitly while retaining the original response.
        choices = tuple((response, dict(action, permission=None)
                         if action.get('kind') == 'AnnounceSneak' and 'permission' not in action else action)
                        for response, action in choices)
        def packet(item):
            if not item or 'source' not in item:
                return item
            source = item.get('source') or {}
            return dict(item, card=(source.get('object') or {}).get('card'),
                        source_card=source.get('card'))
        raw = obs.native
        normalized = dict(raw, stack=[packet(p) for p in raw['stack']],
                          waiting_triggers=[packet(p) for p in raw['waiting_triggers']],
                          announcing=packet(raw.get('announcing')), resolving=packet(raw.get('resolving')))
        obs = obs._replace(native=normalized) if hasattr(obs, '_replace') else replace(obs, native=normalized)
        return super().encode(obs, choices=choices)

    @staticmethod
    def entity_row(obj, side, *, player=False):
        entry = obj.get('entry_choice') or {}
        return (entity_row(obj, side, player=player) + onehot(entry.get('BasicLandType'), BASICS)
                + named_counter_features(obj.get('counters') or {})
                + sum((numeric(int(obj.get(key, 0)) if player else None)
                       for key in STANDARD_PLAYER_STATS), []))

    def action_roles(self, action):
        roles = dict(action)
        effect = action.get('effect') or {}
        nested = effect.get('ExileReflexive', {}) if isinstance(effect, Mapping) else {}
        obj = (action.get('object') or action.get('sacrifice') or action.get('source')
               or action.get('equipment') or nested.get('source'))
        if obj is not None:
            roles['card'] = obj['card']
        elif action.get('grantor') is not None and 'card' not in action:
            roles['card'] = action['grantor']['card']
        if action.get('beneficiary') is not None:
            roles['target'] = {'Player': action['beneficiary']}
        if action.get('player') is not None:
            roles['target'] = {'Player': action['player']}
        defender = action.get('defender')
        if defender is not None:
            if 'card' in defender:
                roles['target'] = {'Card': defender['card']}
            elif 'Player' in defender:
                roles['target'] = defender
            else:
                roles['target'] = {'Card': next(iter(defender.values()))['card']}
        other = action.get('permission') or action.get('exiled')
        if isinstance(action.get('graveyard'), Mapping):
            other = action['graveyard']
        if other is None and action.get('grantor') is not None and ('card' in action or obj is not None):
            other = action['grantor']
        if isinstance(other, (tuple, list)):
            other = other[0]
        if other is not None:
            roles['attacker'] = other['card']
        return roles

    def action_extras(self, action):
        value = action.get('value')
        entry = value if isinstance(value, Mapping) else {}
        if entry and (len(entry) != 1 or next(iter(entry)) not in CATEGORIES['entry_kind']):
            raise UnsupportedInteraction('Unsupported entry choice: ' + repr(value))
        basic = entry.get('BasicLandType')
        if basic is not None and basic not in BASICS:
            raise UnsupportedInteraction('Unsupported basic land type: ' + repr(basic))
        effect = action.get('effect')
        nested = effect.get('ExileReflexive', {}) if isinstance(effect, Mapping) else {}
        destination = action.get('destination')
        numbers = dict(action, destination_count=(destination.get('BattlefieldThenHand', {}).get('battlefield')
                       if isinstance(destination, Mapping) else None), replacement_ability=nested.get('ability'))
        categories = dict(action, effect_kind=(next(iter(effect)) if isinstance(effect, Mapping)
                          else effect if isinstance(effect, str) else None),
                          creature_type=action.get('creature_type') or entry.get('CreatureType'),
                          entry_kind=next(iter(entry), None), entry_color=entry.get('Color'),
                          entry_parity=entry.get('ManaValueParity'),
                          destination_kind=next(iter(destination)) if isinstance(destination, Mapping) else destination)
        defender = action.get('defender')
        categories['defender_kind'] = (('Battle' if action['kind'] == 'AttackBattle' else 'Planeswalker')
                                      if defender and 'card' in defender else next(iter(defender), None)
                                      if defender else None)
        categorical = []
        for field, allowed in CATEGORIES.items():
            selected = categories.get(field)
            if selected is not None and selected not in allowed:
                raise UnsupportedInteraction(f'Unsupported {field}: {selected!r}')
            categorical += onehot(selected, allowed)
        face = action.get('face')
        if face is not None and face not in FACES:
            raise UnsupportedInteraction('Unsupported cast face: ' + repr(face))
        cost = action.get('cost')
        cost = cost if isinstance(cost, Mapping) else {}
        permission = action.get('permission')
        permanent = permission[1] if isinstance(permission, (tuple, list)) else action.get('permanent_type')
        choice_card = (action.get('token') or entry.get('CardName')
                       or (action.get('card') if isinstance(action.get('card'), str) else None))
        return (sum((numeric(v) for v in action.get('production', (None,) * 6)), [])
                + sum((numeric(v) if isinstance(v, (int, float)) else numeric(None)
                       for v in (numbers.get(k) for k in NUMBER_FIELDS)), [])
                + numeric(None if entry else value) + onehot(basic, BASICS)
                + onehot(destination if isinstance(destination, str) else None, DESTINATIONS)
                + sum((numeric(int(action[k])) if isinstance(action.get(k), bool) else numeric(None)
                       for k in BOOL_FIELDS), [])
                + onehot(face, FACES)
                + sum((numeric(v) for v in cost.get('colored', (None,) * 6)), [])
                + numeric(cost.get('generic')) + onehot(permanent, PERMANENT_TYPES)
                + categorical + [float(choice_card is not None)]
                + (list(card_traits.traits_for(choice_card)) if choice_card is not None
                   else [0.] * card_traits.TRAIT_WIDTH)
                + named_counter_features(action.get('counter_payment') or {}))

    def manifest(self):
        result = super().manifest()
        result.update(feature_version=6, decision_aliases=self.decision_aliases, scope='Pinned Standard action schema including all ten Hartford main decks',
                      extra_action_fields=dict(production=6, numbers=NUMBER_FIELDS,
                          booleans=BOOL_FIELDS, basics=BASICS, destinations=DESTINATIONS,
                          faces=FACES, categories=CATEGORIES, choice_card_traits=True, dynamic_cost=('colored', 'generic'), permanent_types=PERMANENT_TYPES), live_entry_basic_types=BASICS,
                      named_counter_state_and_payments=NAMED_COUNTERS,
                      standard_player_stats=STANDARD_PLAYER_STATS,
                      legal_visible_choice_pools=('search_choice.cards', 'library_choice.remaining', 'library_selection.remaining'),
                      observation_pipeline='python')
        result.pop('fingerprint')
        result['fingerprint'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
        return result
