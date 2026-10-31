"""Small, deterministic teaching opponents for the three Standard decks.

Keep two-to-five-land hands (stop mulliganing at five cards), play lands,
crack fetches immediately, cast affordable spells
on an empty stack in our main phase, attack with everything, and never block. The
engine supplies legal actions and completable mana payments. No search, combat
math, hidden state, or learned policy is involved.
"""
from ..core import Response
from ..errors import UnsupportedInteraction
from ..neural.card_traits import FACTS, FEATURE_NAMES, traits_for
from .standard_decks import STANDARD_DECKS

FETCHES = frozenset(('Fabled Passage', 'Elven Passage', 'Escape Tunnel'))
PUMPS = frozenset(('Ancestral Anger', 'Wild Ride', 'Turn Inside Out', 'Full Bore',
                   'Magic Damper', 'Secret Identity', "Grove's Bounty",
                   "Dreadmaw's Ire", 'Shore Up', "Meltstrider's Resolve", 'Defend the Rider'))
REMOVAL = frozenset(('Erode', 'Bitter Triumph', 'Into the Flood Maw', 'Octopus Form',
                     'Seam Rip', 'Sheltered by Ghosts', 'Pyrrhic Strike'))
BASICS = ('Plains', 'Island', 'Swamp', 'Mountain', 'Forest')
PREFERRED_BASICS = {'mono-green-landfall': ('Forest',),
                    'izzet-prowess': ('Island', 'Mountain'),
                    'bw-lifegain': ('Plains', 'Swamp')}
PRODUCES = tuple(FEATURE_NAMES.index('produces.' + c) for c in
                 ('white', 'blue', 'black', 'red', 'green'))
CASTS = frozenset(('AnnounceCast', 'AnnounceExile', 'AnnouncePlotted', 'AnnounceAdventure',
                   'AnnounceOmen', 'AnnounceBack', 'AnnounceFlashback', 'AnnounceFree',
                   'Cast', 'BeginCast', 'CastExiledDuringResolution', 'PlayGraveyardPermanent'))
MANA = frozenset(('TapMana', 'TapGrantedMana', 'AnnounceMana', 'ActivateMana', 'ChooseMana'))
PAYMENTS = frozenset(('PayCosts', 'PayResolutionMana', 'PayAttackCost', 'PayPlot', 'PayFaceUp'))


def _name(obj):
    return (obj.properties.get('card') or obj.name or '') if obj else ''


def _creature(obj):
    return bool(obj.properties.get('kinds', 0) & 1)


class StandardAggro:
    """One shared policy, with only colors and a few card-specific target rules."""

    def __init__(self, deck_id):
        if deck_id not in STANDARD_DECKS:
            raise ValueError('Unknown Standard baseline deck: ' + str(deck_id))
        self.deck_id = deck_id

    def _basic_order(self, obs, basic):
        preferred = PREFERRED_BASICS[self.deck_id]
        if basic not in preferred:
            return (1, 0, basic)
        color = BASICS.index(basic)
        sources = 0
        for obj in obs.objects.values():
            if obj.zone != 'battlefield' or obj.controller != obs.actor:
                continue
            chosen = (obj.properties.get('entry_choice') or {}).get('BasicLandType')
            sources += int(chosen == basic or traits_for(_name(obj))[PRODUCES[color]] > 0)
        return (0, sources, preferred.index(basic))

    @staticmethod
    def _spell_name(action, obj):
        name = _name(obj)
        if action['kind'] in ('AnnounceAdventure', 'AnnounceOmen', 'AnnounceBack'):
            return name.split(' // ')[-1]
        return name

    @staticmethod
    def _cast_allowed(obs, name):
        own = [o for o in obs.objects.values() if o.zone == 'battlefield'
               and o.controller == obs.actor and _creature(o)]
        if name in PUMPS:
            return bool(own) and obs.phase == 'precombat_main'
        if name == 'Burn Together':
            # Attack first, then use the creature as the burn spell's source.
            return bool(own) and obs.phase == 'postcombat_main'
        if name in REMOVAL:
            return any(o.zone == 'battlefield' and o.controller != obs.actor
                       and o.properties.get('kinds', 0) & (1 | 64) for o in obs.objects.values())
        return True

    @staticmethod
    def _target_order(obs, objects, source, target):
        if 'Player' in target:
            # These decks' targeted player damage goes upstairs.
            return (-1 if target['Player'] != obs.actor.value else 3, 0)
        obj = objects.get(target.get('Card'))
        if obj is None:
            return (1, 0)  # Stack targets and other mandatory legal targets.
        harmful = source in REMOVAL or source == 'Burn Together' and obs.native.get('targeting_group') != 0
        friendly = obj.controller == obs.actor
        preferred = friendly != harmful
        ready = not obj.properties.get('tapped') and not obj.properties.get('summoning_sick')
        return (0 if preferred else 2, 0 if ready else 1)

    def choose(self, obs):
        if obs.backend != 'rustmage' or obs.decision is None:
            raise UnsupportedInteraction('Standard baselines require a RustMage decision')
        options = obs.decision.options
        objects = {int(handle.value): obj for handle, obj in obs.objects.items()}
        hand = [o for o in objects.values() if o.zone == 'hand' and o.controller == obs.actor]
        lands = sum(bool(o.properties.get('kinds', 0) & 2) for o in hand)
        if obs.decision.kind == 'mulligan':
            # RustMage bottoms the redraw before offering Keep/Mulligan again,
            # so the private hand itself tells us when we have reached five.
            keep = len(hand) <= 5 or 2 <= lands <= 5
            preferred = 'Keep' if keep else 'Mulligan'
            for option in options:
                if option.kind == 'finite' and option.constraints['action']['kind'] == preferred:
                    return Response(option.id)
        if obs.decision.kind == 'bottom':
            bottoms = [o for o in options if o.kind == 'finite'
                       and o.constraints['action']['kind'] == 'Bottom']
            if bottoms:
                def bottom_order(option):
                    obj = objects.get(option.constraints['action']['card'])
                    land = bool(obj and obj.properties.get('kinds', 0) & 2)
                    # Keep a small mana base and cheap spells; no matchup logic.
                    preferred = land if lands > 3 else not land
                    return (not preferred, -FACTS.get(_name(obj), {}).get('mana_value', 0), option.id)
                return Response(min(bottoms, key=bottom_order).id)
        pending = obs.native.get('announcing') or obs.native.get('resolving') or {}
        source = pending.get('source_card', '')
        main = (obs.active_player == obs.actor
                and obs.phase in ('precombat_main', 'postcombat_main')
                and not obs.native.get('stack')
                and not any(o.zone == 'stack' for o in objects.values()))

        # Use complete numeric domains without enumerating huge ranges.
        for option in options:
            if option.kind == 'finite':
                continue
            c = option.constraints
            if option.id in ('domain:damage_assignment', 'domain:target_amount'):
                return Response(option.id, c['maximum'])
            if option.id == 'domain:x_choice':
                return Response(option.id, max(r['maximum'] for r in c['intervals']))
            if option.id == 'domain:source_counter_choice':
                return Response(option.id, c['maximum'])
            if option.id == 'domain:mana_lot_spend':
                return Response(option.id, max(c['values']) if c.get('values') else c['maximum'])
            if option.id == 'domain:mana_payment':
                payment = list(c['required'])
                left = c['generic']
                for i in (5, 0, 1, 2, 3, 4):
                    amount = min(left, c['available'][i] - payment[i])
                    payment[i] += amount
                    left -= amount
                if left == 0:
                    return Response(option.id, payment)

        ranked = []
        for index, option in enumerate(options):
            if option.kind != 'finite':
                continue
            action = option.constraints['action']
            kind = action['kind']
            obj = objects.get(action.get('card'))
            name = self._spell_name(action, obj)
            rank, order = None, ()
            if kind in ('Concede', 'Mulligan', 'Block', 'CancelAttackDeclaration', 'AnnouncePlot'):
                continue
            if kind in ('Keep', 'FinishOpeningHand', 'PutTrigger') or kind in PAYMENTS:
                rank = 0
            elif kind == 'PlayLand':
                rank = 0
                order = self._basic_order(obs, name)
            elif kind in ('AnnounceActivate', 'Activate', 'BeginActivate'):
                if name in FETCHES and action['ability'] == 0:
                    rank = 1
            elif kind in CASTS:
                if (main or kind == 'CastExiledDuringResolution') and self._cast_allowed(obs, name):
                    rank = 3
                    # Establish creatures before spending one-turn buffs/cantrips.
                    order = (0 if obj and _creature(obj) and kind != 'AnnounceAdventure' else 1,
                             FACTS.get(name, {}).get('mana_value', 0), name)
            elif kind in MANA:
                if obs.decision.kind != 'priority':
                    rank = 2
            elif kind == 'ChooseTarget':
                rank = 0
                order = self._target_order(obs, objects, source, action['target'])
            elif kind == 'Attack':
                rank = 0
            elif kind in ('AttackPlaneswalker', 'AttackBattle'):
                rank = 1
            elif kind in ('ChooseKicker', 'ChooseOffspring', 'ChooseCasualty', 'ChooseBlight', 'ChooseTeamwork'):
                rank = 0 if not action['pay'] else 2
            elif kind == 'ChooseBargain':
                rank = 0 if action['sacrifice'] is None else 2
            elif kind == 'ChooseGift':
                rank = 0 if action['beneficiary'] is None else 2
            elif kind == 'ChooseOptional':
                rank = 0 if action['accept'] else 2
            elif kind == 'ChooseManaRollback':
                rank = 0 if action['keep'] else 2
            elif kind == 'ChooseEntryLife':
                rank = 0 if action['pay'] else 2
            elif kind == 'ChooseEntryValue':
                rank = 0
                basic = action['value'].get('BasicLandType')
                order = self._basic_order(obs, basic) if basic else ()
            elif kind == 'ChooseMode':
                rank = 0
                order = (int(action['mode'] != (1 if source == 'Secret Identity' else 0)),)
            elif kind in ('ChooseSearch', 'ChooseSearchBattlefield'):
                rank = 0
                order = self._basic_order(obs, name)
            elif kind == 'ChooseExplore':
                rank = 0 if not action['graveyard'] else 1
            elif kind == 'ChooseAdditionalLife':
                rank = 0
            elif kind == 'AssignDamage':
                rank = 0
                order = (-action['amount'],)
            elif kind.startswith('Finish') or kind.startswith('Decline') or kind.startswith('Abort'):
                rank = 5
            elif kind == 'Pass':
                rank = 10
            elif obs.decision.kind != 'priority':
                # Mandatory selections: first legal card/mode/order. No tuning.
                rank = 2
            if rank is not None:
                ranked.append(((rank, order, index), option))
        if not ranked:
            raise UnsupportedInteraction('No simple Standard response for ' + obs.decision.kind)
        return Response(min(ranked, key=lambda row: row[0])[1].id)
