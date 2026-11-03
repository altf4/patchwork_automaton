"""Versioned visible identities and narrow typed-rule features. No TensorFlow."""
from functools import lru_cache
from importlib.resources import files
import hashlib
import json
import math
import numpy as np
from . import card_traits

ENTRY_KINDS=('never','always','land_count_at_most','land_count_at_least','pay_life',
    'controls_basic','controls_subtype','opponents_turn','first_three_turns','low_life',
    'controls_planeswalker','behold_subtype','chosen_x_at_most')
SUBTYPES=('Plains','Island','Swamp','Mountain','Forest')
FEATURE_NAMES=tuple('entry.'+k for k in ENTRY_KINDS)+(
    'entry.threshold','entry.life','entry.other_subtype')+tuple('entry.subtype.'+k for k in SUBTYPES)+(
    'counter.simple_complete','counter.partial','counter.hard','counter.tax_fixed','counter.tax_dynamic',
    'counter.tax_generic','counter.exile','target.spell','target.included_creature','target.excluded_creature',
    'target.min_mana','target.max_mana','target.max_known')+tuple('counter.tax_color.'+k for k in 'WUBRGC')+(
    'counter.multiple_or_modal',)
CARDS=tuple(sorted(card_traits.CARDS))
CARD_IDS={name:i+1 for i,name in enumerate(CARDS)}
VOCABULARY_SHA256=hashlib.sha256(json.dumps(CARDS,separators=(',',':')).encode()).hexdigest()

def card_id(name):
    return CARD_IDS.get(card_traits.ALIASES.get(name,name),0) if name else 0

def _number(x):
    return math.log1p(float(x))/5

def vector(descriptor):
    out=dict.fromkeys(FEATURE_NAMES,0.)
    entry=descriptor['entry'];kind=entry['kind']
    if kind not in ENTRY_KINDS:raise ValueError('Unknown land-entry descriptor')
    out['entry.'+kind]=1.
    out['entry.threshold']=_number(entry.get('threshold',0))
    out['entry.life']=_number(entry.get('amount',0))
    for subtype in entry.get('subtypes',[]):
        out['entry.subtype.'+subtype if subtype in SUBTYPES else 'entry.other_subtype']=1.
    spells=descriptor['spells'];counters=[(s,c) for s in spells for c in s['counters']]
    if len(counters)>1 or any(s['mode'] is not None for s,c in counters):out['counter.multiple_or_modal']=1.
    # A partial marker must never masquerade as a complete rule description.
    for spell,counter in counters:
        complete=spell['simple_counter_complete']
        out['counter.simple_complete' if complete else 'counter.partial']=1.
        kind=counter['kind']
        out[{'hard':'counter.hard','unless_pay':'counter.tax_fixed','unless_pay_dynamic':'counter.tax_dynamic'}[kind]]=1.
        out['counter.tax_generic']=max(out['counter.tax_generic'],_number(counter.get('generic',0)))
        out['counter.exile']=max(out['counter.exile'],float(counter.get('exile',False)))
        for color,n in zip('WUBRGC',counter.get('colored',[0]*6)):
            out['counter.tax_color.'+color]=max(out['counter.tax_color.'+color],_number(n))
        for group in spell['targets']:
            rule=group['rule'];kind=rule['kind']
            if kind in ('spell','spell_kinds','spell_characteristics'):out['target.spell']=1.
            if kind=='spell_kinds':
                out['target.included_creature']=float(bool(rule['included']&1))
                out['target.excluded_creature']=float(bool(rule['excluded']&1))
            if kind=='spell_characteristics':
                out['target.min_mana']=_number(rule['min_mana'])
                if rule['max_mana'] is not None:
                    out['target.max_known']=1.;out['target.max_mana']=_number(rule['max_mana'])
    return np.asarray([out[k] for k in FEATURE_NAMES],np.float32)

@lru_cache(maxsize=1)
def catalogue():
    data=json.loads(files('patchwork_automaton').joinpath('data/hartford-input-descriptors.json').read_text())
    payload={k:data[k] for k in ('schema','cards')}
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if digest!=data['manifest']['payload_sha256']:raise ValueError('Input descriptor content hash mismatch')
    if data['manifest']['feature_traits_sha256']!=card_traits.MANIFEST['traits_sha256']:
        raise ValueError('Input descriptors belong to another feature catalogue')
    if set(data['cards'])!=set(CARDS):raise ValueError('Input descriptor identity vocabulary mismatch')
    return data

def descriptor_matrix():
    data=catalogue()
    return np.stack([np.zeros(len(FEATURE_NAMES),np.float32),*[vector(data['cards'][c]) for c in CARDS]])

def manifest():
    return dict(vocabulary_sha256=VOCABULARY_SHA256,cards=len(CARDS)+1,
        descriptor_sha256=catalogue()['manifest']['payload_sha256'],feature_names=FEATURE_NAMES,
        descriptors=catalogue()['manifest'])
