#!/usr/bin/env python3
"""Export lossy numerical traits from evaluated RustMage rules, without changing it.

The helper compiles inside an immutable git archive, so Rust evaluates constants
and typed programs. Oracle prose, Rust Debug output, card names and IDs are never
used as model features. Run --prepare-only to stage the guarded build command.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
HELPER = Path(__file__).with_name('rustmage-card-traits.rs')
DEFAULT_OUTPUT = ROOT / 'src/patchwork_automaton/data/rustmage-card-traits.json'
SCHEMA = 1
COLORS = ('white', 'blue', 'black', 'red', 'green', 'colorless')
TYPES = ('creature', 'land', 'sorcery', 'instant', 'artifact', 'enchantment', 'planeswalker', 'kindred', 'battle')
KEYWORDS = ('flying', 'reach', 'haste', 'vigilance', 'defender', 'flash', 'lifelink', 'deathtouch',
    'trample', 'first_strike', 'double_strike', 'cant_block', 'hexproof', 'indestructible',
    'cant_be_blocked', 'menace', 'max_one_blocker', 'min_three_blockers', 'no_token_blockers',
    'changeling', 'skulk', 'no_power_le_two_blockers', 'no_power_ge_three_blockers',
    'no_greater_power_blockers', 'prevent_combat_to', 'prevent_combat_by', 'no_human_blockers',
    'protection_everything', 'protection_instant_sorcery', 'only_haste_blockers', 'crew_saddle_plus_two')
FLAGS = ('metadata_known', 'implemented', 'rules_known', 'projection_partial', 'legendary', 'basic',
    'token', 'alternate_face', 'power_dynamic', 'toughness_dynamic', 'x_cost', 'hybrid_cost',
    'phyrexian_cost', 'snow_cost', 'other_cost_symbol', 'conditional', 'modal', 'optional',
    'once_per_turn', 'sorcery_activation', 'nonmana_additional_cost', 'cost_reduction',
    'enters_tapped', 'conditional_entry', 'dynamic_amount', 'unknown_rules')
ALT_KINDS = ('kicker', 'offspring', 'flashback', 'plot', 'warp', 'disguise', 'sneak', 'harmonize',
    'tiered', 'additional', 'adventure', 'omen', 'prepared', 'back_face', 'hybrid_alternative')
TRIGGERS = ('entry', 'death', 'attack', 'cast', 'noncreature_cast', 'instant_sorcery_cast',
    'landfall', 'targeted', 'damage', 'lifegain', 'draw', 'discard', 'step', 'counter', 'ward')
TARGETS = ('source', 'attached', 'target', 'self_player', 'opponent', 'each', 'creature',
    'player', 'permanent', 'stack', 'graveyard', 'own', 'opposing')
COST_FLAGS = ('tap', 'sacrifice', 'discard', 'life', 'exile', 'remove_counter', 'loyalty', 'mana')
EFFECTS = ('damage', 'draw', 'gain_life', 'lose_life', 'discard', 'destroy', 'exile', 'bounce',
    'reanimate', 'token', 'counter', 'mana', 'buff', 'grant_keyword', 'counter_spell', 'tap',
    'untap', 'search', 'scry', 'surveil', 'mill', 'exile_play', 'control', 'copy')
NUMERICS = ('mana_value', 'generic_cost', *('cost_' + c for c in COLORS), 'x_symbols',
    'power', 'toughness', 'starting_loyalty', 'alt_mana_min', 'alt_mana_max', 'alt_generic_min', 'alt_generic_max',
    'spell_abilities', 'triggered_abilities', 'activated_abilities', 'static_abilities',
    'unknown_nodes', 'known_nodes', 'damage_min', 'damage_max', 'draw_min', 'draw_max',
    'gain_life_max', 'lose_life_max', 'token_max', 'counter_max', 'mana_max',
    'power_bonus_min', 'power_bonus_max', 'toughness_bonus_min', 'toughness_bonus_max',
    'spell_damage_max', 'spell_draw_max', 'triggered_damage_max', 'triggered_draw_max',
    'activated_damage_max', 'activated_draw_max')


def feature_names():
    names = list(FLAGS)
    for prefix, labels in [('type', TYPES), ('color', COLORS), ('keyword', KEYWORDS),
                           ('alternative', ALT_KINDS), ('trigger', TRIGGERS),
                           ('recipient', TARGETS), ('cost', COST_FLAGS), ('effect', EFFECTS)]:
        names += [prefix + '.' + name for name in labels]
    names += ['hybrid_option.' + c for c in COLORS]
    names += ['phyrexian_option.' + c for c in COLORS[:5]]
    names += ['alternative_color.' + c for c in COLORS]
    names += ['produces.' + c for c in COLORS]
    for name in NUMERICS:
        names += [name + '.log', name + '.known']
    return names


def numeric(value):
    if value is None or isinstance(value, bool):
        return [0.0, 0.0]
    value = float(value)
    if not math.isfinite(value):
        return [0.0, 0.0]
    return [math.copysign(math.log1p(abs(value)), value) / 5.0, 1.0]


def symbol_traits(cost):
    """Read the formal mana-symbol alphabet, never Oracle rules prose."""
    symbols = re.findall(r'\{([^}]+)\}', cost)
    return dict(x_cost=any(s in ('X', 'Y', 'Z') for s in symbols),
        hybrid_cost=any('/' in s and not s.endswith('/P') for s in symbols),
        phyrexian_cost=any(s.endswith('/P') for s in symbols), snow_cost='S' in symbols,
        other_cost_symbol=any(not (s.isdecimal() or s in 'WUBRGCXYZS' or
            all(p in ('W', 'U', 'B', 'R', 'G', 'C', '2', 'P') for p in s.split('/')))
            for s in symbols))


def vector(card):
    flags = set(card['flags'])
    flags.update(k for k, v in symbol_traits(card['mana_cost']).items() if v)
    flags.add('metadata_known')
    if card['implemented']:
        flags.add('implemented')
    if card['rules_known']:
        flags.add('rules_known')
    # Summary projection is deliberately partial even when every visited node
    # is categorized: it does not preserve sequence, targets or branch binding.
    flags.add('projection_partial')
    if card['unknown_nodes'] or not card['rules_known']:
        flags.add('unknown_rules')
    row = [float(k in flags) for k in FLAGS]
    row += [float(bool(card['kinds'] & (1 << i))) for i in range(len(TYPES))]
    row += [float(bool(card['colors'] & (1 << i))) for i in range(5)] + [float(card['colors'] == 0)]
    row += [float(bool(card['keywords'] & (1 << i))) for i in range(len(KEYWORDS))]
    alternatives = {x['kind'] for x in card['alternative_costs']}
    row += [float(k in alternatives) for k in ALT_KINDS]
    for prefix, labels in [('trigger', TRIGGERS), ('recipient', TARGETS), ('cost', COST_FLAGS), ('effect', EFFECTS)]:
        row += [float(prefix + '.' + k in flags) for k in labels]
    symbols = re.findall(r'\{([^}]+)\}', card['mana_cost'])
    row += [float(any('/' in s and not s.endswith('/P') and c in s.split('/') for s in symbols)) for c in 'WUBRGC']
    row += [float(any(s.endswith('/P') and c in s.split('/') for s in symbols)) for c in 'WUBRG']
    row += [float(any(c.get('cost', {}).get('colored', [0] * 6)[i] > 0
        for c in card['alternative_costs'])) for i in range(6)]
    row += [float(bool(card['produced_colors'] & (1 << i))) for i in range(6)]
    values = dict(card['numbers'], mana_value=card['mana_value'], generic_cost=card['cost']['generic'],
        x_symbols=card['x_symbols'], power=card['power'], toughness=card['toughness'],
        starting_loyalty=card['starting_loyalty'], unknown_nodes=card['unknown_nodes'], known_nodes=card['known_nodes'])
    values.update(('cost_' + c, v) for c, v in zip(COLORS, card['cost']['colored']))
    costs = [x['mana_value'] for x in card['alternative_costs'] if x.get('mana_value') is not None]
    values['alt_mana_min'] = min(costs) if costs else None
    values['alt_mana_max'] = max(costs) if costs else None
    generic = [x['cost']['generic'] for x in card['alternative_costs'] if 'cost' in x]
    values['alt_generic_min'] = min(generic) if generic else None
    values['alt_generic_max'] = max(generic) if generic else None
    for name in NUMERICS:
        row += numeric(values.get(name))
    assert len(row) == len(feature_names())
    return row


def git(source, *args):
    return subprocess.check_output(['git', '-C', str(source), *args])


def prepare(source, revision):
    revision = git(source, 'rev-parse', revision).decode().strip()
    helper_bytes = HELPER.read_bytes()
    helper_hash = hashlib.sha256(helper_bytes).hexdigest()
    folder = ROOT / '.local/rustmage-traits-export' / (revision[:12] + '-' + helper_hash[:12])
    marker = folder / '.traits-export-prepared.json'
    if not marker.exists():
        if folder.exists():
            raise RuntimeError('Incomplete exporter directory; preserve it and choose a new helper revision: ' + str(folder))
        folder.mkdir(parents=True)
        archive = git(source, 'archive', revision)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(folder, filter='data')
        # The child module can read private typed rules without changing their
        # visibility or requiring serialization of the entire execution IR.
        (folder / 'src/card_traits_export.rs').write_bytes(helper_bytes)
        with (folder / 'src/cards.rs').open('a') as f:
            f.write('\n#[path = "card_traits_export.rs"]\npub mod card_traits_export;\n')
        with (folder / 'src/lib.rs').open('a') as f:
            f.write('\npub use cards::card_traits_export;\n')
        bindir = folder / 'src/bin'
        bindir.mkdir(exist_ok=True)
        (bindir / 'export_card_traits.rs').write_text('fn main() { println!("{}", rustmage::card_traits_export::export()); }\n')
        marker.write_text(json.dumps(dict(revision=revision, helper_sha256=helper_hash), indent=2) + '\n')
    return folder, revision, helper_hash


def finish(raw, source, revision, helper_hash, output):
    coverage = json.loads(git(source, 'show', revision + ':data/standard/coverage.json'))
    if raw.get('catalogue_version') != coverage['catalogue_version']:
        raise ValueError('Raw descriptor catalogue differs from the requested engine revision')
    metadata_bytes = git(source, 'show', revision + ':data/catalogue-identities.json')
    metadata = json.loads(metadata_bytes)
    facts_bytes = git(source, 'show', revision + ':data/standard/cards.json')
    facts = {c['oracle_id']: c for c in json.loads(facts_bytes)['cards']}
    by_id = {c['index']: c for c in metadata['cards']}
    identity_counts = Counter(c['identity'] for c in raw['cards'])
    duplicate_identities = {name for name, count in identity_counts.items() if count > 1}
    for card in raw['cards']:
        card['serialization_identity'] = card['identity']
        if card['identity'] in duplicate_identities:
            # Face-down manifest and disguise deliberately serialize as the
            # same empty name. Preserve both public recipes via their explicit
            # engine aliases; an empty visible identity remains unknown.
            candidates = [alias for alias, index in metadata['aliases'] if index == card['index'] and alias]
            if not candidates:
                raise ValueError('Ambiguous serialization identity has no distinct engine alias')
            card['identity'] = sorted(candidates)[0]
    identities = {c['index']: c['identity'] for c in raw['cards']}
    assert len(raw['cards']) == len(by_id) == len(identities)
    assert len(set(identities.values())) == len(identities), 'duplicate serialization identity'
    aliases = {alias: identities[index] for alias, index in metadata['aliases'] if alias not in duplicate_identities}
    for identity in identities.values():
        aliases[identity] = identity
    # One immutable archive avoids thousands of independent Git processes.
    with tarfile.open(fileobj=io.BytesIO(git(source, 'archive', revision, 'src/card_definitions'))) as tar:
        definitions = {member.name: tar.extractfile(member).read() for member in tar if member.isfile()}
    for card in raw['cards']:
        entry = by_id[card['index']]
        path = 'src/card_definitions/' + entry['path']
        definition = definitions[path]
        provenance = json.loads(definitions[str(Path(path).with_suffix('.json'))])
        card['provenance'] = dict(definition=path, definition_sha256=hashlib.sha256(definition).hexdigest(),
            complete=provenance['complete'], oracle_id=entry['oracle_id'], category=entry['category'])
        # Printed structured facts remain useful even for unimplemented cards.
        # A dynamic printed stat is unknown here; live observations supply it.
        fact = facts.get(entry['oracle_id'])
        if fact:
            faces = fact.get('card_faces', [])
            face = next((f for f in faces if f['name'] == card['name']), None)
            if face is None and entry['category'] == 'physical':
                face = faces[0] if faces else fact
            if face:
                card['mana_cost'] = face.get('mana_cost', card['mana_cost'])
                for axis in ('power', 'toughness'):
                    printed = face.get(axis)
                    if printed is not None:
                        if re.fullmatch(r'-?\d+', printed):
                            card[axis] = int(printed)
                        else:
                            card[axis] = None
                            card['flags'].append(axis + '_dynamic')
        # Exact formal symbol sequence and fixed obligations; hybrid choices are
        # retained separately, rather than falsely requiring both colors.
        symbols = re.findall(r'\{([^}]+)\}', card['mana_cost'])
        card['mana_symbols'] = symbols
        card['cost'] = dict(colored=[symbols.count(c) for c in 'WUBRGC'],
            generic=sum(int(s) for s in symbols if s.isdecimal()))
        card['x_symbols'] = sum(s in ('X', 'Y', 'Z') for s in symbols)
        card['mana_value'] = sum(int(s) if s.isdecimal() else 0 if s in ('X', 'Y', 'Z')
            else 2 if s.startswith('2/') else 1 for s in symbols)
        card['features'] = vector(card)
    manifest = dict(schema=SCHEMA, rustmage_revision=revision, catalogue_version=raw['catalogue_version'],
        observation_schema=int(git(source, 'show', revision + ':data/observation-schema.txt')),
        extractor_sha256=helper_hash, exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
        printed_facts_sha256=hashlib.sha256(facts_bytes).hexdigest(),
        feature_names_sha256=hashlib.sha256(json.dumps(feature_names(), separators=(',', ':')).encode()).hexdigest(),
        card_count=len(raw['cards']), trait_width=len(feature_names()),
        ambiguous_serialization_identities=sorted(duplicate_identities),
        projection='Typed rule summary; unknown branches and amounts explicit; never an exact rules replacement.')
    result = dict(schema=SCHEMA, manifest=manifest, feature_names=feature_names(), aliases=aliases,
        unknown_traits=[0.0] * len(feature_names()), cards=raw['cards'])
    payload = dict(feature_names=result['feature_names'], aliases=aliases, unknown_traits=result['unknown_traits'],
        cards=[(c['identity'], c['features'], c['mana_value']) for c in raw['cards']])
    manifest['traits_sha256'] = hashlib.sha256(json.dumps(payload, sort_keys=True,
        ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':')) + '\n')
    temporary.replace(output)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT / '.local/rustmage')
    p.add_argument('--revision', default='HEAD')
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--raw-json', type=Path, help='Reuse a completed export from this exact helper/source; no Cargo.')
    args = p.parse_args()
    folder, revision, helper_hash = prepare(args.source, args.revision)
    command = [sys.executable, 'scripts/resource_guard.py', '--', 'cargo', 'build', '--locked', '--bin', 'export_card_traits']
    if args.prepare_only:
        print(json.dumps(dict(directory=str(folder), command=command), indent=2))
        return
    if args.raw_json:
        # A source-only raw JSON file cannot authenticate its exporter. Reuse
        # only the fixed output next to the matching prepared source/helper.
        if args.raw_json.resolve() != (folder / 'card-traits-raw.json').resolve():
            raise ValueError('--raw-json must be the raw output inside this exact prepared export directory')
        marker = json.loads((folder / '.traits-export-prepared.json').read_text())
        if marker != dict(revision=revision, helper_sha256=helper_hash):
            raise ValueError('Prepared exporter identity mismatch')
        copied_hash = hashlib.sha256((folder / 'src/card_traits_export.rs').read_bytes()).hexdigest()
        if copied_hash != helper_hash:
            raise ValueError('Prepared helper bytes differ from the requested extractor')
        raw = json.loads(args.raw_json.read_text())
    else:
        subprocess.run(command, cwd=folder, check=True,
            env=dict(os.environ, CARGO_INCREMENTAL='0', CARGO_PROFILE_DEV_DEBUG='0'))
        raw_bytes = subprocess.check_output([str(folder / 'target/debug/export_card_traits')], cwd=folder)
        (folder / 'card-traits-raw.json').write_bytes(raw_bytes)
        raw = json.loads(raw_bytes)
    print(json.dumps(finish(raw, args.source, revision, helper_hash, args.output), indent=2))


if __name__ == '__main__':
    main()
