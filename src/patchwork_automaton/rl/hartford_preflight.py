"""Fail-closed, metadata-only admission for the ten-deck Hartford experiment.

This module imports no numerical runtime, loads no native library, and writes
no files. Passing establishes compatible recorded artifacts and card coverage;
it does not replace the gameplay smoke test or the engine's rules tests.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from ..backends.rustmage import OBSERVATION_SCHEMA, RUSTMAGE_REVISION
from ..runtime_engine import RUNTIME_REVISION, runtime_catalogue_version


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'src/patchwork_automaton/data'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def direct_sources_sha256(root=ROOT):
    """Fingerprint build inputs, independent of their absolute checkout path."""
    root = Path(root)
    source = root / 'native/direct-standard'
    paths = [source / 'Cargo.toml', source / 'Cargo.lock', *sorted((source / 'src').rglob('*.rs'))]
    payload = [(str(p.relative_to(root)), file_sha256(p)) for p in paths]
    return hashlib.sha256(json.dumps(payload, separators=(',', ':')).encode()).hexdigest()


def hartford_preflight(*, checkout=None, traits_path=None, playback=None,
                      library_path=None, build_manifest_path=None):
    """Return a JSON-safe readiness report; never initialize models or games.

    Paths default to the same selected artifacts as the playback adapters.
    Direct builds write their sidecar with scripts/build-direct-standard.py;
    Python builds use scripts/rustmage-local.py's patchwork-build.json.
    """
    errors = []
    report = dict(ready=False, errors=errors, deck_catalog='hartford', decks={},
                  missing_mainboard_cards=[], unimplemented_mainboard_cards=[])

    def read(path, label):
        try:
            return json.loads(Path(path).read_text())
        except (OSError, ValueError) as error:
            errors.append(f'{label}: {error}')
            return {}

    mode = playback or os.environ.get('PATCHWORK_STANDARD_PLAYBACK', 'direct')
    report['playback'] = mode
    if mode not in ('direct', 'python'):
        errors.append('Playback must be direct or python')
        return report
    default_checkout = (ROOT / '.local/direct-standard' / f'rustmage-{RUNTIME_REVISION[:12]}'
                        if mode == 'direct' else Path(os.environ.get('RUSTMAGE_ROOT', '.local/rustmage')))
    engine_root = Path(checkout or default_checkout).resolve()
    traits_path = Path(traits_path or DATA / 'rustmage-card-traits.json').resolve()
    library_name = ({'linux': 'libpatchwork_direct_standard.so', 'darwin': 'libpatchwork_direct_standard.dylib',
                     'win32': 'patchwork_direct_standard.dll'} if mode == 'direct' else
                    {'linux': 'librustmage.so', 'darwin': 'librustmage.dylib', 'win32': 'rustmage.dll'})[sys.platform]
    default_library = (ROOT / '.local/direct-standard' / f'target-{RUNTIME_REVISION[:12]}-capped256' / 'release' / library_name if mode == 'direct'
                       else engine_root / 'target/release' / library_name)
    library_path = Path(library_path or os.environ.get(
        'DIRECT_STANDARD_LIBRARY' if mode == 'direct' else 'RUSTMAGE_LIBRARY', default_library)).resolve()
    build_path = Path(build_manifest_path or (
        library_path.with_suffix('.build.json') if mode == 'direct' else engine_root / 'target/patchwork-build.json'))
    report['engine'] = dict(checkout=str(engine_root), library=str(library_path),
                            build_manifest=str(build_path), revision=RUNTIME_REVISION,
                            observation_schema=OBSERVATION_SCHEMA)

    decklist_path = DATA / 'hartford-standard-decklists.json'
    decklist = read(decklist_path, 'Hartford deck archive')
    decks = decklist.get('decks', {})
    if len(decks) != 10:
        errors.append(f'Hartford requires exactly ten decks; found {len(decks)}')
    try:
        report['decklist_sha256'] = file_sha256(decklist_path)
    except OSError:
        pass
    main_names = set()
    for deck_id, deck in decks.items():
        for section, expected in (('mainboard', 60), ('sideboard', 15)):
            rows = deck.get(section, [])
            valid = all(isinstance(c.get('name'), str) and c['name'] and
                        type(c.get('quantity')) is int and c['quantity'] > 0 for c in rows)
            count = sum(c['quantity'] for c in rows) if valid else 0
            if count != expected:
                errors.append(f'{deck_id}: expected {expected} valid {section} cards, got {count}')
            if section == 'mainboard':
                main_names.update(c['name'] for c in rows if isinstance(c.get('name'), str))
        report['decks'][deck_id] = dict(mainboard_cards=sum(c.get('quantity', 0) for c in deck.get('mainboard', [])
                                                        if type(c.get('quantity')) is int),
                                       supported_mainboard_cards=0, missing=[], unimplemented=[])

    traits = read(traits_path, 'Card traits')
    manifest = traits.get('manifest', {})
    coverage = read(engine_root / 'data/standard/coverage.json', 'Selected engine coverage')
    build = read(build_path, 'Selected native build provenance (rebuild the selected library to generate it)')
    for key, expected in (('rustmage_revision', RUSTMAGE_REVISION), ('observation_schema', OBSERVATION_SCHEMA)):
        if manifest.get(key) != expected:
            errors.append(f'Card traits {key} does not match bot pin {expected}')
    try:
        expected_catalogue = runtime_catalogue_version(manifest)
    except ValueError as error:
        errors.append(str(error))
        expected_catalogue = None
    if not manifest.get('catalogue_version') or coverage.get('catalogue_version') != expected_catalogue:
        errors.append('Selected engine catalogue does not match card traits')
    try:
        schema = int((engine_root / 'data/observation-schema.txt').read_text())
        if schema != OBSERVATION_SCHEMA:
            errors.append(f'Selected engine schema {schema} does not match bot schema {OBSERVATION_SCHEMA}')
        marker = engine_root / '.direct-standard-revision'
        revision = (marker.read_text().strip() if marker.is_file() else subprocess.check_output(
            ['git', '-C', str(engine_root), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip())
        if revision != RUNTIME_REVISION:
            errors.append(f'Selected engine revision {revision} does not match runtime pin {RUNTIME_REVISION}')
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        errors.append(f'Cannot verify selected engine revision/schema: {error}')
    try:
        if file_sha256(engine_root / 'data/catalogue-identities.json') != manifest.get('metadata_sha256'):
            errors.append('Selected engine identity metadata does not match card traits')
        payload = dict(feature_names=traits['feature_names'], aliases=traits['aliases'], unknown_traits=traits['unknown_traits'],
                       cards=[(c['identity'], c['features'], c['mana_value']) for c in traits['cards']])
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        if digest != manifest.get('traits_sha256'):
            errors.append('Card trait feature contents do not match their manifest')
    except (OSError, KeyError, TypeError) as error:
        errors.append(f'Cannot verify card trait identity/content: {error}')

    trait_cards = {c['identity']: c for c in traits.get('cards', [])}
    coverage_cards = {c['name']: c for c in coverage.get('cards', [])}
    aliases = traits.get('aliases', {})
    missing, unimplemented, supported = set(), set(), set()
    for name in sorted(main_names):
        canonical = aliases.get(name, name)
        card = trait_cards.get(canonical)
        entry = coverage_cards.get(name) or coverage_cards.get(canonical) or coverage_cards.get(name.split(' // ')[0])
        if card is None or entry is None:
            missing.add(name)
        elif card.get('implemented') is not True or entry.get('implemented') is not True or card.get('provenance', {}).get('complete') is not True:
            unimplemented.add(name)
        else:
            provenance = card['provenance']
            try:
                if file_sha256(engine_root / provenance['definition']) != provenance.get('definition_sha256'):
                    errors.append(f'{name}: authored card definition differs from exported traits')
                else:
                    supported.add(name)
            except (OSError, KeyError) as error:
                errors.append(f'{name}: cannot verify authored definition: {error}')
    report.update(unique_mainboard_cards=len(main_names), missing_mainboard_cards=sorted(missing),
                  unimplemented_mainboard_cards=sorted(unimplemented))
    if missing:
        errors.append(f'{len(missing)} mainboard card identities are missing from traits or selected engine coverage')
    if unimplemented:
        errors.append(f'{len(unimplemented)} mainboard cards lack complete implementation')
    for deck_id, deck in decks.items():
        row = report['decks'][deck_id]
        row.update(supported_mainboard_cards=sum(c['quantity'] for c in deck['mainboard']
                                                if c['name'] in supported and type(c.get('quantity')) is int),
                   missing=sorted({c['name'] for c in deck['mainboard']} & missing),
                   unimplemented=sorted({c['name'] for c in deck['mainboard']} & unimplemented))

    for key, expected in (('revision', RUNTIME_REVISION), ('observation_schema', OBSERVATION_SCHEMA)):
        if build.get(key) != expected:
            errors.append(f'Selected native build {key} does not match bot pin {expected}')
    try:
        digest = file_sha256(library_path)
        report['engine']['library_sha256'] = digest
        if digest != build.get('sha256'):
            errors.append('Selected native library hash does not match its build provenance')
        if mode == 'direct':
            for key, expected in (('catalogue_version', expected_catalogue),
                                  ('coverage_sha256', file_sha256(engine_root / 'data/standard/coverage.json')),
                                  ('traits_file_sha256', file_sha256(traits_path)),
                                  ('direct_sources_sha256', direct_sources_sha256())):
                if build.get(key) != expected:
                    errors.append(f'Selected direct build {key} is stale or missing')
    except OSError as error:
        errors.append(f'Cannot verify selected native build: {error}')
    report['ready'] = not errors
    return report


def require_hartford_ready(report):
    """Reject every failed admission; there is deliberately no skip flag."""
    if not report.get('ready'):
        raise ValueError('Hartford preflight failed: ' + '; '.join(report.get('errors', ['missing readiness report'])))
    return report
