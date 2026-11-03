"""Coverage and artifact identity gate without TensorFlow or native playback."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from patchwork_automaton.rl import hartford_preflight as preflight


class HartfordPreflightTests(unittest.TestCase):
    def setUp(self):
        from patchwork_automaton import runtime_engine
        compatibility = patch.object(runtime_engine, 'RUNTIME_PIN', {})
        compatibility.start()
        self.addCleanup(compatibility.stop)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.engine = self.root / 'engine'
        (self.engine / 'data/standard').mkdir(parents=True)
        (self.engine / 'cards').mkdir()
        (self.engine / '.direct-standard-revision').write_text(preflight.RUNTIME_REVISION)
        (self.engine / 'data/observation-schema.txt').write_text(str(preflight.OBSERVATION_SCHEMA))
        (self.engine / 'data/catalogue-identities.json').write_text('{}')
        self.library = self.root / 'library.so'
        self.library.write_bytes(b'test native artifact')
        self.build_path = self.root / 'build.json'
        self.traits_path = self.root / 'traits.json'
        self.decks = json.loads((preflight.DATA / 'hartford-standard-decklists.json').read_text())['decks']
        names = sorted({c['name'] for d in self.decks.values() for c in d['mainboard']})
        self.first_name = names[0]
        cards = []
        for index, name in enumerate(names):
            definition = f'cards/{index}.rs'
            (self.engine / definition).write_text('complete authored rules fixture')
            cards.append(dict(identity=name, name=name, implemented=True, features=[], mana_value=0,
                              provenance=dict(complete=True, definition=definition,
                                  definition_sha256=preflight.file_sha256(self.engine / definition))))
        self.coverage = dict(catalogue_version='fixture-catalogue',
                             cards=[dict(name=n, implemented=True) for n in names])
        self.traits = dict(feature_names=[], aliases={}, unknown_traits=[], cards=cards,
                           manifest=dict(rustmage_revision=preflight.RUSTMAGE_REVISION,
                               observation_schema=preflight.OBSERVATION_SCHEMA,
                               catalogue_version='fixture-catalogue',
                               metadata_sha256=preflight.file_sha256(self.engine / 'data/catalogue-identities.json')))
        self.write_artifacts()

    def write_artifacts(self):
        payload = dict(feature_names=self.traits['feature_names'], aliases=self.traits['aliases'],
                       unknown_traits=self.traits['unknown_traits'],
                       cards=[(c['identity'], c['features'], c['mana_value']) for c in self.traits['cards']])
        self.traits['manifest']['traits_sha256'] = hashlib.sha256(json.dumps(
            payload, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        self.traits_path.write_text(json.dumps(self.traits))
        (self.engine / 'data/standard/coverage.json').write_text(json.dumps(self.coverage))
        self.build_path.write_text(json.dumps(dict(revision=preflight.RUNTIME_REVISION,
            observation_schema=preflight.OBSERVATION_SCHEMA, catalogue_version=self.coverage['catalogue_version'],
            coverage_sha256=preflight.file_sha256(self.engine / 'data/standard/coverage.json'),
            sha256=preflight.file_sha256(self.library), traits_file_sha256=preflight.file_sha256(self.traits_path),
            direct_sources_sha256=preflight.direct_sources_sha256())))

    def audit(self):
        return preflight.hartford_preflight(checkout=self.engine, traits_path=self.traits_path,
            playback='direct', library_path=self.library, build_manifest_path=self.build_path)

    def test_complete_matching_artifacts_pass_all_ten_decks(self):
        report = self.audit()
        self.assertTrue(report['ready'], report['errors'])
        self.assertEqual(len(report['decks']), 10)
        self.assertEqual(report['unique_mainboard_cards'], 164)
        self.assertTrue(all(d['supported_mainboard_cards'] == 60 for d in report['decks'].values()))
        self.assertIs(preflight.require_hartford_ready(report), report)

    def test_rules_revision_can_change_while_feature_provenance_stays_pinned(self):
        with patch.object(preflight, 'RUNTIME_REVISION', '1' * 40):
            (self.engine / '.direct-standard-revision').write_text(preflight.RUNTIME_REVISION)
            self.write_artifacts()
            report = self.audit()
            self.assertTrue(report['ready'], report['errors'])
            self.assertEqual(report['engine']['revision'], '1' * 40)
            self.traits['manifest']['rustmage_revision'] = '1' * 40
            self.write_artifacts()
            self.assertFalse(self.audit()['ready'])

    def test_runtime_catalogue_requires_explicit_saved_feature_compatibility(self):
        from patchwork_automaton import runtime_engine
        pin = dict(feature_catalogue_version='fixture-catalogue',
                   runtime_catalogue_version='fixed-rules-catalogue')
        with patch.object(runtime_engine, 'RUNTIME_PIN', pin):
            self.coverage['catalogue_version'] = 'fixed-rules-catalogue'
            self.write_artifacts()
            self.assertTrue(self.audit()['ready'])
            pin['feature_catalogue_version'] = 'unrelated-features'
            self.assertFalse(self.audit()['ready'])

    def test_missing_and_unimplemented_identities_are_reported_without_silent_substitution(self):
        missing = self.traits['cards'].pop(0)['identity']
        incomplete = self.traits['cards'][0]
        incomplete['implemented'] = False
        self.write_artifacts()
        report = self.audit()
        self.assertEqual(report['missing_mainboard_cards'], [missing])
        self.assertEqual(report['unimplemented_mainboard_cards'], [incomplete['identity']])
        with self.assertRaisesRegex(ValueError, 'Hartford preflight failed'):
            preflight.require_hartford_ready(report)

    def test_engine_coverage_and_authored_definition_must_agree(self):
        self.coverage['cards'][0]['implemented'] = False
        self.write_artifacts()
        self.assertEqual(self.audit()['unimplemented_mainboard_cards'], [self.first_name])
        self.coverage['cards'][0]['implemented'] = True
        self.write_artifacts()
        (self.engine / self.traits['cards'][0]['provenance']['definition']).write_text('stale source')
        self.assertTrue(any('definition differs' in e for e in self.audit()['errors']))

    def test_stale_schema_catalogue_revision_and_library_are_rejected(self):
        (self.engine / 'data/observation-schema.txt').write_text('196')
        (self.engine / '.direct-standard-revision').write_text('different-revision')
        self.coverage['catalogue_version'] = 'different-catalogue'
        self.write_artifacts()
        self.library.write_bytes(b'different library')
        errors = self.audit()['errors']
        for phrase in ('engine schema', 'engine revision', 'engine catalogue', 'library hash'):
            self.assertTrue(any(phrase in e for e in errors), errors)

    def test_missing_binary_provenance_fails_closed(self):
        self.build_path.unlink()
        report = self.audit()
        self.assertFalse(report['ready'])
        self.assertTrue(any('native build provenance' in e for e in report['errors']))

    def test_metadata_changed_after_build_requires_new_provenance(self):
        # This change preserves catalogue identity and all mainboard coverage.
        self.coverage['extra_generated_metadata'] = 'different export'
        (self.engine / 'data/standard/coverage.json').write_text(json.dumps(self.coverage))
        self.assertTrue(any('coverage_sha256' in e for e in self.audit()['errors']))

    def test_python_playback_checks_the_selected_library_and_engine(self):
        report = preflight.hartford_preflight(checkout=self.engine, traits_path=self.traits_path,
            playback='python', library_path=self.library, build_manifest_path=self.build_path)
        self.assertTrue(report['ready'], report['errors'])
        other = self.root / 'other-library.so'
        other.write_bytes(b'other library')
        with patch.dict('os.environ', {'RUSTMAGE_LIBRARY': str(other)}):
            report = preflight.hartford_preflight(checkout=self.engine, traits_path=self.traits_path,
                playback='python', build_manifest_path=self.build_path)
        self.assertFalse(report['ready'])
        self.assertEqual(report['engine']['library'], str(other))
        self.assertTrue(any('library hash' in e for e in report['errors']))

    def test_altered_deck_card_counts_are_rejected(self):
        data = self.root / 'data'
        data.mkdir()
        decks = json.loads((preflight.DATA / 'hartford-standard-decklists.json').read_text())
        decks['decks']['dimir-midrange']['mainboard'][0]['quantity'] += 1
        (data / 'hartford-standard-decklists.json').write_text(json.dumps(decks))
        with patch.object(preflight, 'DATA', data):
            errors = self.audit()['errors']
        self.assertTrue(any('dimir-midrange: expected 60' in e for e in errors))

    def test_import_and_real_rejection_do_not_import_tensorflow(self):
        command = ('import sys; from patchwork_automaton.rl.hartford_preflight import hartford_preflight; '
                   'hartford_preflight(); assert not any(n.startswith("tensorflow") for n in sys.modules)')
        result = subprocess.run([sys.executable, '-c', command], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
