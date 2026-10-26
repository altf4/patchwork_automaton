"""Host diagnostics observe ownership and allocator metadata without neural deps."""
import copy
import ctypes
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

from examples import benchmark_host_memory as diagnostic


class HostMemoryDiagnosticTests(unittest.TestCase):
    def test_explicit_glibc_abi_and_passive_snapshots(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            allocator = diagnostic.LibcAllocator()
            self.assertEqual(ctypes.sizeof(diagnostic.Mallinfo2), 10 * ctypes.sizeof(ctypes.c_size_t))
            self.assertEqual(allocator.libc.mallinfo2.argtypes, [])
            self.assertIs(allocator.libc.mallinfo2.restype, diagnostic.Mallinfo2)
            self.assertEqual(allocator.libc.malloc_info.argtypes, [ctypes.c_int, ctypes.c_void_p])
            self.assertIs(allocator.libc.malloc_info.restype, ctypes.c_int)
            self.assertEqual(allocator.libc.malloc_trim.argtypes, [ctypes.c_size_t])
            info = allocator.mallinfo()
            self.assertEqual(len(info), 10)
            self.assertTrue(all(type(value) is int and value >= 0 for value in info.values()))
            self.assertGreaterEqual(info['arena'], info['uordblks'])
            xml = allocator.xml(root/'malloc.xml')
            self.assertGreaterEqual(xml['heaps'], 1)
            self.assertGreater(xml['system']['current'], 0)
            self.assertEqual(xml['xml_sha256'], diagnostic.sha(root/'malloc.xml'))
            self.assertTrue((root/'malloc.xml').read_text().startswith('<malloc'))
            smaps = diagnostic.smaps_summary('/proc/self/smaps', root/'smaps.txt')
            self.assertGreater(sum(group.get('Rss', 0) for group in smaps['groups_kib'].values()), 0)
            self.assertEqual(smaps['sha256'], diagnostic.sha(root/'smaps.txt'))

    def test_python_size_does_not_follow_weakrefs_or_cycles(self):
        class Holder:
            pass
        holder = Holder()
        holder.payload = bytearray(8 * 1024**2)
        values = [weakref.ref(holder)]
        values.append(values)
        self.assertLess(diagnostic.retained_python_size(values), 1024)

    def test_post_series_probes_require_every_natural_update(self):
        with tempfile.TemporaryDirectory() as directory:
            observer = diagnostic.HostMemoryObserver(Path(directory))
            with self.assertRaisesRegex(RuntimeError, 'All five'):
                observer.interventions()
            self.assertFalse(observer.phases)
            self.assertFalse(observer.completed)

    def initialized_observer(self, directory):
        observer = diagnostic.HostMemoryObserver(Path(directory))
        class Owner:
            pass
        model, trainer = Owner(), Owner()
        state = dict(sha256=dict(model='model', actor_adam='actor', critic_adam='critic'),
                     actor_steps=123, critic_steps=456, pcg=dict(state=789))
        calls = []
        def state_function(model_argument, trainer_argument):
            self.assertIs(model_argument, model)
            self.assertIs(trainer_argument, trainer)
            calls.append(1)
            return copy.deepcopy(state)
        observer.bind(model, trainer, state, state_function)
        observer.boundaries = [dict(label='natural-before-series')] + [
            dict(label=f'natural-round-{r:04d}-{side}')
            for r in diagnostic.soak.ROUNDS for side in ('before', 'after')]
        return observer, model, trainer, state, calls

    def test_post_series_probe_order_and_exact_learning_state(self):
        with tempfile.TemporaryDirectory() as directory:
            observer, model, trainer, state, calls = self.initialized_observer(directory)
            operations = []
            with patch.object(diagnostic.time, 'sleep', lambda seconds: operations.append(('idle', seconds))), \
                 patch.object(diagnostic.gc, 'collect', lambda: operations.append(('gc', None)) or 17), \
                 patch.object(observer.allocator, 'trim', lambda: operations.append(('trim', 0)) or 1):
                observer.interventions()
            self.assertEqual(operations, [('idle', 2.), ('gc', None), ('trim', 0)])
            self.assertEqual(len(calls), 6)
            self.assertTrue(observer.completed)
            self.assertEqual([p['name'] for p in observer.phases], ['idle-control', 'gc-collect', 'malloc-trim'])
            self.assertTrue(all(p['state_before'] == p['state_after'] == state for p in observer.phases))
            self.assertTrue(all(p['model_both_adams_pcg_unchanged'] for p in observer.phases))
            self.assertTrue(all(not p['used_to_pass_normal_lifecycle_gates'] for p in observer.phases))
            self.assertTrue(observer.report('passed')['no_tracemalloc'])
            with self.assertRaisesRegex(RuntimeError, 'repeated'):
                observer.interventions()

    def test_intervention_state_mutation_stays_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            observer, model, trainer, state, calls = self.initialized_observer(directory)
            with patch.object(diagnostic.time, 'sleep', lambda _: state.update(model='changed')):
                with self.assertRaisesRegex(RuntimeError, 'changed during'):
                    observer.interventions()
            self.assertFalse(observer.completed)


if __name__ == '__main__':
    unittest.main()
