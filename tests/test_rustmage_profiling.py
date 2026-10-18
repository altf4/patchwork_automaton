"""Accounting checks for the optional collection profiler."""
from concurrent.futures import ThreadPoolExecutor
import threading
import unittest
from unittest.mock import patch

from test_neural import AVAILABLE

if AVAILABLE:
    from examples.profile_rustmage import StageTimer


@unittest.skipUnless(AVAILABLE, 'Install neural dependencies with uv')
class ProfilingAccountingTests(unittest.TestCase):
    def test_nested_time_is_not_double_counted(self):
        timer = StageTimer()
        with patch('examples.profile_rustmage.time.perf_counter', side_effect=[0.,1.,4.,10.]), \
                patch('examples.profile_rustmage.time.thread_time', side_effect=[0.,1.,3.,8.]):
            with timer.span('outer'):
                with timer.span('inner'):
                    pass
        rows = timer.summary()
        self.assertEqual(rows['outer']['exclusive_thread_cpu_seconds'],6.)
        self.assertEqual(rows['inner']['exclusive_thread_cpu_seconds'],2.)
        self.assertEqual(rows['outer']['exclusive_wall_seconds'],7.)
        self.assertEqual(sum(r['exclusive_wall_seconds'] for r in rows.values()),10.)

    def test_exception_restores_stack(self):
        timer = StageTimer()
        with self.assertRaises(ValueError):
            with timer.span('failed'):
                raise ValueError('expected')
        with timer.span('next'):
            pass
        self.assertEqual({k:v['calls'] for k,v in timer.summary().items()}, {'failed':1,'next':1})

    def test_concurrent_threads_have_independent_stacks(self):
        timer = StageTimer(); barrier = threading.Barrier(2)
        def worker(_):
            with timer.span('worker'):
                barrier.wait(timeout=5)
                with timer.span('child'):
                    sum(range(1000))
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(worker,range(2)))
        rows = timer.summary()
        self.assertEqual(rows['worker']['calls'],2)
        self.assertEqual(rows['child']['calls'],2)
        self.assertAlmostEqual(sum(r['exclusive_thread_cpu_seconds'] for r in rows.values()),
                               rows['worker']['inclusive_thread_cpu_seconds'])
