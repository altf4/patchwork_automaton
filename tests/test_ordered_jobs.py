"""Bounded concurrency, canonical result order and cleanup on all exit paths."""
import threading
import unittest
from patchwork_automaton.neural.ordered_jobs import ordered_jobs


class OrderedJobsTests(unittest.TestCase):
    def test_parallel_execution_preserves_source_order(self):
        barrier = threading.Barrier(2,timeout=10)
        def work(i):
            if i < 2: barrier.wait()
            return i*i
        with ordered_jobs(work,range(12),workers=2,capacity=3) as results:
            self.assertEqual(list(results),[i*i for i in range(12)])

    def test_early_exit_bounds_work_and_joins_workers(self):
        started = []; finished = []
        def work(i):
            started.append(i); finished.append(i)
            return i
        with ordered_jobs(work,range(10000),workers=2,capacity=3) as results:
            self.assertEqual(next(results),0)
        self.assertLessEqual(len(started),4)
        self.assertEqual(sorted(started),sorted(finished))
        self.assertFalse(any(t.name.startswith('ppo-jobs') for t in threading.enumerate()))

    def test_worker_error_propagates_and_joins_other_workers(self):
        barrier = threading.Barrier(2,timeout=10)
        finished = threading.Event()
        def work(i):
            if i < 2: barrier.wait()
            if i == 0: raise RuntimeError('worker failed')
            finished.set()
            return i
        with self.assertRaisesRegex(RuntimeError,'worker failed'):
            with ordered_jobs(work,range(100),workers=2,capacity=2) as results:
                list(results)
        self.assertTrue(finished.is_set())
        self.assertFalse(any(t.name.startswith('ppo-jobs') for t in threading.enumerate()))

    def test_source_error_also_cleans_up(self):
        def source():
            yield 0
            raise RuntimeError('source failed')
        with self.assertRaisesRegex(RuntimeError,'source failed'):
            with ordered_jobs(lambda i:i,source(),workers=1,capacity=2) as results:
                list(results)
        self.assertFalse(any(t.name.startswith('ppo-jobs') for t in threading.enumerate()))

    def test_rejects_unbounded_or_invalid_capacity(self):
        for workers,capacity in ((0,1),(1,0),(3,2),(1,True)):
            with self.assertRaises(ValueError):
                with ordered_jobs(lambda i:i,[],workers=workers,capacity=capacity): pass
