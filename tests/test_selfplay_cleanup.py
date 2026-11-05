"""A failed inference role must not prevent cleanup of the other role."""
import asyncio
from types import SimpleNamespace
import unittest

from patchwork_automaton.rl.standard_selfplay_processes import StandardSelfplayCollector


class SelfplayCleanupTests(unittest.TestCase):
    def collector(self, failures):
        calls = []

        def step(name):
            calls.append(name)
            if name in failures:
                raise failures[name]

        async def close_role(name):
            step(name)

        collector = StandardSelfplayCollector.__new__(StandardSelfplayCollector)
        collector.closed = False
        collector.pool = SimpleNamespace(shutdown=lambda **kwargs: step('pool'))
        collector.replicas = [SimpleNamespace(close=lambda name=n: close_role(name))
                              for n in ('learner', 'opponent')]
        collector.connections = [[SimpleNamespace(close=lambda: step('server'))]]
        collector.worker_connections = [[SimpleNamespace(close=lambda: step('worker'))]]
        return collector, calls

    def test_failed_first_role_still_closes_other_role_and_all_connections(self):
        failure = RuntimeError('learner cleanup failed')
        collector, calls = self.collector({'learner': failure})
        with self.assertRaises(RuntimeError) as caught:
            asyncio.run(collector.close())
        self.assertIs(caught.exception, failure)
        self.assertEqual(calls, ['pool', 'learner', 'opponent', 'server', 'worker'])
        asyncio.run(collector.close())
        self.assertEqual(len(calls), 5)

    def test_multiple_cleanup_errors_are_retained_after_every_resource_is_attempted(self):
        errors = {'pool': RuntimeError('pool'), 'learner': ValueError('learner'),
                  'server': OSError('server')}
        collector, calls = self.collector(errors)
        with self.assertRaises(BaseExceptionGroup) as caught:
            asyncio.run(collector.close())
        self.assertEqual(caught.exception.exceptions, tuple(errors.values()))
        self.assertEqual(calls, ['pool', 'learner', 'opponent', 'server', 'worker'])


if __name__ == '__main__':
    unittest.main()
