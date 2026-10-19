"""Bounded ordered background work, always joined before leaving its scope."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager


@contextmanager
def ordered_jobs(function, items, *, workers, capacity, name='ppo-jobs'):
    """Yield results in input order with at most capacity outstanding futures.

    The consumer may hold one additional yielded result. Input iteration stays
    on the calling thread. On failure or early exit, pending work is cancelled
    and running work is joined before the caller can mutate model weights or
    close shared rollout files.
    """
    if type(workers) is not int or type(capacity) is not int or not 1 <= workers <= capacity:
        raise ValueError('Expected positive workers and capacity >= workers')
    source = iter(items)
    pending = deque()
    executor = ThreadPoolExecutor(max_workers=workers,thread_name_prefix=name)
    def submit_next():
        try:
            item = next(source)
        except StopIteration:
            return
        pending.append(executor.submit(function,item))
    def results():
        while pending:
            result = pending.popleft().result()
            submit_next()
            yield result
    stream = results()
    try:
        for _ in range(capacity): submit_next()
        yield stream
    finally:
        stream.close()
        for future in pending: future.cancel()
        executor.shutdown(wait=True,cancel_futures=True)
