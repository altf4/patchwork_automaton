"""Bounded POSIX pipe exchanges and disposable endpoints for native games.

Only a persistent worker touches its reusable inference connections. A game
child receives fresh proxy connections, so killing that child cannot leave a
partially consumed frame on a connection that a later game will inherit.
"""
from contextlib import contextmanager
import io
import math
from multiprocessing.reduction import ForkingPickler
import os
import select
import struct
import threading
import time


_CHUNK_BYTES = 64 * 1024
_CLEANUP_SECONDS = 121.


def _deadline(timeout):
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout < 0:
        raise ValueError('Finite nonnegative IPC timeout required')
    return time.monotonic() + timeout


@contextmanager
def _nonblocking(connection):
    """The endpoint must have exactly one owning thread for the whole exchange."""
    if os.name != 'posix' or not hasattr(select, 'poll'):
        raise RuntimeError('Deadline-bounded inference pipes require POSIX poll support')
    fd = connection.fileno()
    blocking = os.get_blocking(fd)
    os.set_blocking(fd, False)
    try:
        yield fd
    finally:
        # A failed owner may invalidate the endpoint during bounded shutdown.
        if not connection.closed:
            os.set_blocking(fd, blocking)


def _ready(fd, deadline, *, writing=False):
    poller = select.poll()
    poller.register(fd, select.POLLOUT if writing else select.POLLIN)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Inference IPC frame timed out')
        try:
            ready = poller.poll(max(1, math.ceil(min(remaining, .1)*1000)))
        except InterruptedError:
            continue
        if ready:
            return


def _write(fd, data, deadline):
    view = memoryview(data)
    while view:
        _ready(fd, deadline, writing=True)
        try:
            count = os.write(fd, view[:_CHUNK_BYTES])
        except (BlockingIOError, InterruptedError):
            continue
        if count <= 0:
            raise BrokenPipeError('Inference IPC write made no progress')
        view = view[count:]


def _read(fd, size, deadline):
    # Do not preallocate an untrusted announced length. Retain multiprocessing's
    # complete 4/8-byte framing without truncating observations or action sets.
    result = io.BytesIO()
    remaining = size
    while remaining:
        _ready(fd, deadline)
        try:
            chunk = os.read(fd, min(remaining, _CHUNK_BYTES))
        except (BlockingIOError, InterruptedError):
            continue
        if not chunk:
            if remaining == size:
                raise EOFError('Inference IPC peer closed')
            raise OSError('Inference IPC peer closed during a frame')
        result.write(chunk)
        remaining -= len(chunk)
    return result.getvalue()


def send_message(connection, value, timeout=120):
    """Send one stdlib-compatible pickle frame within one absolute deadline."""
    deadline = _deadline(timeout)
    payload = ForkingPickler.dumps(value)
    size = len(payload)
    header = struct.pack('!i', size) if size <= 0x7fffffff else struct.pack('!iQ', -1, size)
    with _nonblocking(connection) as fd:
        _write(fd, header, deadline)
        _write(fd, payload, deadline)


def recv_message(connection, timeout=120):
    """Receive an entire frame, including its body, before the deadline."""
    deadline = _deadline(timeout)
    with _nonblocking(connection) as fd:
        size, = struct.unpack('!i', _read(fd, 4, deadline))
        if size == -1:
            size, = struct.unpack('!Q', _read(fd, 8, deadline))
        elif size < 0:
            raise ValueError('Invalid negative inference IPC frame length')
        payload = _read(fd, size, deadline)
    if time.monotonic() >= deadline:
        raise TimeoutError('Inference IPC frame timed out')
    return ForkingPickler.loads(payload)


class InferenceCleanupError(RuntimeError):
    """A completed result is retained even when its transport cannot be reused."""

    def __init__(self, message, *, completed_row=None):
        super().__init__(message)
        self.completed_row = completed_row


class InferenceIsolation:
    """Prepare pipes before fork; start parent-owned forwarding only afterward."""

    def __init__(self, context, options):
        from .rustmage_processes import _RemoteInference
        self.options = dict(options)
        self.relays = []
        self.stopping = threading.Event()
        self.started = self.closed = False
        clients = {}
        descriptors = set()
        try:
            for role in ('batcher', 'opponent_batcher'):
                batcher = options.get(role)
                if batcher is None:
                    continue
                if not isinstance(batcher, _RemoteInference):
                    raise TypeError('Isolated games require worker-owned remote inference')
                if batcher.broken or batcher.connection.closed:
                    raise RuntimeError('Cannot reuse broken game inference')
                if id(batcher) not in clients:
                    fd = batcher.connection.fileno()
                    if fd in descriptors:
                        raise ValueError('Different inference clients share one endpoint')
                    descriptors.add(fd)
                    server, child = context.Pipe()
                    client = _RemoteInference(child)
                    self.relays.append(dict(batcher=batcher, server=server, client=client,
                                            thread=None, error=None))
                    clients[id(batcher)] = client
                self.options[role] = clients[id(batcher)]
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                error.add_note('Inference isolation setup cleanup also failed: '+repr(cleanup_error))
            raise

    def child_close_connections(self):
        return tuple(connection for relay in self.relays
                     for connection in (relay['batcher'].connection, relay['server']))

    def invalidate(self):
        for relay in self.relays:
            relay['batcher'].invalidate()

    def _forward(self, relay):
        from ..neural.inference_processes import bridge_requests
        try:
            bridge_requests(relay['server'], relay['batcher'], self.stopping)
        except BaseException as error:
            relay['error'] = error
            relay['batcher'].invalidate()
        finally:
            relay['server'].close()

    def start(self):
        if self.started or self.closed:
            raise RuntimeError('Inference isolation is unavailable')
        self.started = True
        # These copies are no longer needed after fork. Closing them is required
        # for a dead game child to yield EOF instead of stranding a proxy reader.
        for relay in self.relays:
            relay['client'].connection.close()
        try:
            for relay in self.relays:
                thread = threading.Thread(target=self._forward, args=(relay,),
                                          name='game-inference-proxy', daemon=True)
                relay['thread'] = thread
                thread.start()
        except BaseException as error:
            self.invalidate()
            try:
                self.close()
            except BaseException as cleanup_error:
                error.add_note('Inference isolation startup cleanup also failed: '+repr(cleanup_error))
            raise

    def close(self, timeout=_CLEANUP_SECONDS):
        if self.closed:
            return
        deadline = _deadline(timeout)
        self.closed = True
        self.stopping.set()
        errors = []
        for relay in self.relays:
            try:
                relay['client'].connection.close()
            except BaseException as error:
                errors.append(repr(error))
            thread = relay['thread']
            if thread is not None and thread.ident is not None:
                try:
                    thread.join(max(0., deadline-time.monotonic()))
                except BaseException as error:
                    errors.append(repr(error))
                if thread.is_alive():
                    relay['batcher'].invalidate()
                    try:
                        relay['server'].close()
                    except BaseException as error:
                        errors.append(repr(error))
                    # No new game can be forked after this error. Try to reap a
                    # thread whose bounded operation noticed invalidation.
                    try:
                        thread.join(1.)
                    except BaseException as error:
                        errors.append(repr(error))
                    errors.append('Inference proxy did not drain before cleanup deadline')
            try:
                relay['server'].close()
            except BaseException as error:
                errors.append(repr(error))
            if relay['error'] is not None:
                errors.append(repr(relay['error']))
            if relay['batcher'].broken:
                errors.append('Persistent inference endpoint is poisoned')
        if errors:
            for relay in self.relays:
                relay['batcher'].invalidate()
            raise InferenceCleanupError('; '.join(errors))
