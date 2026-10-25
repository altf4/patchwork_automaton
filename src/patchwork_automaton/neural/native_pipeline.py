"""Owned Rust observations and zero-copy NumPy views for TensorFlow input.

The engine's redacted JSON response is parsed directly from its C allocation.
Python materializes the full object tree only for callers that explicitly need
it (e.g. a baseline policy). Numerical buffers retain their Rust allocation owner
through NumPy's ctypes base; advancing/closing a game cannot invalidate them.
"""
from collections.abc import Mapping
import ctypes as C
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
import threading
from types import MethodType

import numpy as np
import orjson

from ..core import Response, freeze
from ..errors import EventOverflow, UnsupportedInteraction

KEYS = ('entities', 'card_ids', 'entity_mask', 'semantics', 'semantic_mask',
        'relations', 'context', 'candidates', 'candidate_refs', 'candidate_mask',
        'selection', 'history', 'history_cards', 'history_mask')


def enabled():
    value = os.environ.get('PATCHWORK_OBSERVATION_PIPELINE', 'python')
    if value not in ('python', 'rust'):
        raise ValueError('PATCHWORK_OBSERVATION_PIPELINE must be python or rust')
    return value == 'rust'


class View(C.Structure):
    _fields_ = [('data', C.c_void_p), ('length', C.c_size_t), ('rank', C.c_size_t),
                ('dims', C.c_size_t * 4), ('dtype', C.c_uint32), ('key', C.c_uint32)]


@lru_cache(maxsize=1)
def library():
    root = Path(__file__).resolve().parents[3]
    filename = {'linux': 'libpatchwork_observation_pipeline.so',
                'darwin': 'libpatchwork_observation_pipeline.dylib',
                'win32': 'patchwork_observation_pipeline.dll'}[sys.platform]
    path = Path(os.environ.get('PATCHWORK_OBSERVATION_LIBRARY',
                              root / '.local/observation-target/release' / filename))
    if not path.is_file():
        raise FileNotFoundError('Build the Rust observation pipeline with '
                                'scripts/build-observation-pipeline.py: ' + str(path))
    lib = C.CDLL(str(path))
    signatures = {
        'pp_abi_version': ([], C.c_uint32), 'pp_error': ([], C.c_char_p),
        'pp_configure': ([C.c_char_p], C.c_bool),
        'pp_parse': ([C.c_void_p, C.c_bool], C.c_void_p),
        'pp_observation_free': ([C.c_void_p], None),
        'pp_metadata': ([C.c_void_p], C.c_char_p), 'pp_json': ([C.c_void_p], C.c_char_p),
        'pp_fingerprint': ([C.c_void_p], C.c_void_p),
        'pp_choices': ([C.c_void_p, C.c_bool], C.c_char_p),
        'pp_encode': ([C.c_void_p, C.c_bool], C.c_void_p),
        'pp_arrays_free': ([C.c_void_p], None),
        'pp_arrays_bytes': ([C.c_void_p], C.c_size_t),
        'pp_arrays_views': ([C.c_void_p, C.POINTER(C.c_size_t)], C.POINTER(View)),
        'pp_history_new': ([C.c_size_t], C.c_void_p), 'pp_history_free': ([C.c_void_p], None),
        'pp_history_observe': ([C.c_void_p, C.c_void_p], C.c_bool),
        'pp_history_take': ([C.c_void_p], C.c_void_p),
        'pp_collate': ([C.POINTER(View), C.c_size_t], C.c_void_p),
    }
    for name, (args, result) in signatures.items():
        fn = getattr(lib, name); fn.argtypes = args; fn.restype = result
    if lib.pp_abi_version() != 1:
        raise RuntimeError('Unsupported observation pipeline ABI')
    from .rustmage_features import RustMageEncoder, CARDS, SEMANTICS, ACTION_FIELDS
    from .features import ZONES
    manifest = RustMageEncoder().manifest()
    config = dict(fingerprint=manifest['fingerprint'], schema=manifest['observation_schema'],
                  cards=CARDS, semantics=[SEMANTICS[c] for c in CARDS],
                  mana_values=[manifest['facts'][c]['mana_value'] for c in CARDS],
                  actions=list(ACTION_FIELDS.items()), history_zones=ZONES,
                  **{k: manifest[k] for k in ('zones', 'steps', 'decisions', 'counters', 'flags', 'player_stats')})
    if not lib.pp_configure(orjson.dumps(config)):
        raise RuntimeError(lib.pp_error().decode())
    return lib


def check(value, lib=None):
    if not value:
        message = (lib or library()).pp_error().decode()
        if 'history overflow' in message:
            raise EventOverflow(message)
        raise UnsupportedInteraction(message)
    return value


class Owner:
    def __init__(self, pointer, free):
        self.pointer, self.free = pointer, free

    def __del__(self):
        if self.pointer:
            self.free(self.pointer)
            self.pointer = None


def arrays(pointer):
    lib = library()
    owner = Owner(check(pointer, lib), lib.pp_arrays_free)
    owner.native_nbytes = lib.pp_arrays_bytes(owner.pointer)
    count = C.c_size_t()
    views = lib.pp_arrays_views(owner.pointer, C.byref(count))
    result = {}
    for v in views[:count.value]:
        ctype = (C.c_float, C.c_int32, C.c_uint8)[v.dtype]
        storage = (ctype * v.length).from_address(v.data)
        storage._allocation_owner = owner
        array = np.ctypeslib.as_array(storage)
        if v.dtype == 2:
            array = array.view(np.bool_)
        result[KEYS[v.key]] = array.reshape(tuple(v.dims[:v.rank]))
    return result


class NativeObservation(Mapping):
    def __init__(self, pointer):
        self.lib = library()
        self.owner = Owner(check(pointer, self.lib), self.lib.pp_observation_free)
        self.metadata = orjson.loads(self.lib.pp_metadata(pointer))
        self._full = None
        self._choices = {}
        self._lock = threading.RLock()

    @classmethod
    def from_raw(cls, raw):
        from ..backends.rustmage import _plain
        payload = orjson.dumps(_plain(raw))
        return cls(library().pp_parse(payload, False))

    def full(self):
        with self._lock:
            if self._full is None:
                self._full = orjson.loads(self.lib.pp_json(self.owner.pointer))
            return self._full

    def __getitem__(self, key):
        if key in self.metadata:
            return self.metadata[key]
        return self.full()[key]

    def __iter__(self): return iter(self.full())
    def __len__(self): return len(self.full())

    def fingerprint(self):
        with self._lock:
            return C.string_at(self.lib.pp_fingerprint(self.owner.pointer), 32)

    def native_choices(self, allow_concede=False):
        with self._lock:
            if allow_concede not in self._choices:
                payload = check(self.lib.pp_choices(self.owner.pointer, allow_concede), self.lib)
                self._choices[allow_concede] = tuple((Response(c['option'], c['value']),
                                                     freeze(c['action']) if c['option'].startswith('action:') else c['action'])
                                                   for c in orjson.loads(payload))
            return self._choices[allow_concede]

    def encoded(self, token, allow_concede=False):
        from .rustmage_features import EncodedRustDecision
        with self._lock:
            choices = self.native_choices(allow_concede)
            values = arrays(self.lib.pp_encode(self.owner.pointer, allow_concede))
        return EncodedRustDecision(values, tuple(c[0] for c in choices), tuple(c[1] for c in choices), token)


class NativeHistory:
    def __init__(self, registry=None, limit=4096):
        self.limit, self._states, self._lock = limit, {}, threading.Lock()
        self.lib = library()

    def reset(self):
        with self._lock: self._states.clear()

    def _state(self, obs):
        key = obs.game, obs.actor
        if key not in self._states:
            self._states[key] = Owner(self.lib.pp_history_new(self.limit), self.lib.pp_history_free)
        return self._states[key].pointer

    def observe(self, obs):
        raw = obs.native if isinstance(obs.native, NativeObservation) else NativeObservation.from_raw(obs.native)
        # Serialize readers with the observation's mutable legal-choice cache,
        # including callers that share an observation between policy instances.
        with self._lock, raw._lock:
            check(self.lib.pp_history_observe(self._state(obs), raw.owner.pointer), self.lib)

    def take(self, obs):
        with self._lock:
            return arrays(self.lib.pp_history_take(self._state(obs)))


def collate(items):
    if not items: raise ValueError('Cannot collate an empty batch')
    views = (View * (len(items) * len(KEYS)))()
    keepalive = []
    dtypes = {np.dtype('float32'): 0, np.dtype('int32'): 1, np.dtype('bool'): 2}
    for b, item in enumerate(items):
        for k, key in enumerate(KEYS):
            value = np.ascontiguousarray(item[key])
            if value.dtype not in dtypes or not 1 <= value.ndim <= 3:
                raise ValueError('Native batching requires float32/int32/bool records of rank 1–3')
            keepalive.append(value)
            v = views[b * len(KEYS) + k]
            v.data, v.length, v.rank, v.dtype, v.key = value.ctypes.data, value.size, value.ndim, dtypes[value.dtype], k
            v.dims[:value.ndim] = value.shape
    try:
        return arrays(library().pp_collate(views, len(items)))
    except UnsupportedInteraction as e:
        raise ValueError(str(e)) from e


def enable_native_observations(game):
    """Intercept the existing JSON ABI without changing engine state or its pin."""
    lib = library()
    original = game.request
    engine_errors = original.__func__.__globals__

    def request(self, op, **fields):
        if op not in ('observe', 'step', 'act'):
            return original(op, **fields)
        with self._lock:
            if not self._handle:
                raise engine_errors['EngineError']('game is closed')
            payload = json.dumps(dict(op=op, **fields)).encode()
            pointer = self._lib.rm_request(self._handle, payload)
            if not pointer:
                raise engine_errors['EngineError']('null native response')
            try:
                observation = lib.pp_parse(pointer, True)
                if observation:
                    return {'observation': NativeObservation(observation)}
                response = orjson.loads(C.string_at(pointer))
                if response.get('error'):
                    if response.get('capacity'):
                        raise engine_errors['CapacityError'](response['error'], response['capacity']['resource'])
                    raise engine_errors['EngineError'](response['error'])
                return response
            finally:
                self._lib.rm_string_free(pointer)

    game.request = MethodType(request, game)
