"""Typed in-process RustMage bridge for the verified Standard feature schema."""
import ctypes as C
from functools import lru_cache
import json
import os
from pathlib import Path
import sys
import numpy as np

from . import card_traits
from .native_pipeline import KEYS
from .standard_features import StandardEncoder

# This bridge implements the pinned Standard encoder exactly. A width match is
# insufficient: action ordering or semantic definitions can change in place.
FINGERPRINT = 'bddb155245929c0dd3a3d59ab48c85f724739dd5767ae86a67146faf2abc9ac6'
ENGINE_REVISION = 'e732a4bb42774fc20672dc2aad3952f9d1b57ae7'


class Metadata(C.Structure):
    _fields_ = [('revision', C.c_uint64), ('turn', C.c_uint32),
                ('viewer', C.c_uint32), ('actor', C.c_int32),
                ('active', C.c_uint32), ('outcome', C.c_int32),
                ('actions', C.c_size_t), ('objects', C.c_size_t),
                ('life0', C.c_int64), ('life1', C.c_int64)]


class View(C.Structure):
    _fields_ = [('data', C.c_void_p), ('length', C.c_size_t),
                ('rank', C.c_size_t), ('dims', C.c_size_t * 4),
                ('dtype', C.c_uint32), ('key', C.c_uint32)]


def check(lib, value):
    if not value:
        raise RuntimeError(lib.ds_error().decode())
    return value


def library_path():
    root = Path(__file__).resolve().parents[3]
    filename = {'linux': 'libpatchwork_direct_standard.so',
                'darwin': 'libpatchwork_direct_standard.dylib',
                'win32': 'patchwork_direct_standard.dll'}[sys.platform]
    return Path(os.environ.get('DIRECT_STANDARD_LIBRARY', root / '.local/direct-standard/target-e732a4bb4277-capped256/release' / filename)).resolve()


@lru_cache(maxsize=1)
def library():
    from ..backends.rustmage import RUSTMAGE_REVISION
    if RUSTMAGE_REVISION != ENGINE_REVISION:
        raise RuntimeError('Direct Standard bridge uses a different RustMage revision; rebuild and verify it before playback')
    encoder = StandardEncoder()
    if encoder.manifest()['fingerprint'] != FINGERPRINT:
        raise RuntimeError('Direct Standard bridge does not support this feature schema; update and verify the encoder')
    path = library_path()
    if not path.is_file():
        raise FileNotFoundError(f'Direct Standard playback library not found at {path}; run python3 scripts/build-direct-standard.py')
    lib = C.CDLL(str(path.resolve()))
    definitions = [
        ('ds_version', [], C.c_uint32), ('ds_error', [], C.c_char_p),
        ('ds_new', [C.c_char_p], C.c_void_p), ('ds_free', [C.c_void_p], None),
        ('ds_observe', [C.c_void_p, C.c_uint32], C.c_void_p),
        ('ds_view_free', [C.c_void_p], None),
        ('ds_metadata', [C.c_void_p, C.POINTER(Metadata)], C.c_bool),
        ('ds_configure', [C.c_char_p], C.c_bool),
        ('ds_history_observe', [C.c_void_p, C.c_void_p], C.c_bool),
        ('ds_history_take', [C.c_void_p, C.c_void_p], C.c_void_p),
        ('ds_prepare', [C.c_void_p, C.c_bool, C.POINTER(C.c_size_t)], C.c_bool),
        ('ds_apply_choice', [C.c_void_p, C.c_void_p, C.c_size_t], C.c_bool),
        ('ds_apply_script', [C.c_void_p, C.c_void_p, C.c_char_p], C.c_bool),
        ('ds_choice_kind', [C.c_void_p, C.c_size_t], C.c_void_p),
        ('ds_choice_debug_json', [C.c_void_p, C.c_size_t], C.c_void_p),
        ('ds_debug_json', [C.c_void_p], C.c_void_p),
        ('ds_string_free', [C.c_void_p], None),
        ('ds_encode', [C.c_void_p], C.c_void_p),
        ('ds_arrays_views', [C.c_void_p, C.POINTER(C.c_size_t)], C.POINTER(View)),
        ('ds_arrays_free', [C.c_void_p], None)]
    for name, args, result in definitions:
        fn = getattr(lib, name); fn.argtypes = args; fn.restype = result
    if lib.ds_version() != 1:
        raise RuntimeError('Unsupported direct Standard bridge ABI')
    names = (*card_traits.CARDS, *card_traits.ALIASES)
    cfg = dict(fingerprint=FINGERPRINT, actions=list(encoder.actions),
        mana_values={n: card_traits.FACTS[card_traits.ALIASES.get(n,n)]['mana_value'] for n in names},
        traits={n: card_traits.traits_for(n) for n in names},
        identities={n: card_traits.ALIASES.get(n,n) for n in names},
        unknown_traits=card_traits.UNKNOWN_TRAITS)
    check(lib, lib.ds_configure(json.dumps(cfg, separators=(',', ':')).encode()))
    return lib


def validate_model(model):
    if model is None or dict(model.feature_spec) != StandardEncoder.feature_spec:
        raise ValueError('Model feature schema does not match the direct Standard encoder')


class _Owner:
    def __init__(self, lib, pointer):
        self.lib, self.pointer = lib, pointer
    def __del__(self):
        if self.pointer:
            self.lib.ds_arrays_free(self.pointer)
            self.pointer = None


def arrays(lib, pointer):
    owner = _Owner(lib, check(lib, pointer))
    count = C.c_size_t()
    views = check(lib, lib.ds_arrays_views(owner.pointer, C.byref(count)))
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


def string(lib, pointer):
    pointer = check(lib, pointer)
    try:
        return C.string_at(pointer).decode()
    finally:
        lib.ds_string_free(pointer)
