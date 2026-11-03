"""Explicit codec dispatch; v1 remains unchanged and v2 never adopts old records."""
V1='sparse-factored-standard-v1'
V2='sparse-identity-standard-v2'

def is_compact(model):return getattr(model,'input_format',None) in (V1,V2)

def codec(model_or_version):
    version=model_or_version if isinstance(model_or_version,str) else model_or_version.input_format
    if version==V1:
        from . import compact_standard as module
    elif version==V2:
        from . import compact_identity as module
    else:raise ValueError('Unknown compact input representation')
    return module

def neural(model_or_version):
    version=model_or_version if isinstance(model_or_version,str) else model_or_version.input_format
    if version==V1:
        from . import compact_standard_model as module
    elif version==V2:
        from . import compact_identity_model as module
    else:raise ValueError('Unknown compact input representation')
    return module

def record_version(record):return getattr(record,'input_format',V1)
