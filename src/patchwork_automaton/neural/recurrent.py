"""Opt-in execution backend for existing GRU weights; no new parameters.

Select a backend before tracing a model/trainer. Backend selection is a runtime
choice, deliberately not checkpoint state: saved policies still load on CPUs.
"""
import keras
from keras import layers, ops


def configure_gru_backend(model, backend):
    """Select an execution backend before building TensorFlow graphs.

    'cudnn-memory' fuses history/state memory but leaves the semantic GRUs
    unfused: their large flattened entity batch and short sequences benchmark
    poorly with cuDNN. 'cudnn' fuses all GRUs; 'auto' allows Keras fallback.

    Strict cuDNN raises if unavailable instead of silently benchmarking a CPU or
    unfused fallback. Recreate trainer graphs after changing an already traced
    model. Learned weights, optimizer slots and model config are untouched.
    """
    modes = {'unfused': False, 'auto': 'auto', 'cudnn': True, 'cudnn-memory': True}
    if backend not in modes:
        raise ValueError('GRU backend must be unfused, auto, cudnn or cudnn-memory')
    if backend.startswith('cudnn') and keras.backend.backend() != 'tensorflow':
        raise ValueError('Strict cuDNN requires the TensorFlow backend')
    grus = [layer for layer in model._flatten_layers() if isinstance(layer, layers.GRU)]
    if not grus:
        raise ValueError('Model has no GRUs')
    semantic_ids = {id(layer.semantic_gru) for layer in model._flatten_layers()
                    if hasattr(layer, 'semantic_gru')}
    for layer in grus:
        layer.use_cudnn = (False if backend == 'cudnn-memory' and id(layer) in semantic_ids
                          else modes[backend])
        # cuDNN RNNs are not XLA-compatible. None of the current PPO functions
        # use jit_compile; declare this accurately for other model callers too.
        if backend != 'unfused':
            layer.supports_jit = False
    return gru_backend_summary(model)


def gru_backend_summary(model):
    """Report the actual per-layer runtime choices without changing graphs."""
    return [dict(path=layer.path, units=layer.units, use_cudnn=layer.use_cudnn)
            for layer in model._flatten_layers() if isinstance(layer, layers.GRU)]


def masked_gru(layer, inputs, *, mask, **kwargs):
    """Preserve a final-output GRU's empty-row semantics on the cuDNN path.

    cuDNN accepts only nonempty, right-padded sequences. Give empty rows a
    disposable first timestep, then select exact zero outputs for those rows.
    The selection also removes gradients through their disposable computation.
    The model's existing empty-history branch restores its initial memory.
    Nonempty masks are unchanged; cuDNN still rejects holes/left padding.
    """
    if layer.use_cudnn is False:
        return layer(inputs, mask=mask, **kwargs)
    if layer.return_sequences or layer.return_state:
        raise ValueError('masked_gru requires a final-output-only GRU')
    nonempty = ops.any(mask, axis=1, keepdims=True)
    first = ops.arange(ops.shape(mask)[1])[None, :] == 0
    safe_mask = ops.logical_or(mask, ops.logical_and(ops.logical_not(nonempty), first))
    output = layer(inputs, mask=safe_mask, **kwargs)
    return ops.where(nonempty, output, ops.zeros_like(output))
