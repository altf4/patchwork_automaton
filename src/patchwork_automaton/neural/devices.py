"""Explicit TensorFlow placement for independent learners and frozen replicas.

Importing this module does not import TensorFlow. Configure visibility before
constructing tensors/models, then enter device_scope around neural work. Device
scopes are thread-local, so inference executors enter their own scopes too.
"""
from contextlib import contextmanager
import os


def canonical_device(device):
    names = {'cpu': '/CPU:0', 'gpu': '/GPU:0', '/cpu:0': '/CPU:0', '/gpu:0': '/GPU:0'}
    try:
        return names[str(device).lower()]
    except KeyError:
        raise ValueError('Neural device must be cpu or gpu') from None


def configure_runtime(device, *, hide_gpu=False):
    """Configure a process before model construction; GPU requests fail closed.

    CPU replicas hide GPUs even if their spawning learner has CUDA enabled.
    Spawn may already have imported TensorFlow through __main__, but it must not
    have initialized its devices. A late visibility change fails explicitly.
    """
    target = canonical_device(device)
    if hide_gpu and target != '/CPU:0':
        raise ValueError('hide_gpu is only valid for a CPU runtime')
    if hide_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import tensorflow as tf
    if hide_gpu:
        try:
            tf.config.set_visible_devices([], 'GPU')
        except RuntimeError as error:
            raise RuntimeError('CPU inference must hide GPUs before TensorFlow device initialization') from error
    if target == '/GPU:0':
        gpus = tf.config.list_physical_devices('GPU')
        if not gpus:
            raise RuntimeError('GPU requested but TensorFlow sees no GPU; CPU fallback is disabled')
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as error:
                if not tf.config.experimental.get_memory_growth(gpu):
                    raise RuntimeError('Enable GPU memory growth before TensorFlow device initialization') from error
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    return target


@contextmanager
def device_scope(device):
    import tensorflow as tf
    with tf.device(canonical_device(device)):
        yield


def _require_placements(placements, device, label):
    expected = canonical_device(device).lstrip('/').upper()
    if not placements or any(not str(value).upper().endswith(expected) for value in placements.values()):
        raise RuntimeError(f'{label} must be on {canonical_device(device)}; got {placements}')
    return placements


def assert_model_device(model, device):
    """Check variable storage placement after a model has been built/loaded."""
    placements = {}
    for index, variable in enumerate(model.variables):
        value = getattr(variable, 'value', variable)
        if callable(value):
            value = value()
        placements[getattr(variable, 'path', getattr(variable, 'name', str(index)))] = getattr(value, 'device', None)
    return _require_placements(placements, device, 'Model variables')


def assert_output_device(outputs, device):
    """Inspect tensors before NumPy conversion, rather than inferring placement."""
    return _require_placements({key: getattr(value, 'device', None) for key, value in outputs.items()},
                               device, 'Neural output tensors')
