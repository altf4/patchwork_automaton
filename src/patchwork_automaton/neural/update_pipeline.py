"""Runtime-only input-cache profiles for frozen PPO updates.

No TensorFlow state or learning configuration is changed here. Automatic GPU
admission deliberately requires the single, mostly idle 24 GiB-class device
used by the cache validation; other machines retain the host preparation path.
"""
import csv
import os
from pathlib import Path
import re
import shutil
import subprocess


_MIB = 1024**2
_AUTO_TOTAL_MIB = 23 * 1024
_AUTO_FREE_MIB = 22 * 1024
_AUTO_CACHE_MAX_MIB = 8192
_CACHE_ATTRIBUTES = (
    'numerical_prepacking', 'cache_policy_batches',
    'policy_cache_device', 'policy_cache_device_bytes',
    'actor_epoch_cache_device', 'actor_epoch_cache_bytes',
)


def _budget(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer MiB budget')
    return value


def _gpu_inventory():
    """Read a bounded physical inventory without creating a GPU context."""
    executable = shutil.which('nvidia-smi')
    if executable is None:
        fallback = Path('/usr/lib/wsl/lib/nvidia-smi')
        executable = str(fallback) if fallback.is_file() and os.access(fallback, os.X_OK) else None
    if executable is None:
        return None, 'nvidia-smi is unavailable'
    try:
        result = subprocess.run(
            [executable, '--query-gpu=uuid,memory.total,memory.free',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None, 'GPU inventory query failed'
    rows = list(csv.reader(result.stdout.splitlines()))
    if len(rows) != 1 or len(rows[0]) != 3:
        return None, 'GPU inventory is not a single unambiguous device'
    uuid, total, free = (field.strip() for field in rows[0])
    if (re.fullmatch(r'GPU-[0-9a-fA-F-]+', uuid) is None
            or not total.isdecimal() or not free.isdecimal()):
        return None, 'GPU inventory has invalid memory or UUID fields'
    total, free = int(total), int(free)
    if total <= 0 or not 0 <= free <= total:
        return None, 'GPU inventory has inconsistent memory fields'
    return dict(uuid=uuid, total_mib=total, free_mib=free), None


def resolve_update_pipeline(learner_device, profile='auto',
                            policy_cache_mib=8192, actor_epoch_cache_mib=8192):
    """Return JSON-serializable runtime settings and their admission reason.

    ``host`` packs numerical records and caches canonical policy inputs in
    temporary mapped files. ``gpu`` additionally enables the two bounded GPU
    caches; a zero cap disables that cache. ``legacy`` disables all new caches.
    Automatic GPU admission allows at most the validated 8192 MiB per cache.
    An explicit GPU profile bypasses that cap and the capacity gate, so its
    caller must choose budgets that fit alongside other allocations.
    """
    if learner_device not in ('cpu', 'gpu'):
        raise ValueError('Learner device must be cpu or gpu')
    if profile not in ('auto', 'host', 'gpu', 'legacy'):
        raise ValueError('Update pipeline profile must be auto, host, gpu or legacy')
    policy_cache_mib = _budget(policy_cache_mib, 'Policy cache')
    actor_epoch_cache_mib = _budget(actor_epoch_cache_mib, 'Actor epoch cache')
    if profile == 'gpu' and learner_device != 'gpu':
        raise ValueError('GPU update pipeline requires a GPU learner')

    resolved = profile
    inventory = None
    if profile == 'auto':
        resolved = 'host'
        if learner_device == 'cpu':
            reason = 'CPU learner uses host input preparation'
        elif max(policy_cache_mib, actor_epoch_cache_mib) > _AUTO_CACHE_MAX_MIB:
            reason = 'Cache caps above the validated 8192 MiB each require the explicit gpu profile'
        else:
            inventory, failure = _gpu_inventory()
            visible = os.environ.get('CUDA_VISIBLE_DEVICES')
            if failure:
                reason = failure
            elif visible is not None and visible not in ('0', inventory['uuid']):
                reason = 'CUDA_VISIBLE_DEVICES does not unambiguously select the inventory device'
            elif inventory['total_mib'] < _AUTO_TOTAL_MIB:
                reason = 'GPU total memory is below the automatic 23 GiB admission threshold'
            elif inventory['free_mib'] < _AUTO_FREE_MIB:
                reason = 'GPU free memory is below the automatic 22 GiB admission threshold'
            else:
                resolved = 'gpu'
                reason = 'Single visible GPU meets the validated total and free memory thresholds'
    else:
        reason = f'Explicit {profile} profile'

    gpu = resolved == 'gpu'
    return dict(
        requested_profile=profile, resolved_profile=resolved, reason=reason,
        learner_device=learner_device, gpu_memory=inventory,
        requested_policy_cache_mib=policy_cache_mib,
        requested_actor_epoch_cache_mib=actor_epoch_cache_mib,
        numerical_prepacking=resolved != 'legacy',
        cache_policy_batches=resolved != 'legacy',
        policy_cache_device='/GPU:0' if gpu and policy_cache_mib else None,
        policy_cache_device_bytes=policy_cache_mib * _MIB if gpu else 0,
        actor_epoch_cache_device='/GPU:0' if gpu and actor_epoch_cache_mib else None,
        actor_epoch_cache_bytes=actor_epoch_cache_mib * _MIB if gpu else 0,
    )


def apply_update_pipeline(trainer, settings):
    """Apply only existing cache attributes; never touch model, Adam or RNG."""
    missing = [name for name in _CACHE_ATTRIBUTES if not hasattr(trainer, name)]
    if missing:
        raise ValueError(f'Trainer does not support update pipeline attributes: {missing}')
    # Resolve the whole assignment before mutating anything, so a missing
    # settings field cannot leave a partially configured trainer.
    values = {name: settings[name] for name in _CACHE_ATTRIBUTES}
    for name, value in values.items():
        setattr(trainer, name, value)
