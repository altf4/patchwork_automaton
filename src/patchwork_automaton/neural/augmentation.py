"""Numeric observation augmentation, applied before policy rollout collection."""
import numpy as np
from .features import ABILITY_ID_SLICE


def mask_identities(inputs, rng, probability):
    """Hide some current-object identities but retain their complete semantics.

    Source-linked candidate ability IDs are masked together with the source
    card ID. The returned inputs must be used both to sample actions and for
    every PPO epoch on those samples. History and remembered state are untouched.
    """
    if not 0 <= probability <= 1:
        raise ValueError('Identity dropout must be in [0, 1]')
    if not probability:
        return inputs
    masked=(rng.random(inputs['entity_mask'].shape)<probability)&inputs['entity_mask']
    result=dict(inputs,card_ids=inputs['card_ids'].copy(),candidates=inputs['candidates'].copy())
    result['card_ids'][masked]=0
    candidates=np.any(inputs['candidate_refs']*masked[:,None,:],axis=-1)&inputs['candidate_mask']
    identity=result['candidates'][...,ABILITY_ID_SLICE]
    identity[candidates]=0
    identity[...,0][candidates]=1
    return result
