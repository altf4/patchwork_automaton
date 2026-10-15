"""Function-preserving context rescaling for numerical policy warm starts."""
import numpy as np
from .model import PolicyNetwork,FeatureStandardization
from .batching import example_batch
from .features import CONTEXT_WIDTH


def calibrate_context(model,training_contexts):
    """Amplify small factual variations without changing initial predictions.

    Statistics come only from training observations. Compensate both dense
    paths that consume context; the old function is preserved, but subsequent
    optimizer steps work in normalized coordinates. No teacher labels are used.
    """
    if not model.fact_readout:raise ValueError('Context calibration requires fact_readout')
    if not model.built:model(example_batch(model.memory_size))
    config=model.get_config()|{'standardize_inputs':True}
    result=PolicyNetwork.from_config(config)
    result(example_batch(model.memory_size))
    old_layers=[l for l in model.layers if not isinstance(l,FeatureStandardization)]
    new_layers=[l for l in result.layers if not isinstance(l,FeatureStandardization)]
    if len(old_layers)!=len(new_layers):raise ValueError('Incompatible policy structure')
    for old,new in zip(old_layers,new_layers):
        if type(old)!=type(new):raise ValueError('Incompatible policy layer order')
        new.set_weights(old.get_weights())
    old_mean=np.zeros(CONTEXT_WIDTH,np.float32);old_scale=np.ones(CONTEXT_WIDTH,np.float32)
    if model.standardize_inputs:
        for name in ('entity_standardization','candidate_standardization'):
            getattr(result,name).set_weights(getattr(model,name).get_weights())
        old_mean,old_scale=[np.asarray(v) for v in model.context_standardization.get_weights()]
    result.context_standardization.adapt(training_contexts)
    mean,scale=[np.asarray(v) for v in result.context_standardization.get_weights()]
    for layer,offset in ((result.context_projection,model.width),
                         (result.query_projection,int(result.query_projection.kernel.shape[0])-CONTEXT_WIDTH)):
        kernel,bias=layer.get_weights();old=kernel[offset:offset+CONTEXT_WIDTH].copy()
        kernel[offset:offset+CONTEXT_WIDTH]=old*(old_scale/scale)[:,None]
        bias+=((mean-old_mean)*old_scale)@old
        layer.set_weights([kernel,bias])
    return result
