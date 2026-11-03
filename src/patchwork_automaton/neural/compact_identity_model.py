"""Compact v1 plus a zero-output, trainable identity/rule residual. Opt in only."""
import numpy as np
import tensorflow as tf
import keras
from .compact_standard_model import (TensorStore as BaseStore,CompactActorCritic,
    CompactStandardPolicy,CompactStandardValue)
from . import compact_identity as codec
from .identity_features import CARDS,descriptor_matrix,manifest

class TensorStore(BaseStore):
    @classmethod
    def batch_data(cls,d,ids,*,deduplicate=True):
        b=super().batch_data(d,ids,deduplicate=deduplicate)
        nodes,owner,lengths=cls.ranges(d['identity_splits'],ids)
        starts=tf.gather(d['identity_splits'],ids)
        offsets=tf.cumsum(lengths,exclusive=True)
        base_starts=tf.gather(d['node_splits'],ids)
        _,_,base_lengths=cls.ranges(d['node_splits'],ids)
        base_offsets=tf.cumsum(base_lengths,exclusive=True)
        board_counts=tf.math.unsorted_segment_sum(tf.cast(b['board'],tf.int32),b['node_owner'],b['n_examples'])
        local_nodes=nodes-tf.gather(starts,owner)
        b.update(identity_ids=tf.gather(d['identity_ids'],nodes),identity_owner=owner,
            identity_present=local_nodes!=0,
            identity_board=(local_nodes>0)&(local_nodes<=tf.gather(board_counts,owner)),
            identity_base_nodes=tf.gather(d['identity_base_nodes'],nodes)+tf.gather(base_offsets-base_starts,owner),
            identity_roles=tf.gather(d['identity_roles'],b['action_indices'])+
                tf.gather(offsets-starts,b['action_owner'])[:,None],
            identity_context=tf.gather(d['identity_context'],ids)+offsets-starts)
        return b

class IdentityResidual(keras.layers.Layer):
    def __init__(self,width,*,use_identity=True,use_rules=False,**kwargs):
        super().__init__(**kwargs)
        self.width=width;self.use_identity=use_identity;self.use_rules=use_rules
        self.identity=keras.layers.Embedding(len(CARDS)+1,width,
            embeddings_initializer=keras.initializers.RandomNormal(stddev=.02))
        self.rules=tf.constant(descriptor_matrix())
        self.node=keras.layers.Dense(width,activation='relu')
        self.hidden=keras.layers.Dense(64,activation='relu')
        self.output_layer=keras.layers.Dense(1,kernel_initializer='zeros',bias_initializer='zeros')

    def encode(self,b,nodes,g):
        ids=b['identity_ids'];known=tf.cast(ids!=0,tf.float32)[:,None]
        identity=self.identity(ids)*known*float(self.use_identity)
        rules=tf.gather(self.rules,ids)*float(self.use_rules)
        # The old encoder's live object facts remain available in this branch.
        x=self.node(tf.concat([tf.gather(nodes,b['identity_base_nodes']),identity,rules],axis=-1))
        x=x*tf.cast(b['identity_present'],tf.float32)[:,None]
        board=tf.cast(b['identity_board'],tf.float32)[:,None]
        pooled=tf.math.unsorted_segment_sum(x*board,b['identity_owner'],b['n_examples'])
        count=tf.math.unsorted_segment_sum(board,b['identity_owner'],b['n_examples'])
        context=tf.gather(x,b['identity_context'])
        return x,tf.concat([g,pooled/tf.maximum(count,1.),pooled/8.,context],axis=-1)

    def call(self,b,nodes,g,*,actor,action_features=None):
        x,global_state=self.encode(b,nodes,g)
        if actor:
            roles=tf.gather(x,b['identity_roles'])
            # Include the base action features (kind, amounts, choices, costs).
            local=tf.gather(global_state,b['action_owner'])
            features=tf.concat([local,tf.reshape(roles,(-1,3*self.width)),action_features],axis=-1)
        else:features=global_state
        return self.output_layer(self.hidden(features))

@keras.saving.register_keras_serializable(package='patchwork')
class IdentityPolicy(CompactStandardPolicy):
    def __init__(self,width=32,identity_width=16,use_identity=True,use_rules=False,**kwargs):
        super().__init__(width=width,reference_readout=True,**kwargs)
        self.residual=IdentityResidual(identity_width,use_identity=use_identity,use_rules=use_rules)
        self.identity_width=identity_width;self.use_identity=use_identity;self.use_rules=use_rules

    def call(self,b):
        nodes,g=self.encode_state(b)
        roles=tf.gather(nodes,b['roles']);global_actions=tf.gather(g,b['action_owner'])
        local=tf.gather(self.bag(self.local_embedding,b,'local',b['local_count']),b['local_map'])
        refs=b['refs']
        parts=[local,global_actions,tf.reshape(roles,(-1,3*self.width)),
            tf.reshape(roles*global_actions[:,None,:],(-1,3*self.width)),
            tf.math.unsorted_segment_sum(tf.gather(nodes,refs[:,1])*b['ref_values'][:,None],refs[:,0],b['n_actions'])]
        old=tf.squeeze(self.score(self.hidden(tf.concat(parts,axis=-1))),axis=-1)
        return old+tf.squeeze(self.residual(b,nodes,g,actor=True,action_features=local),axis=-1)

    def get_config(self):
        result=super().get_config();result.pop('reference_readout')
        return dict(result,identity_width=self.identity_width,use_identity=self.use_identity,use_rules=self.use_rules)

@keras.saving.register_keras_serializable(package='patchwork')
class IdentityValue(CompactStandardValue):
    def __init__(self,width=32,identity_width=16,use_identity=True,use_rules=False,**kwargs):
        super().__init__(width=width,**kwargs)
        self.residual=IdentityResidual(identity_width,use_identity=use_identity,use_rules=use_rules)
        self.identity_width=identity_width;self.use_identity=use_identity;self.use_rules=use_rules

    def call(self,b):
        nodes,g=self.encode_state(b)
        return self.value_head(g)+self.residual(b,nodes,g,actor=False)

    def get_config(self):
        return dict(super().get_config(),identity_width=self.identity_width,use_identity=self.use_identity,use_rules=self.use_rules)

def example_store():
    from .compact_standard_model import example_store as old_example
    out=old_example();n=len(out['board'])
    out.update(identity_splits=np.asarray([0,n],np.int32),identity_ids=np.zeros(n,np.int32),
        identity_base_nodes=np.arange(n,dtype=np.int32),identity_roles=np.zeros_like(out['roles']),
        identity_context=np.zeros(1,np.int32))
    return out

@keras.saving.register_keras_serializable(package='patchwork')
class CompactIdentityActorCritic(CompactActorCritic):
    input_format=codec.VERSION

    def __init__(self,card_count=1,feature_spec=None,actor_width=32,critic_width=32,
                 identity_width=16,use_identity=True,use_rules=False,identity_manifest=None,**kwargs):
        # Construct only the new actor/critic, without unused v1 child layers.
        keras.Model.__init__(self,**kwargs)
        if dict(feature_spec or codec.SPEC)!=dict(codec.SPEC):raise ValueError('Base feature vocabulary mismatch')
        current=manifest()
        if identity_manifest is not None and json_normalized(identity_manifest)!=json_normalized(current):
            raise ValueError('Identity/rule vocabulary mismatch')
        self.card_count=card_count;self.feature_spec=dict(codec.SPEC)
        self.actor_width=actor_width;self.critic_width=critic_width;self.identity_width=identity_width
        self.use_identity=use_identity;self.use_rules=use_rules;self.identity_manifest=current
        args=dict(identity_width=identity_width,use_identity=use_identity,use_rules=use_rules)
        self.actor=IdentityPolicy(actor_width,**args);self.critic=IdentityValue(critic_width,**args)

    @staticmethod
    def example_inputs():
        b=TensorStore(example_store()).batch(tf.constant([0],tf.int32))
        return {k:v.numpy() for k,v in b.items()}

    def get_config(self):
        return dict(super().get_config(),identity_width=self.identity_width,use_identity=self.use_identity,
            use_rules=self.use_rules,identity_manifest=self.identity_manifest)

def json_normalized(value):
    import json
    return json.dumps(value,sort_keys=True)

def transfer_v1(source,*,identity_width=16,use_identity=True,use_rules=False):
    """Copy all old layers exactly; new residual heads initially output zero.

    This starts a new experiment with new optimizer state, not a production resume.
    """
    if type(source) is not CompactActorCritic:raise ValueError('Transfer requires an unchanged compact v1 model')
    model=CompactIdentityActorCritic(source.card_count,source.feature_spec,source.actor_width,source.critic_width,
        identity_width=identity_width,use_identity=use_identity,use_rules=use_rules)
    model(model.example_inputs())
    names=['node_embedding','context_embedding','relation_embedding','node_projection','message_projection','global_projection']
    for old,new in [(source.actor,model.actor),(source.critic,model.critic)]:
        extras=['local_embedding','hidden','score'] if old is source.actor else ['value_head']
        for name in names+extras:
            getattr(new,name).set_weights(getattr(old,name).get_weights())
    return model
