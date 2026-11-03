"""Small sparse-input policy. TensorFlow is imported only by the learner."""
import tensorflow as tf
import keras
from .compact_standard import ROLE, LOCAL, SPEC, VERSION


class TensorStore:
    def __init__(self, arrays):
        self.bytes = sum(a.nbytes for a in arrays.values())
        self.data = {k:tf.constant(v) for k,v in arrays.items()}

    @staticmethod
    def ranges(splits, ids):
        # RaggedRange has only a CPU kernel in the deployed TF build. Build the
        # CSR gather using GPU-supported primitives; no host batch preparation.
        starts = tf.gather(splits,ids)
        lengths = tf.gather(splits,ids+1)-starts
        owners = tf.repeat(tf.range(tf.size(ids)),lengths)
        offsets = tf.range(tf.reduce_sum(lengths))-tf.repeat(tf.cumsum(lengths,exclusive=True),lengths)
        return tf.repeat(starts,lengths)+offsets, owners, lengths

    def batch(self, ids, *, deduplicate=True):
        return self.batch_data(self.data,ids,deduplicate=deduplicate)

    @classmethod
    def batch_data(cls,d,ids,*,deduplicate=True):
        nodes, node_owner, node_lengths = cls.ranges(d['node_splits'],ids)
        actions, action_owner, action_lengths = cls.ranges(d['action_splits'],ids)
        node_offsets = tf.cumsum(node_lengths,exclusive=True)
        out = dict(n_examples=tf.size(ids), n_nodes=tf.size(nodes), n_actions=tf.size(actions),
            action_indices=actions,
            node_owner=node_owner,action_owner=action_owner,board=tf.gather(d['board'],nodes),
            roles=tf.gather(d['roles'],actions)+tf.gather(node_offsets,action_owner)[:,None],
            labels=tf.cumsum(action_lengths,exclusive=True)+tf.gather(d['action'],ids))
        for name, rows in (('nodes',nodes),('local',actions),('context',ids)):
            feature_rows=tf.gather(d[name+'_row_ids'],rows)
            if deduplicate:
                unique, mapping=tf.unique(feature_rows)
            else:
                unique=feature_rows;mapping=tf.range(tf.size(feature_rows))
            out[name+'_map']=mapping
            out[name+'_count']=tf.size(unique)
            positions, owner, _ = cls.ranges(d[name+'_splits'],unique)
            out[name+'_indices'] = tf.gather(d[name+'_indices'],positions)
            out[name+'_values'] = tf.gather(d[name+'_values'],positions)
            out[name+'_owner'] = owner
        edges, edge_owner, _ = cls.ranges(d['edge_splits'],ids)
        edge = tf.gather(d['edges'],edges)
        out['edges'] = tf.concat([edge[:,:2]+tf.gather(node_offsets,edge_owner)[:,None],edge[:,2:]],axis=1)
        refs,ref_owner,_=cls.ranges(d['ref_splits'],ids)
        ref=tf.gather(d['refs'],refs)
        offsets=tf.stack([tf.cumsum(action_lengths,exclusive=True),node_offsets+1],axis=1)
        out['refs']=ref+tf.gather(offsets,ref_owner)
        out['ref_values']=tf.gather(d['ref_values'],refs)
        out['candidate_mask']=tf.sequence_mask(action_lengths,maxlen=tf.reduce_max(action_lengths))
        out['advance_memory']=tf.ones((tf.size(ids),1),tf.float32)
        out['memory_in']=tf.zeros((tf.size(ids),0),tf.float32)
        return out


@keras.saving.register_keras_serializable(package='patchwork')
class CompactStandardPolicy(keras.Model):
    def __init__(self, width=32, reference_readout=False, **kwargs):
        super().__init__(**kwargs)
        self.width = width
        self.reference_readout=reference_readout
        self.node_embedding = keras.layers.Embedding(ROLE,width)
        self.local_embedding = keras.layers.Embedding(LOCAL,width)
        self.context_embedding = keras.layers.Embedding(SPEC['context'],width)
        self.relation_embedding = keras.layers.Embedding(4,width)
        self.node_projection = keras.layers.Dense(width,activation='relu')
        self.message_projection = keras.layers.Dense(width,activation='relu')
        self.global_projection = keras.layers.Dense(width,activation='relu')
        self.hidden = keras.layers.Dense(64,activation='relu')
        self.score = keras.layers.Dense(1,kernel_initializer=keras.initializers.RandomNormal(stddev=.01))

    def bag(self, embedding, batch, name, count):
        x = embedding(batch[name+'_indices'])*batch[name+'_values'][:,None]
        return tf.math.unsorted_segment_sum(x,batch[name+'_owner'],count)

    def encode_state(self,b):
        nodes = self.node_projection(self.bag(self.node_embedding,b,'nodes',b['nodes_count']))
        nodes = tf.gather(nodes,b['nodes_map'])
        edge = b['edges']
        messages = self.message_projection(tf.concat([tf.gather(nodes,edge[:,0]),
            tf.gather(nodes,edge[:,1]),self.relation_embedding(edge[:,2])],axis=-1))
        nodes += tf.math.unsorted_segment_sum(messages,edge[:,0],b['n_nodes'])
        board = tf.cast(b['board'],tf.float32)[:,None]
        summed = tf.math.unsorted_segment_sum(nodes*board,b['node_owner'],b['n_examples'])
        count = tf.math.unsorted_segment_sum(board,b['node_owner'],b['n_examples'])
        context = tf.gather(self.bag(self.context_embedding,b,'context',b['context_count']),b['context_map'])
        global_state = self.global_projection(tf.concat([context,summed/tf.maximum(count,1.),summed/8.],axis=-1))
        return nodes,global_state

    def call(self, b):
        nodes,global_state=self.encode_state(b)
        g = tf.gather(global_state,b['action_owner'])
        roles = tf.gather(nodes,b['roles'])
        local = tf.gather(self.bag(self.local_embedding,b,'local',b['local_count']),b['local_map'])
        parts=[local,g,tf.reshape(roles,(-1,3*self.width)),
               tf.reshape(roles*g[:,None,:],(-1,3*self.width))]
        if self.reference_readout:
            # Preserve the engine's exact pooled object references, including
            # identical object facts with different attachment/target edges.
            refs=b['refs']
            parts.append(tf.math.unsorted_segment_sum(tf.gather(nodes,refs[:,1])*b['ref_values'][:,None],
                refs[:,0],b['n_actions']))
        x = tf.concat(parts,axis=-1)
        return tf.squeeze(self.score(self.hidden(x)),axis=-1)

    def get_config(self):
        return dict(super().get_config(),width=self.width,reference_readout=self.reference_readout)


@keras.saving.register_keras_serializable(package='patchwork')
class CompactStandardValue(keras.Model):
    def __init__(self,width=32,**kwargs):
        super().__init__(**kwargs);self.width=width
        self.node_embedding=keras.layers.Embedding(ROLE,width)
        self.context_embedding=keras.layers.Embedding(SPEC['context'],width)
        self.relation_embedding=keras.layers.Embedding(4,width)
        self.node_projection=keras.layers.Dense(width,activation='relu')
        self.message_projection=keras.layers.Dense(width,activation='relu')
        self.global_projection=keras.layers.Dense(width,activation='relu')
        self.value_head=keras.Sequential([keras.layers.Dense(64,activation='gelu'),keras.layers.Dense(1)])

    bag=CompactStandardPolicy.bag
    encode_state=CompactStandardPolicy.encode_state

    def call(self,b):
        _,state=self.encode_state(b)
        return self.value_head(state)

    def get_config(self):return dict(super().get_config(),width=self.width)


def example_store():
    from .batching import example_batch
    from .compact_standard import encode,pack
    arrays={k:v[0] for k,v in example_batch(0,SPEC).items() if k not in ('memory_in','advance_memory')}
    return pack([encode(arrays)])


def example_inputs():
    store=TensorStore(example_store())
    return {k:v.numpy() for k,v in store.batch(tf.constant([0],tf.int32)).items()}


@keras.saving.register_keras_serializable(package='patchwork')
class CompactActorCritic(keras.Model):
    """Independent stateless actor/critic over compact current observations."""
    deterministic_forward=True
    separate_value=True
    input_format=VERSION
    stateless=True
    memory_size=actor_memory=critic_memory=0

    def __init__(self,card_count=1,feature_spec=None,actor_width=32,critic_width=32,**kwargs):
        super().__init__(**kwargs)
        if dict(feature_spec or SPEC)!=dict(SPEC):raise ValueError('Compact feature vocabulary mismatch')
        self.card_count=card_count;self.feature_spec=dict(SPEC)
        self.actor_width=actor_width;self.critic_width=critic_width
        self.actor=CompactStandardPolicy(actor_width,reference_readout=True)
        self.critic=CompactStandardValue(critic_width)

    @staticmethod
    def example_inputs():return example_inputs()

    def build(self,input_shape):
        self.call({k:tf.convert_to_tensor(v) for k,v in self.example_inputs().items()})
        super().build(input_shape)

    @staticmethod
    def dense_logits(flat,b):
        starts=tf.math.unsorted_segment_min(tf.range(tf.size(flat)),b['action_owner'],b['n_examples'])
        columns=tf.range(tf.size(flat))-tf.gather(starts,b['action_owner'])
        return tf.tensor_scatter_nd_update(tf.fill(tf.shape(b['candidate_mask']),tf.constant(-1.e9,tf.float32)),
            tf.stack([b['action_owner'],columns],axis=1),flat)

    def actor_output(self,b,training=False):
        return dict(logits=self.dense_logits(self.actor(b,training=training),b),
            value=tf.zeros((b['n_examples'],1),tf.float32),memory_out=tf.zeros((b['n_examples'],0),tf.float32))

    def value_output(self,b,training=False):
        return dict(value=self.critic(b,training=training),memory_out=tf.zeros((b['n_examples'],0),tf.float32))

    def call(self,b,training=False):
        actor=self.actor_output(b,training=training)
        return dict(actor,value=self.critic(b,training=training))

    def get_config(self):
        return dict(super().get_config(),card_count=self.card_count,feature_spec=self.feature_spec,
            actor_width=self.actor_width,critic_width=self.critic_width)


def objective(logits, batch):
    owners, n = batch['action_owner'], batch['n_examples']
    maximum = tf.math.unsorted_segment_max(logits,owners,n)
    total = tf.math.unsorted_segment_sum(tf.exp(logits-tf.gather(maximum,owners)),owners,n)
    loss = maximum+tf.math.log(total)-tf.gather(logits,batch['labels'])
    indices = tf.range(tf.size(logits),dtype=tf.int32)
    predicted = tf.math.unsorted_segment_min(tf.where(logits==tf.gather(maximum,owners),
        indices,tf.fill(tf.shape(indices),tf.int32.max)),owners,n)
    return loss, tf.cast(predicted==batch['labels'],tf.float32), predicted
