"""Keras 3 entity-attention / recurrent actor-critic with candidate scoring."""
import keras
from keras import layers, ops
import numpy as np
from .recurrent import masked_gru


@keras.saving.register_keras_serializable(package='patchwork')
class FeatureStandardization(layers.Layer):
    """Frozen per-feature training statistics, with bounded amplification."""
    def build(self, input_shape):
        shape=(int(input_shape[-1]),)
        self.mean=self.add_weight(name='mean',shape=shape,initializer='zeros',trainable=False)
        self.inverse_std=self.add_weight(name='inverse_std',shape=shape,initializer='ones',trainable=False)
        super().build(input_shape)

    def adapt(self, values):
        values=np.asarray(values,np.float32)
        if values.ndim!=2 or len(values)==0 or not np.isfinite(values).all():
            raise ValueError('Expected finite training feature rows')
        if not self.built:
            self.build(values.shape)
        mean=values.mean(axis=0);std=values.std(axis=0)
        self.mean.assign(mean)
        # Constant features stay at unit scale, including unseen ability IDs.
        self.inverse_std.assign(np.where(std>1e-6,1/np.maximum(std,.1),1.))

    def call(self, values):
        return (values-self.mean)*self.inverse_std


@keras.saving.register_keras_serializable(package='patchwork')
class PolicyNetwork(keras.Model):
    # Forward execution has no dropout, batch-statistic updates or other
    # training-mode side effects. PPO can reuse one output for loss and KL.
    # Revisit this contract if stochastic/stateful layers are introduced.
    deterministic_forward = True

    def __init__(self, card_count, width=128, memory_size=256, blocks=3, candidate_interactions=False, player_readout=False, stable_heads=False, standardize_inputs=False, isolate_value=False, fact_readout=False, feature_spec=None, value_head_enabled=True, **kwargs):
        super().__init__(**kwargs)
        self.card_count, self.width, self.memory_size, self.blocks = card_count, width, memory_size, blocks
        self.candidate_interactions = candidate_interactions
        self.player_readout = player_readout
        self.stable_heads = stable_heads
        self.standardize_inputs = standardize_inputs
        self.isolate_value = isolate_value
        self.fact_readout = fact_readout
        self.feature_spec = feature_spec
        self.value_head_enabled = value_head_enabled
        if standardize_inputs:
            self.entity_standardization = FeatureStandardization()
            self.candidate_standardization = FeatureStandardization()
            if fact_readout: self.context_standardization = FeatureStandardization()
        if width % 4: raise ValueError('Attention width must be divisible by four')
        self.card_embedding = layers.Embedding(card_count, 64)
        self.semantic_projection = layers.Dense(64, activation='relu')
        self.semantic_gru = layers.GRU(64, use_cudnn=False)
        self.entity_projection = layers.Dense(width, activation='relu')
        self.relation_projection = layers.Dense(width, activation='relu')
        self.attention = [layers.MultiHeadAttention(4, width // 4) for _ in range(blocks)]
        self.attention_norm = [layers.LayerNormalization() for _ in range(blocks)]
        self.feedforward = [keras.Sequential([layers.Dense(width * 2, activation='relu'), layers.Dense(width)]) for _ in range(blocks)]
        self.feedforward_norm = [layers.LayerNormalization() for _ in range(blocks)]
        self.context_projection = layers.Dense(width, activation='relu')
        self.history_projection = layers.Dense(width, activation='relu')
        self.history_memory = layers.GRU(memory_size, use_cudnn=False)
        self.memory = layers.GRU(memory_size, return_state=True, use_cudnn=False)
        self.candidate_projection = layers.Dense(width, activation='relu')
        self.query_projection = layers.Dense(width, activation='relu')
        self.score_hidden = layers.Dense(width, activation='gelu' if stable_heads else 'tanh')
        if stable_heads:
            # Fixed-scale normalization prevents products and the scoring head
            # from saturating as candidate/query projection magnitudes grow.
            self.candidate_norm = layers.LayerNormalization(center=False, scale=False)
            self.query_norm = layers.LayerNormalization(center=False, scale=False)
            self.score_norm = layers.LayerNormalization(center=False, scale=False)
        # Small random logits keep an untrained policy close to uniform without
        # replacing its decisions with an external random-action controller.
        self.score = layers.Dense(1, kernel_initializer=keras.initializers.RandomNormal(stddev=.01))
        self.value_head = layers.Dense(1, activation=None if stable_heads else 'tanh') if value_head_enabled else None

    def build(self, input_shape):
        # Explicitly create child weights before checkpoint restoration. Letting
        # a subclassed model defer this until its first call loses saved weights.
        dummy = {}
        for key, shape in input_shape.items():
            shape = tuple(1 if d is None else d for d in shape)
            dtype = 'bool' if key.endswith('_mask') else 'int32' if key in ('card_ids', 'history_cards') else 'float32'
            dummy[key] = ops.ones(shape, dtype=dtype) if key.endswith('_mask') else ops.zeros(shape, dtype=dtype)
        self.call(dummy)
        super().build(input_shape)

    def call(self, inputs, training=False):
        entity_features = self.entity_standardization(inputs['entities']) if self.standardize_inputs else inputs['entities']
        candidate_features = self.candidate_standardization(inputs['candidates']) if self.standardize_inputs else inputs['candidates']
        context_features = (self.context_standardization(inputs['context'])
                            if self.standardize_inputs and self.fact_readout else inputs['context'])
        shape = ops.shape(inputs['semantics'])
        semantic = self.semantic_projection(inputs['semantics'])
        semantic = ops.reshape(semantic, (-1, shape[2], 64))
        semantic_mask = ops.reshape(inputs['semantic_mask'], (-1, shape[2]))
        semantic = masked_gru(self.semantic_gru, semantic, mask=semantic_mask, training=training)
        semantic = ops.reshape(semantic, (shape[0], shape[1], 64))
        entity = self.entity_projection(ops.concatenate([
            self.card_embedding(inputs['card_ids']), entity_features, semantic], axis=-1))
        valid = ops.cast(inputs['entity_mask'], entity.dtype)
        entity = entity * valid[:, :, None]
        messages = []
        for channel in range(4):
            links = inputs['relations'][:, :, :, channel] * valid[:, None, :]
            messages.append(ops.matmul(links, entity) / ops.maximum(ops.sum(links, axis=-1, keepdims=True), 1.))
        entity = entity + self.relation_projection(ops.concatenate(messages, axis=-1))
        mask = inputs['entity_mask'][:, :, None] & inputs['entity_mask'][:, None, :]
        for attention, norm, ff, ff_norm in zip(self.attention, self.attention_norm, self.feedforward, self.feedforward_norm):
            entity = norm(entity + attention(entity, entity, attention_mask=mask, training=training))
            entity = ff_norm(entity + ff(entity, training=training)) * valid[:, :, None]
        pooled = ops.sum(entity, axis=1) / ops.maximum(ops.sum(valid, axis=1, keepdims=True), 1.)
        history = self.history_projection(ops.concatenate([
            self.card_embedding(inputs['history_cards']), inputs['history']], axis=-1))
        remembered = masked_gru(self.history_memory, history, mask=inputs['history_mask'],
                                         initial_state=inputs['memory_in'], training=training)
        # An entirely masked Keras GRU returns zero output, even with a nonzero
        # initial state. Empty observation deltas must preserve decision memory.
        remembered = ops.where(ops.any(inputs['history_mask'],axis=1,keepdims=True),
                               remembered,inputs['memory_in'])
        context = self.context_projection(ops.concatenate([pooled, context_features, inputs['selection']], axis=-1))
        _, updated = self.memory(context[:, None, :], initial_state=remembered, training=training)
        state = inputs['advance_memory'] * updated + (1. - inputs['advance_memory']) * inputs['memory_in']
        references = ops.matmul(inputs['candidate_refs'], entity)
        candidate_inputs = [references, candidate_features]
        if self.fact_readout:
            candidate_inputs += [ops.matmul(inputs['candidate_refs'],entity_features)]
        candidates = self.candidate_projection(ops.concatenate(candidate_inputs, axis=-1))
        query_inputs = [state, context, inputs['selection']]
        if self.player_readout:
            # Encoder orders players self, opponent. Preserve their numeric
            # state directly rather than only through the pooled object set.
            query_inputs += [entity_features[:, 0], entity_features[:, 1]]
        if self.fact_readout: query_inputs += [context_features]
        query = self.query_projection(ops.concatenate(query_inputs, axis=-1))
        if self.stable_heads:
            candidates = self.candidate_norm(candidates)
            query = self.query_norm(query)
        query = ops.broadcast_to(query[:, None, :], ops.shape(candidates))
        # Optional diagnostic alternative: expose state/action products directly,
        # instead of relying on tanh to extract interactions from an additive mix.
        scoring = ops.concatenate([candidates, query, candidates * query], axis=-1) if self.candidate_interactions else candidates + query
        hidden = self.score_hidden(scoring)
        if self.stable_heads:
            hidden = self.score_norm(hidden)
        logits = ops.squeeze(self.score(hidden), axis=-1)
        logits = ops.where(inputs['candidate_mask'], logits, ops.full_like(logits, -1.e9))
        value_input = ops.stop_gradient(state) if self.isolate_value else state
        value = self.value_head(value_input) if self.value_head is not None else ops.zeros_like(state[:, :1])
        return {'logits': logits, 'value': value, 'memory_out': state}

    def get_config(self):
        return {**super().get_config(), 'card_count': self.card_count, 'width': self.width,
                'memory_size': self.memory_size, 'blocks': self.blocks,
                'candidate_interactions': self.candidate_interactions,
                'player_readout': self.player_readout, 'stable_heads': self.stable_heads, 'standardize_inputs': self.standardize_inputs, 'isolate_value': self.isolate_value,
                'fact_readout': self.fact_readout, 'feature_spec': self.feature_spec,
                'value_head_enabled': self.value_head_enabled}


@keras.saving.register_keras_serializable(package='patchwork')
class ValueNetwork(keras.Model):
    """Independent visible-state encoder, recurrent memory and nonlinear value head."""
    def __init__(self, card_count, width=64, memory_size=64, **kwargs):
        super().__init__(**kwargs)
        if width % 4: raise ValueError('Attention width must be divisible by four')
        self.card_count, self.width, self.memory_size = card_count, width, memory_size
        self.card_embedding = layers.Embedding(card_count, 64)
        self.semantic_projection = layers.Dense(64, activation='relu')
        self.semantic_gru = layers.GRU(64, use_cudnn=False)
        self.entity_projection = layers.Dense(width, activation='relu')
        self.relation_projection = layers.Dense(width, activation='relu')
        self.attention = layers.MultiHeadAttention(4, width // 4)
        self.attention_norm = layers.LayerNormalization()
        self.feedforward = keras.Sequential([layers.Dense(width * 2, activation='gelu'), layers.Dense(width)])
        self.feedforward_norm = layers.LayerNormalization()
        self.history_projection = layers.Dense(width, activation='relu')
        self.history_memory = layers.GRU(memory_size, use_cudnn=False)
        self.context_projection = layers.Dense(width, activation='gelu')
        self.memory = layers.GRU(memory_size, return_state=True, use_cudnn=False)
        self.value_head = keras.Sequential([layers.Dense(64, activation='gelu'),
            layers.Dense(64, activation='gelu'),
            layers.Dense(1, kernel_initializer=keras.initializers.RandomNormal(stddev=.01))])

    def call(self, inputs, training=False):
        shape = ops.shape(inputs['semantics'])
        semantic = self.semantic_projection(inputs['semantics'])
        semantic = masked_gru(self.semantic_gru, ops.reshape(semantic, (-1, shape[2], 64)),
            mask=ops.reshape(inputs['semantic_mask'], (-1, shape[2])), training=training)
        semantic = ops.reshape(semantic, (shape[0], shape[1], 64))
        entity = self.entity_projection(ops.concatenate([
            self.card_embedding(inputs['card_ids']), inputs['entities'], semantic], axis=-1))
        valid = ops.cast(inputs['entity_mask'], entity.dtype)
        entity *= valid[:, :, None]
        messages = []
        for channel in range(4):
            links = inputs['relations'][:, :, :, channel] * valid[:, None, :]
            messages.append(ops.matmul(links, entity) / ops.maximum(ops.sum(links, axis=-1, keepdims=True), 1.))
        entity += self.relation_projection(ops.concatenate(messages, axis=-1))
        mask = inputs['entity_mask'][:, :, None] & inputs['entity_mask'][:, None, :]
        entity = self.attention_norm(entity + self.attention(entity, entity, attention_mask=mask, training=training))
        entity = self.feedforward_norm(entity + self.feedforward(entity, training=training)) * valid[:, :, None]
        total = ops.sum(entity, axis=1)
        mean = total / ops.maximum(ops.sum(valid, axis=1, keepdims=True), 1.)
        # A sum preserves multiplicity; a fixed divisor keeps its initial scale modest.
        facts = ops.concatenate([mean, total / 20., inputs['entities'][:, 0],
                                 inputs['entities'][:, 1], inputs['context']], axis=-1)
        context = self.context_projection(ops.concatenate([facts, inputs['selection']], axis=-1))
        history = self.history_projection(ops.concatenate([
            self.card_embedding(inputs['history_cards']), inputs['history']], axis=-1))
        remembered = masked_gru(self.history_memory, history, mask=inputs['history_mask'],
            initial_state=inputs['memory_in'], training=training)
        remembered = ops.where(ops.any(inputs['history_mask'], axis=1, keepdims=True), remembered, inputs['memory_in'])
        _, updated = self.memory(context[:, None, :], initial_state=remembered, training=training)
        state = inputs['advance_memory'] * updated + (1. - inputs['advance_memory']) * inputs['memory_in']
        return {'value': self.value_head(ops.concatenate([state, context, facts], axis=-1)), 'memory_out': state}

    def get_config(self):
        return {**super().get_config(), 'card_count': self.card_count,
                'width': self.width, 'memory_size': self.memory_size}


@keras.saving.register_keras_serializable(package='patchwork')
class SeparateActorCritic(keras.Model):
    """One batching/checkpoint container; no learned weights or memory are shared."""
    deterministic_forward = True
    separate_value = True

    def __init__(self, card_count, feature_spec, actor_width=32, actor_memory=32,
                 actor_blocks=1, critic_width=64, critic_memory=64, **kwargs):
        super().__init__(**kwargs)
        self.card_count, self.feature_spec = card_count, feature_spec
        self.actor_width, self.actor_memory, self.actor_blocks = actor_width, actor_memory, actor_blocks
        self.critic_width, self.critic_memory = critic_width, critic_memory
        self.memory_size = actor_memory + critic_memory
        self.actor = PolicyNetwork(card_count, width=actor_width, memory_size=actor_memory,
            blocks=actor_blocks, candidate_interactions=True, player_readout=True,
            stable_heads=True, fact_readout=True, feature_spec=feature_spec, value_head_enabled=False)
        self.critic = ValueNetwork(card_count, width=critic_width, memory_size=critic_memory)

    def actor_output(self, inputs, training=False):
        return self.actor(dict(inputs, memory_in=inputs['memory_in'][:, :self.actor_memory]), training=training)

    def value_output(self, inputs, training=False):
        return self.critic(dict(inputs, memory_in=inputs['memory_in'][:, self.actor_memory:]), training=training)

    def call(self, inputs, training=False):
        actor = self.actor_output(inputs, training=training)
        critic = self.value_output(inputs, training=training)
        return {'logits': actor['logits'], 'value': critic['value'],
                'memory_out': ops.concatenate([actor['memory_out'], critic['memory_out']], axis=-1)}

    def build(self, input_shape):
        dummy = {}
        for key, shape in input_shape.items():
            shape = tuple(1 if d is None else d for d in shape)
            dtype = 'bool' if key.endswith('_mask') else 'int32' if key in ('card_ids', 'history_cards') else 'float32'
            dummy[key] = ops.ones(shape, dtype=dtype) if key.endswith('_mask') else ops.zeros(shape, dtype=dtype)
        self.call(dummy)
        super().build(input_shape)

    def get_config(self):
        return {**super().get_config(), 'card_count': self.card_count, 'feature_spec': self.feature_spec,
                'actor_width': self.actor_width, 'actor_memory': self.actor_memory, 'actor_blocks': self.actor_blocks,
                'critic_width': self.critic_width, 'critic_memory': self.critic_memory}
