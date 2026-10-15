"""Keras 3 entity-attention / recurrent actor-critic with candidate scoring."""
import keras
from keras import layers, ops


@keras.saving.register_keras_serializable(package='patchwork')
class PolicyNetwork(keras.Model):
    def __init__(self, card_count, width=128, memory_size=256, blocks=3, **kwargs):
        super().__init__(**kwargs)
        self.card_count, self.width, self.memory_size, self.blocks = card_count, width, memory_size, blocks
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
        self.score_hidden = layers.Dense(width, activation='tanh')
        # Small random logits keep an untrained policy close to uniform without
        # replacing its decisions with an external random-action controller.
        self.score = layers.Dense(1, kernel_initializer=keras.initializers.RandomNormal(stddev=.01))
        self.value_head = layers.Dense(1, activation='tanh')

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
        shape = ops.shape(inputs['semantics'])
        semantic = self.semantic_projection(inputs['semantics'])
        semantic = ops.reshape(semantic, (-1, shape[2], 64))
        semantic_mask = ops.reshape(inputs['semantic_mask'], (-1, shape[2]))
        semantic = self.semantic_gru(semantic, mask=semantic_mask, training=training)
        semantic = ops.reshape(semantic, (shape[0], shape[1], 64))
        entity = self.entity_projection(ops.concatenate([
            self.card_embedding(inputs['card_ids']), inputs['entities'], semantic], axis=-1))
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
        remembered = self.history_memory(history, mask=inputs['history_mask'],
                                         initial_state=inputs['memory_in'], training=training)
        context = self.context_projection(ops.concatenate([pooled, inputs['context'], inputs['selection']], axis=-1))
        _, updated = self.memory(context[:, None, :], initial_state=remembered, training=training)
        state = inputs['advance_memory'] * updated + (1. - inputs['advance_memory']) * inputs['memory_in']
        references = ops.matmul(inputs['candidate_refs'], entity)
        candidates = self.candidate_projection(ops.concatenate([references, inputs['candidates']], axis=-1))
        query = self.query_projection(ops.concatenate([state, context, inputs['selection']], axis=-1))
        logits = ops.squeeze(self.score(self.score_hidden(candidates + query[:, None, :])), axis=-1)
        logits = ops.where(inputs['candidate_mask'], logits, ops.full_like(logits, -1.e9))
        return {'logits': logits, 'value': self.value_head(state), 'memory_out': state}

    def get_config(self):
        return {**super().get_config(), 'card_count': self.card_count, 'width': self.width,
                'memory_size': self.memory_size, 'blocks': self.blocks}
