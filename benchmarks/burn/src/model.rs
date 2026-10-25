//! Exact tensor-level port of the pinned SeparateActorCritic architecture.
//! Keras reset-after GRU, population layer variance (epsilon .001), exact GELU,
//! four attention heads, masks, actor scoring and critic facts are preserved.
use burn::{prelude::*, tensor::activation};
use serde::Deserialize;
use std::{collections::BTreeMap, fs, path::Path};

#[derive(Clone, Deserialize)]
pub struct ArraySpec {
    pub offset: usize,
    pub shape: Vec<usize>,
    pub elements: usize,
}

#[derive(Clone, Deserialize)]
pub struct FixtureSpec {
    pub name: String,
}

#[derive(Deserialize)]
pub struct Manifest {
    pub arrays: BTreeMap<String, ArraySpec>,
    pub fixtures: Vec<FixtureSpec>,
    pub weights: Vec<String>,
    pub binary_sha256: String,
}

pub struct Archive {
    pub manifest: Manifest,
    bytes: Vec<u8>,
}

impl Archive {
    pub fn load(path: &Path) -> Self {
        Self {
            manifest: serde_json::from_slice(&fs::read(path.join("manifest.json")).unwrap())
                .unwrap(),
            bytes: fs::read(path.join("arrays.bin")).unwrap(),
        }
    }
    pub fn values(&self, name: &str) -> Vec<f32> {
        let spec = &self.manifest.arrays[name];
        self.bytes[spec.offset..spec.offset + spec.elements * 4]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    }
    pub fn tensor<const D: usize>(&self, name: &str, device: &Device) -> Tensor<D> {
        let spec = &self.manifest.arrays[name];
        assert_eq!(spec.shape.len(), D, "{name}");
        Tensor::from_data(
            TensorData::new(self.values(name), spec.shape.clone()),
            device,
        )
    }
}

#[derive(Clone)]
pub struct Inputs {
    pub entities: Tensor<3>,
    pub card_ids: Tensor<2, Int>,
    pub entity_mask: Tensor<2>,
    pub semantics: Tensor<4>,
    pub semantic_mask: Tensor<3, Bool>,
    pub relations: Tensor<4>,
    pub context: Tensor<2>,
    pub candidates: Tensor<3>,
    pub candidate_refs: Tensor<3>,
    pub candidate_mask: Tensor<2, Bool>,
    pub selection: Tensor<2>,
    pub history: Tensor<3>,
    pub history_cards: Tensor<2, Int>,
    pub history_mask: Tensor<2, Bool>,
    pub memory_in: Tensor<2>,
    pub advance_memory: Tensor<2>,
    pub actions: Tensor<2, Int>,
    pub returns: Tensor<1>,
    pub old_logp: Tensor<1>,
    pub advantages: Tensor<1>,
}

impl Inputs {
    pub fn load(a: &Archive, name: &str, d: &Device) -> Self {
        let key = |k| format!("{name}/inputs/{k}");
        let target = |k| format!("{name}/targets/{k}");
        Self {
            entities: a.tensor(&key("entities"), d),
            card_ids: a.tensor::<2>(&key("card_ids"), d).int(),
            entity_mask: a.tensor(&key("entity_mask"), d),
            semantics: a.tensor(&key("semantics"), d),
            semantic_mask: a.tensor::<3>(&key("semantic_mask"), d).greater_elem(0.5),
            relations: a.tensor(&key("relations"), d),
            context: a.tensor(&key("context"), d),
            candidates: a.tensor(&key("candidates"), d),
            candidate_refs: a.tensor(&key("candidate_refs"), d),
            candidate_mask: a.tensor::<2>(&key("candidate_mask"), d).greater_elem(0.5),
            selection: a.tensor(&key("selection"), d),
            history: a.tensor(&key("history"), d),
            history_cards: a.tensor::<2>(&key("history_cards"), d).int(),
            history_mask: a.tensor::<2>(&key("history_mask"), d).greater_elem(0.5),
            memory_in: a.tensor(&key("memory_in"), d),
            advance_memory: a.tensor(&key("advance_memory"), d),
            actions: a.tensor::<1>(&target("actions"), d).int().unsqueeze_dim(1),
            returns: a.tensor(&target("returns"), d),
            old_logp: a.tensor(&target("old_logp"), d),
            advantages: a.tensor(&target("advantages"), d),
        }
    }
}

#[derive(Clone)]
pub struct Network {
    pub weights: BTreeMap<String, Tensor<1>>,
    pub device: Device,
}

impl Network {
    pub fn load(a: &Archive, device: &Device, train: bool) -> Self {
        let weights = a
            .manifest
            .weights
            .iter()
            .map(|name| {
                let values = a.values(&format!("weights/{name}"));
                let len = values.len();
                let t = Tensor::<1>::from_data(TensorData::new(values, [len]), device);
                (name.clone(), if train { t.require_grad() } else { t })
            })
            .collect();
        Self {
            weights,
            device: device.clone(),
        }
    }

    fn w(&self, name: &str) -> Tensor<1> {
        self.weights[name].clone()
    }

    fn dense<const D: usize>(&self, x: Tensor<D>, prefix: &str) -> Tensor<D> {
        let mut dims = x.dims();
        let input = dims[D - 1];
        let kernel = self.w(&format!("{prefix}/kernel"));
        let output = kernel.dims()[0] / input;
        let rows = dims.iter().product::<usize>() / input;
        dims[D - 1] = output;
        (x.reshape([rows, input])
            .matmul(kernel.reshape([input, output]))
            + self.w(&format!("{prefix}/bias")).reshape([1, output]))
        .reshape(dims)
    }

    fn norm<const D: usize>(&self, x: Tensor<D>, prefix: &str) -> Tensor<D> {
        let dim = D - 1;
        let width = x.dims()[dim];
        let centered = x.clone() - x.mean_dim(dim);
        let variance = (centered.clone() * centered.clone()).mean_dim(dim);
        let mut out = centered / (variance + 0.001).sqrt();
        let mut shape = [1; D];
        shape[dim] = width;
        if self.weights.contains_key(&format!("{prefix}/gamma")) {
            out = out * self.w(&format!("{prefix}/gamma")).reshape(shape)
                + self.w(&format!("{prefix}/beta")).reshape(shape);
        }
        out
    }

    fn embed(&self, ids: Tensor<2, Int>, prefix: &str) -> Tensor<3> {
        let [b, n] = ids.dims();
        let weight = self.w(&format!("{prefix}/card_embedding/embeddings"));
        let vocab = weight.dims()[0] / 64;
        weight
            .reshape([vocab, 64])
            .select(0, ids.reshape([b * n]))
            .reshape([b, n, 64])
    }

    fn gru(
        &self,
        x: Tensor<3>,
        mask: Tensor<2, Bool>,
        mut state: Tensor<2>,
        prefix: &str,
    ) -> Tensor<2> {
        let [batch, time, input] = x.dims();
        let hidden = state.dims()[1];
        let bias = self.w(&format!("{prefix}/bias")).reshape([2, hidden * 3]);
        let projected = (x.reshape([batch * time, input]).matmul(
            self.w(&format!("{prefix}/kernel"))
                .reshape([input, 3 * hidden]),
        ) + bias.clone().slice([0..1, 0..3 * hidden]))
        .reshape([batch, time, 3 * hidden]);
        let recurrent = self
            .w(&format!("{prefix}/recurrent_kernel"))
            .reshape([hidden, 3 * hidden]);
        let rbias = bias.slice([1..2, 0..3 * hidden]);
        for t in 0..time {
            let xp = projected
                .clone()
                .slice([0..batch, t..t + 1, 0..3 * hidden])
                .reshape([batch, 3 * hidden]);
            let hp = state.clone().matmul(recurrent.clone()) + rbias.clone();
            let z = activation::sigmoid(
                xp.clone().slice([0..batch, 0..hidden]) + hp.clone().slice([0..batch, 0..hidden]),
            );
            let r = activation::sigmoid(
                xp.clone().slice([0..batch, hidden..2 * hidden])
                    + hp.clone().slice([0..batch, hidden..2 * hidden]),
            );
            let candidate = (xp.slice([0..batch, 2 * hidden..3 * hidden])
                + r * hp.slice([0..batch, 2 * hidden..3 * hidden]))
            .tanh();
            let next = z.clone() * state.clone() + (z.neg() + 1.) * candidate;
            state = state.mask_where(
                mask.clone()
                    .slice([0..batch, t..t + 1])
                    .expand([batch, hidden]),
                next,
            );
        }
        state
    }

    fn attention(&self, x: Tensor<3>, valid: Tensor<2>, prefix: &str) -> Tensor<3> {
        let [batch, n, width] = x.dims();
        let head = width / 4;
        let q = self
            .dense(x.clone(), &format!("{prefix}/attention/query"))
            .reshape([batch, n, 4, head])
            .swap_dims(1, 2)
            / (head as f32).sqrt();
        let k = self
            .dense(x.clone(), &format!("{prefix}/attention/key"))
            .reshape([batch, n, 4, head])
            .swap_dims(1, 2);
        let v = self
            .dense(x, &format!("{prefix}/attention/value"))
            .reshape([batch, n, 4, head])
            .swap_dims(1, 2);
        let mask = (valid.clone().reshape([batch, 1, n, 1]) * valid.reshape([batch, 1, 1, n]))
            .lower_elem(0.5);
        let scores = q
            .matmul(k.swap_dims(2, 3))
            .mask_fill(mask.expand([batch, 4, n, n]), -1.0e9);
        let attended = activation::softmax(scores, 3)
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([batch, n, width]);
        self.dense(attended, &format!("{prefix}/attention/output"))
    }

    fn encode(&self, x: &Inputs, prefix: &str, width: usize) -> Tensor<3> {
        let [batch, n, semantic_count, _] = x.semantics.dims();
        let sem = activation::relu(self.dense(
            x.semantics.clone(),
            &format!("{prefix}/semantic_projection"),
        ));
        let sem = self
            .gru(
                sem.reshape([batch * n, semantic_count, 64]),
                x.semantic_mask.clone().reshape([batch * n, semantic_count]),
                Tensor::zeros([batch * n, 64], &self.device),
                &format!("{prefix}/semantic_gru"),
            )
            .reshape([batch, n, 64]);
        let entity = activation::relu(self.dense(
            Tensor::cat(
                vec![
                    self.embed(x.card_ids.clone(), prefix),
                    x.entities.clone(),
                    sem,
                ],
                2,
            ),
            &format!("{prefix}/entity_projection"),
        )) * x.entity_mask.clone().unsqueeze_dim(2);
        let mut messages = Vec::new();
        for channel in 0..4 {
            let links = x
                .relations
                .clone()
                .slice([0..batch, 0..n, 0..n, channel..channel + 1])
                .reshape([batch, n, n])
                * x.entity_mask.clone().unsqueeze_dim(1);
            messages.push(links.clone().matmul(entity.clone()) / links.sum_dim(2).clamp_min(1.));
        }
        let entity = entity
            + activation::relu(self.dense(
                Tensor::cat(messages, 2),
                &format!("{prefix}/relation_projection"),
            ));
        let entity = self.norm(
            entity.clone() + self.attention(entity, x.entity_mask.clone(), prefix),
            &format!("{prefix}/attention_norm"),
        );
        let ff = self.dense(entity.clone(), &format!("{prefix}/feedforward/0"));
        let ff = if prefix == "actor" {
            activation::relu(ff)
        } else {
            activation::gelu(ff)
        };
        let ff = self.dense(ff, &format!("{prefix}/feedforward/1"));
        let entity = self.norm(entity + ff, &format!("{prefix}/feedforward_norm"))
            * x.entity_mask.clone().unsqueeze_dim(2);
        assert_eq!(entity.dims()[2], width);
        entity
    }

    fn remembered(&self, x: &Inputs, prefix: &str, start: usize, memory: usize) -> Tensor<2> {
        let batch = x.entities.dims()[0];
        let history = activation::relu(self.dense(
            Tensor::cat(
                vec![
                    self.embed(x.history_cards.clone(), prefix),
                    x.history.clone(),
                ],
                2,
            ),
            &format!("{prefix}/history_projection"),
        ));
        self.gru(
            history,
            x.history_mask.clone(),
            x.memory_in.clone().slice([0..batch, start..start + memory]),
            &format!("{prefix}/history_memory"),
        )
    }

    fn next_state(
        &self,
        x: &Inputs,
        context: Tensor<2>,
        remembered: Tensor<2>,
        prefix: &str,
        start: usize,
        memory: usize,
    ) -> Tensor<2> {
        let batch = x.entities.dims()[0];
        let mask = Tensor::<2>::ones([batch, 1], &self.device).greater_elem(0.5);
        let updated = self.gru(
            context.unsqueeze_dim(1),
            mask,
            remembered,
            &format!("{prefix}/memory"),
        );
        x.advance_memory.clone() * updated
            + (x.advance_memory.clone().neg() + 1.)
                * x.memory_in.clone().slice([0..batch, start..start + memory])
    }

    pub fn actor(&self, x: &Inputs) -> (Tensor<2>, Tensor<2>) {
        let [batch, n, facts] = x.entities.dims();
        let entity = self.encode(x, "actor", 32);
        let pooled = entity.clone().sum_dim(1).reshape([batch, 32])
            / x.entity_mask.clone().sum_dim(1).clamp_min(1.);
        let context = activation::relu(self.dense(
            Tensor::cat(vec![pooled, x.context.clone(), x.selection.clone()], 1),
            "actor/context_projection",
        ));
        let state = self.next_state(
            x,
            context.clone(),
            self.remembered(x, "actor", 0, 32),
            "actor",
            0,
            32,
        );
        let candidate_features = Tensor::cat(
            vec![
                x.candidate_refs.clone().matmul(entity),
                x.candidates.clone(),
                x.candidate_refs.clone().matmul(x.entities.clone()),
            ],
            2,
        );
        let candidates = self.norm(
            activation::relu(self.dense(candidate_features, "actor/candidate_projection")),
            "actor/candidate_norm",
        );
        let query = Tensor::cat(
            vec![
                state.clone(),
                context,
                x.selection.clone(),
                x.entities
                    .clone()
                    .slice([0..batch, 0..1, 0..facts])
                    .reshape([batch, facts]),
                x.entities
                    .clone()
                    .slice([0..batch, 1..2, 0..facts])
                    .reshape([batch, facts]),
                x.context.clone(),
            ],
            1,
        );
        let a = candidates.dims()[1];
        let query = self
            .norm(
                activation::relu(self.dense(query, "actor/query_projection")),
                "actor/query_norm",
            )
            .unsqueeze_dim::<3>(1)
            .expand([batch, a, 32]);
        let scoring = Tensor::cat(
            vec![candidates.clone(), query.clone(), candidates * query],
            2,
        );
        let hidden = self.norm(
            activation::gelu(self.dense(scoring, "actor/score_hidden")),
            "actor/score_norm",
        );
        let logits = self
            .dense(hidden, "actor/score")
            .reshape([batch, a])
            .mask_fill(x.candidate_mask.clone().bool_not(), -1.0e9);
        let _ = n;
        (logits, state)
    }

    pub fn critic(&self, x: &Inputs) -> (Tensor<2>, Tensor<2>) {
        let [batch, _, features] = x.entities.dims();
        let entity = self.encode(x, "critic", 64);
        let total = entity.sum_dim(1).reshape([batch, 64]);
        let mean = total.clone() / x.entity_mask.clone().sum_dim(1).clamp_min(1.);
        let facts = Tensor::cat(
            vec![
                mean,
                total / 20.,
                x.entities
                    .clone()
                    .slice([0..batch, 0..1, 0..features])
                    .reshape([batch, features]),
                x.entities
                    .clone()
                    .slice([0..batch, 1..2, 0..features])
                    .reshape([batch, features]),
                x.context.clone(),
            ],
            1,
        );
        let context = activation::gelu(self.dense(
            Tensor::cat(vec![facts.clone(), x.selection.clone()], 1),
            "critic/context_projection",
        ));
        let state = self.next_state(
            x,
            context.clone(),
            self.remembered(x, "critic", 32, 64),
            "critic",
            32,
            64,
        );
        let value = Tensor::cat(vec![state.clone(), context, facts], 1);
        let value = activation::gelu(self.dense(value, "critic/value_head/0"));
        let value = activation::gelu(self.dense(value, "critic/value_head/1"));
        (self.dense(value, "critic/value_head/2"), state)
    }

    pub fn loss(&self, x: &Inputs, actor: bool) -> Tensor<1> {
        if actor {
            let (logits, _) = self.actor(x);
            let logp = activation::log_softmax(logits, 1);
            let chosen = logp
                .clone()
                .gather(1, x.actions.clone())
                .squeeze_dim::<1>(1);
            let ratio = (chosen - x.old_logp.clone()).exp();
            let unclipped = ratio.clone() * x.advantages.clone();
            let clipped = ratio.clamp(0.8, 1.2) * x.advantages.clone();
            let surrogate = unclipped
                .clone()
                .mask_where(unclipped.greater(clipped.clone()), clipped);
            let entropy = (logp.clone().exp() * logp)
                .sum_dim(1)
                .squeeze_dim::<1>(1)
                .neg();
            // All exported real decisions have at least two legal candidates.
            -surrogate.mean() - entropy.mean() * 0.01
        } else {
            let (value, _) = self.critic(x);
            let error = value.squeeze_dim::<1>(1) - x.returns.clone();
            let mask = x.advance_memory.clone().squeeze_dim::<1>(1);
            (error.clone() * error * mask.clone()).sum() / mask.sum().clamp_min(1.)
        }
    }
}
