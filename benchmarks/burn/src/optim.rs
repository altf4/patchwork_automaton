//! Keras-compatible Adam for flattened named tensors, with global norm clipping.
//! Fresh optimizer state is used in both benchmark implementations.
use crate::model::{Inputs, Network};
use burn::prelude::*;
use std::collections::BTreeMap;

pub fn gradients(
    net: &Network,
    x: &Inputs,
    actor: bool,
) -> (f32, f32, BTreeMap<String, Tensor<1>>) {
    let loss = net.loss(x, actor);
    let value: f32 = loss.clone().into_scalar();
    let mut graph = loss.backward();
    let prefix = if actor { "actor/" } else { "critic/" };
    let grads: BTreeMap<_, _> = net
        .weights
        .iter()
        .filter(|(k, _)| k.starts_with(prefix))
        .map(|(k, w)| {
            (
                k.clone(),
                w.grad_remove(&mut graph)
                    .unwrap_or_else(|| panic!("Missing gradient: {k}")),
            )
        })
        .collect();
    let squares = grads
        .values()
        .map(|g| (g.clone() * g.clone()).sum())
        .collect();
    let norm: f32 = Tensor::cat(squares, 0).sum().sqrt().into_scalar();
    assert!(value.is_finite() && norm.is_finite());
    (value, norm, grads)
}

#[derive(Default)]
pub struct Adam {
    pub first: BTreeMap<String, Tensor<1>>,
    pub second: BTreeMap<String, Tensor<1>>,
    steps: u32,
}

impl Adam {
    pub fn initialized(net: &Network, actor: bool) -> Self {
        let prefix = if actor { "actor/" } else { "critic/" };
        let device = net.device.clone().without_autodiff();
        let mut optimizer = Self::default();
        for (name, weight) in &net.weights {
            if name.starts_with(prefix) {
                optimizer
                    .first
                    .insert(name.clone(), Tensor::zeros(weight.dims(), &device));
                optimizer
                    .second
                    .insert(name.clone(), Tensor::zeros(weight.dims(), &device));
            }
        }
        device.sync().unwrap();
        optimizer
    }

    pub fn apply(&mut self, net: &mut Network, norm: f32, grads: BTreeMap<String, Tensor<1>>) {
        self.steps += 1;
        let alpha = 0.0003_f32 * (1. - 0.999_f32.powi(self.steps as i32)).sqrt()
            / (1. - 0.9_f32.powi(self.steps as i32));
        for (name, grad) in grads {
            let grad = grad / norm.max(1.);
            let m = self
                .first
                .entry(name.clone())
                .or_insert_with(|| Tensor::zeros(grad.dims(), &grad.device()))
                .clone();
            let v = self
                .second
                .entry(name.clone())
                .or_insert_with(|| Tensor::zeros(grad.dims(), &grad.device()))
                .clone();
            let m = m.clone() + (grad.clone() - m) * 0.1;
            let v = v.clone() + (grad.clone() * grad - v) * 0.001;
            let next = net.weights[&name].clone().without_autodiff()
                - m.clone() * alpha / (v.clone().sqrt() + 1.0e-7);
            self.first.insert(name.clone(), m);
            self.second.insert(name.clone(), v);
            net.weights.insert(name, next.autodiff().require_grad());
        }
    }

    pub fn step(&mut self, net: &mut Network, x: &Inputs, actor: bool) -> f32 {
        let (_, norm, grads) = gradients(net, x, actor);
        self.apply(net, norm, grads);
        norm
    }
}
