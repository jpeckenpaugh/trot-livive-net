//! Training: hand-written gradients and plain minibatch SGD.
//!
//! ```text
//! dz2 = p - y
//! dW2 = dz2 · h
//! db2 = dz2
//! dz1 = (dz2 · W2) ⊙ f'(z1)
//! dW1 = dz1 ⊗ x
//! db1 = dz1
//! dW1 += λ · W1
//! dW2 += λ · W2
//! θ  ← θ - lr · dθ
//! ```

use crate::data::Dataset;
use crate::model::{Forward, Model};
use crate::rng::Rng;

#[derive(Clone, Debug)]
pub struct Gradients {
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: f32,
}

/// The loss decomposition and gradient of one minibatch step.
#[derive(Clone, Debug)]
pub struct StepReport {
    pub bce: f32,
    pub penalty: f32,
    pub total: f32,
    pub gradients: Gradients,
}

/// L2 penalty on weights only: `(λ/2) * (||W1||² + ||W2||²)`.
pub fn penalty(m: &Model, lambda: f32) -> f32 {
    let mut sum = 0.0f32;
    for &w in &m.w1 {
        sum += w * w;
    }
    for &w in &m.w2 {
        sum += w * w;
    }
    0.5 * lambda * sum
}

pub fn backward(m: &Model, f: &Forward, y: u8) -> Gradients {
    let (h, v) = (m.hidden, m.vocab);
    let dz2 = f.p - y as f32;

    let mut w2 = vec![0.0f32; h];
    for j in 0..h {
        w2[j] = dz2 * f.h[j];
    }

    let mut w1 = vec![0.0f32; h * v];
    let mut b1 = vec![0.0f32; h];
    for j in 0..h {
        let dz1 = dz2 * m.w2[j] * f.dact[j];
        b1[j] = dz1;
        let row = &mut w1[j * v..j * v + v];
        for i in 0..v {
            row[i] = dz1 * f.x[i];
        }
    }
    Gradients {
        w1,
        b1,
        w2,
        b2: dz2,
    }
}

/// Add the L2 gradient `λ·W` to the weight gradients. Biases are excluded.
pub fn add_l2(m: &Model, g: &mut Gradients, lambda: f32) {
    for (gw, w) in g.w1.iter_mut().zip(m.w1.iter()) {
        *gw += lambda * w;
    }
    for (gw, w) in g.w2.iter_mut().zip(m.w2.iter()) {
        *gw += lambda * w;
    }
}

pub fn sgd_step(m: &mut Model, g: &Gradients, lr: f32) {
    for (w, gw) in m.w1.iter_mut().zip(g.w1.iter()) {
        *w -= lr * gw;
    }
    for (b, gb) in m.b1.iter_mut().zip(g.b1.iter()) {
        *b -= lr * gb;
    }
    for (w, gw) in m.w2.iter_mut().zip(g.w2.iter()) {
        *w -= lr * gw;
    }
    m.b2 -= lr * g.b2;
}

/// One minibatch step. Returns the pre-update loss decomposition and the
/// gradient (including the L2 terms) that was used for the update.
pub fn minibatch_step(
    m: &mut Model,
    ds: &Dataset,
    rng: &mut Rng,
    batch: usize,
    lr: f32,
    lambda: f32,
) -> StepReport {
    let h = m.hidden;
    let v = m.vocab;
    let mut gw1 = vec![0.0f32; h * v];
    let mut gb1 = vec![0.0f32; h];
    let mut gw2 = vec![0.0f32; h];
    let mut gb2 = 0.0f32;
    let mut bce = 0.0f32;

    let n = ds.train_idx.len();
    let take = batch.min(n).max(1);
    for _ in 0..take {
        let idx = ds.train_idx[rng.below(n)];
        let s = &ds.snippets[idx];
        let x = ds.vectorize(&s.text);
        let f = m.forward(&x);
        bce += m.loss(&x, s.label);
        let g = backward(m, &f, s.label);
        for (a, b) in gw1.iter_mut().zip(g.w1.iter()) {
            *a += b;
        }
        for (a, b) in gb1.iter_mut().zip(g.b1.iter()) {
            *a += b;
        }
        for (a, b) in gw2.iter_mut().zip(g.w2.iter()) {
            *a += b;
        }
        gb2 += g.b2;
    }
    let inv = 1.0 / take as f32;
    for a in gw1.iter_mut() {
        *a *= inv;
    }
    for a in gb1.iter_mut() {
        *a *= inv;
    }
    for a in gw2.iter_mut() {
        *a *= inv;
    }
    gb2 *= inv;
    bce *= inv;

    let mut gradients = Gradients {
        w1: gw1,
        b1: gb1,
        w2: gw2,
        b2: gb2,
    };
    add_l2(m, &mut gradients, lambda);

    let penalty = penalty(m, lambda);
    let total = bce + penalty;
    sgd_step(m, &gradients, lr);

    StepReport {
        bce,
        penalty,
        total,
        gradients,
    }
}

/// Accuracy and mean loss over the given indices. The loss includes the
/// current-weight L2 penalty, so `lambda = 0` reproduces plain BCE.
pub fn evaluate(m: &Model, ds: &Dataset, idx: &[usize], lambda: f32) -> (f32, f32) {
    if idx.is_empty() {
        return (0.0, 0.0);
    }
    let mut correct = 0usize;
    let mut loss = 0.0f32;
    for &i in idx {
        let s = &ds.snippets[i];
        let x = ds.vectorize(&s.text);
        let f = m.forward(&x);
        loss += m.loss(&x, s.label);
        let pred = if f.p >= 0.5 { 1 } else { 0 };
        if pred == s.label {
            correct += 1;
        }
    }
    (
        correct as f32 / idx.len() as f32,
        loss / idx.len() as f32 + penalty(m, lambda),
    )
}
