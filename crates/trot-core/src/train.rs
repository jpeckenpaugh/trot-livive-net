//! Training: hand-written gradients and plain minibatch SGD.
//!
//! ```text
//! dz2 = p - y
//! dW2 = dz2 · h
//! db2 = dz2
//! dz1 = (dz2 · W2) ⊙ (1 - h²)
//! dW1 = dz1 ⊗ x
//! db1 = dz1
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
        let dz1 = dz2 * m.w2[j] * (1.0 - f.h[j] * f.h[j]);
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

/// One minibatch step. Returns the mean loss over the batch.
pub fn minibatch_step(m: &mut Model, ds: &Dataset, rng: &mut Rng, batch: usize, lr: f32) -> f32 {
    let h = m.hidden;
    let v = m.vocab;
    let mut gw1 = vec![0.0f32; h * v];
    let mut gb1 = vec![0.0f32; h];
    let mut gw2 = vec![0.0f32; h];
    let mut gb2 = 0.0f32;
    let mut loss = 0.0f32;

    let n = ds.train_idx.len();
    let take = batch.min(n).max(1);
    for _ in 0..take {
        let idx = ds.train_idx[rng.below(n)];
        let s = &ds.snippets[idx];
        let x = ds.vectorize(&s.text);
        let f = m.forward(&x);
        loss += m.loss(&x, s.label);
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
    sgd_step(
        m,
        &Gradients {
            w1: gw1,
            b1: gb1,
            w2: gw2,
            b2: gb2,
        },
        lr,
    );
    loss * inv
}

/// Accuracy and mean loss over the given indices.
pub fn evaluate(m: &Model, ds: &Dataset, idx: &[usize]) -> (f32, f32) {
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
    (correct as f32 / idx.len() as f32, loss / idx.len() as f32)
}
