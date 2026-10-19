//! The model: one hidden layer, `tanh`, one output, sigmoid.
//!
//! ```text
//! z1 = W1 · x + b1      W1 ∈ R^(H×V)   (row-major)
//! h  = tanh(z1)         h  ∈ R^H
//! z2 = W2 · h + b2      W2 ∈ R^H, b2 ∈ R
//! p  = σ(z2)
//! ```

use crate::rng::Rng;

#[derive(Clone, Debug)]
pub struct Model {
    pub hidden: usize,
    pub vocab: usize,
    pub w1: Vec<f32>, // hidden * vocab, row-major
    pub b1: Vec<f32>, // hidden
    pub w2: Vec<f32>, // hidden
    pub b2: f32,
}

#[derive(Clone, Debug)]
pub struct Forward {
    pub x: Vec<f32>,
    pub z1: Vec<f32>,
    pub h: Vec<f32>,
    pub z2: f32,
    pub p: f32,
}

pub fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

impl Model {
    pub fn new(vocab: usize, hidden: usize, rng: &mut Rng) -> Self {
        let a = (6.0 / (vocab + hidden) as f32).sqrt();
        let a2 = (6.0 / (hidden + 1) as f32).sqrt();
        let w1 = (0..hidden * vocab)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * a)
            .collect();
        let b1 = vec![0.0f32; hidden];
        let w2 = (0..hidden)
            .map(|_| (rng.next_f32() * 2.0 - 1.0) * a2)
            .collect();
        Model {
            hidden,
            vocab,
            w1,
            b1,
            w2,
            b2: 0.0,
        }
    }

    pub fn param_count(&self) -> usize {
        self.w1.len() + self.b1.len() + self.w2.len() + 1
    }

    pub fn forward(&self, x: &[f32]) -> Forward {
        let (h, v) = (self.hidden, self.vocab);
        let mut z1 = vec![0.0f32; h];
        for j in 0..h {
            let row = &self.w1[j * v..j * v + v];
            let mut acc = self.b1[j];
            for i in 0..v {
                acc += row[i] * x[i];
            }
            z1[j] = acc;
        }
        let hh: Vec<f32> = z1.iter().map(|&z| z.tanh()).collect();
        let mut z2 = self.b2;
        for j in 0..h {
            z2 += self.w2[j] * hh[j];
        }
        let p = sigmoid(z2);
        Forward {
            x: x.to_vec(),
            z1,
            h: hh,
            z2,
            p,
        }
    }

    /// Binary cross-entropy, clamped for numerical safety.
    pub fn loss(&self, x: &[f32], y: u8) -> f32 {
        let p = self.forward(x).p.clamp(1e-7, 1.0 - 1e-7);
        -(y as f32 * p.ln() + (1.0 - y as f32) * (1.0 - p).ln())
    }
}
