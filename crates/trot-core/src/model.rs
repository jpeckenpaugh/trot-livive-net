//! The model: one hidden layer, a selectable activation, one output, sigmoid.
//!
//! ```text
//! z1 = W1 · x + b1      W1 ∈ R^(H×V)   (row-major)
//! h  = f(z1)            h  ∈ R^H       f ∈ {tanh, relu, leaky_relu}
//! z2 = W2 · h + b2      W2 ∈ R^H, b2 ∈ R
//! p  = σ(z2)
//! ```

use crate::rng::Rng;

/// Slope of the negative branch for [`Activation::LeakyReLU`].
pub const LEAKY_ALPHA: f32 = 0.01;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Activation {
    #[default]
    Tanh,
    ReLU,
    LeakyReLU,
}

impl Activation {
    /// Numeric code used by the raw ABI. Unknown codes fall back to tanh.
    pub fn from_code(code: u32) -> Self {
        match code {
            1 => Self::ReLU,
            2 => Self::LeakyReLU,
            _ => Self::Tanh,
        }
    }

    pub fn code(self) -> u32 {
        match self {
            Self::Tanh => 0,
            Self::ReLU => 1,
            Self::LeakyReLU => 2,
        }
    }

    /// Stable JSON identifier.
    pub fn id(self) -> &'static str {
        match self {
            Self::Tanh => "tanh",
            Self::ReLU => "relu",
            Self::LeakyReLU => "leaky_relu",
        }
    }

    /// ONNX op type for the hidden activation node.
    pub fn onnx_node(self) -> &'static str {
        match self {
            Self::Tanh => "Tanh",
            Self::ReLU => "Relu",
            Self::LeakyReLU => "LeakyRelu",
        }
    }

    /// Forward activation. The leaky slope applies at `z <= 0`.
    pub fn activate(self, z: f32) -> f32 {
        match self {
            Self::Tanh => z.tanh(),
            Self::ReLU => {
                if z > 0.0 {
                    z
                } else {
                    0.0
                }
            }
            Self::LeakyReLU => {
                if z > 0.0 {
                    z
                } else {
                    LEAKY_ALPHA * z
                }
            }
        }
    }

    /// Derivative with respect to `z`, including the exact edge behavior.
    pub fn derivative(self, z: f32) -> f32 {
        match self {
            Self::Tanh => {
                let t = z.tanh();
                1.0 - t * t
            }
            Self::ReLU => {
                if z > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::LeakyReLU => {
                if z > 0.0 {
                    1.0
                } else {
                    LEAKY_ALPHA
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct Model {
    pub hidden: usize,
    pub vocab: usize,
    pub activation: Activation,
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
    pub dact: Vec<f32>,
    pub z2: f32,
    pub p: f32,
}

pub fn sigmoid(z: f32) -> f32 {
    1.0 / (1.0 + (-z).exp())
}

impl Model {
    pub fn new(vocab: usize, hidden: usize, rng: &mut Rng) -> Self {
        Self::with_activation(vocab, hidden, Activation::default(), rng)
    }

    pub fn with_activation(
        vocab: usize,
        hidden: usize,
        activation: Activation,
        rng: &mut Rng,
    ) -> Self {
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
            activation,
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
        let mut hh = vec![0.0f32; h];
        let mut dact = vec![0.0f32; h];
        for j in 0..h {
            hh[j] = self.activation.activate(z1[j]);
            dact[j] = self.activation.derivative(z1[j]);
        }
        let mut z2 = self.b2;
        for j in 0..h {
            z2 += self.w2[j] * hh[j];
        }
        let p = sigmoid(z2);
        Forward {
            x: x.to_vec(),
            z1,
            h: hh,
            dact,
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
