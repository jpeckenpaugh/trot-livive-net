//! One test per activation × L2 cell.
//!
//! CI runs this file as a matrix, selecting a single cell per job:
//!
//! ```text
//! cargo test -p trot-core --test activation_matrix tanh_l2_off -- --exact
//! ```
//!
//! Each cell trains the default corpus, checks that it still learns, verifies
//! the analytic gradient (including the `λ·W` term) against a central
//! difference, and confirms the exported ONNX graph carries the matching
//! activation node.

use trot_core::data::Dataset;
use trot_core::model::{Activation, Model, LEAKY_ALPHA};
use trot_core::onnx;
use trot_core::rng::Rng;
use trot_core::train;

const SEED: u64 = 42;
const CORPUS: usize = 256;
const HIDDEN: usize = 8;
const STEPS: usize = 300;
const LAMBDA: f32 = 0.05;

fn total_loss(m: &Model, x: &[f32], y: u8, lambda: f32) -> f32 {
    m.loss(x, y) + train::penalty(m, lambda)
}

#[allow(clippy::too_many_arguments)]
fn check_param(
    m: &Model,
    x: &[f32],
    y: u8,
    lambda: f32,
    analytic: f32,
    label: &str,
    read: impl Fn(&Model) -> f32,
    write: impl Fn(&mut Model, f32),
) {
    let eps = 1e-3f32;
    let orig = read(m);
    let mut plus = m.clone();
    write(&mut plus, orig + eps);
    let mut minus = m.clone();
    write(&mut minus, orig - eps);
    let numeric =
        (total_loss(&plus, x, y, lambda) - total_loss(&minus, x, y, lambda)) / (2.0 * eps);
    assert!(
        (numeric - analytic).abs() < 1e-3,
        "{label}: analytic {analytic} vs numeric {numeric} (lambda {lambda})"
    );
}

fn check_gradients(m: &Model, ds: &Dataset, lambda: f32) {
    let snippet = &ds.snippets[0];
    let x = ds.vectorize(&snippet.text);
    let y = snippet.label;
    let f = m.forward(&x);
    let mut g = train::backward(m, &f, y);
    train::add_l2(m, &mut g, lambda);

    for k in 0..m.w1.len() {
        check_param(
            m,
            &x,
            y,
            lambda,
            g.w1[k],
            &format!("w1[{k}]"),
            |mm| mm.w1[k],
            |mm, v| mm.w1[k] = v,
        );
    }
    for j in 0..m.b1.len() {
        check_param(
            m,
            &x,
            y,
            lambda,
            g.b1[j],
            &format!("b1[{j}]"),
            |mm| mm.b1[j],
            |mm, v| mm.b1[j] = v,
        );
    }
    for j in 0..m.w2.len() {
        check_param(
            m,
            &x,
            y,
            lambda,
            g.w2[j],
            &format!("w2[{j}]"),
            |mm| mm.w2[j],
            |mm, v| mm.w2[j] = v,
        );
    }
    check_param(m, &x, y, lambda, g.b2, "b2", |mm| mm.b2, |mm, v| mm.b2 = v);
}

fn exercise(activation: Activation, lambda: f32) {
    let ds = Dataset::generated(SEED, CORPUS);
    assert_eq!(ds.vocab_len(), 50);
    assert_eq!(ds.train_idx.len() + ds.test_idx.len(), CORPUS);

    let mut rng = Rng::new(SEED);
    let mut m = Model::with_activation(ds.vocab_len(), HIDDEN, activation, &mut rng);
    let (_, initial_loss) = train::evaluate(&m, &ds, &ds.train_idx, lambda);

    for _ in 0..STEPS {
        train::minibatch_step(&mut m, &ds, &mut rng, 16, 0.3, lambda);
    }

    let (train_acc, train_loss) = train::evaluate(&m, &ds, &ds.train_idx, lambda);
    let (test_acc, test_loss) = train::evaluate(&m, &ds, &ds.test_idx, lambda);
    let cell = format!("{} (lambda {lambda})", activation.id());
    assert!(train_acc > 0.85, "{cell}: train acc {train_acc}");
    assert!(test_acc > 0.80, "{cell}: test acc {test_acc}");
    assert!(test_loss.is_finite(), "{cell}: non-finite test loss");
    assert!(
        train_loss < initial_loss * 0.6,
        "{cell}: loss {initial_loss} -> {train_loss}"
    );

    let penalty = train::penalty(&m, lambda);
    if lambda == 0.0 {
        assert_eq!(penalty, 0.0, "{cell}: L2 off must have zero penalty");
    } else {
        assert!(penalty > 0.0, "{cell}: L2 on must report a penalty");
    }

    check_gradients(&m, &ds, lambda);

    let bytes = onnx::export(&m);
    assert!(bytes.len() > 100, "{cell}: suspiciously small ONNX export");
    let op = activation.onnx_node().as_bytes();
    assert!(
        bytes.windows(op.len()).any(|window| window == op),
        "{cell}: ONNX export is missing the {} node",
        activation.onnx_node()
    );
    if activation == Activation::LeakyReLU {
        assert_eq!(LEAKY_ALPHA, 0.01);
    }

    for value in m.w1.iter().chain(m.b1.iter()).chain(m.w2.iter()) {
        assert!(value.is_finite(), "{cell}: non-finite parameter");
    }
    assert!(m.b2.is_finite(), "{cell}: non-finite b2");
}

#[test]
fn tanh_l2_off() {
    exercise(Activation::Tanh, 0.0);
}

#[test]
fn tanh_l2_on() {
    exercise(Activation::Tanh, LAMBDA);
}

#[test]
fn relu_l2_off() {
    exercise(Activation::ReLU, 0.0);
}

#[test]
fn relu_l2_on() {
    exercise(Activation::ReLU, LAMBDA);
}

#[test]
fn leaky_relu_l2_off() {
    exercise(Activation::LeakyReLU, 0.0);
}

#[test]
fn leaky_relu_l2_on() {
    exercise(Activation::LeakyReLU, LAMBDA);
}
