use trot_core::data::Dataset;
use trot_core::generator;
use trot_core::model::{Activation, Model, LEAKY_ALPHA};
use trot_core::onnx;
use trot_core::rng::Rng;
use trot_core::train;

const ACTIVATIONS: [Activation; 3] = [Activation::Tanh, Activation::ReLU, Activation::LeakyReLU];

fn tiny(activation: Activation) -> Model {
    Model {
        hidden: 2,
        vocab: 3,
        activation,
        w1: vec![1.0, 0.0, -1.0, 0.5, -0.5, 2.0],
        b1: vec![0.1, -0.2],
        w2: vec![0.7, -0.3],
        b2: 0.05,
    }
}

fn total_loss(m: &Model, x: &[f32], y: u8, lambda: f32) -> f32 {
    m.loss(x, y) + train::penalty(m, lambda)
}

fn assert_close(a: f32, b: f32, what: &str) {
    assert!((a - b).abs() < 1e-6, "{what}: {a} vs {b}");
}

#[test]
fn activation_forward_hand_computed() {
    // x = [0, 0, 1] gives z1 = [-0.9, 1.8]; both branches are exercised.
    let x = [0.0, 0.0, 1.0];
    for activation in ACTIVATIONS {
        let m = tiny(activation);
        let f = m.forward(&x);
        assert_close(f.z1[0], -0.9, "z1[0]");
        assert_close(f.z1[1], 1.8, "z1[1]");

        let (h0, h1, d0, d1) = match activation {
            Activation::Tanh => (
                (-0.9f32).tanh(),
                1.8f32.tanh(),
                1.0 - (-0.9f32).tanh().powi(2),
                1.0 - 1.8f32.tanh().powi(2),
            ),
            Activation::ReLU => (0.0, 1.8, 0.0, 1.0),
            Activation::LeakyReLU => (LEAKY_ALPHA * -0.9, 1.8, LEAKY_ALPHA, 1.0),
        };
        assert_close(f.h[0], h0, "h[0]");
        assert_close(f.h[1], h1, "h[1]");
        assert_close(f.dact[0], d0, "dact[0]");
        assert_close(f.dact[1], d1, "dact[1]");

        let z2 = 0.05 + 0.7 * h0 - 0.3 * h1;
        let p = 1.0 / (1.0 + (-z2).exp());
        assert_close(f.z2, z2, "z2");
        assert_close(f.p, p, "p");
    }
}

#[test]
fn activation_edge_behavior_and_codes() {
    assert_eq!(Activation::Tanh.activate(0.0), 0.0);
    assert_eq!(Activation::Tanh.derivative(0.0), 1.0);
    assert_eq!(Activation::ReLU.activate(0.0), 0.0);
    assert_eq!(Activation::ReLU.derivative(0.0), 0.0);
    assert_eq!(Activation::LeakyReLU.activate(0.0), 0.0);
    assert_eq!(Activation::LeakyReLU.derivative(0.0), LEAKY_ALPHA);
    assert_close(Activation::LeakyReLU.activate(-2.0), -0.02, "leaky(-2)");

    assert_eq!(Activation::from_code(0), Activation::Tanh);
    assert_eq!(Activation::from_code(1), Activation::ReLU);
    assert_eq!(Activation::from_code(2), Activation::LeakyReLU);
    assert_eq!(Activation::from_code(99), Activation::Tanh);
    assert_eq!(Activation::Tanh.id(), "tanh");
    assert_eq!(Activation::ReLU.id(), "relu");
    assert_eq!(Activation::LeakyReLU.id(), "leaky_relu");
    assert_eq!(Activation::LeakyReLU.onnx_node(), "LeakyRelu");
}

fn check_gradient(activation: Activation, lambda: f32) {
    let mut m = tiny(activation);
    let x = [0.0, 0.0, 1.0];
    let y = 1u8;
    let f = m.forward(&x);
    let mut g = train::backward(&m, &f, y);
    train::add_l2(&m, &mut g, lambda);
    let eps = 1e-3f32;

    for k in 0..m.w1.len() {
        let orig = m.w1[k];
        m.w1[k] = orig + eps;
        let lp = total_loss(&m, &x, y, lambda);
        m.w1[k] = orig - eps;
        let lm = total_loss(&m, &x, y, lambda);
        m.w1[k] = orig;
        let numeric = (lp - lm) / (2.0 * eps);
        assert!(
            (numeric - g.w1[k]).abs() < 1e-3,
            "{}: w1[{k}] grad {numeric} vs {}",
            activation.id(),
            g.w1[k]
        );
    }

    for j in 0..m.b1.len() {
        let orig = m.b1[j];
        m.b1[j] = orig + eps;
        let lp = total_loss(&m, &x, y, lambda);
        m.b1[j] = orig - eps;
        let lm = total_loss(&m, &x, y, lambda);
        m.b1[j] = orig;
        let numeric = (lp - lm) / (2.0 * eps);
        assert!(
            (numeric - g.b1[j]).abs() < 1e-3,
            "{}: b1[{j}] grad {numeric} vs {}",
            activation.id(),
            g.b1[j]
        );
    }

    for j in 0..m.w2.len() {
        let orig = m.w2[j];
        m.w2[j] = orig + eps;
        let lp = total_loss(&m, &x, y, lambda);
        m.w2[j] = orig - eps;
        let lm = total_loss(&m, &x, y, lambda);
        m.w2[j] = orig;
        let numeric = (lp - lm) / (2.0 * eps);
        assert!(
            (numeric - g.w2[j]).abs() < 1e-3,
            "{}: w2[{j}] grad {numeric} vs {}",
            activation.id(),
            g.w2[j]
        );
    }

    let orig = m.b2;
    m.b2 = orig + eps;
    let lp = total_loss(&m, &x, y, lambda);
    m.b2 = orig - eps;
    let lm = total_loss(&m, &x, y, lambda);
    m.b2 = orig;
    let numeric = (lp - lm) / (2.0 * eps);
    assert!(
        (numeric - g.b2).abs() < 1e-3,
        "{}: b2 grad {numeric} vs {}",
        activation.id(),
        g.b2
    );
}

#[test]
fn gradient_check_all_activations() {
    for activation in ACTIVATIONS {
        check_gradient(activation, 0.0);
    }
}

#[test]
fn l2_gradient_check_all_activations() {
    for activation in ACTIVATIONS {
        check_gradient(activation, 0.05);
    }
}

#[test]
fn loss_penalty_decomposition_and_last_gradients() {
    let mut ds = Dataset::generated(7, 100);
    ds.train_idx = vec![0];
    ds.test_idx = Vec::new();

    for activation in ACTIVATIONS {
        let mut m = Model::with_activation(ds.vocab_len(), 3, activation, &mut Rng::new(3));
        let before = m.clone();
        let mut rng = Rng::new(9);
        let lambda = 0.05f32;
        let lr = 0.1f32;
        let report = train::minibatch_step(&mut m, &ds, &mut rng, 1, lr, lambda);

        let s = &ds.snippets[0];
        let x = ds.vectorize(&s.text);
        let bce = before.loss(&x, s.label);
        let penalty = train::penalty(&before, lambda);
        assert_close(report.bce, bce, "bce");
        assert_close(report.penalty, penalty, "penalty");
        assert_close(report.total, bce + penalty, "total");

        let f = before.forward(&x);
        let bce_grad = train::backward(&before, &f, s.label);
        for k in 0..before.w1.len() {
            let expected = bce_grad.w1[k] + lambda * before.w1[k];
            assert_close(report.gradients.w1[k], expected, "dW1 includes lambda*W1");
            assert_close(
                m.w1[k],
                before.w1[k] - lr * expected,
                "W1 follows theta - lr*dtheta",
            );
        }
        for j in 0..before.w2.len() {
            let expected = bce_grad.w2[j] + lambda * before.w2[j];
            assert_close(report.gradients.w2[j], expected, "dW2 includes lambda*W2");
            assert_close(
                m.w2[j],
                before.w2[j] - lr * expected,
                "W2 follows theta - lr*dtheta",
            );
        }
        for j in 0..before.b1.len() {
            assert_close(report.gradients.b1[j], bce_grad.b1[j], "db1 has no L2");
        }
        assert_close(report.gradients.b2, bce_grad.b2, "db2 has no L2");
    }
}

#[test]
fn zero_lambda_reports_no_penalty() {
    let mut ds = Dataset::generated(7, 100);
    ds.train_idx = vec![0];
    ds.test_idx = Vec::new();
    let mut m = Model::new(ds.vocab_len(), 3, &mut Rng::new(3));
    let before = m.clone();
    let report = train::minibatch_step(&mut m, &ds, &mut Rng::new(9), 1, 0.1, 0.0);
    assert_eq!(report.penalty, 0.0);
    assert_close(report.total, report.bce, "total == bce");

    let s = &ds.snippets[0];
    let x = ds.vectorize(&s.text);
    let f = before.forward(&x);
    let g = train::backward(&before, &f, s.label);
    for (a, b) in report.gradients.w1.iter().zip(g.w1.iter()) {
        assert_close(*a, *b, "dW1 without L2");
    }
}

#[test]
fn seed_determinism() {
    let a = Model::new(20, 8, &mut Rng::new(42));
    let b = Model::new(20, 8, &mut Rng::new(42));
    assert_eq!(a.w1, b.w1);
    assert_eq!(a.w2, b.w2);
}

#[test]
fn generator_is_deterministic_balanced_and_bounded() {
    let a = generator::generate(42, 256);
    let b = generator::generate(42, 256);
    assert_eq!(a.len(), 256);
    assert_eq!(a, b);
    assert_eq!(a.iter().filter(|s| s.label == 1).count(), 128);
    assert_eq!(a.iter().filter(|s| s.label == 0).count(), 128);
    let ds = Dataset::generated(42, 256);
    assert!(ds.vocab_len() <= 50, "vocab {} > 50", ds.vocab_len());
    assert_eq!(ds.train_idx.len(), 205);
    assert_eq!(ds.test_idx.len(), 51);
}

#[test]
fn corpus_controls_respected() {
    let clean = generator::generate_with_noise(7, 256, 0.0);
    assert_eq!(clean.iter().filter(|s| s.label == 1).count(), 128);
    assert_eq!(clean.iter().filter(|s| s.label == 0).count(), 128);

    // Vocabulary is bounded at every supported corpus size.
    for n in [100usize, 256, 1000, 5000] {
        let ds = Dataset::generated_with(11, n, 0.08);
        assert!(ds.vocab_len() <= 50, "n={n} vocab={}", ds.vocab_len());
        assert_eq!(ds.snippets.len(), n);
        assert_eq!(ds.positives() + ds.negatives(), n);
    }

    // Noise is balanced: counts stay 50/50 at the default rate.
    let noisy = Dataset::generated_with(7, 1000, 0.08);
    assert_eq!(noisy.positives(), 500);
    assert_eq!(noisy.negatives(), 500);
}

#[test]
fn loss_decreases_and_learns() {
    let ds = Dataset::generated(42, 256);
    let mut rng = Rng::new(42);
    let mut m = Model::new(ds.vocab_len(), 8, &mut rng);
    let (_, initial) = train::evaluate(&m, &ds, &ds.train_idx, 0.0);
    for _ in 0..400 {
        train::minibatch_step(&mut m, &ds, &mut rng, 16, 0.3, 0.0);
    }
    let (train_acc, final_loss) = train::evaluate(&m, &ds, &ds.train_idx, 0.0);
    assert!(final_loss < initial * 0.6, "loss {initial} -> {final_loss}");
    assert!(train_acc > 0.8, "train acc {train_acc}");
}

// ---- minimal protobuf reader for ONNX assertions -------------------------

enum FieldValue<'a> {
    Varint(u64),
    Bytes(&'a [u8]),
    Float(f32),
}

fn read_varint(buf: &[u8], pos: &mut usize) -> u64 {
    let mut shift = 0;
    let mut value = 0u64;
    loop {
        let b = buf[*pos];
        *pos += 1;
        value |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            return value;
        }
        shift += 7;
    }
}

fn fields(buf: &[u8]) -> Vec<(u32, FieldValue<'_>)> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < buf.len() {
        let key = read_varint(buf, &mut pos);
        let field = (key >> 3) as u32;
        let wire = (key & 7) as u32;
        let value = match wire {
            0 => FieldValue::Varint(read_varint(buf, &mut pos)),
            2 => {
                let len = read_varint(buf, &mut pos) as usize;
                let bytes = &buf[pos..pos + len];
                pos += len;
                FieldValue::Bytes(bytes)
            }
            5 => {
                let bytes = &buf[pos..pos + 4];
                pos += 4;
                FieldValue::Float(f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
            }
            other => panic!("unsupported wire type {other}"),
        };
        out.push((field, value));
    }
    out
}

fn bytes_for(buf: &[u8], field: u32) -> Vec<&[u8]> {
    fields(buf)
        .into_iter()
        .filter_map(|(f, v)| match (f, v) {
            (f, FieldValue::Bytes(b)) if f == field => Some(b),
            _ => None,
        })
        .collect()
}

fn text(buf: &[u8]) -> String {
    String::from_utf8(buf.to_vec()).expect("utf8")
}

fn node_ops(graph: &[u8]) -> Vec<String> {
    bytes_for(graph, 1)
        .into_iter()
        .map(|n| text(bytes_for(n, 4)[0]))
        .collect()
}

fn node_attrs(node: &[u8]) -> Vec<(String, Option<f32>, Option<u64>)> {
    bytes_for(node, 5)
        .into_iter()
        .map(|a| {
            let name = text(bytes_for(a, 1)[0]);
            let mut value = None;
            let mut kind = None;
            for (f, v) in fields(a) {
                match (f, v) {
                    (2, FieldValue::Float(x)) => value = Some(x),
                    (20, FieldValue::Varint(x)) => kind = Some(x),
                    _ => {}
                }
            }
            (name, value, kind)
        })
        .collect()
}

fn initializer_names(graph: &[u8]) -> Vec<String> {
    bytes_for(graph, 5)
        .into_iter()
        .map(|t| text(bytes_for(t, 8)[0]))
        .collect()
}

#[test]
fn onnx_export_maps_each_activation() {
    for activation in ACTIVATIONS {
        let m = tiny(activation);
        let bytes = onnx::export(&m);
        assert!(bytes.len() > 100);
        let model = fields(&bytes);
        let graph = model
            .iter()
            .find_map(|(f, v)| match (f, v) {
                (7, FieldValue::Bytes(b)) => Some(*b),
                _ => None,
            })
            .expect("graph");

        let ops = node_ops(graph);
        assert_eq!(
            ops,
            vec![
                "Gemm".to_string(),
                activation.onnx_node().to_string(),
                "Gemm".to_string(),
                "Sigmoid".to_string()
            ]
        );
        assert_eq!(
            initializer_names(graph),
            vec![
                "W1".to_string(),
                "b1".to_string(),
                "W2".to_string(),
                "b2".to_string()
            ]
        );

        let nodes = bytes_for(graph, 1);
        let attrs = node_attrs(nodes[1]);
        match activation {
            Activation::Tanh | Activation::ReLU => assert!(attrs.is_empty(), "{attrs:?}"),
            Activation::LeakyReLU => {
                assert_eq!(attrs.len(), 1);
                let (name, value, kind) = &attrs[0];
                assert_eq!(name, "alpha");
                assert_eq!(*kind, Some(1), "alpha must be a FLOAT attribute");
                assert_eq!(value.map(f32::to_bits), Some(LEAKY_ALPHA.to_bits()));
            }
        }
    }
}
