use trot_core::data::Dataset;
use trot_core::generator;
use trot_core::model::Model;
use trot_core::onnx;
use trot_core::rng::Rng;
use trot_core::train;

fn tiny() -> Model {
    Model {
        hidden: 2,
        vocab: 3,
        w1: vec![1.0, 0.0, -1.0, 0.5, -0.5, 2.0],
        b1: vec![0.1, -0.2],
        w2: vec![0.7, -0.3],
        b2: 0.05,
    }
}

#[test]
fn forward_hand_computed() {
    let m = tiny();
    let f = m.forward(&[1.0, 0.0, 1.0]);
    let h0 = 0.1f32.tanh();
    let h1 = 2.3f32.tanh();
    let z2 = 0.05 + 0.7 * h0 - 0.3 * h1;
    let p = 1.0 / (1.0 + (-z2).exp());
    assert!((f.z1[0] - 0.1).abs() < 1e-6);
    assert!((f.z1[1] - 2.3).abs() < 1e-6);
    assert!((f.h[0] - h0).abs() < 1e-6);
    assert!((f.z2 - z2).abs() < 1e-6);
    assert!((f.p - p).abs() < 1e-6);
}

#[test]
fn gradient_check() {
    let mut rng = Rng::new(7);
    let mut m = Model::new(4, 3, &mut rng);
    let x = [1.0, 0.0, 1.0, 1.0];
    let y = 1u8;
    let f = m.forward(&x);
    let g = train::backward(&m, &f, y);
    let eps = 1e-3f32;

    // Check every weight by central difference.
    let mut numeric = Vec::new();
    for k in 0..m.w1.len() {
        let orig = m.w1[k];
        m.w1[k] = orig + eps;
        let lp = m.loss(&x, y);
        m.w1[k] = orig - eps;
        let lm = m.loss(&x, y);
        m.w1[k] = orig;
        numeric.push((lp - lm) / (2.0 * eps));
    }
    for (n, a) in numeric.iter().zip(g.w1.iter()) {
        assert!((n - a).abs() < 1e-3, "w1 grad {n} vs {a}");
    }

    for j in 0..m.b1.len() {
        let orig = m.b1[j];
        m.b1[j] = orig + eps;
        let lp = m.loss(&x, y);
        m.b1[j] = orig - eps;
        let lm = m.loss(&x, y);
        m.b1[j] = orig;
        let n = (lp - lm) / (2.0 * eps);
        assert!((n - g.b1[j]).abs() < 1e-3, "b1 grad {n} vs {}", g.b1[j]);
    }

    for j in 0..m.w2.len() {
        let orig = m.w2[j];
        m.w2[j] = orig + eps;
        let lp = m.loss(&x, y);
        m.w2[j] = orig - eps;
        let lm = m.loss(&x, y);
        m.w2[j] = orig;
        let n = (lp - lm) / (2.0 * eps);
        assert!((n - g.w2[j]).abs() < 1e-3, "w2 grad {n} vs {}", g.w2[j]);
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
fn loss_decreases_and_learns() {
    let ds = Dataset::generated(42, 256);
    let mut rng = Rng::new(42);
    let mut m = Model::new(ds.vocab_len(), 8, &mut rng);
    let (_, initial) = train::evaluate(&m, &ds, &ds.train_idx);
    for _ in 0..400 {
        train::minibatch_step(&mut m, &ds, &mut rng, 16, 0.3);
    }
    let (train_acc, final_loss) = train::evaluate(&m, &ds, &ds.train_idx);
    assert!(final_loss < initial * 0.6, "loss {initial} -> {final_loss}");
    assert!(train_acc > 0.8, "train acc {train_acc}");
}

#[test]
fn onnx_export_smoke() {
    let ds = Dataset::generated(42, 256);
    let mut rng = Rng::new(42);
    let mut m = Model::new(ds.vocab_len(), 8, &mut rng);
    for _ in 0..100 {
        train::minibatch_step(&mut m, &ds, &mut rng, 16, 0.3);
    }
    let bytes = onnx::export(&m);
    assert!(bytes.len() > 100);
    for needle in [
        "Gemm",
        "Tanh",
        "Sigmoid",
        "W1",
        "b1",
        "W2",
        "b2",
        "input",
        "probability",
    ] {
        assert!(
            bytes.windows(needle.len()).any(|w| w == needle.as_bytes()),
            "missing {needle}"
        );
    }
}
