//! Train the default model for each activation, export `target/trot_<id>.onnx`,
//! and write reference predictions to `target/trot_<id>_ref.json` so a Python
//! script can verify the exported graph with a standard runtime.
//!
//!     cargo run --release -p trot-core --example export_onnx

use trot_core::data::Dataset;
use trot_core::model::{Activation, Model};
use trot_core::onnx;
use trot_core::rng::Rng;
use trot_core::train;

const ACTIVATIONS: [Activation; 3] = [Activation::Tanh, Activation::ReLU, Activation::LeakyReLU];

fn export_one(ds: &Dataset, activation: Activation) -> (String, String, usize) {
    let mut rng = Rng::new(42);
    let mut m = Model::with_activation(ds.vocab_len(), 8, activation, &mut rng);
    for _ in 0..600 {
        train::minibatch_step(&mut m, ds, &mut rng, 16, 0.3, 0.0);
    }
    let (acc, loss) = train::evaluate(&m, ds, &ds.test_idx, 0.0);
    eprintln!(
        "{}: test acc {acc:.3} loss {loss:.4} vocab {}",
        activation.id(),
        ds.vocab_len()
    );

    let bytes = onnx::export(&m);
    let model_path = format!("target/trot_{}.onnx", activation.id());
    std::fs::write(&model_path, &bytes).expect("write onnx");

    // Reference inputs (presence vectors) and the core's own probabilities.
    let mut inputs = String::new();
    let mut probs = String::new();
    let mut n = 0usize;
    for &i in ds.test_idx.iter().take(25) {
        let x = ds.vectorize(&ds.snippets[i].text);
        let p = m.forward(&x).p;
        inputs.push('[');
        for (k, v) in x.iter().enumerate() {
            if k > 0 {
                inputs.push(',');
            }
            inputs.push_str(&format!("{v:.1}"));
        }
        inputs.push_str("],");
        probs.push_str(&format!("{p:.9},"));
        n += 1;
    }
    let json = format!(
        "{{\"activation\":\"{}\",\"vocab\":{},\"hidden\":{},\"n\":{},\
         \"inputs\":[{}],\"probs\":[{}]}}",
        activation.id(),
        ds.vocab_len(),
        m.hidden,
        n,
        inputs.trim_end_matches(','),
        probs.trim_end_matches(',')
    );
    let ref_path = format!("target/trot_{}_ref.json", activation.id());
    std::fs::write(&ref_path, &json).expect("write ref");

    // Keep the v2 paths pointing at the default (tanh) export.
    if activation == Activation::Tanh {
        std::fs::write("target/trot.onnx", &bytes).expect("write default onnx");
        std::fs::write("target/trot_ref.json", &json).expect("write default ref");
    }

    (model_path, ref_path, bytes.len())
}

fn main() {
    let ds = Dataset::generated(42, 256);
    for activation in ACTIVATIONS {
        let (model_path, ref_path, bytes) = export_one(&ds, activation);
        println!("wrote {model_path} ({bytes} bytes) and {ref_path}");
    }
}
