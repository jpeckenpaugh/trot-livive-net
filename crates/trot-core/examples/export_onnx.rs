//! Train the default model, export `target/trot.onnx`, and write reference
//! predictions to `target/trot_ref.json` so a Python script can verify the
//! exported graph with a standard runtime.
//!
//!     cargo run --release -p trot-core --example export_onnx

use trot_core::data::Dataset;
use trot_core::model::Model;
use trot_core::onnx;
use trot_core::rng::Rng;
use trot_core::train;

fn main() {
    let ds = Dataset::generated(42, 256);
    let mut rng = Rng::new(42);
    let mut m = Model::new(ds.vocab_len(), 8, &mut rng);
    for _ in 0..600 {
        train::minibatch_step(&mut m, &ds, &mut rng, 16, 0.3);
    }
    let (acc, loss) = train::evaluate(&m, &ds, &ds.test_idx);
    eprintln!("test acc {acc:.3} loss {loss:.4} vocab {}", ds.vocab_len());

    let bytes = onnx::export(&m);
    std::fs::write("target/trot.onnx", &bytes).expect("write onnx");

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
        "{{\"vocab\":{},\"hidden\":{},\"n\":{},\"inputs\":[{}],\"probs\":[{}]}}",
        ds.vocab_len(),
        m.hidden,
        n,
        inputs.trim_end_matches(','),
        probs.trim_end_matches(',')
    );
    std::fs::write("target/trot_ref.json", json).expect("write ref");
    println!(
        "wrote target/trot.onnx ({} bytes) and target/trot_ref.json",
        bytes.len()
    );
}
