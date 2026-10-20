//! Raw WebAssembly boundary for TROT.
//!
//! No `wasm-bindgen`: the exports are plain C-ABI functions. JSON results are
//! written into a buffer the host reads with `trot_out_ptr` / `trot_out_len`;
//! the host writes input text into memory it allocates with `trot_alloc`.
//!
//! These are raw ABI entry points: the host owns the validity of any pointer it
//! passes. Marking them `unsafe extern "C"` would not change the exported ABI,
//! so the pointer-deref lint is allowed here explicitly.

#![allow(clippy::not_unsafe_ptr_arg_deref)]

use core::ptr::null_mut;

use trot_core::data::Dataset;
use trot_core::model::{Activation, Model};
use trot_core::onnx;
use trot_core::rng::Rng;
use trot_core::train;

/// ABI version shared by the WASM exports, the JSON payloads, and `app.js`.
pub const ABI_VERSION: u32 = 3;

struct State {
    ds: Dataset,
    model: Model,
    rng: Rng,
    step: u64,
    last_loss: f32,
    history: Vec<f32>,
    seed: u64,
    noise: f32,
    lambda: f32,
    last_report: Option<train::StepReport>,
    last_lr: f32,
    last_batch: u32,
    last_steps: u32,
    last_export_bytes: usize,
}

static mut STATE: *mut State = null_mut();
static mut OUT: *mut Vec<u8> = null_mut();

fn st() -> &'static mut State {
    unsafe { &mut *STATE }
}

fn set_out(bytes: Vec<u8>) {
    unsafe {
        if OUT.is_null() {
            OUT = Box::into_raw(Box::new(bytes));
        } else {
            *OUT = bytes;
        }
    }
}

fn esc(s: &str) -> String {
    let mut r = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => r.push_str("\\\""),
            '\\' => r.push_str("\\\\"),
            '\n' => r.push_str("\\n"),
            '\t' => r.push_str("\\t"),
            _ => r.push(c),
        }
    }
    r
}

fn farr(v: &[f32]) -> String {
    let mut s = String::from("[");
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!("{x}"));
    }
    s.push(']');
    s
}

fn frows(v: &[f32], rows: usize, cols: usize) -> String {
    let mut s = String::from("[");
    for j in 0..rows {
        if j > 0 {
            s.push(',');
        }
        s.push_str(&farr(&v[j * cols..j * cols + cols]));
    }
    s.push(']');
    s
}

fn init_state(seed: u32, n: u32, noise: f32, activation: u32, lambda: f32) {
    let mut n = (n as usize).clamp(100, 5000);
    n -= n % 2;
    let noise = noise.clamp(0.0, 1.0);
    let lambda = lambda.clamp(0.0, 0.1);
    let activation = Activation::from_code(activation);

    let ds = Dataset::generated_with(seed as u64, n, noise);
    let mut rng = Rng::new(seed as u64);
    let model = Model::with_activation(ds.vocab_len(), 8, activation, &mut rng);
    let s = State {
        ds,
        model,
        rng,
        step: 0,
        last_loss: 0.0,
        history: Vec::new(),
        seed: seed as u64,
        noise,
        lambda,
        last_report: None,
        last_lr: 0.0,
        last_batch: 0,
        last_steps: 0,
        last_export_bytes: 0,
    };
    unsafe {
        if STATE.is_null() {
            STATE = Box::into_raw(Box::new(s));
        } else {
            *STATE = s;
        }
    }
}

fn state_json(s: &State) -> String {
    let (train_acc, train_loss) = train::evaluate(&s.model, &s.ds, &s.ds.train_idx, s.lambda);
    let (test_acc, test_loss) = train::evaluate(&s.model, &s.ds, &s.ds.test_idx, s.lambda);
    let vocab_tokens =
        s.ds.vocab
            .iter()
            .map(|t| format!("\"{}\"", esc(t)))
            .collect::<Vec<_>>()
            .join(",");
    let w1_rows = frows(&s.model.w1, s.model.hidden, s.model.vocab);

    format!(
        "{{\"abi\":{},\"activation\":\"{}\",\"lambda\":{},\
         \"hidden\":{},\"vocab\":{},\"vocab_tokens\":[{}],\"step\":{},\"loss\":{},\
         \"train_acc\":{},\"test_acc\":{},\"train_loss\":{},\"test_loss\":{},\
         \"w1\":{},\"b1\":{},\"w2\":{},\"b2\":{},\"params\":{},\
         \"n\":{},\"n_pos\":{},\"n_neg\":{},\"train_n\":{},\"test_n\":{},\
         \"seed\":{},\"noise\":{},\"history\":{}}}",
        ABI_VERSION,
        s.model.activation.id(),
        s.lambda,
        s.model.hidden,
        s.model.vocab,
        vocab_tokens,
        s.step,
        s.last_loss,
        train_acc,
        test_acc,
        train_loss,
        test_loss,
        w1_rows,
        farr(&s.model.b1),
        farr(&s.model.w2),
        s.model.b2,
        s.model.param_count(),
        s.ds.snippets.len(),
        s.ds.positives(),
        s.ds.negatives(),
        s.ds.train_idx.len(),
        s.ds.test_idx.len(),
        s.seed,
        s.noise,
        farr(&s.history),
    )
}

fn train_json(s: &State) -> String {
    let (bce, penalty, total) = match &s.last_report {
        Some(r) => (r.bce, r.penalty, r.total),
        None => (0.0, 0.0, 0.0),
    };
    format!(
        "{{\"abi\":{},\"activation\":\"{}\",\"lambda\":{},\"bce\":{},\"penalty\":{},\
         \"total\":{},\"lr\":{},\"batch\":{},\"steps\":{},\"step\":{}}}",
        ABI_VERSION,
        s.model.activation.id(),
        s.lambda,
        bce,
        penalty,
        total,
        s.last_lr,
        s.last_batch,
        s.last_steps,
        s.step
    )
}

// ---- ABI -----------------------------------------------------------------

#[no_mangle]
pub extern "C" fn trot_alloc(len: usize) -> *mut u8 {
    let mut v: Vec<u8> = Vec::with_capacity(len.max(1));
    let p = v.as_mut_ptr();
    core::mem::forget(v);
    p
}

#[no_mangle]
pub extern "C" fn trot_dealloc(ptr: *mut u8, len: usize) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        drop(Vec::from_raw_parts(ptr, 0, len.max(1)));
    }
}

#[no_mangle]
pub extern "C" fn trot_out_ptr() -> *const u8 {
    unsafe {
        if OUT.is_null() {
            null_mut()
        } else {
            (*OUT).as_ptr()
        }
    }
}

#[no_mangle]
pub extern "C" fn trot_out_len() -> usize {
    unsafe {
        if OUT.is_null() {
            0
        } else {
            (*OUT).len()
        }
    }
}

#[no_mangle]
pub extern "C" fn trot_version() -> u32 {
    ABI_VERSION
}

#[no_mangle]
pub extern "C" fn trot_init(seed: u32, n: u32, noise: f32, activation: u32, lambda: f32) {
    init_state(seed, n, noise, activation, lambda);
    let json = state_json(st());
    set_out(json.into_bytes());
}

#[no_mangle]
pub extern "C" fn trot_configure(activation: u32, lambda: f32) {
    let s = st();
    s.model.activation = Activation::from_code(activation);
    s.lambda = lambda.clamp(0.0, 0.1);
    s.last_report = None;
    s.last_lr = 0.0;
    s.last_batch = 0;
    s.last_steps = 0;
    let json = state_json(s);
    set_out(json.into_bytes());
}

#[no_mangle]
pub extern "C" fn trot_state_json() {
    let json = state_json(st());
    set_out(json.into_bytes());
}

#[no_mangle]
pub extern "C" fn trot_dataset_json() {
    let s = st();
    let mut token_stats = String::from("[");
    for (i, (pos, neg)) in s.ds.token_stats().iter().enumerate() {
        if i > 0 {
            token_stats.push(',');
        }
        token_stats.push_str(&format!(
            "{{\"token\":\"{}\",\"pos\":{},\"neg\":{}}}",
            esc(&s.ds.vocab[i]),
            pos,
            neg
        ));
    }
    token_stats.push(']');

    let mut snippets = String::from("[");
    for (i, sn) in s.ds.snippets.iter().enumerate() {
        if i > 0 {
            snippets.push(',');
        }
        snippets.push_str(&format!(
            "{{\"text\":\"{}\",\"label\":{},\"split\":\"{}\",\"flipped\":{}}}",
            esc(&sn.text),
            sn.label,
            if s.ds.is_test(i) { "test" } else { "train" },
            sn.flipped
        ));
    }
    snippets.push(']');

    let json = format!(
        "{{\"abi\":{},\"n\":{},\"n_pos\":{},\"n_neg\":{},\"train_n\":{},\"test_n\":{},\
         \"vocab\":{},\"seed\":{},\"noise\":{},\"token_stats\":{},\"snippets\":{}}}",
        ABI_VERSION,
        s.ds.snippets.len(),
        s.ds.positives(),
        s.ds.negatives(),
        s.ds.train_idx.len(),
        s.ds.test_idx.len(),
        s.ds.vocab_len(),
        s.seed,
        s.noise,
        token_stats,
        snippets
    );
    set_out(json.into_bytes());
}

/// Input presence vectors and the core's probabilities for the test split.
/// Used by the browser to verify the exported ONNX against a standard runtime.
#[no_mangle]
pub extern "C" fn trot_roundtrip_json() {
    let s = st();
    let mut inputs = String::from("[");
    let mut probs = String::from("[");
    for (k, &i) in s.ds.test_idx.iter().enumerate() {
        if k > 0 {
            inputs.push(',');
            probs.push(',');
        }
        let x = s.ds.vectorize(&s.ds.snippets[i].text);
        inputs.push_str(&farr(&x));
        probs.push_str(&format!("{}", s.model.forward(&x).p));
    }
    inputs.push(']');
    probs.push(']');
    let json = format!(
        "{{\"abi\":{},\"activation\":\"{}\",\"n\":{},\"vocab\":{},\"inputs\":{},\"probs\":{}}}",
        ABI_VERSION,
        s.model.activation.id(),
        s.ds.test_idx.len(),
        s.model.vocab,
        inputs,
        probs
    );
    set_out(json.into_bytes());
}

#[no_mangle]
pub extern "C" fn trot_train(steps: u32, lr: f32, batch: u32) {
    if steps == 0 {
        return;
    }
    let s = st();
    let lr = lr.clamp(0.0, 1.0);
    let batch = batch.clamp(1, 64);
    for _ in 0..steps {
        let report = train::minibatch_step(
            &mut s.model,
            &s.ds,
            &mut s.rng,
            batch as usize,
            lr,
            s.lambda,
        );
        s.step += 1;
        s.last_loss = report.total;
        s.history.push(report.total);
        if s.history.len() > 2000 {
            s.history.drain(0..200);
        }
        s.last_report = Some(report);
    }
    s.last_lr = lr;
    s.last_batch = batch;
    s.last_steps = steps;
    let json = train_json(s);
    set_out(json.into_bytes());
}

#[no_mangle]
pub extern "C" fn trot_gradients_json() {
    let s = st();
    let (dw1, db1, dw2, db2, lr, batch, step, has_grad) = match &s.last_report {
        Some(r) => (
            r.gradients.w1.as_slice(),
            r.gradients.b1.as_slice(),
            r.gradients.w2.as_slice(),
            r.gradients.b2,
            s.last_lr,
            s.last_batch,
            s.step,
            true,
        ),
        None => (&[][..], &[][..], &[][..], 0.0, 0.0, 0, 0, false),
    };

    let (dw1, db1, dw2, db2) = if has_grad {
        (dw1.to_vec(), db1.to_vec(), dw2.to_vec(), db2)
    } else {
        (
            vec![0.0f32; s.model.hidden * s.model.vocab],
            vec![0.0f32; s.model.hidden],
            vec![0.0f32; s.model.hidden],
            0.0f32,
        )
    };

    let json = format!(
        "{{\"abi\":{},\"activation\":\"{}\",\"lambda\":{},\"lr\":{},\"batch\":{},\
         \"step\":{},\"dW1\":{},\"db1\":{},\"dW2\":[{}],\"db2\":[{}],\
         \"update_rule\":\"theta <- theta - lr * dtheta\"}}",
        ABI_VERSION,
        s.model.activation.id(),
        s.lambda,
        lr,
        batch,
        step,
        frows(&dw1, s.model.hidden, s.model.vocab),
        farr(&db1),
        farr(&dw2),
        db2
    );
    set_out(json.into_bytes());
}

#[no_mangle]
pub extern "C" fn trot_trace_json(text_ptr: *const u8, text_len: usize) {
    let bytes = unsafe { core::slice::from_raw_parts(text_ptr, text_len) };
    let text = String::from_utf8_lossy(bytes).to_string();
    let s = st();
    let tokens = trot_core::data::tokenize(&text);
    let x = s.ds.vectorize(&text);
    let f = s.model.forward(&x);
    let v = s.model.vocab;
    let h = s.model.hidden;

    let toks_json = tokens
        .iter()
        .map(|t| format!("\"{}\"", esc(t)))
        .collect::<Vec<_>>()
        .join(",");

    let mut in_vocab = String::from("[");
    let mut per_token = String::new();
    let mut first = true;
    for (k, t) in tokens.iter().enumerate() {
        let idx = s.ds.vocab.binary_search(t).ok();
        if k > 0 {
            in_vocab.push(',');
        }
        in_vocab.push_str(if idx.is_some() { "true" } else { "false" });
        if let Some(i) = idx {
            if !first {
                per_token.push(',');
            }
            first = false;
            let col: Vec<f32> = (0..h).map(|j| s.model.w1[j * v + i]).collect();
            per_token.push_str(&format!(
                "{{\"token\":\"{}\",\"index\":{},\"col\":{}}}",
                esc(t),
                i,
                farr(&col)
            ));
        }
    }
    in_vocab.push(']');

    let per_unit_z2: Vec<f32> = (0..h).map(|j| f.h[j] * s.model.w2[j]).collect();
    let label = if f.p >= 0.5 { 1 } else { 0 };

    let json = format!(
        "{{\"abi\":{},\"text\":\"{}\",\"tokens\":[{}],\"in_vocab\":{},\"per_token\":[{}],\
         \"activation\":\"{}\",\"b1\":{},\"z1\":{},\"h\":{},\"dact\":{},\"w2\":{},\
         \"per_unit_z2\":{},\"z2\":{},\"b2\":{},\"p\":{},\"label\":{}}}",
        ABI_VERSION,
        esc(&text),
        toks_json,
        in_vocab,
        per_token,
        s.model.activation.id(),
        farr(&s.model.b1),
        farr(&f.z1),
        farr(&f.h),
        farr(&f.dact),
        farr(&s.model.w2),
        farr(&per_unit_z2),
        f.z2,
        s.model.b2,
        f.p,
        label
    );
    set_out(json.into_bytes());
}

/// Export the current model as ONNX bytes; returns the byte length.
#[no_mangle]
pub extern "C" fn trot_export_onnx() -> usize {
    let s = st();
    let bytes = onnx::export(&s.model);
    let len = bytes.len();
    s.last_export_bytes = len;
    set_out(bytes);
    len
}

/// Graph anatomy of the most recent export.
#[no_mangle]
pub extern "C" fn trot_export_json() {
    let s = st();
    let activation = s.model.activation;
    let attrs = match activation {
        Activation::LeakyReLU => format!("{{\"alpha\":{}}}", trot_core::model::LEAKY_ALPHA),
        Activation::Tanh | Activation::ReLU => "{}".to_string(),
    };
    let json = format!(
        "{{\"abi\":{},\"activation\":\"{}\",\"activation_node\":\"{}\",\
         \"activation_attrs\":{},\"opset\":{},\"ir_version\":{},\
         \"graph_name\":\"trot\",\"nodes\":[\"Gemm\",\"{}\",\"Gemm\",\"Sigmoid\"],\
         \"initializers\":[{{\"name\":\"W1\",\"dims\":[{},{}],\"type\":\"float32\"}},\
         {{\"name\":\"b1\",\"dims\":[{}],\"type\":\"float32\"}},\
         {{\"name\":\"W2\",\"dims\":[1,{}],\"type\":\"float32\"}},\
         {{\"name\":\"b2\",\"dims\":[1],\"type\":\"float32\"}}],\
         \"input\":{{\"name\":\"input\",\"dims\":[\"batch\",{}],\"type\":\"float32\"}},\
         \"output\":{{\"name\":\"probability\",\"dims\":[\"batch\",1],\"type\":\"float32\"}},\
         \"bytes\":{}}}",
        ABI_VERSION,
        activation.id(),
        activation.onnx_node(),
        attrs,
        onnx::OPSET,
        onnx::IR_VERSION,
        activation.onnx_node(),
        s.model.hidden,
        s.model.vocab,
        s.model.hidden,
        s.model.hidden,
        s.model.vocab,
        s.last_export_bytes
    );
    set_out(json.into_bytes());
}
