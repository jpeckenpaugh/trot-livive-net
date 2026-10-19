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
use trot_core::model::Model;
use trot_core::onnx;
use trot_core::rng::Rng;
use trot_core::train;

struct State {
    ds: Dataset,
    model: Model,
    rng: Rng,
    step: u64,
    last_loss: f32,
    history: Vec<f32>,
    seed: u64,
    noise: f32,
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

fn init_state(seed: u32, n: u32, noise: f32) {
    let n = (n as usize).clamp(100, 5000);
    let ds = Dataset::generated_with(seed as u64, n, noise);
    let mut rng = Rng::new(seed as u64);
    let model = Model::new(ds.vocab_len(), 8, &mut rng);
    let s = State {
        ds,
        model,
        rng,
        step: 0,
        last_loss: 0.0,
        history: Vec::new(),
        seed: seed as u64,
        noise,
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
    let (train_acc, train_loss) = train::evaluate(&s.model, &s.ds, &s.ds.train_idx);
    let (test_acc, test_loss) = train::evaluate(&s.model, &s.ds, &s.ds.test_idx);
    let vocab_tokens =
        s.ds.vocab
            .iter()
            .map(|t| format!("\"{}\"", esc(t)))
            .collect::<Vec<_>>()
            .join(",");

    let mut w1_rows = String::from("[");
    for j in 0..s.model.hidden {
        if j > 0 {
            w1_rows.push(',');
        }
        w1_rows.push_str(&farr(
            &s.model.w1[j * s.model.vocab..j * s.model.vocab + s.model.vocab],
        ));
    }
    w1_rows.push(']');

    format!(
        "{{\"hidden\":{},\"vocab\":{},\"vocab_tokens\":[{}],\"step\":{},\"loss\":{},\
         \"train_acc\":{},\"test_acc\":{},\"train_loss\":{},\"test_loss\":{},\
         \"w1\":{},\"b1\":{},\"w2\":{},\"b2\":{},\"params\":{},\
         \"n\":{},\"n_pos\":{},\"n_neg\":{},\"train_n\":{},\"test_n\":{},\
         \"seed\":{},\"noise\":{},\"history\":{}}}",
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
pub extern "C" fn trot_init(seed: u32, n: u32, noise: f32) {
    init_state(seed, n, noise);
    let json = state_json(st());
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
            "{{\"text\":\"{}\",\"label\":{},\"split\":\"{}\"}}",
            esc(&sn.text),
            sn.label,
            if s.ds.is_test(i) { "test" } else { "train" }
        ));
    }
    snippets.push(']');

    let json = format!(
        "{{\"n\":{},\"n_pos\":{},\"n_neg\":{},\"train_n\":{},\"test_n\":{},\"vocab\":{},\
         \"seed\":{},\"noise\":{},\"token_stats\":{},\"snippets\":{}}}",
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
        "{{\"n\":{},\"vocab\":{},\"inputs\":{},\"probs\":{}}}",
        s.ds.test_idx.len(),
        s.model.vocab,
        inputs,
        probs
    );
    set_out(json.into_bytes());
}

#[no_mangle]
pub extern "C" fn trot_train(steps: u32, lr: f32, batch: u32) -> f32 {
    let s = st();
    let mut last = s.last_loss;
    for _ in 0..steps {
        let l = train::minibatch_step(&mut s.model, &s.ds, &mut s.rng, batch as usize, lr.max(0.0));
        s.step += 1;
        s.history.push(l);
        if s.history.len() > 2000 {
            s.history.drain(0..200);
        }
        last = l;
    }
    s.last_loss = last;
    last
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
        "{{\"text\":\"{}\",\"tokens\":[{}],\"in_vocab\":{},\"per_token\":[{}],\
         \"b1\":{},\"z1\":{},\"h\":{},\"w2\":{},\"per_unit_z2\":{},\"z2\":{},\"b2\":{},\
         \"p\":{},\"label\":{}}}",
        esc(&text),
        toks_json,
        in_vocab,
        per_token,
        farr(&s.model.b1),
        farr(&f.z1),
        farr(&f.h),
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
    let bytes = onnx::export(&st().model);
    let len = bytes.len();
    set_out(bytes);
    len
}

/// Version string so the host can confirm it loaded the right build.
#[no_mangle]
pub extern "C" fn trot_version() -> u32 {
    2
}
