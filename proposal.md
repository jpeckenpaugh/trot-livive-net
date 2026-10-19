# concept.md — TROT

**Transparent Rust ONNX Trainer** · repo: `trot-livive-net`

---

## 1. What This Is

TROT is a small, complete, from-scratch neural network project. A multilayer perceptron is written in plain Rust, compiled to WebAssembly, trained live in the browser, and exported to ONNX. Every parameter and every arithmetic operation is rendered in the DOM or readable in source.

The project exists to demonstrate one thing: **a real model can be built and trained without hiding anything.** The point is not accuracy. The point is that a reader can trace an input vector to an output probability one multiplication at a time, and then take the resulting artifact to any standard runtime and get the same answer.

TROT is a standalone project. It is not a component of a larger application and has no downstream consumer.

---

## 2. Non-Goals

- **Not a framework.** No layers API, no optimizers beyond plain SGD, no graph engine. One architecture, one loss, one update rule.
- **Not a transformer.** No attention, no tokenizers beyond whitespace + case folding, no embeddings beyond a bag-of-words vector.
- **Not a benchmark.** Accuracy is not the deliverable; inspectability is. The generalization test exists to make *train vs. test* visible, not to chase a number.
- **Not a server.** Everything is client-side. The build produces static files.
- **Not an ONNX runtime.** TROT *writes* ONNX. It does not read it. Round-trip verification uses an external runtime (see §10).
- **Not a general Rust ML crate.** The crate is a binary/`cdylib` pair, not a published library. No stability guarantees.

---

## 3. The Model

### 3.1 Task

Binary sentiment classification over short business-review snippets. Input is a sentence. Output is a probability in `[0, 1]` that the sentence is positive.

### 3.2 Architecture

Single hidden layer. Deliberately minimal.

```
x ∈ {0,1}^V           bag-of-words presence vector
z1 = W1 · x + b1      W1 ∈ R^(H×V), b1 ∈ R^H
h  = tanh(z1)         h ∈ R^H
z2 = W2 · h + b2      W2 ∈ R^H, b2 ∈ R
p  = σ(z2)            σ = logistic sigmoid
```

- **H = 8** by default, configurable 1–32.
- **Init:** `W1 ~ U(-a, a)` with `a = sqrt(6 / (V + H))` (Glorot-style), `b = 0`. Seeded, reproducible (§6.4).
- **Presence vector, not counts.** `x_i = 1` if token `i` appears, else `0`. Counts are a possible future toggle; presence is simpler to trace.

### 3.3 Loss and Update

```
L = -(y·log(p) + (1-y)·log(1-p))     binary cross-entropy
```

Gradients by hand, in source, with comments:

```
dz2 = p - y
dW2 = dz2 · h
db2 = dz2
dz1 = (dz2 · W2) ⊙ (1 - h²)
dW1 = dz1 ⊗ x
db1 = dz1
```

Update: plain SGD, `θ ← θ - lr · dθ`.

- **Minibatch SGD**, default batch size **16**, default `lr = 0.2`.
- `lr` adjustable in the UI from `0.01` to `2.0`.
- Batch size adjustable `1`–`64`. Per-sample (`1`) and full-batch remain available as toggles.
- Full-batch over the full corpus is a deliberate, occasional operation (the accuracy check), not the default click.

### 3.4 Per-Token Contribution (for the UI trace)

Two exact decompositions, both rendered in the trace view:

- **Into the hidden layer:** for token `i` present in the input, its contribution to `z1` is `W1[:, i]`. The sum of these columns, plus `b1`, equals `z1`. Exact, no approximation.
- **Into the output:** `z2 = Σ_j h_j · W2[j] + b2`. Each hidden unit's contribution is `h_j · W2[j]`. Exact.

This lets the UI say: *"the token `great` pushed hidden unit 3 by +0.41 and hidden unit 7 by −0.12"* with numbers that add up to what the trace shows.

---

## 4. Data

### 4.1 Two sets, one purpose

The corpus comes in two layers, both committed as *code*, not as bulk JSON:

- **Starter set — 20 hand-authored snippets.** These are the walkthrough's data: the same examples the conceptual stages use, small enough to read in full. They are the pedagogical anchor.
- **Generated set — 1,000 snippets by default (adjustable 100–5,000).** Produced by a seeded template grammar in `data/generator.rs`. This is what the model trains on by default. The data is not a magic artifact; it is `f(seed)`.

Committed files:

```
data/
├── starter.json          # 20 hand-authored snippets, human-editable
└── generator.rs          # seeded grammar; generates the rest on load
```

The generated corpus is **never committed**. It is regenerated deterministically from a constant seed. A reader can change the seed and get a different corpus; a test asserts that the same seed always yields the same bytes.

### 4.2 Starter set (20 snippets)

Format:

```json
[
  { "text": "the service was great", "label": 1 },
  { "text": "the service was terrible", "label": 0 }
]
```

Authoring rules for the starter set:

- **10 positive, 10 negative.** Balanced.
- **3–12 whitespace-separated words each.**
- **Overlapping vocabulary.** At least 12 distinct tokens appear in both classes. Words like *service*, *was*, *the*, *delivery*, *price* appear on both sides.
- **No single token may be a perfect class predictor.**
- **Ordinary business vocabulary only.** No slang, no sarcasm, no negation-heavy constructions in v1.

### 4.3 Generator

A template grammar in Rust. Every generated snippet is a filled template; every template slot draws from a small, hand-authored table. A reader can open `generator.rs` and see every sentence shape and every word the corpus can contain.

Sketch:

```
template      ::= "the" slot1 "was" slot2
                | slot1 "was" slot2
                | "their" slot1 "seemed" slot2
                | ...
slot1         ::= "service" | "delivery" | "price" | "staff" | "food" | ...
slot2         ::= positive_adj | negative_adj
positive_adj  ::= "great" | "fine" | "fast" | "reasonable" | "friendly" | ...
negative_adj  ::= "terrible" | "slow" | "overpriced" | "rude" | "awful" | ...
```

Design rules (these are hard constraints, enforced by tests in §10):

- **Vocabulary bounded.** The grammar must produce `|V| ≤ 50` distinct tokens. This keeps the `W1` weight table (`H × V` = 8 × ~40 = ~320 cells) readable as a grid in the UI.
- **Shared vocabulary across classes.** Function words and the `slot1` nouns appear in both positive and negative snippets. Only the `slot2` adjective carries signal.
- **Multiple discriminating dimensions.** Sentiment, speed, price, staff — so the model weighs several feature families rather than memorizing one word.
- **No perfect single-token predictor.** Removing any one adjective family from the corpus must not collapse test accuracy to chance. Tested (§10.4).
- **Balanced.** 50/50 class split before the train/test partition.
- **Train/test split by snippet.** 80/20, split on snippet identity, not on template. Every template family appears in both halves.

### 4.4 Vocabulary

Derived at runtime from whichever dataset is active.

```
vocab = sorted(set(tokenize(text) for all snippets in active dataset))
tokenize(s) = s.to_lowercase().split_whitespace(), punctuation stripped
```

Vocabulary size `V` is displayed in the UI ("V = 41 tokens, derived from 1,000 snippets"). This is a teaching moment: the vocabulary is not a mysterious artifact, it is `sort(unique(words))`.

### 4.5 Dataset selection

A dropdown in the train view: **Starter (20)** / **Generated (seed 42, N=1000)**. The starter set is the walkthrough's data. The generated set is the default for training and evaluation. The trace view defaults to a snippet from whichever set is active.

The seed and `N` for the generated set are editable fields. Changing either regenerates the corpus and resets the model (with a confirmation prompt).

---

## 5. Repository Layout

```
trot-livive-net/
├── Cargo.toml                  # workspace: core + wasm
├── data/
│   ├── starter.json            # 20 labeled snippets, committed, human-editable
│   └── (generator lives in crates/trot-core/src/generator.rs)
├── crates/
│   ├── trot-core/              # pure Rust, no wasm-bindgen
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── data.rs         # starter loading, tokenize, vocab, dataset union
│   │   │   ├── generator.rs    # seeded template grammar
│   │   │   ├── model.rs        # struct Model, forward, backward
│   │   │   ├── train.rs        # sgd step, minibatch, loss
│   │   │   ├── rng.rs          # seeded PRNG, in-crate
│   │   │   └── onnx.rs         # export path
│   │   └── tests/
│   │       ├── forward.rs
│   │       ├── gradient.rs
│   │       ├── generator.rs
│   │       ├── training.rs
│   │       └── onnx_export.rs
│   └── trot-wasm/              # wasm-bindgen boundary
│       └── src/lib.rs
├── web/
│   ├── index.html
│   ├── app.js
│   ├── style.css
│   └── pkg/                    # generated by wasm-pack, committed
├── Makefile
└── README.md
```

**Why split core and wasm:** `trot-core` has no WASM dependency and can be tested with plain `cargo test`. The WASM crate is a thin adapter. This keeps the model code readable and the boundary small.

---

## 6. Rust Core API

`trot-core` exposes the model as ordinary structs and functions. No traits beyond what's needed for the ONNX writer.

```rust
// data.rs
pub struct Snippet  { pub text: String, pub label: u8 }
pub struct Dataset  {
    pub snippets: Vec<Snippet>,
    pub vocab: Vec<String>,
    pub train_idx: Vec<usize>,
    pub test_idx:  Vec<usize>,
}

impl Dataset {
    pub fn starter() -> Result<Self, Error>;
    pub fn generated(seed: u64, n: usize) -> Self;
    pub fn union(a: &Dataset, b: &Dataset) -> Self;
    pub fn vectorize(&self, text: &str) -> Vec<f32>;   // {0,1}^V
}

// generator.rs
pub struct Grammar { /* template + slot tables, all in source */ }
impl Grammar {
    pub fn default_grammar() -> Self;
    pub fn generate(&self, seed: u64, n: usize) -> Vec<Snippet>;
}

// model.rs
pub struct Model {
    pub w1: Vec<f32>,   // H * V, row-major
    pub b1: Vec<f32>,   // H
    pub w2: Vec<f32>,   // H
    pub b2: f32,
    pub hidden: usize,
    pub vocab: usize,
}

pub struct Forward {
    pub x: Vec<f32>,
    pub z1: Vec<f32>,
    pub h: Vec<f32>,
    pub z2: f32,
    pub p: f32,
}

impl Model {
    pub fn new(vocab: usize, hidden: usize, rng: &mut Rng) -> Self;
    pub fn forward(&self, x: &[f32]) -> Forward;
    pub fn loss(&self, x: &[f32], y: u8) -> f32;
}

// train.rs
pub struct Gradients { pub w1: Vec<f32>, pub b1: Vec<f32>, pub w2: Vec<f32>, pub b2: f32 }
pub fn backward(m: &Model, f: &Forward, y: u8) -> Gradients;
pub fn sgd_step(m: &mut Model, g: &Gradients, lr: f32);
pub fn minibatch_step(m: &mut Model, d: &Dataset, idx: &[usize], lr: f32) -> f32;

// rng.rs — no external dependency
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self;
    pub fn next_f32(&mut self) -> f32;   // [0, 1)
    pub fn next_u32(&mut self) -> u32;
}
```

Everything is `Vec<f32>` in row-major order. No `ndarray`, no `tch`, no `burn`. The matrix-vector products are hand-written loops with a comment above each explaining the index arithmetic. If a reader cannot follow the loop, the loop is wrong.

---

## 7. WASM Boundary

`trot-wasm` compiles to `cdylib` and is exposed via `wasm-bindgen`. The boundary returns JSON strings, not nested JS objects, so the adapter stays trivial and the data shapes are inspectable in the browser console.

### 7.1 Exported functions

```rust
#[wasm_bindgen] pub fn init(seed: u32, dataset: &str, n: usize) -> Result<(), JsValue>;
#[wasm_bindgen] pub fn dataset_json() -> String;               // { snippets, vocab, train_idx, test_idx }
#[wasm_bindgen] pub fn get_state() -> String;                  // { w1, b1, w2, b2, hidden, vocab }
#[wasm_bindgen] pub fn train_step(lr: f32, batch: usize) -> String;   // { loss, steps }
#[wasm_bindgen] pub fn evaluate() -> String;                   // { train: {loss, correct, total}, test: {loss, correct, total} }
#[wasm_bindgen] pub fn trace(text: &str) -> String;            // full forward trace, see §7.2
#[wasm_bindgen] pub fn export_onnx() -> Vec<u8>;               // serialized .onnx
#[wasm_bindgen] pub fn reset(seed: u32);
```

`init(seed, dataset, n)` — `dataset` is `"starter"` or `"generated"`. For `"starter"`, `n` is ignored. For `"generated"`, `n` is the corpus size.

### 7.2 Trace payload

`trace(text)` returns enough for the UI to render every number without re-deriving anything:

```json
{
  "text": "the service was great",
  "tokens": ["the", "service", "was", "great"],
  "in_vocab": [true, true, true, true],
  "x": [0, 0, 1, 0, "... V entries"],
  "per_token_z1": {
    "the": [0.02, -0.11, "..."],
    "great": [0.41, 0.07, "..."]
  },
  "z1": [0.43, -0.04, "..."],
  "h": [0.40, -0.04, "..."],
  "per_unit_z2": [0.31, -0.02, "..."],
  "z2": 0.74,
  "p": 0.677,
  "label_predicted": 1
}
```

The UI never computes model math. It only formats and arranges.

### 7.3 Build

`wasm-pack build crates/trot-wasm --target web --out-dir ../../web/pkg`

The generated `web/pkg/` is **committed** so a clone of the repo can be served without a Rust toolchain. Contributors rebuild it via `make wasm`. Learners never touch it.

### 7.4 Reproducibility

`init(seed, ...)` and `reset(seed)` must produce **bit-identical weights** across native and WASM targets, and `generated(seed, n)` must produce **byte-identical corpora** across native and WASM targets. This is a hard requirement: the same seed on `cargo test` and in the browser must yield the same first loss and the same dataset. Tested (§10.3).

---

## 8. ONNX Export

### 8.1 Purpose

The `.onnx` file exists to prove three things, in this order:

1. **Portability** — the weights leave the project and load in a standard runtime.
2. **Inspectability** — the file opens in Netron and the graph and initializers are readable by eye.
3. **Round-trip fidelity** — reload and get identical predictions, to within `1e-6`.

It is not a feed into another application. It is the artifact form of the "Loaded Model" concept.

### 8.2 Graph

Minimal, single input, single output.

```
input  : [batch, V]  float32
Gemm   : W1 (H×V) transposed, b1 (H)      → [batch, H]
Tanh   :                                    → [batch, H]
Gemm   : W2 (1×H), b2 (1)                  → [batch, 1]
Sigmoid:                                    → [batch, 1]  "probability"
output : "probability"
```

Opset 13. IR version 8. Initializers named `W1`, `b1`, `W2`, `b2` so Netron labels them exactly as the source does.

### 8.3 Implementation

Primary path: **`onnx-export-rs`**, which targets inference-only Rust models, emits `Gemm` + `Sigmoid`, and narrows to `f32`. If its API cannot express a 1-output `Gemm` cleanly, the fallback is a **hand-written minimal protobuf encoder** (~200 lines) covering only the messages needed: `ModelProto`, `GraphProto`, `NodeProto`, `TensorProto`, `ValueInfoProto`. The encoder lives in `onnx.rs` behind a `trait OnnxWriter` so either path can be swapped without touching `model.rs`.

### 8.4 Constants

```rust
// onnx.rs
pub const OPSET: i64 = 13;
pub const IR_VERSION: i64 = 8;
pub const INPUT_NAME: &str = "input";
pub const OUTPUT_NAME: &str = "probability";
pub const INITIALIZER_NAMES: [&str; 4] = ["W1", "b1", "W2", "b2"];
```

Asserted in tests (§10.2).

---

## 9. UI

Single page, vanilla HTML/CSS/JS. No framework, no build step for the front end. Four views behind a tab bar.

### 9.1 Train

- Dataset selector: **Starter (20)** / **Generated (seed 42, N=1000)**. Seed and `N` fields for the generated set, with a regenerate confirmation.
- Loss curve on a `<canvas>`, current loss printed beside it.
- **Two accuracy readouts:** train accuracy and test accuracy, side by side. This is the headline addition that a larger corpus enables — the learner watches train climb toward 100% while test plateaus, and the concept of generalization becomes visible.
- Controls: **Train one step** (one minibatch), **Train 10**, **Train 100**, learning rate slider, batch size input, **Reset** (with seed field).
- Weight table below: `W1` as an `H × V` grid, `b1` and `W2` as rows. Every cell is a number. Cells flash on update so the learner sees which weights moved.

### 9.2 Data

- Filter by class, by template (for the generated set), by token.
- Token frequency table, per class: for every token, its count in positive snippets and in negative snippets. This is where a learner confirms for themselves that the vocabulary is shared and no single token is a cheat code.
- The 20 starter snippets are displayed in full alongside the generated corpus, so the relationship between the two layers is visible.
- Corpus statistics: `N`, `|V|`, class balance, train/test split sizes, template family distribution.

### 9.3 Trace

- Text input, defaulting to a snippet from the active dataset.
- Token list with in-vocabulary markers.
- Per-token contribution table into `z1` (columns of `W1`).
- Hidden pre-activation `z1`, activation `h`.
- Per-unit contribution into `z2`, the logit, the sigmoid, the final label.
- Every number formatted to 3 decimals but with a hover title showing full `f32` precision.

### 9.4 Export

- **Export .onnx** button (downloads the file).
- A summary panel: graph nodes, initializer shapes, opset, IR version, file size.
- A note pointing to Netron for full inspection.
- A **Round-trip check** button that loads the exported bytes into `onnxruntime-web` (loaded from CDN, only on this view) and compares predictions against the in-lab model for the entire active dataset. Displays pass/fail with max absolute difference.

The CDN load on the export view is the only external runtime dependency in the project, and it is confined to a feature that *verifies* the artifact rather than producing it.

---

## 10. Verification

Tests are the contract. An agent building this should treat §10 as the acceptance criteria.

### 10.1 Core model (`cargo test -p trot-core`)

| Test | Assertion |
|---|---|
| `forward_hand_computed` | With a 3-token vocab and 2 hidden units, a fixed weight set and fixed input produce a forward pass matching a value computed by hand in the test. |
| `gradient_check` | Analytic gradients from `backward` match central-difference numerical gradients within `1e-4` relative error, on 5 random inputs, for every parameter. |
| `loss_decreases` | 200 minibatch SGD steps on the generated corpus reduce full-dataset loss at 20 checkpoints, with `lr = 0.2`, `batch = 16`. |
| `seed_determinism` | `Model::new(42, ...)` twice produces byte-identical weights. |
| `vocab_derivation` | Vocab equals `sorted(set(tokenize(t)))` over the active dataset, exactly. |

### 10.2 ONNX (`cargo test -p trot-core --test onnx_export`)

| Test | Assertion |
|---|---|
| `parses` | Exported bytes parse as a valid `ModelProto` using the `prost`-generated ONNX definitions in `dev-dependencies` only. |
| `constants` | Opset, IR version, node names, initializer names, and shapes match §8.4 exactly. |
| `weights_match` | Decoded initializers equal the model's `w1`, `b1`, `w2`, `b2` to within `1e-6`. |

### 10.3 WASM/native parity (`crates/trot-wasm/tests/`)

| Test | Assertion |
|---|---|
| `seed_parity` | A native test binary and a `wasm-bindgen-test` both call `init(42, "generated", 1000)` and produce the same `get_state()` JSON (byte-identical after rounding to 6 decimals). |
| `corpus_parity` | `Dataset::generated(42, 1000)` yields byte-identical `Vec<Snippet>` on native and WASM. |
| `train_parity` | 100 minibatch steps from seed 42 produce the same loss sequence on both targets. |

### 10.4 Generator (`cargo test -p trot-core --test generator`)

| Test | Assertion |
|---|---|
| `deterministic` | `Grammar::default_grammar().generate(42, 1000)` twice yields byte-identical corpora. |
| `vocab_bounded` | `|V| ≤ 50` for the default grammar at any `N` in `100..=5000`. |
| `balanced` | 50/50 class split before train/test partition. |
| `split_by_snippet` | Train and test index sets are disjoint, cover all snippets, and every template family appears in both. |
| `shared_vocab` | At least 12 distinct tokens appear in both positive and negative snippets. |
| `no_single_token_classifier` | For each token `t`, remove all snippets containing `t`; the remaining corpus still trains to ≥ 85% test accuracy within 500 steps. Catches degenerate grammars. |

### 10.5 Training behavior

| Test | Assertion |
|---|---|
| `generalizes` | Default model (`H=8`, `lr=0.2`, `batch=16`) reaches ≥ 90% train and ≥ 80% test accuracy on the generated corpus within 500 minibatch steps. |
| `starter_overfits` | The 20 hand-authored snippets are exactly memorizable (20/20) within 2,000 minibatch steps at `batch=1`. Keeps the "learn the data exactly" sanity check alive. |

### 10.6 Round-trip (manual + CI if feasible)

The export view's round-trip check must pass on the default trained model: ONNX Runtime Web and the in-lab model agree on every snippet in the active dataset to `1e-6`.

### 10.7 Lint and format

`cargo fmt --check`, `cargo clippy -- -D warnings` in CI. No `#[allow(...)]` without a comment explaining why.

---

## 11. Build and Tooling

```make
make test      # cargo test --workspace
make lint      # fmt check + clippy
make wasm      # wasm-pack build, updates web/pkg/
make serve     # python3 -m http.server in web/
make all       # test, lint, wasm, then print the serve URL
```

Toolchain pinned in `rust-toolchain.toml` (`stable`, with `wasm32-unknown-unknown` and `wasm-bindgen-cli`/`wasm-pack` noted in the README). CI runs `make test` and `make lint` on push.

**Zero runtime dependencies** in `trot-core` except `serde` + `serde_json` for starter loading and WASM boundary payloads. `onnx-export-rs` (or `prost` + hand-rolled encoder) is the only other dependency, and it is confined to `onnx.rs`. `prost`-generated ONNX bindings may appear in `dev-dependencies` for tests only.

The generator has **no dependencies**. It is hand-written template filling.

The front end has no build step, no bundler, no npm. `web/app.js` imports the generated `pkg/trot_wasm.js` as an ES module directly.

---

## 12. Milestones

Each milestone is independently reviewable. An agent should not proceed past a milestone until its acceptance criteria pass.

**M0 — Scaffold.** Workspace, both crates, `data/starter.json` authored to §4.2, `generator.rs` with the default grammar, `make test` runs (with placeholder tests), CI green.

**M1 — Core model, native only.** `data.rs`, `generator.rs`, `model.rs`, `train.rs`, `rng.rs`. All §10.1, §10.4, and §10.5 tests pass. No WASM, no ONNX. This is the milestone that answers "is the math right" and "is the data sound."

**M2 — WASM boundary.** `trot-wasm` exposes `init`, `get_state`, `train_step`, `evaluate`. `web/index.html` shows a live loss curve, train/test accuracy readouts, and a weight table. §10.3 parity tests pass. No data browser, no trace view, no export yet.

**M3 — Data browser.** Data view with filters, token frequency table, corpus statistics, and the starter set shown alongside the generated corpus.

**M4 — Trace view.** `trace()` implemented, per-token and per-unit contributions rendered. Every number in the UI comes from the JSON payload; no math in JS.

**M5 — ONNX export.** `onnx.rs`, `export_onnx()`, export view with summary panel. §10.2 tests pass. The exported file opens cleanly in Netron.

**M6 — Round-trip.** ONNX Runtime Web loaded on the export view, round-trip check passes for the entire active dataset. README finalized with screenshots and a "read the source" guide pointing at the relevant functions.

**Post-v1 (explicitly deferred):** momentum, label noise toggle, negation handling, count vectors instead of presence, multi-class, a second hidden layer, saved model checkpoints, a JS-only port for comparison.

---

## 13. Design Principles (the tie-breakers)

When a decision is ambiguous, these resolve it.

1. **If a reader can't follow the loop, the loop is wrong.** Hand-written matvecs with index comments beat clever iterators.
2. **Every number on screen came from Rust.** The UI formats; it does not compute. If JS is doing arithmetic on model state, the boundary is wrong.
3. **The artifact is inspectable too.** ONNX initializers are named after the source struct fields. Netron should read like the source.
4. **The data is code, not a blob.** The corpus is `f(seed)` over a readable grammar. If a change requires committing generated rows, it is out of scope.
5. **Small enough to hold in your head.** `|V| ≤ 50`, `H = 8`, a grammar a reader can read in one sitting. If a change makes the model harder to trace by hand, it is out of scope for v1.
6. **Reproducible across targets.** Same seed, same weights, same corpus, native and WASM. No exceptions.
7. **Dependencies are confined.** `serde` at the boundary, ONNX crate in `onnx.rs`, `prost` in dev-deps. Nothing else.

---

## 14. Success Criteria

The project is complete when a person with a browser and no Rust toolchain can:

1. Open `web/index.html` from a static host.
2. See the model's initial weights as a table of numbers.
3. Click **Train** and watch those numbers change while the loss curve descends and **train and test accuracy diverge** — the generalization gap made visible.
4. Open the data view and read the generator's grammar, the token frequency table, and the starter set side by side, confirming that no single token is a cheat code.
5. Type a sentence into the trace view and follow its path from tokens, through `W1` columns, into `h`, through `W2`, out of the sigmoid — every number on the page coming from the Rust core.
6. Click **Export .onnx**, open the file in Netron, and see `W1`, `b1`, `W2`, `b2` named exactly as they were in the source.
7. Click **Round-trip check** and see the ONNX runtime agree with the in-lab model on every snippet in the active dataset.
8. Read the Rust source and find nothing hidden.

When those eight steps work, the question "where is the model code?" has an answer that is a URL, a repository, a generator function, and a file format.
