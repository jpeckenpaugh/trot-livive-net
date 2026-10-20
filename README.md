# TROT — Transparent Rust ONNX Trainer

**Live:** <https://trot.livive.net/>

TROT is a small neural network you can read, run, and export. The model is
written in plain Rust — forward pass, backward pass, loss, weight update — and
compiled to WebAssembly so it trains in the browser with no server, no Python,
and no install. Each step renders its numbers to the page: weight matrices,
activations, gradients, loss, one multiplication at a time. Train it, trace a
single forward pass by hand, then export the result to ONNX and load it in any
standard runtime. The point is not the model's accuracy. The point is that
nothing is hidden.

- **The model is the code.** `crates/trot-core/src/model.rs` and `train.rs` are
  hand-written loops with index comments. No `ndarray`, no `burn`, no autodiff.
  The hidden activation is selectable (`tanh`, `relu`, `leaky_relu`), and L2
  regularization can be switched on with an explicit `lambda`.
- **The data is the code.** The corpus is a deterministic function of seed and
  size over the template grammar in `generator.rs`: a fixed 50-token lexicon,
  balanced positive/negative sentences, unique text, and an explicit label-noise
  rate. Defaults are seed 42 / 256 snippets (205 train / 51 test) and 0% noise;
  the page accepts 100–5,000 snippets and a 0–20% noise slider with an 8%
  preset. Noise flips an equal number of labels from each class, so the final
  classes stay 50/50. Flips are selected after the corpus shuffle but before
  the train/test split, so a flipped row can land in either split. Every flipped
  row is marked `flipped: true` in the data payload; its clean polarity is
  recoverable as `flipped ? 1 - label : label`, and it is badged in the Data
  panel. No dataset blob.
- **The artifact is inspectable too.** The `.onnx` is written by a ~200-line
  hand-rolled protobuf encoder (`onnx.rs`): four nodes, opset 13, IR 8,
  initializers named `W1`, `b1`, `W2`, `b2`. See
  [onnx_details.md](onnx_details.md) for the message tree, wire encoding,
  tensor layout, failure modes, and validation. The file opens in
  [Netron](https://netron.app/) and reads like the source.
- **The export is verified in place.** *Round-trip check* loads the exported
  bytes into `onnxruntime-web` (CDN, on demand) and compares every test-set
  prediction against the in-browser model.
- **The corpus is browsable.** The Data panel lists generated snippets with
  class/token filters, split membership, and per-token positive/negative counts;
  up to 400 filtered rows are shown at a time.

## What the page shows

The page has six panels:

1. **Train** — activation, optional L2 with live `lambda`, learning rate, batch
   size, seed, corpus size, label noise (slider plus an 8% preset), metrics, the
   BCE + penalty decomposition, and the loss curve. The page boots with a
   300-step run so the demo is immediately useful; Reset returns to random
   weights over the active corpus.
2. **Export** — writes `trot.onnx`, shows the graph anatomy reported by the Rust
   core, and runs the browser round-trip check.
3. **Data** — generated snippets with class and token filters, split membership,
   a badge on intentionally flipped labels, and per-token positive/negative
   counts.
4. **Weights** — the full `W1` grid (`H × V`) plus `b1` and `W2`.
5. **Gradients** — `dW1`, `db1`, `dW2`, and `db2` from the last minibatch,
   including the L2 term when enabled. Hovering or focusing a cell shows
   `theta -> dtheta -> theta - lr*dtheta` for the next step.
6. **Trace a prediction** — tokens, the `W1` column for each in-vocabulary
   token, `z1`, `h`, the activation derivative, `W2`, per-unit contributions,
   `z2`, and `p`.

## Architecture

```text
x ∈ {0,1}^V           bag-of-words presence vector (V = 50)
z1 = W1 · x + b1      W1 ∈ R^(8×50), b1 ∈ R^8
h  = f(z1)            f ∈ {tanh, relu, leaky_relu}
z2 = W2 · h + b2      W2 ∈ R^8, b2 ∈ R
p  = σ(z2)            binary cross-entropy
loss = BCE + (λ/2)(||W1||² + ||W2||²)   λ = 0 by default
θ ← θ - lr · dθ       minibatch SGD, biases are not penalized
```

Backpropagation is written out by hand in `train.rs`, including the
activation derivative and the L2 gradient `λ·W` on the weights. The WASM
boundary (`crates/trot-wasm`) is raw C-ABI — no `wasm-bindgen`. Results cross
the boundary as JSON strings; the UI formats numbers and never computes model
math.

## Build, test, run

```sh
make test     # cargo test -p trot-core
make lint     # cargo fmt --check + cargo clippy -D warnings
make wasm     # build trot.wasm and copy it to the repo root
make export   # train/export tanh, relu, leaky_relu; validate all with onnxruntime
make serve    # python3 -m http.server 8000  → http://localhost:8000
make all      # test, lint, wasm
```

`trot.wasm` is committed so the site can be served without a Rust toolchain.

### No C toolchain? (some sandboxes)

If the host has no `cc` linker, test natively against musl with the bundled LLD:

```sh
rustup target add aarch64-unknown-linux-musl
CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_LINKER=rust-lld \
  cargo test -p trot-core --target aarch64-unknown-linux-musl
```

## Verification

```sh
cargo test -p trot-core        # forward vs hand-computed, gradient checks,
                               # learning, determinism, generator bounds, onnx smoke
cargo run --release -p trot-core --example export_onnx
python3 scripts/validate_onnx.py
```

`export_onnx` trains the default corpus for each activation and writes
`target/trot_tanh.onnx`, `target/trot_relu.onnx`, and
`target/trot_leaky_relu.onnx` (plus reference predictions). The tanh export is
also copied to `target/trot.onnx`, the v2-compatible default path.
`validate_onnx.py` loads all three files with `onnx`, runs
`onnx.checker.check_model`, checks opset/IR, graph/initializer/node names and
shapes, and compares 25 predictions per model against the Rust core with
`onnxruntime`. The script requires agreement within `1e-6`; current measured
maximum differences are all below `1.5e-7`.

## Repository layout

```text
crates/trot-core/src/
  rng.rs        SplitMix64 — deterministic on native and Wasm
  data.rs       tokenize, fixed 50-token vocabulary, 80/20 split, presence vectors
  generator.rs  seeded template grammar, collocation rules, explicit label noise
  model.rs      forward pass, selectable activation
  train.rs      backward pass, L2 penalty, minibatch SGD
  onnx.rs       hand-written ONNX protobuf encoder
crates/trot-wasm/src/lib.rs   raw C-ABI boundary and JSON payloads
docs/abi-v3.md                frozen ABI v3 contract and corpus spec
onnx_details.md               ONNX message tree, wire format, and validation
index.html / app.js / style.css   the page (no framework, no bundler)
trot.wasm     committed build output
```

## Read the source

| Question | File |
| --- | --- |
| Where is the forward pass? | `crates/trot-core/src/model.rs` |
| Where are the gradients? | `crates/trot-core/src/train.rs` |
| Where does the data come from? | `crates/trot-core/src/generator.rs` |
| Where is the ONNX written? | `crates/trot-core/src/onnx.rs` |
| How does the ONNX encoder work? | [onnx_details.md](onnx_details.md) |
| Where is the UI contract? | `docs/abi-v3.md` |
| Where does the browser call in? | `crates/trot-wasm/src/lib.rs` |

## License

MIT — see [LICENSE](LICENSE). `concept.md` and `proposal.md` are the original
design notes.
