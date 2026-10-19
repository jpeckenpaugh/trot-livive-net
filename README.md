# TROT — Transparent Rust ONNX Trainer

**Live:** <https://trot.livive.net/>

TROT is a small neural network you can read, run, and export. The model is
written in plain Rust — forward pass, backward pass, loss, weight update — and
compiled to WebAssembly so it trains in the browser with no server, no Python,
and no install. Each step renders its numbers to the page: weight matrices,
activations, loss, one multiplication at a time. Train it, trace a single
forward pass by hand, then export the result to ONNX and load it in any standard
runtime. The point is not the model's accuracy. The point is that nothing is
hidden.

- **The model is the code.** `crates/trot-core/src/model.rs` and `train.rs` are
  hand-written loops with index comments. No `ndarray`, no `burn`, no autodiff.
- **The data is the code.** The 256-sentence corpus is `f(seed)` over the
  template grammar in `generator.rs` — 128 positive, 128 negative, ~8% of labels
  deliberately flipped so train and test accuracy diverge. Seed, corpus size
  (100–5,000), and the noise toggle are on the page; no dataset blob.
- **The artifact is inspectable too.** The `.onnx` is written by a ~200-line
  hand-rolled protobuf encoder (`onnx.rs`): four nodes, opset 13, IR 8,
  initializers named `W1`, `b1`, `W2`, `b2`. It opens in
  [Netron](https://netron.app/) and reads like the source.
- **The export is verified in place.** *Round-trip check* loads the exported
  bytes into `onnxruntime-web` (CDN, on demand) and compares every test-set
  prediction against the in-browser model.
- **The corpus is browsable.** The Data panel lists all generated snippets with
  class/token filters, split membership, and per-token positive/negative counts.

## Architecture

```text
x ∈ {0,1}^V           bag-of-words presence vector (V = 50)
z1 = W1 · x + b1      W1 ∈ R^(8×50), b1 ∈ R^8
h  = tanh(z1)
z2 = W2 · h + b2      W2 ∈ R^8, b2 ∈ R
p  = σ(z2)            binary cross-entropy, minibatch SGD
```

The WASM boundary (`crates/trot-wasm`) is raw C-ABI — no `wasm-bindgen`. Results
cross the boundary as JSON strings; the UI formats numbers and never computes
model math.

## Build, test, run

```sh
make test     # cargo test -p trot-core
make lint     # cargo fmt --check + cargo clippy -D warnings
make wasm     # build trot.wasm and copy it to the repo root
make export   # train, write target/trot.onnx, validate with onnxruntime
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
cargo test -p trot-core        # forward vs hand-computed, gradient check,
                               # learning, determinism, generator bounds, onnx smoke
cargo run --release -p trot-core --example export_onnx
python3 scripts/validate_onnx.py
```

`validate_onnx.py` loads `target/trot.onnx` with `onnxruntime`, checks the
protobuf, opset/IR, node and initializer names, and compares 25 predictions
against the Rust core — currently max difference `~1e-7`.

## Repository layout

```text
crates/trot-core/src/
  rng.rs        SplitMix64 — deterministic on native and Wasm
  data.rs       tokenize, vocabulary, 80/20 split, presence vectors
  generator.rs  the seeded template grammar (256 sentences)
  model.rs      forward pass
  train.rs      backward pass + minibatch SGD
  onnx.rs       hand-written ONNX protobuf encoder
crates/trot-wasm/src/lib.rs   raw C-ABI boundary
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
| Where does the browser call in? | `crates/trot-wasm/src/lib.rs` |

## License

MIT — see [LICENSE](LICENSE). `concept.md` and `proposal.md` are the original
design notes.
