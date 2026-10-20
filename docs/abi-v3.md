# TROT ABI v3 contract and corpus specification

**Status:** frozen for S1/S2/S3/S4 on branch `trot-improvements`.
**Scope:** interface and corpus contract only. This document is the S0 deliverable; it
does not change model, training, corpus, ONNX, or UI behavior by itself.
**Parent:** AIA-47, approved plan revision `2d372c65-4864-49c1-adde-4b20ac7ac25c`, §3 (S0).

If an implementation needs to change anything below, stop and agree a v4 contract
rather than silently changing a field, shape, or behavior. S1, S2, and S3 may work
in parallel only because the items in this file are fixed.

---

## 1. Invariants

| Item | v3 value |
|---|---|
| ABI version | **3** |
| Hidden units `H` | **8** |
| Vocabulary size `V` | **50** (fixed lexicon, not the observed token set) |
| Parameter count | **417** = `8*50 + 8 + 8 + 1` |
| Default activation | **`tanh`** |
| Default L2 | **off**, `lambda = 0.0` |
| Default label noise | **0%** (`0.0`) |
| Corpus seed / size / split | seed **42**, **256** snippets, **205 train / 51 test** |
| ONNX | opset **13**, IR **8** |
| Graph | `input -> Gemm -> activation -> Gemm -> Sigmoid -> probability` |
| Initializers | `W1`, `b1`, `W2`, `b2` |

Non-goals for v3: no new runtime dependency, no optimizer beyond plain SGD, no
additional hidden layer, no static dataset, and no model math in JavaScript.

The v3 corpus vocabulary is always exactly the 50-token lexicon in §8.2,
lexicographically sorted. A corpus size change or seed change must not change
`V`, `H`, or any JSON shape. If a future corpus changes the lexicon, that is an
ABI v4 change.

### 1.1 Version constant and handshake

Single source of truth in Rust:

```rust
// crates/trot-wasm/src/lib.rs
pub const ABI_VERSION: u32 = 3;
```

`trot_version()` returns `ABI_VERSION`. Every JSON payload carries
`"abi": 3` as an integer. On boot `app.js` must do:

```js
const ABI_VERSION = 3;
if (ex.trot_version() !== ABI_VERSION) {
  throw new Error(`trot.wasm ABI ${ex.trot_version()} != app ABI ${ABI_VERSION}`);
}
```

The existing build string in the header (`· wasm build v3`) is display only; the
version check above is the contract.

---

## 2. WASM boundary

The boundary stays raw C-ABI (no `wasm-bindgen`). JSON results are written to the
global output buffer read with `trot_out_ptr()` / `trot_out_len()`. The buffer is
overwritten by every call; the host must copy it before the next call.

### 2.1 Exports

| Export | Signature | Output | Behavior |
|---|---|---|---|
| `trot_version` | `() -> u32` | — | returns `3` |
| `trot_init` | `(seed: u32, n: u32, noise: f32, activation: u32, lambda: f32) -> void` | `state` JSON | rebuild corpus + model; clears step/history/gradients |
| `trot_configure` | `(activation: u32, lambda: f32) -> void` | `state` JSON | live config change; **does not** reset weights/corpus; clears stored gradients |
| `trot_state_json` | `() -> void` | `state` JSON | current model/corpus metadata, weights, metrics |
| `trot_dataset_json` | `() -> void` | `dataset` JSON | snippets, split, token statistics, flips |
| `trot_train` | `(steps: u32, lr: f32, batch: u32) -> void` | `train` JSON | runs minibatch SGD using the configured activation/lambda |
| `trot_gradients_json` | `() -> void` | `gradients` JSON | last-minibatch gradients, lr, update rule |
| `trot_trace_json` | `(text_ptr: *const u8, text_len: usize) -> void` | `trace` JSON | forward pass for one input string |
| `trot_export_onnx` | `() -> usize` | ONNX bytes | writes bytes to the output buffer, returns length |
| `trot_export_json` | `() -> void` | `export` JSON | graph anatomy for the last/current export |
| `trot_roundtrip_json` | `() -> void` | roundtrip JSON | test-set inputs and Rust probabilities |
| `trot_alloc` / `trot_dealloc` | existing | — | input text allocation |
| `trot_out_ptr` / `trot_out_len` | existing | — | output buffer access |

`trot_train` changes from returning `f32` (v2) to `void` and writing the `train`
payload. The S3 `app.js` must perform the v3 handshake; do not mix a v2 `app.js`
with a v3 `trot.wasm`.

### 2.2 Numeric configuration codes

JSON always uses string activation identifiers. The raw WASM boundary uses
`u32` codes:

| Code | Identifier | Default |
|---|---|---|
| `0` | `"tanh"` | yes |
| `1` | `"relu"` | |
| `2` | `"leaky_relu"` | |

Any other code is treated as `0` (`tanh`). The state payload always reports the
effective identifier, never the code.

### 2.3 Clamping and normalization

These are the only clamps/normalizations in the ABI. They make the wire values
predictable; the UI must display the effective values from the payloads.

| Value | Rule |
|---|---|
| `n` | clamp to `[100, 5000]`, then force even with `n -= n % 2` |
| `noise` | clamp to `[0.0, 1.0]`; UI exposes `[0.0, 0.20]` |
| `lambda` | clamp to `[0.0, 0.1]` |
| `lr` | clamp to `[0.0, 1.0]` |
| `batch` | clamp to `[1, 64]` |
| `steps` | `0` is a no-op; otherwise run the requested count |
| `seed` | `u32` as supplied |

`trot_init` must set `state.n` to the effective even size and `state.noise` to
the effective noise, not the raw arguments.

---

## 3. JSON conventions

- UTF-8, no BOM, no trailing bytes. One complete JSON value per call.
- All payloads include `"abi": 3`.
- Numbers are finite JSON numbers; no `NaN`, `Infinity`, or `-Infinity`.
  Rust `f32` values are serialized with the standard `format!("{x}")` shortest
  representation; JS parses them as doubles and formats for display.
- Matrices are **row-major**: `w1[j][i]` is `W1[j * V + i]`; `dW1[j][i]` is the
  same element. `state.w2` and `trace.w2` are 1-D length `H`; `gradients.dW2` is
  explicitly 1×H per the S0 requirement.
- `label` is `0 | 1` (0 negative, 1 positive). `split` is `"train" | "test"`.
  `flipped` is a JSON boolean.
- `activation` is `"tanh" | "relu" | "leaky_relu"`.
- Consumers must ignore unknown fields; producers must emit every required field
  listed here. A missing required field is a contract violation.
- Field order is not significant. Arrays are ordered as specified.

### 3.1 Names `app.js` will read

The following names are the stable UI-facing surface. S3 may add display-only DOM
state, but it must source these values from the payloads:

```text
state:     abi, activation, lambda, hidden, vocab, vocab_tokens, step, loss,
           train_acc, test_acc, train_loss, test_loss, w1, b1, w2, b2,
           params, n, n_pos, n_neg, train_n, test_n, seed, noise, history
dataset:   abi, n, n_pos, n_neg, train_n, test_n, vocab, seed, noise,
           token_stats[].token/pos/neg,
           snippets[].text/label/split/flipped
train:     abi, activation, lambda, bce, penalty, total, lr, batch, steps, step
gradients: abi, activation, lambda, lr, batch, step, dW1, db1, dW2, db2,
           update_rule
trace:     abi, text, tokens, in_vocab, per_token[].token/index/col,
           activation, b1, z1, h, dact, w2, per_unit_z2, z2, b2, p, label
export:    abi, activation, activation_node, activation_attrs, opset, ir_version,
           graph_name, nodes, initializers[].name/dims/type,
           input{name,dims,type}, output{name,dims,type}, bytes
roundtrip: abi, activation, n, vocab, inputs, probs
```

---

## 4. Payloads

Shapes use `H = 8`, `V = 50`. Examples are abbreviated with `...` where an array
repeats; the shapes are authoritative.

### 4.1 `state`

Produced by `trot_init`, `trot_configure`, and `trot_state_json`.

```json
{
  "abi": 3,
  "activation": "tanh",
  "lambda": 0.0,
  "hidden": 8,
  "vocab": 50,
  "vocab_tokens": ["and", "awful", "..."],
  "step": 310,
  "loss": 0.1832,
  "train_acc": 0.9659,
  "test_acc": 0.9020,
  "train_loss": 0.1710,
  "test_loss": 0.2214,
  "w1": [[0.012, -0.044, "..."], "..."],
  "b1": [0.01, "...", 8 values],
  "w2": [0.21, "...", 8 values],
  "b2": -0.03,
  "params": 417,
  "n": 256,
  "n_pos": 128,
  "n_neg": 128,
  "train_n": 205,
  "test_n": 51,
  "seed": 42,
  "noise": 0.0,
  "history": [0.69, 0.51, "..."]
}
```

| Field | Type | Shape | Meaning |
|---|---|---|---|
| `abi` | integer | — | `3` |
| `activation` | string | — | effective activation identifier |
| `lambda` | number | — | effective L2 coefficient in `[0, 0.1]` |
| `hidden` | integer | — | `H = 8` |
| `vocab` | integer | — | `V = 50` |
| `vocab_tokens` | string[] | `V` | sorted fixed lexicon; column order for `w1` |
| `step` | integer | — | completed minibatch steps |
| `loss` | number | — | total loss of the most recent minibatch; `0.0` before any step |
| `train_acc` | number | — | training-split accuracy in `[0,1]` |
| `test_acc` | number | — | test-split accuracy in `[0,1]` |
| `train_loss` | number | — | mean BCE over train + current penalty |
| `test_loss` | number | — | mean BCE over test + current penalty |
| `w1` | number[][] | `H × V` | row-major `W1` |
| `b1` | number[] | `H` | hidden bias |
| `w2` | number[] | `H` | output weights (logically 1×H) |
| `b2` | number | — | output bias |
| `params` | integer | — | always `417` in v3 |
| `n` | integer | — | effective corpus size |
| `n_pos` / `n_neg` | integer | — | final-label counts (after flips) |
| `train_n` / `test_n` | integer | — | split sizes |
| `seed` | integer | — | corpus seed |
| `noise` | number | — | effective noise fraction in `[0,1]` |
| `history` | number[] | ≤ 2000 | total loss per step; when over 2000, drop the oldest 200 |

`state.loss` is the total of the last minibatch. It is historical: changing
`lambda` or `activation` through `trot_configure` does not rewrite it. The
per-step decomposition is in the `train` payload.

### 4.2 `dataset`

Produced by `trot_dataset_json`. `n` is the effective even corpus size.

```json
{
  "abi": 3,
  "n": 256,
  "n_pos": 128,
  "n_neg": 128,
  "train_n": 205,
  "test_n": 51,
  "vocab": 50,
  "seed": 42,
  "noise": 0.0,
  "token_stats": [
    {"token": "and", "pos": 3, "neg": 4},
    "... 50 entries, sorted like vocab_tokens ..."
  ],
  "snippets": [
    {
      "text": "The service was great.",
      "label": 1,
      "split": "train",
      "flipped": false
    },
    "... n entries ..."
  ]
}
```

| Field | Type | Shape | Meaning |
|---|---|---|---|
| `abi` | integer | — | `3` |
| `n` | integer | — | effective even corpus size |
| `n_pos` / `n_neg` | integer | — | final-label counts |
| `train_n` / `test_n` | integer | — | split sizes |
| `vocab` | integer | — | `50` |
| `seed` | integer | — | corpus seed |
| `noise` | number | — | effective noise fraction |
| `token_stats` | object[] | `V` | sorted like `vocab_tokens`; per-token snippet counts by final label |
| `token_stats[].token` | string | — | token |
| `token_stats[].pos` | integer | — | snippets with final label 1 containing the token |
| `token_stats[].neg` | integer | — | snippets with final label 0 containing the token |
| `snippets` | object[] | `n` | generated corpus in dataset order |
| `snippets[].text` | string | — | canonical sentence, capitalized, final period |
| `snippets[].label` | integer | `0 | 1` | final label, **after** any intentional flip |
| `snippets[].split` | string | `"train" | "test"` | split membership |
| `snippets[].flipped` | boolean | — | `true` iff this label was intentionally flipped |

`token_stats` uses presence (a token counts once per snippet), not term
frequency, and uses the final label. A token in the fixed lexicon that does not
appear in a small corpus reports `pos = neg = 0`.

`flipped` is metadata, not a different label. The clean (pre-flip) polarity is
recoverable as `flipped ? 1 - label : label`.

### 4.3 `train`

Produced by `trot_train` after at least one step. If `steps = 0`, the call is a
no-op and the output buffer is unchanged.

```json
{
  "abi": 3,
  "activation": "tanh",
  "lambda": 0.0,
  "bce": 0.1832,
  "penalty": 0.0,
  "total": 0.1832,
  "lr": 0.3,
  "batch": 16,
  "steps": 10,
  "step": 310
}
```

| Field | Type | Meaning |
|---|---|---|
| `abi` | integer | `3` |
| `activation` | string | activation used for the last step |
| `lambda` | number | effective L2 used for the last step |
| `bce` | number | mean binary cross-entropy over the last minibatch, pre-update |
| `penalty` | number | `(lambda/2) * (||W1||² + ||W2||²)` at pre-update weights |
| `total` | number | `bce + penalty` |
| `lr` | number | effective learning rate used |
| `batch` | integer | effective batch size used |
| `steps` | integer | number of steps performed by this call |
| `step` | integer | global step count after the call |

When `lambda = 0`, `penalty = 0.0` and `total = bce`. The decomposition is for the
**last** step only, not an average over the call.

### 4.4 `gradients`

Produced by `trot_gradients_json`. Gradients are the last minibatch average used
for the last SGD update. They include the L2 term:

```text
dW1 = dBCE/dW1 + lambda * W1       (H × V)
dW2 = dBCE/dW2 + lambda * W2       (1 × H)
db1 = dBCE/db1                     (H)
db2 = dBCE/db2                     (1)
```

```json
{
  "abi": 3,
  "activation": "tanh",
  "lambda": 0.0,
  "lr": 0.3,
  "batch": 16,
  "step": 310,
  "dW1": [[0.004, -0.001, "..."], "..."],
  "db1": [0.002, "...", 8 values],
  "dW2": [[0.031, "...", 8 values]],
  "db2": [-0.012],
  "update_rule": "theta <- theta - lr * dtheta"
}
```

| Field | Type | Shape | Meaning |
|---|---|---|---|
| `abi` | integer | — | `3` |
| `activation` | string | — | activation used for the gradient |
| `lambda` | number | — | effective L2 used for the gradient |
| `lr` | number | — | learning rate used for the last update; `0.0` if no step has run |
| `batch` | integer | — | batch size used; `0` if no step has run |
| `step` | integer | — | global step of the gradient; `0` if no step has run |
| `dW1` | number[][] | `H × V` | row-major gradient including `lambda * W1` |
| `db1` | number[] | `H` | hidden-bias gradient; no L2 |
| `dW2` | number[][] | `1 × H` | output-weight gradient including `lambda * W2` |
| `db2` | number[] | `1` | output-bias gradient; no L2 |
| `update_rule` | string | — | exact literal `theta <- theta - lr * dtheta` |

If no training step has occurred (including after `trot_init` or
`trot_configure`), all gradient arrays are zeros, `lr = 0.0`, `batch = 0`, and
`step = 0`. `activation` and `lambda` still report the current configuration.

The update rule is applied elementwise after the gradients are reported:

```text
W1 <- W1 - lr * dW1
b1 <- b1 - lr * db1
W2 <- W2 - lr * dW2
b2 <- b2 - lr * db2
```

For the S3 hover detail, `theta_after = theta_before - lr * dtheta` is the
**next-step projection** from the displayed gradient; it is not the final weight
after a multi-step click.

### 4.5 `trace`

Produced by `trot_trace_json(ptr, len)`. Input text is tokenized exactly as the
corpus: lowercase, split on whitespace, keep alphanumerics, drop empty tokens.

```json
{
  "abi": 3,
  "text": "The service was great.",
  "tokens": ["the", "service", "was", "great"],
  "in_vocab": [true, true, true, true],
  "per_token": [
    {"token": "the", "index": 44, "col": ["...", 8 values]},
    {"token": "service", "index": 39, "col": ["...", 8 values]}
  ],
  "activation": "tanh",
  "b1": ["...", 8 values],
  "z1": ["...", 8 values],
  "h": ["...", 8 values],
  "dact": ["...", 8 values],
  "w2": ["...", 8 values],
  "per_unit_z2": ["...", 8 values],
  "z2": 0.84,
  "b2": -0.03,
  "p": 0.70,
  "label": 1
}
```

| Field | Type | Shape | Meaning |
|---|---|---|---|
| `abi` | integer | — | `3` |
| `text` | string | — | original input text |
| `tokens` | string[] | token count | tokenized input, in order |
| `in_vocab` | boolean[] | token count | parallel to `tokens` |
| `per_token` | object[] | in-vocab count | one entry per in-vocab token, input order |
| `per_token[].token` | string | — | token |
| `per_token[].index` | integer | — | index into `vocab_tokens` |
| `per_token[].col` | number[] | `H` | `W1[:, index]` |
| `activation` | string | — | selected activation identifier |
| `b1` | number[] | `H` | hidden bias |
| `z1` | number[] | `H` | hidden pre-activation |
| `h` | number[] | `H` | `activation(z1)` |
| `dact` | number[] | `H` | `activation'(z1)`, elementwise |
| `w2` | number[] | `H` | output weights |
| `per_unit_z2` | number[] | `H` | `h[j] * w2[j]` |
| `z2` | number | — | `sum(per_unit_z2) + b2` |
| `b2` | number | — | output bias |
| `p` | number | — | `sigmoid(z2)` |
| `label` | integer | `0 | 1` | `p >= 0.5 ? 1 : 0` |

Out-of-vocabulary tokens appear in `tokens` with `in_vocab: false`, are ignored
by the presence vector, and have no `per_token` entry. `dact` is always present,
for both correlated and uncorrelated activations.

### 4.6 `export`

Two calls are involved:

1. `trot_export_onnx() -> usize` writes the ONNX bytes to the output buffer and
   returns the byte length.
2. `trot_export_json()` writes graph metadata to the output buffer.

`app.js` must copy the bytes from step 1 before calling step 2, because the
metadata overwrites the output buffer.

```json
{
  "abi": 3,
  "activation": "tanh",
  "activation_node": "Tanh",
  "activation_attrs": {},
  "opset": 13,
  "ir_version": 8,
  "graph_name": "trot",
  "nodes": ["Gemm", "Tanh", "Gemm", "Sigmoid"],
  "initializers": [
    {"name": "W1", "dims": [8, 50], "type": "float32"},
    {"name": "b1", "dims": [8], "type": "float32"},
    {"name": "W2", "dims": [1, 8], "type": "float32"},
    {"name": "b2", "dims": [1], "type": "float32"}
  ],
  "input": {"name": "input", "dims": ["batch", 50], "type": "float32"},
  "output": {"name": "probability", "dims": ["batch", 1], "type": "float32"},
  "bytes": 1234
}
```

| Field | Type | Meaning |
|---|---|---|
| `abi` | integer | `3` |
| `activation` | string | current activation identifier |
| `activation_node` | string | `Tanh`, `Relu`, or `LeakyRelu` |
| `activation_attrs` | object | `{}` for Tanh/ReLU; `{"alpha": 0.01}` for Leaky ReLU |
| `opset` | integer | `13` |
| `ir_version` | integer | `8` |
| `graph_name` | string | `"trot"` |
| `nodes` | string[] | exactly `["Gemm", <activation>, "Gemm", "Sigmoid"]` |
| `initializers` | object[] | exactly `W1`, `b1`, `W2`, `b2` in that order |
| `initializers[].name` | string | initializer name |
| `initializers[].dims` | integer[] | tensor shape |
| `initializers[].type` | string | `"float32"` |
| `input` | object | `{name:"input", dims:["batch", 50], type:"float32"}` |
| `output` | object | `{name:"probability", dims:["batch", 1], type:"float32"}` |
| `bytes` | integer | byte length of the most recent `trot_export_onnx()` result; `0` if none |

### 4.7 `roundtrip`

Existing browser ONNX verification payload, retained with the version tag and
activation name:

```json
{
  "abi": 3,
  "activation": "tanh",
  "n": 51,
  "vocab": 50,
  "inputs": [[0.0, 1.0, "..."], "..."],
  "probs": [0.12, "..."]
}
```

`n` is the test-split size, `inputs` is `n × V` presence vectors, and `probs` is
length `n`. `trot_roundtrip_json` uses the current model and activation. The
browser compares `onnxruntime-web` output to `probs` within `1e-6`.

---

## 5. Activation identifiers and math

### 5.1 Forward and derivative

All values are `f32`. `z` is an element of `z1`; `h` is the corresponding element
of `h`; `dact` is the corresponding element of `dact`.

| Identifier | Code | Forward `h = f(z)` | Derivative `f'(z)` |
|---|---|---|---|
| `tanh` | `0` | `tanh(z)` | `1 - h*h` (equivalently `1 - tanh(z)²`) |
| `relu` | `1` | `z > 0 ? z : 0` | `z > 0 ? 1 : 0` |
| `leaky_relu` | `2` | `z > 0 ? z : 0.01 * z` | `z > 0 ? 1 : 0.01` |

Exact edge behavior:

- **ReLU at `z == 0`:** forward `0`, derivative `0`.
- **Leaky ReLU at `z == 0`:** forward `0`, derivative **`0.01`**. The leaky slope
  applies at `z <= 0`; for `z > 0` the slope is `1`.
- **Tanh at `z == 0`:** forward `0`, derivative `1`.
- `leaky_relu` alpha is exactly `0.01f32` and is not configurable in v3.

### 5.2 Backward pass

The hidden-layer delta is generalized from v2:

```text
dz2 = p - y
dW2 = dz2 * h
db2 = dz2
dz1[j] = dz2 * W2[j] * dact[j]
dW1[j][i] = dz1[j] * x[i]
db1[j] = dz1[j]
```

With `tanh`, `dact[j] = 1 - h[j]²`, which is identical to v2 behavior. No other
part of the backward pass changes.

### 5.3 Switching activation

`trot_configure(activation, lambda)` changes the activation of the live model
without resetting weights, corpus, step counter, or history. The next forward
pass, trace, train step, and ONNX export use the new activation. The stored
`gradients` payload is cleared (zeros, `step = 0`, `lr = 0.0`, `batch = 0`)
because old gradients are not valid under the new derivative.

`trot_init` also accepts an activation code and builds the new model with it.

---

## 6. L2 semantics

### 6.1 Definition

For weights only:

```text
penalty = (lambda / 2) * (||W1||² + ||W2||²)
        = (lambda / 2) * (sum(W1[j][i]^2) + sum(W2[j]^2))
```

Biases `b1` and `b2` are **excluded**. There is no penalty on activations or
inputs.

### 6.2 Loss and gradients

For a minibatch:

```text
bce     = mean over the batch of BCE(p, y)
penalty = (lambda / 2) * (||W1||² + ||W2||²)
total   = bce + penalty
```

The penalty is a property of the weights and is added once to the mean BCE; it is
not averaged per example. With `lambda = 0`, `penalty = 0` and `total = bce`.

The gradient used by SGD is the gradient of `total`:

```text
dW1 = dBCE/dW1 + lambda * W1
dW2 = dBCE/dW2 + lambda * W2
db1 = dBCE/db1
db2 = dBCE/db2
```

`gradients.dW1` and `gradients.dW2` include the `lambda * W` terms;
`gradients.db1` and `gradients.db2` do not. The `train` payload reports the
decomposition at the **pre-update** weights of the last step; `gradients` is the
gradient of that same step.

`state.train_loss` and `state.test_loss` are total losses:
`mean_BCE(split) + penalty(current weights)`. At `lambda = 0` they are exactly the
v2 BCE losses.

### 6.3 Defaults, range, and ONNX

- Default: **off**, `lambda = 0.0`.
- Effective ABI range: `[0.0, 0.1]`, clamped. Out-of-range values are clamped and
  echoed as effective values.
- UI range: `0.0` to `0.1`, step `0.001`; the checkbox controls whether the slider
  is active. Unchecked forces `lambda = 0.0`.
- ONNX: **unaffected**. L2 is training-only; no penalty node, initializer, or
  attribute is exported. `export.nodes`, `export.initializers`, and the graph shape
  are identical for on/off at the same activation.

---

## 7. ONNX mapping

### 7.1 Activation mapping

| ABI identifier | ONNX node | Attributes |
|---|---|---|
| `tanh` | `Tanh` | none |
| `relu` | `Relu` | none |
| `leaky_relu` | `LeakyRelu` | `alpha = 0.01` (FLOAT) |

The graph contains exactly one activation node, in position 2 of the four-node
chain:

```text
input [batch,50]
  -> Gemm(W1, b1, transB=1) -> z1 [batch,8]
  -> <Tanh|Relu|LeakyRelu>  -> h  [batch,8]
  -> Gemm(W2, b2, transB=1) -> z2 [batch,1]
  -> Sigmoid                 -> probability [batch,1]
```

### 7.2 Frozen ONNX constants

| Item | Value |
|---|---|
| Opset | `13` |
| IR version | `8` |
| Graph name | `trot` |
| Input name | `input` |
| Output name | `probability` |
| Intermediate values | `z1`, `h`, `z2` |
| Initializers | `W1` `[8,50]`, `b1` `[8]`, `W2` `[1,8]`, `b2` `[1]` |
| Data type | `FLOAT` (`float32`) |
| Gemm attribute | `transB = 1` on both Gemm nodes |
| Leaky ReLU attribute | `alpha = 0.01f32` |

The Leaky ReLU `alpha` attribute is an `AttributeProto` `FLOAT` (field 2, type
`FLOAT`), not an integer. The existing hand-written protobuf encoder needs an
`attr_float` helper for this; no new dependency is allowed.

`lambda` and label noise never appear in the ONNX graph. The export metadata
payload (`export`) must report the activation node and attributes that are
actually in the bytes.

### 7.3 Validation

`scripts/validate_onnx.py` must validate all three activations:

- protobuf validity and ONNX checker;
- opset 13 / IR 8;
- initializer names exactly `["W1", "b1", "W2", "b2"]` and shapes
  `[8,50]`, `[8]`, `[1,8]`, `[1]`;
- node op types exactly `["Gemm", <activation>, "Gemm", "Sigmoid"]`;
- `LeakyRelu` has `alpha == 0.01`; `Tanh`/`Relu` have no alpha;
- input/output names and types;
- `onnxruntime` predictions match the Rust core within `1e-6` for each activation.

The in-browser round-trip uses the same `1e-6` tolerance.

---

## 8. Corpus generator specification

The corpus is generated, not hand-edited. S1 changes the generator and
regenerates; it does not patch rows. The default corpus is seed 42, 256 snippets,
0% noise.

### 8.1 Corpus constants

| Constant | Value |
|---|---|
| `N_MIN`, `N_MAX` | `100`, `5000` |
| Effective `n` | clamped, then forced even |
| `NOISE_DEFAULT` | `0.0` |
| `NOISE_PRESET` | `0.08` |
| UI noise max | `0.20` |
| Test fraction | `0.20`, `test_n = round(0.2 * n)` |
| Default seed | `42` |
| Default size | `256` |
| Split seed | `seed ^ 0x5EED_5EED` |
| Tokenizer | lowercase; whitespace split; keep alphanumerics; drop empty |
| Vocabulary | fixed 50-token lexicon, sorted |
| Duplicate retry limit | `64` attempts per snippet |
| Unique candidate space | **5,794** sentences |

The unique candidate space is 22 single-adjective templates × 203 legal
noun/adjective pairs (4,466) plus 1,328 ordered same-polarity adjective pairs for
the two-adjective template, for **5,794**. The per-class spaces are 3,164
positive and 2,630 negative, both at least the 2,500 needed at `n = 5000`, so the
generator can guarantee unique sentences for every supported `n`.

### 8.2 Fixed lexicon

Exactly these 50 tokens, sorted ascending. `vocab_tokens[i]` is column `i` of
`W1`. All templates use only this lexicon.

```text
and, awful, bad, broken, clean, delivery, dirty, disappointing, excellent,
experience, fast, felt, fine, food, found, frankly, friendly, good, great,
helpful, honestly, i, late, overall, overpriced, packaging, pleasant, poor,
price, product, quality, quick, really, reasonable, reliable, room, rude,
seemed, selection, service, slow, staff, support, terrible, the, today,
unhelpful, value, wait, was
```

Counts: 12 function words, 14 nouns, 24 adjectives. The removed v2 tokens `a`
and `very` are gone; `honestly` and `frankly` are added so the fixed lexicon is
still exactly 50. The vocabulary is the lexicon, not the set of tokens observed
in a particular small sample, so `V = 50` for every seed and size.

### 8.3 Adjective categories

Each adjective belongs to exactly one category and one polarity.

| Category | Positive adjectives | Negative adjectives |
|---|---|---|
| manner | `friendly`, `helpful` | `rude`, `unhelpful` |
| speed | `fast`, `quick` | `slow`, `late` |
| condition | `clean` | `dirty`, `broken` |
| price/value | `reasonable` | `overpriced` |
| general evaluation | `great`, `fine`, `excellent`, `good`, `pleasant`, `reliable` | `terrible`, `awful`, `poor`, `bad`, `disappointing` |

Totals: 12 positive adjectives, 12 negative adjectives. "Absolute" adjectives
such as `excellent`, `terrible`, and `broken` are never prefixed with `very`.

### 8.4 Per-noun compatibility

A noun may only take adjectives from the categories and subsets below. `all`
means every adjective in that category, both polarities. `—` means none.

| Noun | Manner | Speed | Condition | Price/value | General evaluation |
|---|---|---|---|---|---|
| `service` | all | `fast`, `quick`, `slow` | — | all | all |
| `delivery` | — | all | — | all | all |
| `price` | — | — | — | all | all |
| `staff` | all | — | — | — | all |
| `food` | — | — | — | all | all |
| `room` | — | — | `clean`, `dirty` | all | all |
| `wait` | — | `fast`, `quick`, `slow` | — | — | all |
| `quality` | — | — | — | `reasonable` | all |
| `experience` | — | — | — | — | all |
| `product` | — | — | all | all | all |
| `support` | all | all | — | — | all |
| `packaging` | — | — | all | — | all |
| `selection` | — | — | — | — | all |
| `value` | — | — | — | all | all |

This table is the complete compatibility contract. Examples that are
intentionally invalid and must never be generated: `The price was fast.`, `The
experience was friendly.`, `The food was clean.`, `The room was broken.`, `The
selection was overpriced.`, `The wait was late.`, `The staff was fast.`

The generator samples both the noun and the adjective set from this table; it
does not rely on a separate "grammar checker" at runtime.

### 8.5 Template list

All templates are capitalized, end with a single final period, use `the` (never
`a`/`an`), and contain no `very`. `{noun}` is one of the 14 nouns. `{adj}` is a
compatible adjective of the requested polarity. `{adj2}` is a second, distinct
adjective of the **same noun and same polarity**, used only by T21.

| # | Template |
|---|---|
| T1 | `The {noun} was {adj}.` |
| T2 | `The {noun} seemed {adj}.` |
| T3 | `I found the {noun} {adj}.` |
| T4 | `I felt the {noun} was {adj}.` |
| T5 | `Overall the {noun} was {adj}.` |
| T6 | `Today the {noun} seemed {adj}.` |
| T7 | `I found the {noun} {adj} today.` |
| T8 | `Overall I found the {noun} {adj}.` |
| T9 | `The {noun} really was {adj}.` |
| T10 | `Honestly the {noun} was {adj}.` |
| T11 | `Frankly the {noun} seemed {adj}.` |
| T12 | `The {noun} was {adj} overall.` |
| T13 | `Today I found the {noun} {adj}.` |
| T14 | `Overall the {noun} seemed {adj}.` |
| T15 | `I really felt the {noun} was {adj}.` |
| T16 | `The {noun} seemed {adj} today.` |
| T17 | `Honestly I found the {noun} {adj}.` |
| T18 | `Frankly the {noun} was {adj}.` |
| T19 | `The {noun} was {adj} today.` |
| T20 | `I felt the {noun} was {adj} today.` |
| T21 | `The {noun} was {adj} and {adj2}.` |
| T22 | `Honestly the {noun} seemed {adj}.` |
| T23 | `Frankly I felt the {noun} was {adj}.` |

T1–T20 and T22–T23 are single-adjective (22 templates). T21 is the only
two-adjective template. The template index is the row number minus one and is
used by the fallback scan.

### 8.6 Deterministic generation algorithm

The corpus text must be identical for the same `seed`, `n`, and `noise`; changing
only `noise` must not change the clean text, only labels and `flipped` flags.

1. Effective `n` and `noise` are clamped as in §2.3.
2. Create the corpus RNG: `Rng::new(seed)`. This RNG is independent of the model
   initialization/training RNG, which also starts at `Rng::new(seed)` but is
   created after corpus generation.
3. For `i in 0..n`:
   - `positive = (i % 2 == 0)`. Before shuffling, positive and negative snippets
     alternate exactly.
   - Draw `template = rng.below(23)`.
   - Draw `noun = NOUNS[rng.below(14)]`.
   - Build the compatible adjective pool for `noun` and the requested polarity
     from §8.4.
   - For a single-adjective template, draw `adj` uniformly from that pool.
   - For T21, draw `adj` uniformly, then draw `adj2` uniformly from the same pool
     excluding `adj`.
   - Format the template.
   - If the exact text is already in the accepted set, retry from the template
     draw, up to 64 attempts total (including the first).
   - If still duplicated, use the deterministic fallback in §8.7.
   - Append `Snippet { text, label: positive ? 1 : 0, flipped: false }` and record
     the text as accepted.
4. Shuffle the corpus with Fisher-Yates using the same RNG:
   `for i in (1..n).rev() { let j = rng.below(i + 1); swap(i, j); }`.
5. Apply label noise as in §8.8. This happens after the shuffle and uses the same
   RNG; no base-corpus draws are affected by the noise value.

The model RNG is a separate stream: `Model::new` consumes `Rng::new(seed)` after
the dataset is built, and `State.rng` continues from there for training. Corpus
generation must not consume the training RNG.

### 8.7 Duplicate rejection and fallback

Duplicate means exact string equality against the accepted set. The retry bound
is 64. The fallback is deterministic and polarity-preserving:

- Precompute (lazily is fine) the ordered candidate list for each polarity:
  template index T1..T23, noun index in §8.4 order, adjective index in category
  order, and for T21 the second adjective index. Single-adjective candidates come
  in template/noun/adjective order; T21 candidates follow in
  template/noun/adj/adj2 order.
- Candidate counts are 3,164 positive and 2,630 negative.
- Maintain a cursor per polarity. On fallback for snippet `i`, start at
  `cursor = (i * 97) % candidates.len()`, scan forward cyclically for the first
  candidate not in the accepted set, accept it, and set the cursor to the next
  position.
- For `n <= 5000`, each class needs at most 2,500 unique sentences and each
  per-class candidate space is larger, so an unused candidate always exists. If
  an implementation ever exhausts the space (only possible outside the supported
  range), it must fail loudly (panic native / trap wasm) rather than silently
  emit a duplicate.

The fallback is not a second grammar; it selects from the same templates and
compatibility table.

### 8.8 Label noise and flips

- Clean label: positive adjective(s) -> `1`; negative adjective(s) -> `0`.
- Effective `noise` is clamped to `[0,1]`. UI range is `0.00`–`0.20`.
- `k = round(n * noise / 2.0)` per class. `n` is even, so both classes have
  `n/2` snippets. Clamp `k` to `n/2`.
- With `noise = 0.0`, `k = 0`; no labels are flipped and every `flipped` is
  `false`.
- With `n = 256`, `noise = 0.08`: `k = round(10.24) = 10` per class, **20 flips
  total**. With `noise = 0.20`: `k = round(25.6) = 26` per class, **52 flips**.
- Flip selection is uniform without replacement **within each class** over the
  already-shuffled corpus:
  1. Collect the indices of final-clean label `1` and label `0` in corpus order.
  2. Fisher-Yates shuffle each index list with the corpus RNG.
  3. Take the first `k` from each list; toggle the label (`0 -> 1` or `1 -> 0`)
     and set `flipped = true`.
- Because `k` is equal for both classes, `n_pos` and `n_neg` remain `n/2`.
- Selection happens after the corpus shuffle, so the v2 defect (all flips
  clustered in the first rows/class block) cannot recur. The audit must check
  that flips are not the first `k` indices of each class in generation order.
- `flipped` is independent of the train/test split; a flipped snippet keeps its
  split assignment.

### 8.9 Tokenization, vocabulary, and split

- Tokenization is unchanged from v2: lowercase, split whitespace, keep
  alphanumerics, drop empty tokens. Punctuation is removed; `I` becomes `i`.
- Vocabulary is the fixed 50-token lexicon in §8.2, sorted ascending. It is not
  rebuilt from the observed tokens. `W1` always has 50 columns, including columns
  for tokens absent from a small sample.
- Presence vector `x in {0,1}^V`: one for each vocabulary token present.
- Split: Fisher-Yates shuffle `0..n` with `Rng::new(seed ^ 0x5EED_5EED)`;
  `test_n = round(0.2 * n)`; first `test_n` indices are test, the rest train;
  `train_idx` is sorted ascending, `test_idx` keeps the shuffled order (v2
  behavior). For the default `n = 256`, `test_n = 51`, `train_n = 205`.
- Splits do not depend on labels or `flipped`.

### 8.10 Corpus audit checks

S1 must add automated checks and a `corpus_audit` example that prints every
snippet with flags for manual review. At minimum, assert:

1. `n` is even, in `[100,5000]`; default seed 42 / n 256 / noise 0.
2. Every sentence starts with an ASCII uppercase letter and ends with exactly one
   `.`; no double spaces.
3. No `a`, `an`, or `very` tokens; no `i the ...` sequence.
4. Every noun/adjective pair is legal in §8.4; T21's two adjectives are distinct
   and same-polarity.
5. Exact duplicate text count is zero for every supported `n` (the fallback
   guarantees it).
6. Clean-label polarity is 100% consistent: positive adjectives -> 1, negative ->
   0 before flips.
7. `flipped` count is exactly `2*k`; `k` per class; `n_pos = n_neg = n/2`;
   flips are not the first `k` of a class in generation order.
8. Noise 0 -> zero flips; 8% preset at n=256 -> exactly 20 flips (10/class);
   20% at n=256 -> 52 flips (26/class).
9. Same seed/n/noise -> byte-identical snippets, labels, and flags. Same
   seed/n but different noise -> identical text and clean polarity, differing
   only in labels/flags.
10. Vocabulary is exactly the 50-token sorted lexicon; every token is in the
    lexicon; `W1` is 8×50 and `params = 417`.
11. Token statistics match the final labels and count presence once per snippet.
12. The `corpus_audit` output contains no flagged rows for the default corpus.

---

## 9. S3 panel outline

The UI keeps the existing no-framework structure. All model math and every number
shown come from Rust through the ABI; `app.js` only formats.

### 9.1 Train panel

Add to the existing `#panel-train`:

| Control | ID | Type / range | Default | Behavior |
|---|---|---|---|---|
| Activation | `#activation` | `<select>`: `tanh`, `relu`, `leaky_relu` | `tanh` | calls `trot_configure(code, lambda)`, re-renders state/trace; does not reset weights |
| L2 checkbox | `#l2` | `<input type="checkbox">` | unchecked | checked enables `#lambda`; unchecked forces `lambda = 0.0` |
| Lambda | `#lambda` | `<input type="range" min="0" max="0.1" step="0.001">` | `0.000` | output `#lambda-out`; calls `trot_configure`; disabled when L2 unchecked |
| Lambda formula hint | `#lambda-formula` | text | — | `penalty = (λ/2)(‖W1‖² + ‖W2‖²)` |
| BCE | `#m-bce` | text | `—` | from `train.bce` after a train click |
| Penalty | `#m-penalty` | text | `—` | from `train.penalty` |
| Total | `#m-total` | text | `—` | from `train.total`; replaces the v2 `#m-loss` display |
| Noise | `#noise` | `<input type="range" min="0" max="20" step="1">` | `0` | percent; output `#noise-out` |
| Noise preset | `#noise-preset` | `<button>` | — | sets `#noise` to 8 and invokes the normal Regenerate flow (with confirmation) |
| Noise note | `#noise-note` | text | — | explains intentional flips and the 8% preset |

The loss line must read exactly `BCE + penalty = total` with the three values.
When L2 is off, `penalty` displays `0.000`. Keep the existing Train 1/10/100,
Reset, learning-rate, batch, seed, corpus-size, and Regenerate controls.

Behavior:

- Boot: `trot_init(42, 256, 0.0, 0, 0.0)` then `trot_train(300, 0.3, 16)`, so
  the demo is trained and clean by default.
- `Regenerate`: `trot_init(seed, n, noise, activationCode, lambda)` after the
  existing confirmation; updates state, dataset, trace, metrics; clears
  gradients.
- `Reset`: re-initializes over the active corpus with its seed/n/noise and the
  current activation/lambda; does not regenerate the corpus.
- Activation or lambda change: `trot_configure`; updates state, metrics, trace;
  clears the gradients panel and sets `#grad-state` to “train to refresh”.
- Noise changes only affect the next Regenerate.

### 9.2 Gradients panel

New `<section id="panel-gradients">`, heading `5 · Gradients`; Trace becomes
`6 · Trace a prediction`.

| Element | ID | Content |
|---|---|---|
| `dW1` grid | `#grad-w1` | `H × V` table, row headers `h0..h7`, column headers `vocab_tokens` |
| Small gradients | `#grad-small` | rows: `dW2` (1×8), `db1` (8), `db2` (1) |
| Learning rate | `#grad-lr` | `lr = 0.300` from `gradients.lr` |
| Step | `#grad-step` | `step = 310` from `gradients.step` |
| Update rule | `#grad-rule` | exact text `theta <- theta - lr * dtheta` |
| L2 note | `#grad-l2` | `λ = 0.000 (dW includes λW)` or `L2 off` |
| Detail | `#grad-detail` | hover/focus detail: `θ → dθ → θ − lr·dθ (next step)` |
| Empty state | `#grad-state` | “train to refresh” when `gradients.step === 0` |

`dW1` and `dW2` already include `λW`; the panel labels that fact rather than
recomputing. The hover detail is a convenience: the full gradient values and the
update rule must be readable without hovering. Do not put 400 table cells in the
tab order.

### 9.3 Data flip badge

Add a `flipped` column to `#snippet-table`:

- Header `flipped`.
- `true` -> a text badge `flipped` with `title` and `aria-label` “intentional
  label noise”; `false` -> `—`.
- Keep `label` as the final label. Do not recolor or hide flipped rows.
- `#data-stats` includes `X intentional flips`.
- The existing class/token filters continue to work on `label` and `text`.
- The badge must not rely on color alone.

### 9.4 Export anatomy table

Add `#export-anatomy` (a table) populated from `trot_export_json()` after
`trot_export_onnx()`:

| Row | Example |
|---|---|
| graph | `trot` |
| opset / IR | `13 / 8` |
| activation | `tanh` -> `Tanh` |
| nodes | `Gemm → Tanh → Gemm → Sigmoid` |
| initializers | `W1 [8,50]`, `b1 [8]`, `W2 [1,8]`, `b2 [1]` |
| input | `input float32[batch,50]` |
| output | `probability float32[batch,1]` |
| bytes | `1234` |

The table must use the activation node actually in the bytes; for Leaky ReLU
show `LeakyRelu (alpha = 0.01)`.

### 9.5 Trace

Keep the existing trace tables. Change the hidden row label to
`h = <activation>(z1)` using `trace.activation`, and add a row
`dact = <activation>'(z1)` using `trace.dact`. Keep the token -> `W1` column ->
`z1`, `W2`, `per_unit_z2`, `z2`, `p`, and verdict rows.

### 9.6 Keyboard and label expectations

- Use native `<select>`, `<input type="checkbox">`, `<input type="range">`, and
  `<button>`; no clickable `<div>` controls.
- Every control has a visible `<label for="...">` or an accessible name. Range
  outputs use `<output for="...">` and update while the range moves.
- Select: arrows/Home/End; checkbox: Space; range: arrows/Home/End/Page; buttons:
  Enter/Space.
- Visible focus indicator on every new control; preserve existing focus styles.
- `#noise-out` reads `0%`, `8%`, `20%`; `#lambda-out` reads `0.000`–`0.100`.
- The flip badge has text, not just color. Table headers use `<th scope>`.
- The hover detail is not the only way to read a gradient; the table values and
  update rule are always visible.
- When L2 is unchecked, `#lambda` is disabled and `#lambda-out` shows `0.000`.

---

## 10. File-level change map

S1 and S2 touch different core files but both touch `crates/trot-wasm/src/lib.rs`,
`crates/trot-core/tests/core.rs`, and (for wiring) `app.js`. Land S1 first or
split the WASM changes by function; rebase on `trot-improvements`; never
force-push. `trot.wasm` is rebuilt only in S5.

### S1 — corpus

| File | Change |
|---|---|
| `crates/trot-core/src/generator.rs` | fixed lexicon; category pools; noun compatibility table; 23 templates; deterministic generation; duplicate rejection + fallback; uniform per-class flips; `flipped`; default noise 0.0; preset 0.08 |
| `crates/trot-core/src/data.rs` | `Snippet.flipped: bool`; fixed 50-token vocab; token stats by final label; keep v2 split/tokenization |
| `crates/trot-wasm/src/lib.rs` | dataset JSON `flipped`; init default noise; effective n/noise; S1 slice only (ABI v3 completion is S2) |
| `crates/trot-core/tests/core.rs` | deterministic corpus, lexicon, compatibility, uniqueness, banned patterns, capitalization/punctuation, flip counts/spread, noise 0 vs 0.08 vs 0.20 |
| `crates/trot-core/examples/corpus_audit.rs` | new audit printer, one line per snippet with flags |
| `scripts/check.sh` | optionally run the corpus audit on the default corpus |
| `index.html`, `app.js`, `style.css` | Data flip badge and noise control slice (coordinate with S3) |

### S2 — core + ONNX

| File | Change |
|---|---|
| `crates/trot-core/src/model.rs` | `Activation` enum (`Tanh`, `ReLU`, `LeakyReLU(0.01)`); `Model.activation`; forward; `activate`; `activation_derivative`; `Forward.dact` if useful |
| `crates/trot-core/src/train.rs` | derivative-generalized backward; L2 penalty/gradient; last-minibatch gradient storage; return `bce`/`penalty`/`total` |
| `crates/trot-core/src/onnx.rs` | activation node selection; `attr_float` for LeakyReLU alpha; unchanged consts/names/shapes |
| `crates/trot-core/src/lib.rs` | export `Activation` if it lives outside `model` |
| `crates/trot-wasm/src/lib.rs` | ABI v3 constant/exports; `trot_configure`; new `state`/`dataset`/`train`/`gradients`/`trace`/`export` payloads; version handshake |
| `crates/trot-core/tests/core.rs` | hand-computed forward/derivative per activation; central-difference gradient checks per activation; L2 gradient check; penalty decomposition; ONNX smoke per activation |
| `crates/trot-core/examples/export_onnx.rs` | export each activation for validation |
| `scripts/validate_onnx.py` | activation matrix, node/alpha checks, round-trip for all three |
| `app.js` | switch to v3 calls/payloads (S3 completes the rendering) |

### S3 — UI

| File | Change |
|---|---|
| `index.html` | activation/L2/lambda/noise controls; loss breakdown; Gradients panel; flip column; export anatomy table; Trace derivative row |
| `app.js` | v3 handshake; `trot_init`/`trot_configure`/`trot_train` calls; gradient rendering; hover detail; export metadata; badge; activation-aware trace |
| `style.css` | gradient grid, badge, anatomy table, control states, focus styles; reuse existing tables/grids |

### S4 — docs

| File | Change |
|---|---|
| `docs/onnx_details.md` | new: protobuf encoder, field/tag map, operator mapping, raw_data layout, failure modes, validation |
| `README.md` | configurable corpus, gradients panel, activations, L2, noise semantics, new defaults |
| `concept.md` | gradients are real; activation is selectable; noise is explicit; no stale claims |

### S5 — CI/deploy

| File | Change |
|---|---|
| `.github/workflows/ci.yml` | activation × L2 test matrix; corpus audit; ONNX validation for all three activations; wasm build |
| `scripts/check.sh`, `Makefile` | same matrix locally; keep existing gates |
| `trot.wasm` | rebuild and commit after S2/S3 land |
| Pages/deploy | live smoke: 0% and 8% noise, L2 on/off, all activations, trace, export, round-trip |

---

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Fixed 50-token lexicon changes W1 columns and all baselines | S1/S2 land together on `trot-improvements`; tests assert `V=50`/`W1 8×50`; re-baseline only after the corpus gate. |
| S1 and S2 both edit `trot-wasm`/`app.js` | Land S1 first or split by function; rebase, never force-push; rebuild `trot.wasm` only in S5. |
| Random generation with rejection slows down near `n=5000` | Per-class candidate spaces (3,164/2,630) are small; bounded retry + cursor fallback; benchmark the 5,000 case in S1. |
| Duplicate fallback accidentally changes label polarity | Fallback scans only same-polarity candidates; assert clean polarity in the audit. |
| Noise 0% hides the generalization-gap lesson | Keep the 8% preset one click away; explain the control; badge flips; keep train/test metrics visible. |
| Flips cluster again | Select uniformly per class after the corpus shuffle; audit rejects the first-`k` pattern. |
| L2 reporting drifts from the gradient | Single `total = bce + penalty` definition; gradient check includes `lambda*W`; `train` reports pre-update decomposition; `gradients` includes `lambda*W`. |
| Activation switch leaves stale gradients | `trot_configure` clears stored gradients; UI shows “train to refresh”. |
| ONNX LeakyReLU alpha missing or wrong type | `attr_float` helper; `validate_onnx.py` asserts `alpha == 0.01` for LeakyRelu and no alpha for Tanh/Relu. |
| JSON payload grows (8×50 grids) | ~400 numbers per grid; acceptable for the demo; the UI reuses the existing table rendering. |
| New controls are mouse-only or unlabeled | Native controls, labels, focus rings, text badge, keyboard checks in S3/S5. |
| S0 contract drifts during implementation | This file is normative; any field/shape/behavior change requires a v4 agreement before implementation. |

---

## 12. Frozen decisions

1. ABI **3**; `trot_version() == 3`; every payload has `"abi": 3`.
2. `H = 8`, `V = 50`, `params = 417`; fixed sorted lexicon in §8.2.
3. State `w1` is `8×50`, `b1` is `8`, `w2` is `8`, `b2` is scalar.
   Gradients are `dW1 8×50`, `db1 8`, `dW2 1×8`, `db2 [1]`.
4. Activation identifiers are `tanh` (default), `relu`, `leaky_relu`; numeric
   codes `0/1/2`; Leaky ReLU alpha is exactly `0.01`, including slope `0.01` at
   `z <= 0`.
5. L2 is `penalty = (λ/2)(||W1||² + ||W2||²)`, weights only; gradients add
   `λW`; default off; ABI λ range `[0,0.1]`; ONNX unaffected.
6. ONNX is `Gemm -> Tanh|Relu|LeakyRelu -> Gemm -> Sigmoid`, opset 13, IR 8,
   initializers `W1,b1,W2,b2` with shapes `[8,50]`, `[8]`, `[1,8]`, `[1]`.
7. Corpus defaults: seed 42, n 256, noise **0%**; 8% preset; UI noise 0–20%;
   templates/lexicon/compatibility per §8; duplicate rejection with bounded
   retry and deterministic fallback; uniform per-class flips; `flipped` flag.
8. S3 panels: Train (activation, L2/λ, BCE/penalty/total, noise/preset),
   Gradients, Data flip badge, Export anatomy, activation-aware trace, accessible
   native controls.
9. S1/S2/S3 file ownership and sequencing per §10; no feature behavior changes
   in S0 beyond this document.

No open questions. If an implementation discovers a conflict, reopen this
contract rather than choosing locally.
