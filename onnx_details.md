# ONNX export details

TROT exports its trained model as a real ONNX `ModelProto`. The encoder is
`crates/trot-core/src/onnx.rs` (about 220 lines including comments and helpers).
This document explains why it is hand-written, exactly which protobuf fields it
emits, how the tensors are laid out, what can go wrong, and how the result is
validated.

The exported contract is fixed in [`docs/abi-v3.md`](docs/abi-v3.md): opset 13,
IR version 8, graph name `trot`, four nodes, and initializers named `W1`, `b1`,
`W2`, `b2`.

## Why the encoder is hand-written

The exported graph is tiny and completely fixed: one `Gemm`, one hidden
activation, another `Gemm`, and one `Sigmoid`. The model has no optional
subgraphs, no custom operators, and no dynamic control flow. Encoding that
subset needs only the small set of protobuf messages listed below and a handful
of wire primitives.

Writing those primitives directly buys three things:

- **No dependency.** `trot-core` has an empty `[dependencies]` table. There is
  no `prost`, no `protobuf`, and no generated-code build step, so the crate
  still builds for native tests and `wasm32-unknown-unknown` with no codegen
  tooling.
- **Auditability.** The encoder is small enough to read beside the model. A
  reader can follow the same graph in Rust, in the exported bytes, and in
  Netron.
- **A stable, minimal surface.** The exporter emits only the fields TROT needs.
  It cannot silently grow a dependency or emit fields nobody reviewed.

The tradeoff is deliberate: this is not a general protobuf or ONNX writer. It
does not read files, does not support arbitrary operators or attributes, and
does not encode sparse tensors, external data, training graphs, or multiple
graphs. Adding a new operator means extending this file and its tests.

## Message tree

`onnx::export` builds the following tree. Field numbers are the ONNX/protobuf
field numbers used by the encoder.

```text
ModelProto
  ir_version        = 8                         (field 1, varint)
  producer_name     = "trot"                    (field 2, string)
  producer_version  = CARGO_PKG_VERSION         (field 3, string; 0.1.0 today)
  graph             = GraphProto                (field 7, message)
  opset_import      = OperatorSetIdProto        (field 8, message)
    version         = 13                        (field 2, varint)

GraphProto
  node              = NodeProto[4]              (field 1, repeated message)
  name              = "trot"                    (field 2, string)
  initializer       = TensorProto[4]            (field 5, repeated message)
  input             = ValueInfoProto            (field 11, message)
  output            = ValueInfoProto            (field 12, message)

NodeProto
  input             = string[]                  (field 1, repeated string)
  output            = string[]                  (field 2, repeated string)
  op_type           = string                    (field 4, string)
  attribute         = AttributeProto[]          (field 5, repeated message)

TensorProto
  dims              = int64[]                   (field 1, repeated varint)
  data_type         = 1 (FLOAT)                 (field 2, varint)
  name              = string                    (field 8, string)
  raw_data          = little-endian f32 bytes   (field 9, bytes)

ValueInfoProto
  name              = string                    (field 1, string)
  type              = TypeProto                 (field 2, message)

TypeProto
  tensor_type       = TensorTypeProto           (field 1, message)

TensorTypeProto
  elem_type         = 1 (FLOAT)                 (field 1, varint)
  shape             = TensorShapeProto          (field 2, message)

TensorShapeProto
  dim               = Dimension[]               (field 1, repeated message)

Dimension
  dim_value         = int64                     (field 1, varint)
  dim_param         = string                    (field 2, string)
```

The operator-set `domain` is omitted, which means the default ONNX domain `""`.
The only emitted opset is `("", 13)`.

The four nodes are always emitted in this order:

```text
Gemm(input, W1, b1) -> z1
f(z1)               -> h        f is Tanh, Relu, or LeakyRelu
Gemm(h, W2, b2)     -> z2
Sigmoid(z2)         -> probability
```

`Gemm` carries `transB = 1` in both layers. The hidden node depends on the
selected activation:

| Rust `Activation` | ONNX `op_type` | Attributes |
| --- | --- | --- |
| `Tanh` | `Tanh` | none |
| `ReLU` | `Relu` | none |
| `LeakyReLU` | `LeakyRelu` | `alpha` (FLOAT) = `0.01` |

`LEAKY_ALPHA` in `model.rs` is the single source of truth for that `0.01`, and
the same value is shown in the browser's export anatomy.

## Wire encoding

Protobuf encodes each field as a tag followed by a value. The tag is a varint
whose low three bits select the wire type:

| Wire type | Meaning | Used for |
| --- | --- | --- |
| `0` | varint | integers and enums |
| `2` | length-delimited | strings, bytes, embedded messages |
| `5` | 32-bit fixed | `float` attribute values |

The encoder's helpers map directly onto those cases:

```text
tag(field, wire)          = varint((field << 3) | wire)
field_varint(field, n)    = tag(field, 0) + varint(n)
field_bytes(field, b)     = tag(field, 2) + varint(len(b)) + b
field_str(field, s)       = field_bytes(field, s.as_bytes())
field_msg(field, body)    = field_bytes(field, body)
field_float(field, value) = tag(field, 5) + value.to_le_bytes()
```

`put_varint` writes the unsigned integer seven bits at a time, least-significant
group first, setting the continuation bit (`0x80`) on every byte except the
last. All values encoded here (field numbers, lengths, dims, enum values, and
the non-negative attribute integers) are non-negative.

Strings are UTF-8. Repeated scalar fields (`dims`) and repeated message fields
(`node`, `initializer`, `input`, `output`, `attribute`) are emitted one tagged
entry per element; the encoder does not use packed repeated encoding. Protobuf
field order is not semantically significant, but the encoder's output is
deterministic because the emission order is fixed.

### Attributes

`AttributeProto` has two encoded value fields in this exporter:

- `name` (field 1, string)
- `f` (field 2, fixed32 float) or `i` (field 3, varint)
- `type` (field 20, varint): `FLOAT = 1`, `INT = 2`

`Gemm.transB` is an `INT` attribute with value `1`. `LeakyRelu.alpha` is a
`FLOAT` attribute with value `0.01`. `Tanh` and `Relu` carry no attributes.

### Shapes and types

The input is declared in `value_info` as FLOAT `[batch, V]`, where `batch` is a
symbolic dimension (`dim_param = "batch"`) and `V = 50` is fixed. The output is
FLOAT `[batch, 1]`. Initializer dimensions are concrete:

| Initializer | ONNX dims | Meaning |
| --- | --- | --- |
| `W1` | `[8, 50]` | hidden unit rows, vocabulary columns |
| `b1` | `[8]` | hidden bias |
| `W2` | `[1, 8]` | output row, hidden columns |
| `b2` | `[1]` | output bias |

The initializers use `data_type = 1` (FLOAT) and `raw_data`, not `float_data`.

## `raw_data` layout

`raw_data` is one opaque byte string containing the tensor's elements in
row-major order, each as a 4-byte little-endian IEEE-754 `f32`. There is no
padding, header, or per-element tag.

- `W1` is already stored in Rust as `hidden * vocab`, row-major, with row `j`
  holding hidden unit `j` across vocabulary columns `0..V`. It is written
  verbatim, so `W1[j][i] = W1_rust[j * V + i]`.
- `b1` is written in hidden-unit order `h0..h7`.
- `W2` is written in hidden-unit order, while its ONNX dims are `[1, 8]`.
- `b2` is a single `f32`.

Because `raw_data` has no internal type or length markers, the byte length must
equal `product(dims) * 4`. The current encoder derives the bytes from the model
vectors, so for a valid `Model` the length follows automatically; a mismatch
would be rejected by the ONNX checker or runtime.

`f32::to_le_bytes` makes the host endianness irrelevant. ONNX defines
`raw_data` as little-endian for this element type.

## What the graph computes

For a batch of presence vectors `X` with shape `[batch, V]`:

```text
Z1 = X * W1^T + b1             # Gemm, transB = 1: [batch,50] x [50,8] -> [batch,8]
H  = f(Z1)                     # elementwise hidden activation
Z2 = H * W2^T + b2             # Gemm, transB = 1: [batch,8] x [8,1]  -> [batch,1]
P  = sigmoid(Z2)               # probability
```

This matches the Rust forward pass in `model.rs`: `z1 = W1 x + b1`,
`h = f(z1)`, `z2 = W2 h + b2`, `p = sigmoid(z2)`. The ONNX model contains no
loss, gradients, optimizer state, or corpus information; it is inference only.

## Failure modes

The encoder does not validate its own output. These are the ways an export can
become invalid or misleading:

- **Malformed protobuf.** A wrong tag/wire combination, an incorrect
  length-delimited size, a truncated varint, or a field written with the wrong
  wire type. The Rust encoder is the only writer, so this is a regression risk
  when the encoder changes; the unit test parses the bytes back.
- **`raw_data` shape mismatch.** Bytes not equal to `product(dims) * 4`, or a
  model whose vector lengths do not match `hidden * vocab`, `hidden`, and so
  on. ONNX load/checking or the runtime will reject the tensor.
- **Missing or wrong `transB`.** Without `transB = 1`, `Gemm` would interpret
  `W1` as `[8,50]` multiplied directly against `[batch,50]`, producing a shape
  error (or, for square matrices, silently wrong math).
- **Wrong activation node or attribute.** `LeakyRelu` without its `alpha`
  implements a different slope; `Tanh`/`Relu` with unexpected attributes are
  rejected by the validator.
- **Wrong opset or IR version.** The file declares opset 13 and IR 8. A runtime
  that only understands an earlier opset may reject it; a mismatched contract
  makes the graph non-reproducible even if it happens to load.
- **Wrong element type or shape metadata.** `data_type` and `elem_type` must be
  FLOAT, and the declared input/output shapes must match what the nodes
  produce. The `batch` dimension is symbolic; the vocabulary dimension and the
  output width are fixed at 50 and 1.
- **Missing initializer or node names.** The graph is wired by name (`input`,
  `W1`, `b1`, `z1`, `h`, `W2`, `b2`, `z2`, `probability`). Renaming without
  updating the consumers breaks the graph.
- **Endianness or finite-value assumptions.** The encoder writes little-endian
  `f32` values. It does not add NaN/Inf checks; finite training weights are
  expected. Round-trip comparisons use a tolerance because JS numbers are
  doubles and ONNX runtimes may reassociate floating-point operations.
- **Scope creep.** This encoder cannot represent arbitrary ONNX models,
  operators, attributes, external tensors, or multiple graphs. It is an
  exporter for exactly this model.

## How validation is performed

Validation happens at three levels.

1. **Rust unit test.** `crates/trot-core/tests/core.rs` contains a minimal
   protobuf reader and `onnx_export_maps_each_activation`. For all three
   activations it parses the exported bytes and asserts the node order
   (`Gemm`, activation, `Gemm`, `Sigmoid`), initializer names, and the
   `LeakyRelu` alpha type/value. This catches encoder regressions without a
   Python dependency.

2. **ONNX checker and a standard runtime.** `scripts/validate_onnx.py` loads
   each `target/trot_<activation>.onnx` with `onnx`, calls
   `onnx.checker.check_model`, verifies opset 13 / IR 8 / graph name, checks
   initializer names and shapes, checks every node's inputs, outputs, and
   attributes, and checks the declared input/output names, types, and shapes.
   It then runs 25 test-set presence vectors through `onnxruntime` and compares
   the probabilities with the Rust reference values, requiring a maximum
   absolute difference of at most `1e-6`. The reference files are produced by
   `cargo run --release -p trot-core --example export_onnx`, which trains each
   activation for 600 steps on the default corpus before exporting.

3. **Browser round-trip.** The page's *Round-trip check* exports the current
   in-browser model, asks the Rust core for every test-split input vector and
   probability, loads the bytes into `onnxruntime-web` (version 1.19.2, loaded
   from a CDN on demand), and requires every test-set prediction to match
   within `1e-6`. The result is shown as PASS or FAIL next to the Export panel.
   This path requires network access for the CDN; the checker script above is
   the offline verification path.

CI runs the Rust tests, builds the Wasm, and then runs the export example and
`scripts/validate_onnx.py`, so all three activation variants are checked on
every pull request and on pushes to `main`.
