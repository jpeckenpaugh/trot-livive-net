#!/usr/bin/env node
// Smoke-test the committed wasm artifact without a browser.
//
// GitHub Pages serves `trot.wasm`; this script loads exactly that file, checks
// the ABI exports, and exercises every activation with L2 off and on through
// the same JSON entry points app.js uses.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const wasmPath = process.argv[2]
  ? resolve(process.argv[2])
  : join(root, "trot.wasm");
const bytes = readFileSync(wasmPath);

assert.ok(
  bytes.length > 100_000 && bytes.length < 1_000_000,
  `trot.wasm size ${bytes.length} bytes is outside the plausible range`,
);

const { instance } = await WebAssembly.instantiate(bytes, {});
const ex = instance.exports;
const dec = new TextDecoder();
const enc = new TextEncoder();

const REQUIRED_EXPORTS = [
  "memory",
  "trot_version",
  "trot_init",
  "trot_configure",
  "trot_state_json",
  "trot_dataset_json",
  "trot_train",
  "trot_gradients_json",
  "trot_trace_json",
  "trot_export_onnx",
  "trot_export_json",
  "trot_roundtrip_json",
  "trot_alloc",
  "trot_dealloc",
  "trot_out_ptr",
  "trot_out_len",
];
for (const name of REQUIRED_EXPORTS) {
  assert.ok(ex[name] !== undefined, `wasm is missing export ${name}`);
}

function outText() {
  const len = ex.trot_out_len();
  const ptr = ex.trot_out_ptr();
  return dec.decode(new Uint8Array(ex.memory.buffer, ptr, len));
}

function outBytes(len) {
  const ptr = ex.trot_out_ptr();
  return new Uint8Array(ex.memory.buffer, ptr, len).slice();
}

function read(fn) {
  fn();
  return JSON.parse(outText());
}

const state = () => read(() => ex.trot_state_json());
const dataset = () => read(() => ex.trot_dataset_json());
const gradients = () => read(() => ex.trot_gradients_json());
const exportMeta = () => read(() => ex.trot_export_json());
const roundtrip = () => read(() => ex.trot_roundtrip_json());

assert.equal(ex.trot_version(), 3, "ABI version");

const ACTIVATIONS = [
  ["tanh", 0, "Tanh"],
  ["relu", 1, "Relu"],
  ["leaky_relu", 2, "LeakyRelu"],
];

function finite(values, label) {
  for (const value of values) {
    assert.ok(Number.isFinite(value), `${label} contains ${value}`);
  }
}

for (const [name, code, node] of ACTIVATIONS) {
  for (const lambda of [0, 0.05]) {
    const label = `${name} L2=${lambda}`;
    ex.trot_init(42, 256, 0.0, code, lambda);
    const initial = read(() => {});
    assert.equal(initial.abi, 3, `${label}: ABI`);
    assert.equal(initial.activation, name, `${label}: activation`);
    assert.equal(initial.lambda, lambda, `${label}: lambda`);
    assert.equal(initial.vocab, 50, `${label}: vocab`);
    assert.equal(initial.n, 256, `${label}: corpus size`);
    assert.equal(initial.train_n + initial.test_n, 256, `${label}: split`);

    ex.trot_train(25, 0.3, 16);
    const train = read(() => {});
    assert.equal(train.step, 25, `${label}: step`);
    assert.equal(train.batch, 16, `${label}: batch`);
    finite([train.bce, train.penalty, train.total], `${label}: train report`);
    if (lambda === 0) assert.equal(train.penalty, 0, `${label}: L2 off penalty`);
    else assert.ok(train.penalty > 0, `${label}: L2 on penalty`);

    const after = state();
    assert.equal(after.step, 25, `${label}: state step`);
    assert.equal(after.activation, name, `${label}: state activation`);
    assert.equal(after.lambda, lambda, `${label}: state lambda`);
    assert.equal(after.w1.length, 8, `${label}: W1 rows`);
    assert.equal(after.w1[0].length, 50, `${label}: W1 columns`);
    finite(after.w1.flat(), `${label}: W1`);
    finite(after.b1, `${label}: b1`);
    finite(after.w2, `${label}: w2`);
    finite([after.b2, after.train_acc, after.test_acc], `${label}: metrics`);

    const g = gradients();
    assert.equal(g.dW1.length, 8, `${label}: dW1 rows`);
    assert.equal(g.dW1[0].length, 50, `${label}: dW1 columns`);
    assert.equal(g.lambda, lambda, `${label}: gradient lambda`);
    finite(g.dW1.flat(), `${label}: dW1`);
    finite(g.db1, `${label}: db1`);
    finite(g.dW2[0], `${label}: dW2`);

    const d = dataset();
    assert.equal(d.snippets.length, 256, `${label}: snippets`);
    assert.equal(d.token_stats.length, 50, `${label}: token stats`);
    assert.equal(d.snippets.filter((s) => s.flipped).length, 0, `${label}: flips`);

    const text = enc.encode("the service was great");
    const ptr = ex.trot_alloc(text.length);
    new Uint8Array(ex.memory.buffer, ptr, text.length).set(text);
    ex.trot_trace_json(ptr, text.length);
    ex.trot_dealloc(ptr, text.length);
    const trace = JSON.parse(outText());
    assert.equal(trace.activation, name, `${label}: trace activation`);
    assert.equal(trace.tokens.length, 4, `${label}: trace tokens`);
    assert.equal(trace.z1.length, 8, `${label}: trace z1`);
    assert.equal(trace.dact.length, 8, `${label}: trace derivative`);
    finite([...trace.z1, ...trace.h, ...trace.dact, trace.p], `${label}: trace`);

    const n = ex.trot_export_onnx();
    const model = outBytes(n);
    assert.equal(model.length, n, `${label}: export bytes`);
    assert.ok(n > 1_000, `${label}: export size ${n}`);
    const meta = exportMeta();
    assert.equal(meta.activation, name, `${label}: export activation`);
    assert.equal(meta.activation_node, node, `${label}: export node`);
    assert.equal(meta.opset, 13, `${label}: opset`);
    assert.equal(meta.ir_version, 8, `${label}: IR version`);
    assert.deepEqual(meta.nodes, ["Gemm", node, "Gemm", "Sigmoid"], `${label}: graph`);
    assert.equal(meta.bytes, n, `${label}: export metadata bytes`);
    if (name === "leaky_relu") {
      assert.equal(meta.activation_attrs.alpha, 0.01, `${label}: LeakyRelu alpha`);
    } else {
      assert.deepEqual(meta.activation_attrs, {}, `${label}: activation attrs`);
    }

    const rt = roundtrip();
    assert.equal(rt.activation, name, `${label}: round-trip activation`);
    assert.equal(rt.inputs.length, rt.n, `${label}: round-trip rows`);
    assert.equal(rt.inputs[0].length, 50, `${label}: round-trip vocab`);
    assert.equal(rt.probs.length, rt.n, `${label}: round-trip probabilities`);
    finite(rt.probs, `${label}: round-trip probabilities`);

    const nextCode = (code + 1) % ACTIVATIONS.length;
    ex.trot_configure(nextCode, 0);
    assert.equal(state().activation, ACTIVATIONS[nextCode][0], `${label}: configure`);
  }
}

ex.trot_init(42, 256, 0.08, 0, 0);
const noisy = dataset();
assert.equal(noisy.snippets.length, 256, "noise corpus size");
assert.equal(noisy.snippets.filter((s) => s.flipped).length, 20, "8% flips at n=256");
assert.equal(noisy.snippets.filter((s) => s.label === 1).length, 128, "noise balance");

console.log(
  `OK  trot.wasm ${bytes.length} bytes: ABI 3, 3 activations x 2 L2 states, ` +
    "export metadata, trace, gradients, round-trip, and 8% noise verified",
);
