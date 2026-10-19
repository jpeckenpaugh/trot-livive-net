// TROT — the UI formats numbers. It never computes model math; every value on
// the page comes from the Rust core through the WebAssembly boundary.

const $ = (id) => document.getElementById(id);
const dec = new TextDecoder();
const enc = new TextEncoder();

let ex = null;
let state = null;
let prevW1 = null;
let datasetInfo = null;
let maxAbs = 1;

async function boot() {
  const res = await fetch("trot.wasm");
  const buf = await res.arrayBuffer();
  const { instance } = await WebAssembly.instantiate(buf, {});
  ex = instance.exports;
  $("build").textContent = `· wasm build v${ex.trot_version()}`;

  ex.trot_init(42);
  datasetInfo = JSON.parse(readOut());
  // Start from a trained model so the demo is immediately useful; Reset goes
  // back to random weights.
  ex.trot_train(300, 0.3, 16);
  state = readState();
  await refresh(true);

  $("dataset-line").textContent =
    `${datasetInfo.n} generated sentences · ${datasetInfo.n_pos} positive / ${datasetInfo.n_neg} negative` +
    ` · |V| = ${datasetInfo.vocab} · ${datasetInfo.train_n} train / ${datasetInfo.test_n} test` +
    ` · ~8% label noise (deliberate, so train and test can diverge)` +
    ` · trained 300 steps on load · Reset starts from random weights`;

  bind();
}

function readOut() {
  const len = ex.trot_out_len();
  if (len === 0) return "";
  const ptr = ex.trot_out_ptr();
  return dec.decode(new Uint8Array(ex.memory.buffer, ptr, len));
}

function readOutBytes() {
  const len = ex.trot_out_len();
  const ptr = ex.trot_out_ptr();
  return new Uint8Array(ex.memory.buffer, ptr, len).slice();
}

function readState() {
  ex.trot_state_json();
  return JSON.parse(readOut());
}

function withInput(str, fn) {
  const bytes = enc.encode(str);
  const ptr = ex.trot_alloc(bytes.length);
  new Uint8Array(ex.memory.buffer, ptr, bytes.length).set(bytes);
  try {
    fn(ptr, bytes.length);
  } finally {
    ex.trot_dealloc(ptr, bytes.length);
  }
}

const f3 = (x) => Number(x).toFixed(3);
const pct = (x) => `${(Number(x) * 100).toFixed(1)}%`;

function cell(text, cls) {
  const td = document.createElement("td");
  td.textContent = text;
  td.title = text;
  if (cls) td.className = cls;
  return td;
}

function weightStyle(v) {
  const t = Math.min(1, Math.abs(v) / maxAbs);
  const color = v >= 0 ? "248,113,113" : "96,165,250";
  return `rgba(${color},${(0.08 + 0.5 * t).toFixed(3)})`;
}

function renderWeights(s) {
  maxAbs = Math.max(1e-6, ...s.w1.flat().map((v) => Math.abs(v)));
  const table = $("w1");
  table.textContent = "";
  $("w1-shape").textContent = `${s.hidden} × ${s.vocab}`;

  const head = document.createElement("tr");
  head.appendChild(document.createElement("th"));
  for (const tok of s.vocab_tokens) {
    const th = document.createElement("th");
    th.className = "token";
    th.textContent = tok;
    head.appendChild(th);
  }
  table.appendChild(head);

  const flat = s.w1.flat();
  s.w1.forEach((row, j) => {
    const tr = document.createElement("tr");
    const rh = document.createElement("th");
    rh.className = "rowhead";
    rh.textContent = `h${j}`;
    tr.appendChild(rh);
    row.forEach((v, i) => {
      const td = cell(f3(v));
      td.style.background = weightStyle(v);
      const k = j * s.vocab + i;
      if (prevW1 && prevW1[k] !== v) {
        td.classList.add("flash");
        setTimeout(() => td.classList.remove("flash"), 650);
      }
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });
  prevW1 = flat;

  const small = $("b1w2");
  small.textContent = "";
  const header = document.createElement("tr");
  header.appendChild(document.createElement("th"));
  for (let j = 0; j < s.hidden; j++) {
    const th = document.createElement("th");
    th.textContent = `h${j}`;
    header.appendChild(th);
  }
  small.appendChild(header);
  for (const [label, row] of [["b1", s.b1], ["W2", s.w2]]) {
    const tr = document.createElement("tr");
    const rh = document.createElement("th");
    rh.className = "rowhead";
    rh.textContent = label;
    tr.appendChild(rh);
    for (const v of row) tr.appendChild(cell(f3(v)));
    small.appendChild(tr);
  }
}

function renderMetrics(s) {
  $("m-step").textContent = s.step;
  $("m-loss").textContent = f3(s.loss);
  $("m-train").textContent = pct(s.train_acc);
  $("m-test").textContent = pct(s.test_acc);
  $("params").textContent = `· ${s.params} params`;
  drawLoss(s.history);
}

function drawLoss(history) {
  const c = $("loss");
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth;
  const h = c.height;
  c.width = w * dpr;
  c.height = h * dpr;
  const g = c.getContext("2d");
  g.scale(dpr, dpr);
  g.clearRect(0, 0, w, h);
  if (!history || history.length < 2) return;
  const min = Math.min(...history);
  const max = Math.max(...history);
  const span = Math.max(1e-6, max - min);
  g.strokeStyle = "#5eead4";
  g.lineWidth = 1.5;
  g.beginPath();
  history.forEach((v, i) => {
    const x = (i / (history.length - 1)) * (w - 8) + 4;
    const y = h - 6 - ((v - min) / span) * (h - 14);
    if (i === 0) g.moveTo(x, y);
    else g.lineTo(x, y);
  });
  g.stroke();
}

function render(s) {
  renderMetrics(s);
  renderWeights(s);
}

async function refresh(initial) {
  state = readState();
  render(state);
  if (initial) trace();
}

function train(steps) {
  const lr = parseFloat($("lr").value);
  const batch = parseInt($("batch").value, 10) || 16;
  ex.trot_train(steps, lr, batch);
  refresh(false);
}

function renderTrace(t) {
  const out = $("trace-out");
  out.classList.remove("muted");
  out.textContent = "";
  const hidden = state.hidden;

  const tokens = document.createElement("h3");
  tokens.textContent = "tokens";
  out.appendChild(tokens);
  const tokLine = document.createElement("div");
  t.tokens.forEach((tok, i) => {
    const span = document.createElement("span");
    span.textContent = tok + " ";
    span.title = t.in_vocab[i] ? "in vocabulary" : "out of vocabulary (ignored)";
    if (!t.in_vocab[i]) span.className = "oov";
    tokLine.appendChild(span);
  });
  out.appendChild(tokLine);

  const h1 = document.createElement("h3");
  h1.textContent = "token → W1 column → z1";
  out.appendChild(h1);
  const t1 = document.createElement("table");
  const r1 = document.createElement("tr");
  r1.appendChild(document.createElement("th"));
  for (let j = 0; j < hidden; j++) {
    const th = document.createElement("th");
    th.textContent = `h${j}`;
    r1.appendChild(th);
  }
  t1.appendChild(r1);
  for (const pt of t.per_token) {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.textContent = pt.token;
    tr.appendChild(th);
    for (const v of pt.col) tr.appendChild(cell(f3(v)));
    t1.appendChild(tr);
  }
  const bRow = document.createElement("tr");
  bRow.appendChild(Object.assign(document.createElement("th"), { textContent: "b1" }));
  for (const v of t.b1) bRow.appendChild(cell(f3(v)));
  t1.appendChild(bRow);
  const zRow = document.createElement("tr");
  zRow.appendChild(Object.assign(document.createElement("th"), { textContent: "z1" }));
  for (const v of t.z1) zRow.appendChild(cell(f3(v)));
  t1.appendChild(zRow);
  out.appendChild(t1);

  const h2 = document.createElement("h3");
  h2.textContent = "hidden → output";
  out.appendChild(h2);
  const t2 = document.createElement("table");
  const r2 = document.createElement("tr");
  r2.appendChild(document.createElement("th"));
  for (let j = 0; j < hidden; j++) {
    const th = document.createElement("th");
    th.textContent = `h${j}`;
    r2.appendChild(th);
  }
  t2.appendChild(r2);
  for (const [label, row] of [
    ["h = tanh(z1)", t.h],
    ["W2", t.w2],
    ["h · W2", t.per_unit_z2],
  ]) {
    const tr = document.createElement("tr");
    tr.appendChild(Object.assign(document.createElement("th"), { textContent: label }));
    for (const v of row) tr.appendChild(cell(f3(v)));
    t2.appendChild(tr);
  }
  out.appendChild(t2);

  const verdict = document.createElement("p");
  verdict.className = "verdict";
  const word = t.label === 1 ? "positive" : "negative";
  verdict.textContent = `z2 = Σ(h·W2) + b2 = ${f3(t.z2)}   p = σ(z2) = ${f3(t.p)}   →  ${word}`;
  out.appendChild(verdict);
}

function trace() {
  const text = $("trace-text").value;
  withInput(text, (ptr, len) => {
    ex.trot_trace_json(ptr, len);
    renderTrace(JSON.parse(readOut()));
  });
}

function exportOnnx() {
  const len = ex.trot_export_onnx();
  const bytes = readOutBytes();
  const blob = new Blob([bytes], { type: "application/octet-stream" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "trot.onnx";
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
  $("export-summary").textContent =
    `trot.onnx · ${bytes.length} bytes\n` +
    `opset 13 · IR 8\n` +
    `nodes      [Gemm, Tanh, Gemm, Sigmoid]\n` +
    `init       [W1 (${state.hidden}×${state.vocab}), b1 (${state.hidden}), W2 (1×${state.hidden}), b2 (1)]\n` +
    `input      input float32[batch, ${state.vocab}]\n` +
    `output     probability float32[batch, 1]\n` +
    `weights    ${state.params} params, trained for ${state.step} steps`;
}

function bind() {
  $("btn-1").onclick = () => train(1);
  $("btn-10").onclick = () => train(10);
  $("btn-100").onclick = () => train(100);
  $("btn-reset").onclick = () => {
    ex.trot_init(42);
    prevW1 = null;
    refresh(false);
  };
  $("btn-trace").onclick = trace;
  $("trace-text").onkeydown = (e) => {
    if (e.key === "Enter") trace();
  };
  $("btn-export").onclick = exportOnnx;
  $("lr").oninput = (e) => {
    $("lr-out").textContent = parseFloat(e.target.value).toFixed(2);
  };
}

boot().catch((err) => {
  $("dataset-line").textContent = `failed to load: ${err}`;
  console.error(err);
});
