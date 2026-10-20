// TROT — the UI formats numbers. It never computes model math; every value on
// the page comes from the Rust core through the WebAssembly boundary.

const $ = (id) => document.getElementById(id);
const dec = new TextDecoder();
const enc = new TextEncoder();

const ABI_VERSION = 3;
const ACTIVATION_CODES = { tanh: 0, relu: 1, leaky_relu: 2 };

let ex = null;
let state = null;
let datasetInfo = null;
let lastTrain = null;
let lastGradients = null;
let lastExported = null;
let prevW1 = null;
let maxAbs = 1;
let filterClass = "all";
let filterToken = "";
let ortPromise = null;

const ORT_VERSION = "1.19.2";
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

async function boot() {
  const res = await fetch("trot.wasm");
  const buf = await res.arrayBuffer();
  const { instance } = await WebAssembly.instantiate(buf, {});
  ex = instance.exports;

  const abi = ex.trot_version();
  if (abi !== ABI_VERSION) {
    throw new Error(`unsupported wasm ABI ${abi}; expected ${ABI_VERSION}`);
  }
  $("build").textContent = `· wasm build v${abi} (ABI ${ABI_VERSION})`;

  ex.trot_init(42, 256, 0.0, ACTIVATION_CODES.tanh, 0.0);
  // Start from a trained model so the demo is immediately useful; Reset goes
  // back to random weights over the active corpus.
  ex.trot_train(300, 0.3, 16);
  lastTrain = JSON.parse(readOut());
  datasetInfo = readDataset();
  state = readState();
  lastGradients = readGradients();

  bind();
  syncControls();
  updateDatasetLine();
  render(state);
  renderLoss();
  renderDataset();
  renderGradients(lastGradients);
  trace();
  renderExportAnatomy(readExportMeta());
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

function readDataset() {
  ex.trot_dataset_json();
  return JSON.parse(readOut());
}

function readGradients() {
  ex.trot_gradients_json();
  return JSON.parse(readOut());
}

function readExportMeta() {
  ex.trot_export_json();
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

function th(text, cls, scope = "col") {
  const el = document.createElement("th");
  el.textContent = text;
  el.scope = scope;
  if (cls) el.className = cls;
  return el;
}

function heatStyle(v, span) {
  const t = Math.min(1, Math.abs(v) / Math.max(1e-9, span));
  const color = v >= 0 ? "248,113,113" : "96,165,250";
  return `rgba(${color},${(0.08 + 0.5 * t).toFixed(3)})`;
}

function weightStyle(v) {
  return heatStyle(v, maxAbs);
}

// ---- train / weights -----------------------------------------------------

function renderWeights(s) {
  maxAbs = Math.max(1e-6, ...s.w1.flat().map((v) => Math.abs(v)));
  const table = $("w1");
  table.textContent = "";
  $("w1-shape").textContent = `${s.hidden} × ${s.vocab}`;

  const head = document.createElement("tr");
  head.appendChild(th(""));
  for (const tok of s.vocab_tokens) {
    head.appendChild(th(tok, "token"));
  }
  table.appendChild(head);

  const flat = s.w1.flat();
  s.w1.forEach((row, j) => {
    const tr = document.createElement("tr");
    tr.appendChild(th(`h${j}`, "rowhead", "row"));
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
  header.appendChild(th(""));
  for (let j = 0; j < s.hidden; j++) header.appendChild(th(`h${j}`));
  small.appendChild(header);
  for (const [label, row] of [["b1", s.b1], ["W2", s.w2]]) {
    const tr = document.createElement("tr");
    tr.appendChild(th(label, "rowhead", "row"));
    for (const v of row) tr.appendChild(cell(f3(v)));
    small.appendChild(tr);
  }
}

function renderMetrics(s) {
  $("m-step").textContent = s.step;
  $("m-train").textContent = pct(s.train_acc);
  $("m-test").textContent = pct(s.test_acc);
  $("params").textContent = `· ${s.params} params`;
  drawLoss(s.history);
}

function renderLoss() {
  const t = lastTrain;
  if (!t || !t.step) {
    $("m-bce").textContent = "—";
    $("m-penalty").textContent = "—";
    $("m-total").textContent = "—";
    return;
  }
  $("m-bce").textContent = f3(t.bce);
  $("m-penalty").textContent = f3(t.penalty);
  $("m-total").textContent = f3(t.total);
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

function syncControls() {
  $("activation").value = state.activation;
  const l2on = Number(state.lambda) > 0;
  $("l2").checked = l2on;
  $("lambda").disabled = !l2on;
  $("lambda").value = l2on ? String(state.lambda) : "0";
  $("lambda-out").textContent = f3(l2on ? state.lambda : 0);

  const noisePct = Math.round(Number(datasetInfo.noise) * 100);
  $("noise").value = String(noisePct);
  $("noise-out").textContent = `${noisePct}%`;
}

function updateDatasetLine() {
  const d = datasetInfo;
  $("dataset-line").textContent =
    `${d.n} generated sentences · ${d.n_pos} positive / ${d.n_neg} negative` +
    ` · |V| = ${d.vocab} · ${d.train_n} train / ${d.test_n} test` +
    ` · seed ${d.seed} · ${Math.round(d.noise * 100)}% label noise`;
}

function train(steps) {
  const lr = parseFloat($("lr").value);
  const batch = parseInt($("batch").value, 10) || 16;
  ex.trot_train(steps, lr, batch);
  lastTrain = JSON.parse(readOut());
  state = readState();
  lastGradients = readGradients();
  invalidateExport();
  render(state);
  renderLoss();
  renderGradients(lastGradients);
  renderExportAnatomy(readExportMeta());
}

function configure() {
  const activation = $("activation").value;
  const l2on = $("l2").checked;
  const lambda = l2on ? parseFloat($("lambda").value) || 0 : 0;
  $("lambda").disabled = !l2on;
  $("lambda-out").textContent = f3(lambda);

  ex.trot_configure(ACTIVATION_CODES[activation] ?? 0, lambda);
  state = readState();
  lastTrain = null;
  lastGradients = readGradients();
  invalidateExport();

  render(state);
  renderLoss();
  renderGradients(lastGradients);
  trace();
  renderExportAnatomy(readExportMeta());
}

function regenerate() {
  const seed = parseInt($("seed").value, 10) || 0;
  const n = Math.min(5000, Math.max(100, parseInt($("corpus-n").value, 10) || 256));
  const noisePct = Math.min(20, Math.max(0, parseInt($("noise").value, 10) || 0));
  const noise = noisePct / 100;
  const activation = $("activation").value;
  const lambda = $("l2").checked ? parseFloat($("lambda").value) || 0 : 0;
  if (
    !window.confirm(
      `Regenerate ${n} sentences from seed ${seed} at ${noisePct}% label noise ` +
        `and reset the model?`,
    )
  ) {
    return;
  }
  $("corpus-n").value = n;
  ex.trot_init(
    seed,
    n,
    noise,
    ACTIVATION_CODES[activation] ?? 0,
    lambda,
  );
  prevW1 = null;
  lastTrain = null;
  datasetInfo = readDataset();
  state = readState();
  lastGradients = readGradients();
  $("corpus-n").value = datasetInfo.n;
  invalidateExport();

  updateDatasetLine();
  render(state);
  renderLoss();
  renderDataset();
  renderGradients(lastGradients);
  trace();
  renderExportAnatomy(readExportMeta());
}

function reset() {
  // Reset the model over the *active* corpus (use Regenerate to change it).
  ex.trot_init(
    datasetInfo.seed,
    datasetInfo.n,
    datasetInfo.noise,
    ACTIVATION_CODES[state.activation] ?? 0,
    state.lambda,
  );
  prevW1 = null;
  lastTrain = null;
  state = readState();
  lastGradients = readGradients();
  invalidateExport();

  render(state);
  renderLoss();
  renderGradients(lastGradients);
  trace();
  renderExportAnatomy(readExportMeta());
}

// ---- data browser --------------------------------------------------------

function renderDataset() {
  const d = datasetInfo;
  const flips = d.snippets.filter((s) => s.flipped).length;
  $("data-stats").textContent =
    `${d.n} snippets · ${d.n_pos} positive / ${d.n_neg} negative · |V| = ${d.vocab}` +
    ` · ${d.train_n} train / ${d.test_n} test · seed ${d.seed}` +
    ` · ${Math.round(d.noise * 100)}% label noise · ${flips} intentional flips`;

  const tokenTable = $("token-table");
  tokenTable.textContent = "";
  const thHead = document.createElement("tr");
  thHead.appendChild(th("token"));
  thHead.appendChild(th("positive", "num"));
  thHead.appendChild(th("negative", "num"));
  tokenTable.appendChild(thHead);
  for (const t of d.token_stats) {
    const tr = document.createElement("tr");
    tr.appendChild(th(t.token, "", "row"));
    tr.appendChild(cell(String(t.pos), "num"));
    tr.appendChild(cell(String(t.neg), "num"));
    tokenTable.appendChild(tr);
  }

  renderSnippets();
}

function flippedCell(flipped) {
  const td = document.createElement("td");
  td.className = "flip-cell";
  if (!flipped) {
    td.textContent = "—";
    return td;
  }
  const badge = document.createElement("span");
  badge.className = "flip-badge";
  badge.textContent = "flipped";
  badge.title = "intentional label noise";
  badge.setAttribute("aria-label", "intentional label noise");
  td.appendChild(badge);
  return td;
}

function renderSnippets() {
  const d = datasetInfo;
  const tok = filterToken.trim().toLowerCase();
  const rows = d.snippets.filter((s) => {
    if (filterClass !== "all" && String(s.label) !== filterClass) return false;
    if (!tok) return true;
    return s.text
      .toLowerCase()
      .split(/\s+/)
      .some((w) => w.replace(/[^a-z0-9]/g, "") === tok);
  });

  const limit = 400;
  const shown = rows.slice(0, limit);
  const table = $("snippet-table");
  table.textContent = "";
  const head = document.createElement("tr");
  head.appendChild(th("label"));
  head.appendChild(th("sentence"));
  head.appendChild(th("split"));
  head.appendChild(th("flipped"));
  table.appendChild(head);
  for (const s of shown) {
    const tr = document.createElement("tr");
    tr.className = `label-${s.label}`;
    tr.appendChild(cell(s.label === 1 ? "positive" : "negative", "tag"));
    tr.appendChild(cell(s.text));
    tr.appendChild(cell(s.split, "split"));
    tr.appendChild(flippedCell(s.flipped));
    table.appendChild(tr);
  }
  $("data-count").textContent =
    `${shown.length} of ${rows.length} shown` + (rows.length > limit ? ` (first ${limit})` : "");
}

// ---- gradients -----------------------------------------------------------

function emptyGradients(h, v) {
  const zeros = Array(h).fill(0);
  return {
    step: 0,
    lr: 0,
    batch: 0,
    lambda: 0,
    dW1: Array.from({ length: h }, () => Array(v).fill(0)),
    db1: zeros,
    dW2: [zeros],
    db2: [0],
    update_rule: "theta <- theta - lr * dtheta",
  };
}

function gradientFor(kind, row, col, g) {
  if (kind === "w1") return g.dW1[row][col];
  if (kind === "w2") return g.dW2[0][col];
  if (kind === "b1") return g.db1[col];
  return g.db2[0];
}

function beforeFor(kind, row, col) {
  if (kind === "w1") return state.w1[row][col];
  if (kind === "w2") return state.w2[col];
  if (kind === "b1") return state.b1[col];
  return state.b2;
}

function gradLabel(kind, row, col) {
  if (kind === "w1") return `W1[h${row}][${state.vocab_tokens[col]}]`;
  if (kind === "w2") return `W2[h${col}]`;
  if (kind === "b1") return `b1[h${col}]`;
  return "b2";
}

function showGradDetail(td) {
  const kind = td.dataset.kind;
  const row = Number(td.dataset.row);
  const col = Number(td.dataset.col);
  const before = beforeFor(kind, row, col);
  const gradient = gradientFor(kind, row, col, lastGradients);
  const lr = Number(lastGradients.lr) || 0;
  // Display-only next-step projection from the Rust-reported weight, gradient,
  // and learning rate. No forward/backward/loss math happens in the browser.
  const next = before - lr * gradient;
  const label = gradLabel(kind, row, col);
  const stepNote = lastGradients.step > 0 ? `${lastGradients.step}` : "train to refresh";
  const detail =
    `${label}: θ = ${f3(before)} → dθ = ${f3(gradient)} → ` +
    `θ − lr·dθ = ${f3(next)} (next step · step ${stepNote})`;
  td.title = detail;
  $("grad-detail").textContent = detail;
}

function gradCell(kind, row, col, gradient, span, rows, cols, first) {
  const td = cell(f3(gradient), "grad-cell");
  td.style.background = heatStyle(gradient, span);
  td.dataset.kind = kind;
  td.dataset.row = String(row);
  td.dataset.col = String(col);
  td.dataset.rows = String(rows);
  td.dataset.cols = String(cols);
  td.tabIndex = first ? 0 : -1;
  td.setAttribute("aria-label", `${gradLabel(kind, row, col)} gradient ${f3(gradient)}`);
  td.title = `${gradLabel(kind, row, col)} · dθ = ${f3(gradient)}`;
  return td;
}

function moveGradFocus(td, key) {
  const table = td.closest("table");
  const row = Number(td.dataset.row);
  const col = Number(td.dataset.col);
  const rows = Number(td.dataset.rows);
  const cols = Number(td.dataset.cols);
  let nextRow = row;
  let nextCol = col;
  if (key === "ArrowRight") nextCol = Math.min(cols - 1, col + 1);
  else if (key === "ArrowLeft") nextCol = Math.max(0, col - 1);
  else if (key === "ArrowDown") nextRow = Math.min(rows - 1, row + 1);
  else if (key === "ArrowUp") nextRow = Math.max(0, row - 1);
  else if (key === "Home") nextCol = 0;
  else if (key === "End") nextCol = cols - 1;
  else return false;

  const next = table.querySelector(
    `td[data-row="${nextRow}"][data-col="${nextCol}"]`,
  );
  if (next) {
    td.tabIndex = -1;
    next.tabIndex = 0;
    next.focus();
    showGradDetail(next);
  }
  return true;
}

function renderGradients(g) {
  const h = state.hidden;
  const v = state.vocab;
  const gradients = g || emptyGradients(h, v);
  const hasStep = gradients.step > 0;

  $("grad-lr").textContent = `lr = ${f3(gradients.lr)}`;
  $("grad-step").textContent = `step = ${gradients.step}`;
  $("grad-l2").textContent =
    Number(gradients.lambda) > 0
      ? `λ = ${f3(gradients.lambda)} (dW includes λW)`
      : "L2 off";
  $("grad-rule").textContent = gradients.update_rule;
  $("grad-state").textContent = hasStep
    ? `last minibatch · batch ${gradients.batch}`
    : "train to refresh";
  $("grad-detail").textContent = hasStep
    ? "Hover or focus a gradient cell for θ → dθ → θ − lr·dθ (next step)."
    : "No minibatch yet. Train to refresh; the grids show zeros until then.";

  const span = Math.max(1e-9, ...gradients.dW1.flat().map((x) => Math.abs(x)));
  const table = $("grad-w1");
  table.textContent = "";
  const head = document.createElement("tr");
  head.appendChild(th(""));
  for (const tok of state.vocab_tokens) head.appendChild(th(tok, "token"));
  table.appendChild(head);
  gradients.dW1.forEach((row, j) => {
    const tr = document.createElement("tr");
    tr.appendChild(th(`h${j}`, "rowhead", "row"));
    row.forEach((gradient, i) => {
      tr.appendChild(gradCell("w1", j, i, gradient, span, h, v, j === 0 && i === 0));
    });
    table.appendChild(tr);
  });

  const small = $("grad-small");
  small.textContent = "";
  const smallHead = document.createElement("tr");
  smallHead.appendChild(th(""));
  for (let j = 0; j < h; j++) smallHead.appendChild(th(`h${j}`));
  small.appendChild(smallHead);

  const smallSpan = Math.max(
    1e-9,
    ...gradients.dW2[0].map((x) => Math.abs(x)),
    ...gradients.db1.map((x) => Math.abs(x)),
    Math.abs(gradients.db2[0]),
  );
  const w2row = document.createElement("tr");
  w2row.appendChild(th("dW2", "rowhead", "row"));
  gradients.dW2[0].forEach((value, c) => {
    w2row.appendChild(gradCell("w2", 0, c, value, smallSpan, 3, h, c === 0));
  });
  small.appendChild(w2row);
  const b1row = document.createElement("tr");
  b1row.appendChild(th("db1", "rowhead", "row"));
  gradients.db1.forEach((value, c) => {
    b1row.appendChild(gradCell("b1", 1, c, value, smallSpan, 3, h, false));
  });
  small.appendChild(b1row);
  const b2row = document.createElement("tr");
  b2row.appendChild(th("db2", "rowhead", "row"));
  b2row.appendChild(gradCell("b2", 2, 0, gradients.db2[0], smallSpan, 3, h, false));
  small.appendChild(b2row);
}

// ---- trace ---------------------------------------------------------------

function renderTrace(t) {
  const out = $("trace-out");
  out.classList.remove("muted");
  out.textContent = "";
  const hidden = state.hidden;
  const act = t.activation || state.activation;

  const activationLine = document.createElement("p");
  activationLine.className = "trace-act";
  activationLine.textContent = `activation = ${act}`;
  out.appendChild(activationLine);

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
  r1.appendChild(th(""));
  for (let j = 0; j < hidden; j++) r1.appendChild(th(`h${j}`));
  t1.appendChild(r1);
  for (const pt of t.per_token) {
    const tr = document.createElement("tr");
    tr.appendChild(th(pt.token, "", "row"));
    for (const v of pt.col) tr.appendChild(cell(f3(v)));
    t1.appendChild(tr);
  }
  const bRow = document.createElement("tr");
  bRow.appendChild(th("b1", "", "row"));
  for (const v of t.b1) bRow.appendChild(cell(f3(v)));
  t1.appendChild(bRow);
  const zRow = document.createElement("tr");
  zRow.appendChild(th("z1", "", "row"));
  for (const v of t.z1) zRow.appendChild(cell(f3(v)));
  t1.appendChild(zRow);
  out.appendChild(t1);

  const h2 = document.createElement("h3");
  h2.textContent = "hidden → output";
  out.appendChild(h2);
  const t2 = document.createElement("table");
  const r2 = document.createElement("tr");
  r2.appendChild(th(""));
  for (let j = 0; j < hidden; j++) r2.appendChild(th(`h${j}`));
  t2.appendChild(r2);
  for (const [label, row] of [
    [`h = ${act}(z1)`, t.h],
    [`dact = ${act}'(z1)`, t.dact],
    ["W2", t.w2],
    ["h · W2", t.per_unit_z2],
  ]) {
    const tr = document.createElement("tr");
    tr.appendChild(th(label, "", "row"));
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

// ---- export / round-trip -------------------------------------------------

function activationNodeLabel(meta) {
  const attrs = meta.activation_attrs || {};
  if (attrs.alpha !== undefined) {
    return `${meta.activation_node} (alpha = ${attrs.alpha})`;
  }
  return meta.activation_node;
}

function invalidateExport() {
  lastExported = null;
  $("export-summary").textContent = "model changed — export again";
}

function renderExportAnatomy(meta) {
  const table = $("export-anatomy");
  table.textContent = "";
  const rows = [
    ["graph", meta.graph_name],
    ["opset / IR", `${meta.opset} / ${meta.ir_version}`],
    ["activation", `${meta.activation} → ${activationNodeLabel(meta)}`],
    ["nodes", meta.nodes.join(" → ")],
    [
      "initializers",
      meta.initializers.map((i) => `${i.name} [${i.dims.join(",")}]`).join(" · "),
    ],
    ["input", `${meta.input.name} ${meta.input.type}[${meta.input.dims.join(",")}]`],
    ["output", `${meta.output.name} ${meta.output.type}[${meta.output.dims.join(",")}]`],
    [
      "bytes",
      lastExported && lastExported.activation === meta.activation
        ? String(lastExported.bytes)
        : "not exported yet",
    ],
  ];
  for (const [key, value] of rows) {
    const tr = document.createElement("tr");
    tr.appendChild(th(key, "anatomy-key", "row"));
    tr.appendChild(cell(String(value)));
    table.appendChild(tr);
  }
}

function exportOnnx() {
  const len = ex.trot_export_onnx();
  const bytes = readOutBytes();
  const meta = readExportMeta();
  lastExported = { activation: meta.activation, bytes: bytes.length };
  renderExportAnatomy(meta);

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
    `opset ${meta.opset} · IR ${meta.ir_version}\n` +
    `activation ${meta.activation} → ${activationNodeLabel(meta)}\n` +
    `nodes      ${meta.nodes.join(" → ")}\n` +
    `init       ${meta.initializers.map((i) => `${i.name} (${i.dims.join("×")})`).join(", ")}\n` +
    `input      ${meta.input.name} float32[${meta.input.dims.join(", ")}]\n` +
    `output     ${meta.output.name} float32[${meta.output.dims.join(", ")}]\n` +
    `weights    ${state.params} params, trained for ${state.step} steps`;
  return len;
}

function loadOrt() {
  if (window.ort) return Promise.resolve(window.ort);
  if (ortPromise) return ortPromise;
  ortPromise = new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = `${ORT_BASE}ort.min.js`;
    s.onload = () => {
      const ort = window.ort;
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.wasmPaths = ORT_BASE;
      resolve(ort);
    };
    s.onerror = () => reject(new Error("could not load onnxruntime-web from the CDN"));
    document.head.appendChild(s);
  });
  return ortPromise;
}

async function roundtrip() {
  const status = $("rt-status");
  status.className = "muted";
  status.textContent = "loading onnxruntime-web…";
  try {
    const ort = await loadOrt();
    const len = ex.trot_export_onnx();
    const modelBytes = readOutBytes();
    ex.trot_roundtrip_json();
    const ref = JSON.parse(readOut());
    const n = ref.n;
    const v = ref.vocab;
    const flat = new Float32Array(n * v);
    for (let i = 0; i < n; i++) flat.set(ref.inputs[i], i * v);

    status.textContent = `running ${n} test-set predictions through onnxruntime-web…`;
    const tensor = new ort.Tensor("float32", flat, [n, v]);
    const session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ["wasm"],
    });
    const out = await session.run({ input: tensor });
    const got = out.probability.data;
    let maxDiff = 0;
    for (let i = 0; i < n; i++) maxDiff = Math.max(maxDiff, Math.abs(got[i] - ref.probs[i]));
    await session.release?.();

    const pass = maxDiff <= 1e-6;
    status.className = pass ? "pass" : "fail";
    status.textContent =
      `${pass ? "PASS" : "FAIL"} · ${n} predictions · ${len} bytes · ` +
      `activation ${ref.activation} · max |Rust − onnxruntime-web| = ${maxDiff.toExponential(2)}`;
  } catch (err) {
    status.className = "fail";
    status.textContent = `round-trip failed: ${err.message || err}`;
    console.error(err);
  }
}

// ---- wiring --------------------------------------------------------------

function bind() {
  $("btn-1").onclick = () => train(1);
  $("btn-10").onclick = () => train(10);
  $("btn-100").onclick = () => train(100);
  $("btn-reset").onclick = reset;
  $("btn-regen").onclick = regenerate;
  $("btn-trace").onclick = trace;
  $("trace-text").onkeydown = (e) => {
    if (e.key === "Enter") trace();
  };
  $("btn-export").onclick = exportOnnx;
  $("btn-roundtrip").onclick = roundtrip;

  $("activation").onchange = configure;
  $("l2").onchange = () => {
    if ($("l2").checked && (parseFloat($("lambda").value) || 0) === 0) {
      $("lambda").value = "0.01";
    }
    $("lambda").disabled = !$("l2").checked;
    configure();
  };
  $("lambda").oninput = (e) => {
    const on = $("l2").checked;
    $("lambda-out").textContent = f3(on ? parseFloat(e.target.value) : 0);
    if (on) configure();
  };
  $("noise").oninput = (e) => {
    $("noise-out").textContent = `${parseInt(e.target.value, 10) || 0}%`;
  };
  $("noise-preset").onclick = () => {
    $("noise").value = "8";
    $("noise-out").textContent = "8%";
    regenerate();
  };

  $("lr").oninput = (e) => {
    $("lr-out").textContent = parseFloat(e.target.value).toFixed(2);
  };
  $("filter-class").onchange = (e) => {
    filterClass = e.target.value;
    renderSnippets();
  };
  $("filter-token").oninput = (e) => {
    filterToken = e.target.value;
    renderSnippets();
  };

  for (const id of ["grad-w1", "grad-small"]) {
    const table = $(id);
    table.addEventListener("mouseover", (e) => {
      const td = e.target.closest("td[data-kind]");
      if (td) showGradDetail(td);
    });
    table.addEventListener("focusin", (e) => {
      const td = e.target.closest("td[data-kind]");
      if (td) showGradDetail(td);
    });
    table.addEventListener("keydown", (e) => {
      const td = e.target.closest("td[data-kind]");
      if (!td) return;
      if (moveGradFocus(td, e.key)) e.preventDefault();
    });
  }
}

boot().catch((err) => {
  $("dataset-line").textContent = `failed to load: ${err.message || err}`;
  console.error(err);
});
