#!/usr/bin/env python3
"""Validate target/trot.onnx against the Rust core's own predictions.

Run after building the example:

    cargo run --release -p trot-core --example export_onnx
    python3 scripts/validate_onnx.py

Checks: protobuf validity, opset/IR, frozen names, and that a standard runtime
agrees with the Rust forward pass to within 1e-6.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

OPSET = 13
IR_VERSION = 8


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    model_path = root / "target" / "trot.onnx"
    ref_path = root / "target" / "trot_ref.json"
    if not model_path.exists() or not ref_path.exists():
        print("missing target/trot.onnx or target/trot_ref.json; run the example first", file=sys.stderr)
        return 2

    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)

    opsets = [(o.domain, o.version) for o in model.opset_import]
    assert (("", OPSET) in opsets), f"opset {opsets}"
    assert model.ir_version == IR_VERSION, model.ir_version
    names = [i.name for i in model.graph.initializer]
    assert names == ["W1", "b1", "W2", "b2"], names
    ops = [n.op_type for n in model.graph.node]
    assert ops == ["Gemm", "Tanh", "Gemm", "Sigmoid"], ops

    ref = json.loads(ref_path.read_text())
    x = np.array(ref["inputs"], dtype=np.float32)
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
    assert inp.name == "input" and out.name == "probability", (inp.name, out.name)
    assert inp.type == "tensor(float)"

    y = sess.run([out.name], {inp.name: x})[0].reshape(-1)
    expected = np.array(ref["probs"], dtype=np.float32)
    max_diff = float(np.abs(y - expected).max())
    assert np.allclose(y, expected, atol=1e-6), f"max diff {max_diff}"

    print(f"OK  {model_path.name}: {model_path.stat().st_size} bytes, "
          f"{ref['n']} predictions, max |Rust - onnxruntime| = {max_diff:.2e}")
    print(f"    opset {OPSET}, IR {IR_VERSION}, initializers {names}, nodes {ops}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
