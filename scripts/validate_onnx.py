#!/usr/bin/env python3
"""Validate the exported ONNX graph for all three activations.

Run after building the example:

    cargo run --release -p trot-core --example export_onnx
    python3 scripts/validate_onnx.py

Checks: protobuf validity, opset/IR, frozen names/shapes, activation node and
attributes, and that a standard runtime agrees with the Rust forward pass to
within 1e-6.
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
HIDDEN = 8
VOCAB = 50
CASES = [
    ("tanh", "Tanh"),
    ("relu", "Relu"),
    ("leaky_relu", "LeakyRelu"),
]


def attr_map(node: onnx.NodeProto) -> dict[str, onnx.AttributeProto]:
    return {a.name: a for a in node.attribute}


def check_activation_node(node: onnx.NodeProto, identifier: str, op: str) -> None:
    attrs = attr_map(node)
    if identifier == "leaky_relu":
        assert op == "LeakyRelu", op
        alpha = attrs.get("alpha")
        assert alpha is not None, "LeakyRelu missing alpha"
        assert alpha.type == onnx.AttributeProto.FLOAT, alpha.type
        assert abs(alpha.f - 0.01) < 1e-7, alpha.f
        assert len(attrs) == 1, sorted(attrs)
    else:
        assert op in ("Tanh", "Relu"), op
        assert not attrs, f"{op} must have no attributes: {sorted(attrs)}"


def check_gemm(node: onnx.NodeProto, inputs: list[str], outputs: list[str]) -> None:
    assert list(node.input) == inputs, list(node.input)
    assert list(node.output) == outputs, list(node.output)
    attrs = attr_map(node)
    assert attrs["transB"].i == 1, attrs["transB"].i


def validate_case(root: Path, identifier: str, op: str) -> tuple[float, int]:
    model_path = root / "target" / f"trot_{identifier}.onnx"
    ref_path = root / "target" / f"trot_{identifier}_ref.json"
    if not model_path.exists() or not ref_path.exists():
        raise AssertionError(
            f"missing {model_path.name} or {ref_path.name}; run the example first"
        )

    model = onnx.load(str(model_path))
    onnx.checker.check_model(model)

    opsets = [(o.domain, o.version) for o in model.opset_import]
    assert ("", OPSET) in opsets, f"opset {opsets}"
    assert model.ir_version == IR_VERSION, model.ir_version
    assert model.graph.name == "trot", model.graph.name

    names = [i.name for i in model.graph.initializer]
    assert names == ["W1", "b1", "W2", "b2"], names
    shapes = {
        i.name: list(i.dims)
        for i in model.graph.initializer
    }
    assert shapes == {
        "W1": [HIDDEN, VOCAB],
        "b1": [HIDDEN],
        "W2": [1, HIDDEN],
        "b2": [1],
    }, shapes

    nodes = list(model.graph.node)
    ops = [n.op_type for n in nodes]
    assert ops == ["Gemm", op, "Gemm", "Sigmoid"], ops
    check_gemm(nodes[0], ["input", "W1", "b1"], ["z1"])
    assert list(nodes[1].input) == ["z1"], list(nodes[1].input)
    assert list(nodes[1].output) == ["h"], list(nodes[1].output)
    check_activation_node(nodes[1], identifier, op)
    check_gemm(nodes[2], ["h", "W2", "b2"], ["z2"])
    assert list(nodes[3].input) == ["z2"], list(nodes[3].input)
    assert list(nodes[3].output) == ["probability"], list(nodes[3].output)

    ref = json.loads(ref_path.read_text())
    assert ref["activation"] == identifier, ref["activation"]
    x = np.array(ref["inputs"], dtype=np.float32)
    assert x.shape == (ref["n"], VOCAB), x.shape

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
    assert inp.name == "input" and out.name == "probability", (inp.name, out.name)
    assert inp.type == "tensor(float)", inp.type
    assert out.type == "tensor(float)", out.type
    assert inp.shape == ["batch", VOCAB], inp.shape
    assert out.shape == ["batch", 1], out.shape

    y = sess.run([out.name], {inp.name: x})[0].reshape(-1)
    expected = np.array(ref["probs"], dtype=np.float32)
    max_diff = float(np.abs(y - expected).max())
    assert np.allclose(y, expected, atol=1e-6), f"{identifier}: max diff {max_diff}"
    return max_diff, model_path.stat().st_size


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    for identifier, op in CASES:
        try:
            max_diff, size = validate_case(root, identifier, op)
        except AssertionError as exc:
            print(f"FAIL {identifier}: {exc}", file=sys.stderr)
            return 1
        print(
            f"OK  trot_{identifier}.onnx: {size} bytes, opset {OPSET}, IR {IR_VERSION}, "
            f"node {op}, max |Rust - onnxruntime| = {max_diff:.2e}"
        )
    print("OK  all three activations match the Rust core within 1e-6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
