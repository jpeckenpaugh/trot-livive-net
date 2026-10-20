//! ONNX export — a minimal, hand-written protobuf encoder.
//!
//! The graph is four nodes and the initializers are named after the struct
//! fields, so the file reads like the source:
//!
//! ```text
//! input [batch,V] → Gemm(W1,b1,transB=1) → Tanh|Relu|LeakyRelu → Gemm(W2,b2,transB=1) → Sigmoid → probability
//! ```
//!
//! We encode only the ten protobuf messages the format actually needs. That is
//! why the crate has zero dependencies.

use crate::model::{Activation, Model, LEAKY_ALPHA};

pub const OPSET: i64 = 13;
pub const IR_VERSION: i64 = 8;
pub const INPUT_NAME: &str = "input";
pub const OUTPUT_NAME: &str = "probability";
pub const INITIALIZER_NAMES: [&str; 4] = ["W1", "b1", "W2", "b2"];

// ---- protobuf primitives -------------------------------------------------

fn put_varint(out: &mut Vec<u8>, mut n: u64) {
    loop {
        let b = (n & 0x7f) as u8;
        n >>= 7;
        if n != 0 {
            out.push(b | 0x80);
        } else {
            out.push(b);
            break;
        }
    }
}

fn tag(out: &mut Vec<u8>, field: u32, wire: u32) {
    put_varint(out, ((field << 3) | wire) as u64);
}

fn field_varint(out: &mut Vec<u8>, field: u32, n: u64) {
    tag(out, field, 0);
    put_varint(out, n);
}

fn field_bytes(out: &mut Vec<u8>, field: u32, bytes: &[u8]) {
    tag(out, field, 2);
    put_varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

fn field_float(out: &mut Vec<u8>, field: u32, value: f32) {
    tag(out, field, 5);
    out.extend_from_slice(&value.to_le_bytes());
}

fn field_str(out: &mut Vec<u8>, field: u32, s: &str) {
    field_bytes(out, field, s.as_bytes());
}

fn field_msg(out: &mut Vec<u8>, field: u32, body: &[u8]) {
    field_bytes(out, field, body);
}

// ---- ONNX messages -------------------------------------------------------

fn tensor(name: &str, dims: &[i64], data: &[f32]) -> Vec<u8> {
    let mut t = Vec::new();
    field_str(&mut t, 8, name); // name
    for &d in dims {
        field_varint(&mut t, 1, d as u64); // dims
    }
    field_varint(&mut t, 2, 1); // data_type = FLOAT
    let mut raw = Vec::with_capacity(data.len() * 4);
    for &v in data {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    field_bytes(&mut t, 9, &raw); // raw_data
    t
}

fn attr_int(name: &str, value: i64) -> Vec<u8> {
    let mut a = Vec::new();
    field_str(&mut a, 1, name);
    field_varint(&mut a, 3, value as u64);
    field_varint(&mut a, 20, 2); // AttributeProto::INT
    a
}

fn attr_float(name: &str, value: f32) -> Vec<u8> {
    let mut a = Vec::new();
    field_str(&mut a, 1, name);
    field_float(&mut a, 2, value);
    field_varint(&mut a, 20, 1); // AttributeProto::FLOAT
    a
}

fn node(op: &str, inputs: &[&str], outputs: &[&str], attrs: &[Vec<u8>]) -> Vec<u8> {
    let mut n = Vec::new();
    for i in inputs {
        field_str(&mut n, 1, i);
    }
    for o in outputs {
        field_str(&mut n, 2, o);
    }
    field_str(&mut n, 4, op); // op_type
    for a in attrs {
        field_msg(&mut n, 5, a);
    }
    n
}

fn activation_attrs(activation: Activation) -> Vec<Vec<u8>> {
    match activation {
        Activation::LeakyReLU => vec![attr_float("alpha", LEAKY_ALPHA)],
        Activation::Tanh | Activation::ReLU => Vec::new(),
    }
}

fn dimension(v: i64) -> Vec<u8> {
    let mut d = Vec::new();
    field_varint(&mut d, 1, v as u64);
    d
}

fn dimension_param(p: &str) -> Vec<u8> {
    let mut d = Vec::new();
    field_str(&mut d, 2, p);
    d
}

fn value_info(name: &str, dims: &[Dim]) -> Vec<u8> {
    let mut shape = Vec::new();
    for d in dims {
        let dim = match d {
            Dim::Fixed(v) => dimension(*v),
            Dim::Param(p) => dimension_param(p),
        };
        field_msg(&mut shape, 1, &dim);
    }
    let mut tensor_type = Vec::new();
    field_varint(&mut tensor_type, 1, 1); // elem_type = FLOAT
    field_msg(&mut tensor_type, 2, &shape);
    let mut type_proto = Vec::new();
    field_msg(&mut type_proto, 1, &tensor_type); // TypeProto.tensor_type

    let mut vi = Vec::new();
    field_str(&mut vi, 1, name);
    field_msg(&mut vi, 2, &type_proto);
    vi
}

enum Dim {
    Fixed(i64),
    Param(&'static str),
}

/// Serialize the trained model as an ONNX `ModelProto`.
pub fn export(m: &Model) -> Vec<u8> {
    let h = m.hidden as i64;
    let v = m.vocab as i64;

    let mut graph = Vec::new();
    field_str(&mut graph, 2, "trot"); // graph name

    let gemm1 = node(
        "Gemm",
        &[INPUT_NAME, "W1", "b1"],
        &["z1"],
        &[attr_int("transB", 1)],
    );
    let activation = node(
        m.activation.onnx_node(),
        &["z1"],
        &["h"],
        &activation_attrs(m.activation),
    );
    let gemm2 = node(
        "Gemm",
        &["h", "W2", "b2"],
        &["z2"],
        &[attr_int("transB", 1)],
    );
    let sigmoid = node("Sigmoid", &["z2"], &[OUTPUT_NAME], &[]);
    for n in [&gemm1, &activation, &gemm2, &sigmoid] {
        field_msg(&mut graph, 1, n);
    }

    for t in [
        tensor("W1", &[h, v], &m.w1),
        tensor("b1", &[h], &m.b1),
        tensor("W2", &[1, h], &m.w2),
        tensor("b2", &[1], std::slice::from_ref(&m.b2)),
    ] {
        field_msg(&mut graph, 5, &t);
    }

    field_msg(
        &mut graph,
        11,
        &value_info(INPUT_NAME, &[Dim::Param("batch"), Dim::Fixed(v)]),
    );
    field_msg(
        &mut graph,
        12,
        &value_info(OUTPUT_NAME, &[Dim::Param("batch"), Dim::Fixed(1)]),
    );

    let mut model = Vec::new();
    field_varint(&mut model, 1, IR_VERSION as u64);
    field_str(&mut model, 2, "trot"); // producer_name
    field_str(&mut model, 3, env!("CARGO_PKG_VERSION")); // producer_version
    field_msg(&mut model, 7, &graph);

    let mut opset = Vec::new();
    field_varint(&mut opset, 2, OPSET as u64);
    field_msg(&mut model, 8, &opset);

    model
}
