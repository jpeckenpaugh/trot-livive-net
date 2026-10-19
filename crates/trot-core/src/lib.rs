//! trot-core — a tiny multilayer perceptron written from scratch.
//!
//! Everything here is plain `Vec<f32>` and hand-written loops. No math library,
//! no BLAS, no autodiff. If a reader can follow a `for` loop, they can follow
//! the model.
//!
//! The model loops use explicit indices on purpose: the index arithmetic is the
//! documentation. Clippy would rather see iterator adapters, but adapters hide
//! the `j * V + i` addressing that makes the model readable.

#![allow(clippy::needless_range_loop)]

pub mod data;
pub mod generator;
pub mod model;
pub mod onnx;
pub mod rng;
pub mod train;
