.PHONY: all test lint wasm export serve clean

all: test lint wasm

test:
	cargo test -p trot-core

lint:
	cargo fmt --all -- --check
	cargo clippy --workspace --all-targets -- -D warnings

wasm:
	cargo build --release -p trot-wasm --target wasm32-unknown-unknown
	cp target/wasm32-unknown-unknown/release/trot_wasm.wasm trot.wasm
	@echo "wrote trot.wasm ($$(wc -c < trot.wasm) bytes)"

export:
	cargo run --release -p trot-core --example export_onnx
	python3 scripts/validate_onnx.py

serve:
	@echo "http://localhost:8000"
	python3 -m http.server 8000

clean:
	cargo clean
