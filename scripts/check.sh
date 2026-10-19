#!/usr/bin/env bash
#
# Quality gates for TROT: fmt, clippy, tests, and ONNX round-trip validation.
# Usage: scripts/check.sh [--no-onnx]
set -uo pipefail
cd "$(dirname "$0")/.."

fail=0
skip_onnx=0
[ "${1:-}" = "--no-onnx" ] && skip_onnx=1

echo "==> cargo fmt --all -- --check"
cargo fmt --all -- --check && echo "    fmt: ok" || { echo "    fmt: FAILED"; fail=1; }

echo
echo "==> cargo clippy --workspace --all-targets -- -D warnings"
if cargo clippy --workspace --all-targets -- -D warnings; then
  echo "    clippy: clean"
else
  echo "    clippy: FAILED"
  fail=1
fi

echo
echo "==> cargo test -p trot-core"
if cargo test -p trot-core; then
  echo "    tests: ok"
else
  echo "    tests: FAILED"
  fail=1
fi

if [ "$skip_onnx" -eq 0 ]; then
  echo
  echo "==> onnx round-trip (rust core vs onnxruntime)"
  if cargo run --release -q -p trot-core --example export_onnx >/dev/null && \
     python3 scripts/validate_onnx.py; then
    echo "    onnx: ok"
  else
    echo "    onnx: FAILED"
    fail=1
  fi
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "ALL CHECKS PASSED"
else
  echo "CHECKS FAILED"
fi
exit "$fail"
