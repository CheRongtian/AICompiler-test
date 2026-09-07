#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MODEL="${1:-test_model}"

PYTHON="$ROOT/.venv_ai_compiler/bin/python3"

"$PYTHON" "$ROOT/tools/export_pytorch.py" \
  --model "$ROOT/tests/models/${MODEL}.py" \
  --output "$ROOT/build/${MODEL}.tmc"

"$ROOT/build/TensorMetalCompiler" \
  --import-pytorch "$ROOT/build/${MODEL}.tmc"