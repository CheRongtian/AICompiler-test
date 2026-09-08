#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MODEL="${1:-test_model}"

PYTHON="$ROOT/.venv_ai_compiler/bin/python3"
MODEL_PATH="$ROOT/tests/models/${MODEL}.py"
if [[ "$MODEL" == "test_model" ]]; then
  MODEL_PATH="$ROOT/workloads/pytorch/test_model.py"
fi

"$PYTHON" "$ROOT/tools/export_pytorch.py" \
  --model "$MODEL_PATH" \
  --output "$ROOT/build/${MODEL}.tmc"

"$ROOT/build/TensorMetalCompiler" \
  --import-pytorch "$ROOT/build/${MODEL}.tmc"
