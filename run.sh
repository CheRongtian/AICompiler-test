#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$ROOT/.env"
PYTHON="$ROOT/.venv_ai_compiler/bin/python3"
BUILD_DIR="$ROOT/build"
MANIFEST="$BUILD_DIR/transformer.tmc"
REQUEST="$BUILD_DIR/advisor_request.json"
RESPONSE="$BUILD_DIR/advisor_response.json"
COMPILER="$BUILD_DIR/TensorMetalCompiler"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing $ENV_FILE. Create it from .env.example and fill in URL and API key." >&2
  exit 1
fi

endpoint=""
api_key=""
while IFS='=' read -r name value; do
  case "$name" in
    TMC_LLM_ENDPOINT) endpoint="$value" ;;
    TMC_LLM_API_KEY) api_key="$value" ;;
  esac
done < "$ENV_FILE"

if [[ -z "$endpoint" ]]; then
  echo "TMC_LLM_ENDPOINT is empty in .env." >&2
  exit 1
fi
if [[ -z "$api_key" ]]; then
  echo "TMC_LLM_API_KEY is empty in .env." >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing project Python environment: $PYTHON" >&2
  exit 1
fi

"$PYTHON" "$ROOT/tools/export_pytorch.py" \
  --output "$MANIFEST"

cmake -S "$ROOT" -B "$BUILD_DIR"
cmake --build "$BUILD_DIR" --target TensorMetalCompiler

"$COMPILER" --import-pytorch "$MANIFEST" \
  --emit-advisor-request "$REQUEST"

"$PYTHON" "$ROOT/tools/llm_advisor.py" \
  --input "$REQUEST" \
  --output "$RESPONSE"

"$COMPILER" --import-pytorch "$MANIFEST" \
  --advisor-response "$RESPONSE"
