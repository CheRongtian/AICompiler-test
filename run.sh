#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-advisor}"
ENV_FILE="$ROOT/.env"
PYTHON="$ROOT/.venv_ai_compiler/bin/python3"
BUILD_DIR="$ROOT/build"
MANIFEST="$BUILD_DIR/transformer.tmc"
DECODE_MANIFEST="$BUILD_DIR/transformer_decode.tmc"
DECODER_LLM_MANIFEST="$BUILD_DIR/decoder_llm.tmc"
REQUEST="$BUILD_DIR/advisor_request.json"
RESPONSE="$BUILD_DIR/advisor_response.json"
COMPILER="$BUILD_DIR/TensorMetalCompiler"

if [[ "$MODE" != "advisor" && "$MODE" != "generate" && "$MODE" != "decode" && "$MODE" != "decoder-llm" ]]; then
  echo "Usage: ./run.sh [advisor|generate|decode|decoder-llm]" >&2
  exit 1
fi

advisor_url=""
kernel_generator_url=""
api_key=""
if [[ "$MODE" == "advisor" || "$MODE" == "generate" ]]; then
  if [[ ! -f "$ENV_FILE" ]]; then
    echo "Missing $ENV_FILE. Create it from .env.example and fill in URL and API key." >&2
    exit 1
  fi
  while IFS='=' read -r name value; do
    case "$name" in
      TMC_LLM_ADVISOR_URL) advisor_url="$value" ;;
      TMC_LLM_KERNEL_GENERATOR_URL) kernel_generator_url="$value" ;;
      TMC_LLM_API_KEY) api_key="$value" ;;
    esac
  done < "$ENV_FILE"
  if [[ -z "$api_key" ]]; then
    echo "TMC_LLM_API_KEY is empty in .env." >&2
    exit 1
  fi
  if [[ "$MODE" == "advisor" && -z "$advisor_url" ]]; then
    echo "TMC_LLM_ADVISOR_URL is empty in .env." >&2
    exit 1
  fi
  if [[ "$MODE" == "generate" && -z "$kernel_generator_url" ]]; then
    echo "TMC_LLM_KERNEL_GENERATOR_URL is empty in .env." >&2
    exit 1
  fi
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing project Python environment: $PYTHON" >&2
  exit 1
fi

cmake -S "$ROOT" -B "$BUILD_DIR"
cmake --build "$BUILD_DIR" --target TensorMetalCompiler

case "$MODE" in
  advisor)
    "$PYTHON" "$ROOT/tools/export_pytorch.py" \
      --output "$MANIFEST"

    "$COMPILER" --import-pytorch "$MANIFEST" \
      --emit-advisor-request "$REQUEST"

    "$PYTHON" "$ROOT/tools/llm_advisor.py" \
      --input "$REQUEST" \
      --output "$RESPONSE"

    "$COMPILER" --import-pytorch "$MANIFEST" \
      --advisor-response "$RESPONSE"
    ;;
  generate)
    "$PYTHON" "$ROOT/tools/llm_kernel_generator.py" \
      --compiler "$COMPILER" \
      --work-dir "$BUILD_DIR/generated_kernels"
    ;;
  decode)
    "$PYTHON" "$ROOT/tools/export_transformer_decode.py" \
      --output "$DECODE_MANIFEST"

    "$COMPILER" --transformer-decode "$DECODE_MANIFEST"
    ;;
  decoder-llm)
    "$PYTHON" "$ROOT/tools/export_decoder_llm.py" \
      --output "$DECODER_LLM_MANIFEST"

    "$COMPILER" --decoder-llm "$DECODER_LLM_MANIFEST"
    ;;
esac
