#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-advisor}"
PATTERN="${2:-silu_mul}"
ENV_FILE="$ROOT/.env"
PYTHON="$ROOT/.venv_ai_compiler/bin/python3"
BUILD_DIR="$ROOT/build"
MANIFEST="$BUILD_DIR/transformer.tmc"
DECODE_MANIFEST="$BUILD_DIR/transformer_decode.tmc"
DECODER_LLM_MANIFEST="$BUILD_DIR/decoder_llm.tmc"
KV_MANIFEST="$BUILD_DIR/kv_cache.tmc"
REQUEST="$BUILD_DIR/advisor_request.json"
RESPONSE="$BUILD_DIR/advisor_response.json"
COMPILER="$BUILD_DIR/TensorMetalCompiler"
DECODER_DTYPE="${TMC_DECODER_DTYPE:-fp32}"
DECODER_CHECKPOINT="${TMC_DECODER_CHECKPOINT:-}"
DECODER_SEED="${TMC_DECODER_SEED:-11}"

DECODER_ARGS=(--seed "$DECODER_SEED")
[[ -n "$DECODER_CHECKPOINT" ]] && DECODER_ARGS+=(--checkpoint "$DECODER_CHECKPOINT")
[[ -n "${TMC_DECODER_HIDDEN_SIZE:-}" ]] && DECODER_ARGS+=(--hidden-size "$TMC_DECODER_HIDDEN_SIZE")
[[ -n "${TMC_DECODER_HEADS:-}" ]] && DECODER_ARGS+=(--heads "$TMC_DECODER_HEADS")
[[ -n "${TMC_DECODER_LAYERS:-}" ]] && DECODER_ARGS+=(--layers "$TMC_DECODER_LAYERS")
[[ -n "${TMC_DECODER_INTERMEDIATE_SIZE:-}" ]] && DECODER_ARGS+=(--intermediate-size "$TMC_DECODER_INTERMEDIATE_SIZE")
[[ -n "${TMC_DECODER_VOCAB_SIZE:-}" ]] && DECODER_ARGS+=(--vocab-size "$TMC_DECODER_VOCAB_SIZE")
[[ -n "${TMC_DECODER_CAPACITY:-}" ]] && DECODER_ARGS+=(--capacity "$TMC_DECODER_CAPACITY")
[[ -n "${TMC_DECODER_PREFILL_LENGTH:-}" ]] && DECODER_ARGS+=(--prefill-length "$TMC_DECODER_PREFILL_LENGTH")
[[ -n "${TMC_DECODER_DECODE_COUNT:-}" ]] && DECODER_ARGS+=(--decode-count "$TMC_DECODER_DECODE_COUNT")
[[ -n "${TMC_DECODER_EPSILON:-}" ]] && DECODER_ARGS+=(--epsilon "$TMC_DECODER_EPSILON")

export_decoder_llm() {
  local requested_dtype="${1:-$DECODER_DTYPE}"
  local effective_dtype
  effective_dtype="$("$COMPILER" --resolve-storage "$requested_dtype")"
  "$PYTHON" "$ROOT/tools/export_decoder_llm.py" \
    --output "$DECODER_LLM_MANIFEST" \
    --dtype "$requested_dtype" \
    --effective-dtype "$effective_dtype" \
    "${DECODER_ARGS[@]}"
}

benchmark_decoder_pair() {
  local requested_dtype="${1:-$DECODER_DTYPE}"
  local effective_dtype
  export_decoder_llm "$requested_dtype"
  effective_dtype="$("$COMPILER" --resolve-storage "$requested_dtype")"

  "$PYTHON" "$ROOT/tools/benchmark_decoder_mps.py" \
    --warmup 2 --samples 10 \
    --dtype "$requested_dtype" \
    --effective-dtype "$effective_dtype" \
    "${DECODER_ARGS[@]}"

  "$COMPILER" --benchmark-decoder-llm "$DECODER_LLM_MANIFEST" \
    --kernel-library "$BUILD_DIR/generated_kernels" \
    --warmup 2 --samples 10

  "$PYTHON" "$ROOT/tools/benchmark_decoder_mps.py" \
    --warmup 2 --samples 10 --token-only \
    --dtype "$requested_dtype" \
    --effective-dtype "$effective_dtype" \
    "${DECODER_ARGS[@]}"
  "$COMPILER" --benchmark-decoder-tokens "$DECODER_LLM_MANIFEST" \
    --kernel-library "$BUILD_DIR/generated_kernels" \
    --warmup 2 --samples 10
}

if [[ "$MODE" != "advisor" && "$MODE" != "generate" && "$MODE" != "decode" && "$MODE" != "decoder-llm" && "$MODE" != "paged-kv" && "$MODE" != "serving" && "$MODE" != "benchmark" && "$MODE" != "benchmark-precision" && "$MODE" != "precision" && "$MODE" != "ablation" && "$MODE" != "regression" ]]; then
  echo "Usage: ./run.sh [advisor|generate [pattern|decoder-all|decoder-model]|decode|decoder-llm|paged-kv|serving|benchmark|benchmark-precision|precision|ablation|regression]" >&2
  exit 1
fi

advisor_url=""
kernel_generator_url=""
api_key=""
if [[ "$MODE" == "advisor" || "$MODE" == "generate" || "$MODE" == "ablation" ]]; then
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
  if [[ "$MODE" != "generate" && -z "$advisor_url" ]]; then
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
    if [[ "$PATTERN" == "decoder-all" ]]; then
      patterns=(decoder_gemv_64_64 decoder_gemv_64_128 decoder_gemv_128_64
                decoder_rope decoder_residual_rmsnorm decoder_gated_mlp)
    elif [[ "$PATTERN" == "decoder-model" ]]; then
      model_hidden="${TMC_DECODER_HIDDEN_SIZE:-64}"
      model_intermediate="${TMC_DECODER_INTERMEDIATE_SIZE:-128}"
      model_vocab="${TMC_DECODER_VOCAB_SIZE:-128}"
      effective_dtype="$("$COMPILER" --resolve-storage "$DECODER_DTYPE")"
      dtype_suffix=""
      [[ "$effective_dtype" == "fp16" ]] && dtype_suffix="_fp16"
      [[ "$effective_dtype" == "bf16" ]] && dtype_suffix="_bf16"
      patterns=()
      matrix_shapes=("$model_hidden $model_hidden"
                     "$model_hidden $model_intermediate"
                     "$model_intermediate $model_hidden"
                     "$model_hidden $model_vocab")
      for matrix_shape in "${matrix_shapes[@]}"; do
        read -r input_size output_size <<< "$matrix_shape"
        candidate="decoder_gemv_${input_size}_${output_size}${dtype_suffix}"
        if (( input_size * output_size > 8388608 )); then
          echo "Generated contract skipped: $candidate exceeds the 8M-element fixture limit."
          continue
        fi
        if [[ " ${patterns[*]} " != *" $candidate "* ]]; then
          patterns+=("$candidate")
        fi
      done
      if [[ "$model_hidden" == "64" && "$model_intermediate" == "128" ]]; then
        patterns+=("decoder_rope${dtype_suffix}"
                   "decoder_residual_rmsnorm${dtype_suffix}"
                   "decoder_gated_mlp${dtype_suffix}")
      fi
    else
      patterns=("$PATTERN")
    fi
    for pattern in "${patterns[@]}"; do
      "$PYTHON" "$ROOT/tools/llm_kernel_generator.py" \
        --compiler "$COMPILER" \
        --pattern "$pattern" \
        --work-dir "$BUILD_DIR/generated_kernels/$pattern"
    done
    ;;
  decode)
    "$PYTHON" "$ROOT/tools/export_transformer_decode.py" \
      --output "$DECODE_MANIFEST"

    "$COMPILER" --transformer-decode "$DECODE_MANIFEST"
    ;;
  decoder-llm)
    export_decoder_llm

    "$COMPILER" --decoder-llm "$DECODER_LLM_MANIFEST" \
      --kernel-library "$BUILD_DIR/generated_kernels"
    ;;
  paged-kv)
    export_decoder_llm

    "$COMPILER" --paged-kv "$DECODER_LLM_MANIFEST" \
      --page-size 4 \
      --chunk-size 3 \
      --kernel-library "$BUILD_DIR/generated_kernels"
    ;;
  serving)
    export_decoder_llm

    "$COMPILER" --serving "$DECODER_LLM_MANIFEST" \
      --kernel-library "$BUILD_DIR/generated_kernels"
    ;;
  benchmark)
    benchmark_decoder_pair "$DECODER_DTYPE"
    ;;
  benchmark-precision)
    for precision_dtype in fp32 fp16 bf16; do
      echo "Decoder MPS/Metal benchmark: $precision_dtype"
      benchmark_decoder_pair "$precision_dtype"
    done
    ;;
  precision)
    for precision_dtype in fp32 fp16 bf16; do
      echo "Decoder precision regression: $precision_dtype"
      export_decoder_llm "$precision_dtype"
      "$COMPILER" --decoder-llm "$DECODER_LLM_MANIFEST" \
        --kernel-library "$BUILD_DIR/generated_kernels"
    done
    ;;
  ablation)
    "$PYTHON" "$ROOT/tools/export_pytorch.py" \
      --output "$MANIFEST"

    echo "Advisor ablation: OFF"
    "$COMPILER" --benchmark-import-pytorch "$MANIFEST"

    "$COMPILER" --import-pytorch "$MANIFEST" \
      --emit-advisor-request "$REQUEST"

    "$PYTHON" "$ROOT/tools/llm_advisor.py" \
      --input "$REQUEST" \
      --output "$RESPONSE"

    echo "Advisor ablation: ON"
    "$COMPILER" --benchmark-import-pytorch "$MANIFEST" \
      --advisor-response "$RESPONSE"

    export_decoder_llm
    echo "Decoder generated-kernel ablation: OFF vs ON"
    "$COMPILER" --benchmark-decoder-llm "$DECODER_LLM_MANIFEST" \
      --kernel-library "$BUILD_DIR/generated_kernels" \
      --warmup 2 --samples 10

    echo "Decoder fusion ablation: OFF vs ON"
    "$COMPILER" --benchmark-decoder-fusions "$DECODER_LLM_MANIFEST" \
      --kernel-library "$BUILD_DIR/generated_kernels" \
      --warmup 2 --samples 10
    ;;
  regression)
    echo "Regression: Metal baseline, autotuning, TensorIR, and fusion"
    "$COMPILER"

    echo "Regression: PyTorch importer"
    regression_models=(
      "$ROOT/workloads/pytorch/test_model.py"
      "$ROOT/tests/models/test_linear.py"
      "$ROOT/tests/models/test_mlp.py"
    )
    for regression_model_path in "${regression_models[@]}"; do
      regression_model="$(basename "$regression_model_path" .py)"
      "$PYTHON" "$ROOT/tools/export_pytorch.py" \
        --model "$regression_model_path" \
        --output "$BUILD_DIR/${regression_model}.tmc"
      "$COMPILER" --import-pytorch "$BUILD_DIR/${regression_model}.tmc"
    done

    echo "Regression: stateful fixed-capacity KV cache"
    "$PYTHON" "$ROOT/tools/export_kv_cache.py" --output "$KV_MANIFEST"
    "$COMPILER" --kv-cache "$KV_MANIFEST"

    echo "Regression: Advisor protocol"
    "$PYTHON" "$ROOT/tools/export_pytorch.py" --output "$MANIFEST"
    "$COMPILER" --import-pytorch "$MANIFEST" \
      --emit-advisor-request "$BUILD_DIR/regression_advisor_request.json"

    echo "Regression: generated-kernel contract"
    "$COMPILER" --emit-kernel-contract \
      "$BUILD_DIR/regression_silu_mul_contract.json" --pattern silu_mul

    echo "Regression: stateful Transformer decode"
    "$PYTHON" "$ROOT/tools/export_transformer_decode.py" \
      --output "$DECODE_MANIFEST"
    "$COMPILER" --transformer-decode "$DECODE_MANIFEST"

    echo "Regression: decoder-only runtime"
    export_decoder_llm
    "$COMPILER" --decoder-llm "$DECODER_LLM_MANIFEST" \
      --kernel-library "$BUILD_DIR/generated_kernels"

    echo "Regression: chunked prefill and paged KV cache"
    "$COMPILER" --paged-kv "$DECODER_LLM_MANIFEST" \
      --page-size 4 \
      --chunk-size 3 \
      --kernel-library "$BUILD_DIR/generated_kernels"

    echo "Regression: continuous batching and preemption"
    "$COMPILER" --serving "$DECODER_LLM_MANIFEST" \
      --kernel-library "$BUILD_DIR/generated_kernels"

    echo "Regression: decoder storage precision"
    for precision_dtype in fp32 fp16 bf16; do
      export_decoder_llm "$precision_dtype"
      "$COMPILER" --decoder-llm "$DECODER_LLM_MANIFEST" \
        --kernel-library "$BUILD_DIR/generated_kernels"
    done

    echo "Regression: PASS"
    ;;
esac
