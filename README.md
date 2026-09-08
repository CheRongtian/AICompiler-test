# AICompiler-test

## Project Structure

```
AICompiler/
├── apps/
│   ├── generated_kernel_main.cpp
│   ├── generated_kernel_main.hpp
│   ├── decoder_benchmark_main.cpp
│   ├── decoder_benchmark_main.hpp
│   ├── decoder_llm_main.cpp
│   ├── decoder_llm_main.hpp
│   ├── main.cpp
│   ├── kv_cache_main.cpp
│   ├── kv_cache_main.hpp
│   ├── paged_kv_main.cpp
│   ├── paged_kv_main.hpp
│   ├── pytorch_import_main.cpp
│   ├── pytorch_import_main.hpp
│   ├── tensor_graph_examples.cpp
│   ├── tensor_graph_examples.hpp
│   ├── transformer_decode_main.cpp
│   ├── transformer_decode_main.hpp
│   └── tensor_metal_main.cpp
├── src/
│   ├── analyzer/
│   │   ├── GraphAnalyzer.*
│   │   └── PatternAnalyzer.*
│   ├── backend/metal/
│   │   ├── FusionMetalEmitter.cpp
│   │   ├── GraphMetalEmitter.cpp
│   │   ├── KVCacheMetalEmitter.*
│   │   ├── DecodeGEMVMetalEmitter.*
│   │   ├── DecoderLLMMetalEmitter.*
│   │   ├── MetalEmitter.*
│   │   ├── MetalRuntime.*
│   │   ├── RMSNormBaseline.cpp
│   │   └── TransformerDecodeMetalEmitter.*
│   ├── benchmark/
│   │   └── Benchmark.*
│   ├── importer/
│   │   ├── KVCacheImporter.*
│   │   ├── DecoderLLMImporter.*
│   │   ├── PyTorchImporter.*
│   │   └── TransformerDecodeImporter.*
│   ├── llm/
│   │   ├── AdvisorProtocol.*
│   │   ├── DecoderKernelContracts.cpp
│   │   ├── GeneratedKernelProtocol.*
│   │   └── KernelContract.*
│   ├── planner/
│   │   ├── KernelPlan.*
│   │   ├── DecodeGEMVTuner.*
│   │   ├── DecoderLLMPlan.*
│   │   ├── KVCachePlan.*
│   │   ├── RegionPlan.*
│   │   ├── RMSNormTuner.*
│   │   └── TransformerDecodePlan.*
│   ├── runtime/
│   │   ├── GeneratedKernelAdmission.*
│   │   ├── DecoderLLMExecutor.*
│   │   ├── GraphExecutor.*
│   │   ├── KVCacheState.*
│   │   ├── KernelRegistry.*
│   │   ├── StatefulExecutor.*
│   │   └── TransformerDecodeExecutor.*
│   ├── tensor/
│   │   ├── TensorIR.*
│   │   ├── DecoderLLM.hpp
│   │   └── TransformerDecode.hpp
│   ├── validation/
│   │   ├── GraphReference.*
│   │   └── Validator.*
│   ├── AST.hpp
│   ├── lexer.*
│   ├── parser.*
│   ├── codegen.*
│   ├── TopLevel.*
│   ├── JITLib.*
│   └── KaleidoscopeJIT.hpp
├── workloads/pytorch/
│   ├── transformer.py
│   ├── KVCache.py
│   ├── MOE.py
│   ├── decoder_llm.py
│   ├── test_model.py
│   └── transformer_kv_benchmark.py
├── tools/
│   ├── export_kv_cache.py
│   ├── export_decoder_llm.py
│   ├── export_transformer_decode.py
│   ├── benchmark_decoder_mps.py
│   ├── llm_advisor.py
│   ├── llm_kernel_generator.py
│   └── export_pytorch.py
├── prompts/
│   ├── metal_advisor_en.md
│   ├── metal_advisor_zh.md
│   ├── metal_kernel_generator_en.md
│   └── metal_kernel_generator_zh.md
├── config/
│   └── kernel_principles.json
├── docs/papers/
│   └── 2606.07665v2.pdf
├── tests/models/
│   │     ├── test_linear.py
│   │     └── test_mlp.py
│   └── test.sh
├── .env.example
├── CMakeLists.txt
├── run.sh
└── README.md
```

## 0. Preparation
This stage verifies Metal device discovery, runtime MSL compilation, kernel lookup, and compute pipeline creation without GPU dispatch.

- Test:
```bash
mkdir build
cd build
cmake ..
make
./TensorMetalCompiler
```
- Outcome:
```bash
Metal device: Apple M3 Pro
Library compile: PASS
Kernel lookup: PASS
Pipeline creation: PASS
threadExecutionWidth: 32
maxTotalThreadsPerThreadgroup: 1024
```

## 1. Implementation

### Metal execution baseline

- Added runtime MSL compilation, compute pipeline creation, buffer management, GPU dispatch, readback, and timing.
- Validated VectorAdd and fp32 RMSNorm for `[1, 4096]` and `[3, 4097]` against CPU references.

### RMSNorm autotuning and admission

- Generates RMSNorm candidates with 64, 128, and 256 threads per threadgroup.
- Applies hardware filtering, compilation, interface checking, numerical validation, warmup, and benchmarking.
- Admits a candidate only when it beats the fixed GPU baseline by at least 1.05x in two rounds; otherwise uses the baseline fallback.

### Tensor graph compiler

- Added a static TensorIR graph with fp16/fp32 types, explicit strides, views, broadcasting, reduction axes, dependency analysis, and kernel planning.
- Supports Add, Mul, MatMul, RMSNorm, LayerNorm, ReLU, SiLU, Softmax, RoPE, reductions, reshape/view/transpose/contiguous, MaskedFill, Slice, Embedding, and Gather.
- Executes unfused multi-node graphs with GPU-resident intermediate buffers and reuses RMSNorm autotuning.

### Region fusion and admission

- Forms regions with explicit external inputs and outputs and conservative alias/effect boundaries.
- Generates fused candidates for Add + RMSNorm, Residual Add + LayerNorm, SiLU + Mul, and Linear + ReLU.
- Compares fused candidates with an unfused same-command-buffer baseline and admits only candidates that pass numerical validation and the performance threshold.

### Static PyTorch frontend

- Uses `torch.export` to convert PyTorch models into the existing TensorIR.
- Supports built-in workloads and custom test models through an `export_case()` interface.
- Runs imported graphs through analysis, fusion/admission, Metal execution, and numerical validation against PyTorch.
- Custom PyTorch tests are stored under `tests/models/` and can be executed with `tests/test.sh`.

```bash
./tests/test.sh
./tests/test.sh test_linear
./tests/test.sh test_mlp
```

### Stateful KV cache runtime

- Traces explicit prefill and decode entries from `KVCache.py` and recognizes the two axis-2 `aten.cat` cache appends.
- Lowers cache growth to fixed-capacity key/value Metal buffers with a runtime-managed valid length.
- Compiles reusable prefill and single-token decode command sequences while preserving the stateless TensorGraph path.
- Validates every output and logical cache prefix against the PyTorch reference, checks safe rejection at fixed-buffer capacity, and reports GPU latency.

```bash
python3 tools/export_kv_cache.py --output build/kv_cache.tmc
cmake --build build --target TensorMetalCompiler
./build/TensorMetalCompiler --kv-cache build/kv_cache.tmc
```

### LLM-guided candidate search

- Exports static region, tensor, hardware, and legal-candidate summaries as JSON.
- Lets an external LLM rank existing RMSNorm and fusion candidates without generating MSL.
- Applies strict response parsing, candidate legality checks, and a two-candidate search budget per optimization kind.
- Retains hardware filtering, numerical validation, benchmarking, admission, and deterministic fallback inside the compiler.

```bash
./run.sh advisor
```

Copy either Advisor prompt under `prompts/` into the remote Advisor Agent configuration, then fill `TMC_LLM_ADVISOR_URL` and the shared `TMC_LLM_API_KEY` in the ignored project-local `.env` file. `run.sh` exports the Transformer workload, configures and builds the compiler, requests LLM advice, and executes advised compilation. The API client sends only the current compiler request JSON as a user message. API connectivity is external to the compiler; malformed or unavailable advice falls back to deterministic planning.

### LLM-generated Metal kernel admission

- Uses compiler-owned fp32 contracts for SiLU + Mul and interleaved Q/K RoPE, including cases, buffer roles, constants, dispatch, and admission thresholds.
- Reads `config/kernel_principles.json` at runtime. Every Workflow request carries the immutable contract, principles, all previous attempts, and compiler feedback.
- Shares compilation, reflected ABI checks, per-output numerical/guard validation, warmup, and admission across patterns.
- Selects a template baseline per case and requires at least 1.05x speedup in two paired rounds on every case; otherwise retries up to three times and keeps the template fallback.

```bash
./run.sh generate
./run.sh generate rope
```

Copy either updated `prompts/metal_kernel_generator_zh.md` or `prompts/metal_kernel_generator_en.md` into the shared remote Kernel Generator Workflow. Configure `TMC_LLM_KERNEL_GENERATOR_URL` and `TMC_LLM_API_KEY` in `.env`. Prompts stay on the Workflow; Python sends runtime data only. Remote memory is unnecessary. Requests are stored under `build/generated_kernels/<pattern>/`; the compiler writes `admitted.json` only after all admission checks pass. A failed generation preserves any existing admitted artifact.

### Multi-output Metal execution

- Binds consecutive input buffers, output buffers, and optional packed uint32 constants.
- Supports typed multi-output interface checking, reusable PreparedExecution/PreparedSequence, and readback of all outputs.
- Exercises the generated-kernel path with separate rotated Q and K outputs; the existing single-output entry points use the same runtime implementation.

### Stateful Transformer decode

- Exports a two-layer cached decoder reference with external encoder memory using the existing PyTorch workload.
- Compiles token embedding, positional encoding, cached causal self-attention, cross-attention, residual LayerNorm, ReLU FFN, LM head, and argmax into reusable Metal sequences.
- Precomputes each layer's cross-attention memory K/V once and maintains independent fixed-capacity self-attention K/V buffers per layer.
- Separates multi-token prefill from single-token decode, feeds each generated token into the next step, and keeps intermediate tensors on GPU buffers.
- Validates logits, selected tokens, every layer's logical cache prefix, cache length, storage reuse, and safe capacity rejection against PyTorch references.
- Reports prefill GPU time and median single-token decode GPU time.

```bash
./run.sh decode
```

This workload follows the current PyTorch model's sinusoidal positional encoding, LayerNorm, cross-attention, and ReLU FFN semantics. The decoder-only workload below uses RoPE, RMSNorm, and a gated MLP.

### Decoder-only LLM and Decode GEMV

- Adds a two-layer decoder-only PyTorch workload with pre-norm RMSNorm, interleaved RoPE, causal self-attention, SwiGLU, LM head, and autoregressive token feedback.
- Imports fixed fp32 parameters and PyTorch references for an 8-token prefill followed by eight single-token decode steps.
- Keeps one fixed-capacity K/V cache per layer and compiles reusable prefill and decode Metal command sequences.
- Autotunes Decode GEMV candidates for each Linear shape and admits candidates only after hardware, interface, numerical, warmup, and paired performance checks; prefill uses the generic Linear kernel.
- Validates logits, generated tokens, every logical cache prefix, cache length, and cache storage reuse.

```bash
./run.sh decoder-llm
```

### Admitted kernels in decoder execution

- Adds decode-only contracts for GEMV (`decoder_gemv_64_64`, `decoder_gemv_64_128`, `decoder_gemv_128_64`), `decoder_rope`, `decoder_residual_rmsnorm`, and `decoder_gated_mlp`.
- Decoder RoPE reads the actual position tables and int cache length. Residual RMSNorm produces both the residual sum and normalized tensor. Gated MLP replaces Gate/Up projections plus SiLU×Mul; Down projection remains a separate GEMV.
- Benchmarks generated kernels against template sequences in one command buffer, including scalar/vector GEMV candidates. Group reductions use one threadgroup per work item.
- The registry matches the complete current contract and device recorded by local admission, then rechecks pipeline creation, ABI, buffer sizes, dtype and dispatch at binding time. Missing or incompatible artifacts use templates. These are trusted local admission records.
- Matching kernels bind actual decoder buffers in the reusable decode sequence. Prefill retains its template path. Standalone `silu_mul` and `rope` artifacts are not decoder replacements.
- The runtime audit reports selected node, implementation, function, completed calls and dispatches; loaded-but-unused artifacts are reported separately. Counts cover the replacement regions.

Update the Kernel Generator system prompt for the group-dispatch rules, then run:

```bash
./run.sh generate decoder-all
./run.sh decoder-llm
```

To generate one pattern, use e.g. `./run.sh generate decoder_residual_rmsnorm`.
Generation uses the same remote Workflow for all six patterns. Decoder execution itself makes no API calls. Look for `generated` entries with nonzero `completed_calls` and `Decoder-only validation: PASS`. If all candidates fall back, the audit reports zero generated dispatches; that does not establish successful generated-kernel integration on hardware.

### Benchmark and ablation

- Uses two warmup runs and ten measured runs, with alternating template/generated order for the Metal decoder comparison.
- Reports prefill and decode p50/p90, GPU command time, CPU submit-to-completion time, end-to-end time, tokens/s, and fixed KV storage.
- Compares template-only and generated-enabled decoder plans and prints a measured-only runtime usage audit.
- Runs the PyTorch reference on MPS with synchronized timing over the same logits/token readback and autoregressive token-feedback scope; Metal GPU timestamps and MPS synchronized wall timing are reported separately.
- Advisor ablation benchmarks deterministic exhaustive search and LLM-guided Top-K plans. The regression command covers local compiler/runtime paths without contacting either remote Workflow.

```bash
./run.sh benchmark
./run.sh ablation
./run.sh regression
```

### Chunked prefill and paged KV cache

- Adds a single-request paged KV layout with fixed token pages, a logical-to-physical block table, incremental page allocation, reset, and safe capacity rejection.
- Compiles reusable Metal append and causal-attention kernels that translate logical token positions through the block table.
- Runs the decoder prefill in three-token chunks across four-token page boundaries, then continues single-token decode on the same cache.
- Reconstructs logical K/V prefixes for validation against the existing PyTorch reference and checks full-capacity overflow without changing cache state.

```bash
./run.sh paged-kv
```

This stage intentionally covers one request. Continuous batching, preemption, and multi-request page scheduling remain in the following serving stage.

AgentCompile evaluates CUDA/A800 mechanisms. This project evaluates the corresponding compiler and runtime principles on Apple M3 Pro, Metal, and unified memory; the reported measurements describe these Metal implementations.
