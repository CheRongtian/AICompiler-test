# AICompiler-test

## Project Structure

```
AICompiler/
├── apps/
│   ├── main.cpp
│   ├── pytorch_import_main.cpp
│   ├── pytorch_import_main.hpp
│   ├── tensor_graph_examples.cpp
│   ├── tensor_graph_examples.hpp
│   └── tensor_metal_main.cpp
├── src/
│   ├── analyzer/
│   │   ├── GraphAnalyzer.*
│   │   └── PatternAnalyzer.*
│   ├── backend/metal/
│   │   ├── FusionMetalEmitter.cpp
│   │   ├── GraphMetalEmitter.cpp
│   │   ├── MetalEmitter.*
│   │   ├── MetalRuntime.*
│   │   └── RMSNormBaseline.cpp
│   ├── benchmark/
│   │   └── Benchmark.*
│   ├── importer/
│   │   └── PyTorchImporter.*
│   ├── planner/
│   │   ├── KernelPlan.*
│   │   ├── RegionPlan.*
│   │   └── RMSNormTuner.*
│   ├── runtime/
│   │   └── GraphExecutor.*
│   ├── tensor/
│   │   └── TensorIR.*
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
│   └── transformer_kv_benchmark.py
├── tools/
│   └── export_pytorch.py
├── docs/papers/
│   └── 2606.07665v2.pdf
├── tests/models/
│   │     ├── test_linear.py
│   │     ├── test_mlp.py
│   │     └── test_model.py
│   └── test.sh
├── CMakeLists.txt
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
