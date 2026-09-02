# AICompiler-test
## Project Structure
```
AICompiler/
├── apps/
│   ├── main.cpp
│   └── tensor_metal_main.cpp
├── src/
│   ├── backend/metal/
│   │   ├── MetalRuntime.hpp
│   │   └── MetalRuntime.mm
│   ├── AST.hpp
│   ├── lexer.*
│   ├── parser.*
│   ├── codegen.*
│   ├── TopLevel.* 
│   └── JITLib.*
├── workloads/pytorch/
├── docs/papers/
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
