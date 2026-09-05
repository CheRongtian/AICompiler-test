cmake_minimum_required(VERSION 3.17)

project(AICompiler LANGUAGES C CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

llvm_map_components_to_libnames(llvm_libs
  native
  Core
  Support
  Passes
  InstCombine
  ScalarOpts
  TransformUtils
  Analysis
  ExecutionEngine
  OrcJIT
  Object
)

list(REMOVE_DUPLICATES llvm_libs)

add_executable(KaleidoscopeExercise
  apps/main.cpp
  src/lexer.cpp
  src/parser.cpp
  src/TopLevel.cpp
  src/codegen.cpp
  src/JITLib.cpp
  src/AST.hpp
  src/lexer.hpp
  src/parser.hpp
  src/TopLevel.hpp
  src/codegen.hpp
  src/JITLib.hpp
  src/KaleidoscopeJIT.hpp
)

target_include_directories(KaleidoscopeExercise PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_libraries(KaleidoscopeExercise PRIVATE ${llvm_libs})
target_link_options(KaleidoscopeExercise PRIVATE
  "-Wl,-no_warn_duplicate_libraries"
)

find_library(METAL_FRAMEWORK Metal)
find_library(FOUNDATION_FRAMEWORK Foundation)

if(NOT METAL_FRAMEWORK OR NOT FOUNDATION_FRAMEWORK)
  message(FATAL_ERROR "Metal.framework and Foundation.framework are required")
endif()

add_executable(TensorMetalCompiler
  apps/tensor_metal_main.cpp
  src/tensor/TensorIR.hpp
  src/tensor/TensorIR.cpp
  src/backend/metal/MetalEmitter.hpp
  src/backend/metal/MetalEmitter.cpp
  src/backend/metal/RMSNormBaseline.cpp
  src/backend/metal/MetalRuntime.hpp
  src/backend/metal/MetalRuntime.mm
  src/validation/Validator.hpp
  src/validation/Validator.cpp
  src/benchmark/Benchmark.hpp
  src/benchmark/Benchmark.cpp
  src/planner/RMSNormTuner.hpp
  src/planner/RMSNormTuner.cpp
)

target_include_directories(TensorMetalCompiler PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_options(TensorMetalCompiler PRIVATE
  $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>
)

target_link_libraries(TensorMetalCompiler PRIVATE
  ${METAL_FRAMEWORK}
  ${FOUNDATION_FRAMEWORK}
)
