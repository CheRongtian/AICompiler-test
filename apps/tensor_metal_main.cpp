#include "backend/metal/MetalRuntime.hpp"
#include "decoder_llm_main.hpp"
#include "generated_kernel_main.hpp"
#include "kv_cache_main.hpp"
#include "pytorch_import_main.hpp"
#include "tensor_graph_examples.hpp"
#include "transformer_decode_main.hpp"
#include "validation/Validator.hpp"

#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr const char *kVectorAddShader = R"metal(
#include <metal_stdlib>
using namespace metal;

kernel void vector_add(device const float *lhs [[buffer(0)]],
                       device const float *rhs [[buffer(1)]],
                       device float *output [[buffer(2)]],
                       constant uint &element_count [[buffer(3)]],
                       uint index [[thread_position_in_grid]]) {
  if (index < element_count) {
    output[index] = lhs[index] + rhs[index];
  }
}
)metal";

const char *passFail(bool passed) {
  return passed ? "PASS" : "FAIL";
}

bool printPipelineResult(const tensor::metal::ComputePipelineResult &result) {
  std::cout << "Library compile: " << passFail(result.libraryCompilePassed) << '\n';
  std::cout << "Kernel lookup: " << passFail(result.kernelLookupPassed) << '\n';
  std::cout << "Pipeline creation: " << passFail(result.pipelineCreationPassed) << '\n';
  std::cout << "threadExecutionWidth: " << result.threadExecutionWidth << '\n';
  std::cout << "maxTotalThreadsPerThreadgroup: "
            << result.maxTotalThreadsPerThreadgroup << '\n';
  if (!result.errorMessage.empty()) {
    std::cerr << "Metal error: " << result.errorMessage << '\n';
  }
  return result.pipelineCreationPassed;
}

bool validateOutput(const tensor::metal::ExecutionResult &result,
                    const std::vector<double> &reference,
                    double absoluteTolerance, double relativeTolerance) {
  std::cout << "GPU execution: " << passFail(result.executionPassed) << '\n';
  if (!result.executionPassed) {
    std::cerr << "Metal error: " << result.errorMessage << '\n';
    return false;
  }

  if (result.gpuExecutionTimeUs.has_value()) {
    std::cout << "GPU command buffer time (us): "
              << *result.gpuExecutionTimeUs << '\n';
  } else {
    std::cout << "GPU command buffer time (us): Unavailable "
                 "(Metal returned missing or invalid timestamps)\n";
  }
  std::cout << "CPU submit-to-completion time (us): "
            << result.cpuSubmitToCompletionTimeUs << '\n';

  const auto validation = tensor::validation::compare(
      result.output, reference, absoluteTolerance, relativeTolerance);
  std::cout << "Tolerance: atol=" << absoluteTolerance
            << ", rtol=" << relativeTolerance << '\n';
  std::cout << "Max absolute error: " << validation.maxAbsoluteError << '\n';
  std::cout << "Numerical validation: " << passFail(validation.passed) << '\n';
  if (!validation.passed) {
    std::cerr << "Validation error: " << validation.errorMessage << '\n';
  }
  return validation.passed;
}

bool runVectorAddCase(const tensor::metal::MetalRuntime &runtime,
                      std::size_t elementCount, std::size_t threadsPerGroup) {
  std::vector<float> lhs(elementCount);
  std::vector<float> rhs(elementCount);
  std::vector<double> reference(elementCount);
  for (std::size_t i = 0; i < elementCount; ++i) {
    lhs[i] = static_cast<float>(static_cast<int>(i % 257) - 128) * 0.25f;
    rhs[i] = static_cast<float>(static_cast<int>(i % 113) - 56) * 0.125f;
    reference[i] = static_cast<double>(lhs[i]) + static_cast<double>(rhs[i]);
  }

  std::cout << "VectorAdd N=" << elementCount << '\n';
  const tensor::metal::DispatchSize dispatch{
      (elementCount + threadsPerGroup - 1) / threadsPerGroup, threadsPerGroup};
  const auto result = runtime.run(
      {{lhs.data(), lhs.size()}, {rhs.data(), rhs.size()}}, elementCount,
      dispatch, {static_cast<std::uint32_t>(elementCount)});
  return validateOutput(result, reference, 1e-6, 1e-6);
}


} // namespace

int main(int argc, char **argv) {
  try {
    tensor::metal::MetalRuntime runtime;
    const std::string deviceName = runtime.deviceName();
    std::cout << "Metal device: "
              << (deviceName.empty() ? "Unavailable" : deviceName) << '\n';

    if (argc == 3 && std::string(argv[1]) == "--import-pytorch") {
      return runImportedPyTorchGraph(runtime, argv[2], std::cout) ? 0 : 1;
    }
    if (argc == 5 && std::string(argv[1]) == "--import-pytorch" &&
        std::string(argv[3]) == "--emit-advisor-request") {
      PyTorchAdvisorOptions advisor;
      advisor.requestOutputPath = argv[4];
      return runImportedPyTorchGraph(runtime, argv[2], std::cout, advisor) ? 0 : 1;
    }
    if (argc == 5 && std::string(argv[1]) == "--import-pytorch" &&
        std::string(argv[3]) == "--advisor-response") {
      PyTorchAdvisorOptions advisor;
      advisor.responsePath = argv[4];
      return runImportedPyTorchGraph(runtime, argv[2], std::cout, advisor) ? 0 : 1;
    }
    if (argc == 3 && std::string(argv[1]) == "--kv-cache") {
      return runKVCacheWorkload(runtime, argv[2], std::cout) ? 0 : 1;
    }
    if (argc == 3 && std::string(argv[1]) == "--transformer-decode") {
      return runTransformerDecodeWorkload(runtime, argv[2], std::cout) ? 0 : 1;
    }
    if (argc == 3 && std::string(argv[1]) == "--decoder-llm") {
      return runDecoderLLMWorkload(runtime, argv[2], std::cout) ? 0 : 1;
    }
    if (argc == 5 && std::string(argv[1]) == "--decoder-llm" &&
        std::string(argv[3]) == "--kernel-library") {
      return runDecoderLLMWorkload(runtime, argv[2], std::cout, argv[4]) ? 0 : 1;
    }
    if (argc == 3 && std::string(argv[1]) == "--emit-kernel-contract") {
      return emitGeneratedKernelContract(runtime, argv[2], std::cout) ? 0 : 1;
    }
    if (argc == 5 && std::string(argv[1]) == "--emit-kernel-contract" &&
        std::string(argv[3]) == "--pattern") {
      return emitGeneratedKernelContract(runtime, argv[2], std::cout, argv[4]) ? 0 : 1;
    }
    if (argc == 7 && std::string(argv[1]) == "--admit-generated-kernel" &&
        std::string(argv[3]) == "--feedback-output" &&
        std::string(argv[5]) == "--pattern") {
      return runGeneratedKernelAdmission(runtime, argv[2], argv[4], std::cout, argv[6]) ? 0 : 1;
    }
    if (argc == 9 && std::string(argv[1]) == "--admit-generated-kernel" &&
        std::string(argv[3]) == "--feedback-output" &&
        std::string(argv[5]) == "--pattern" &&
        std::string(argv[7]) == "--artifact-output") {
      return runGeneratedKernelAdmission(runtime, argv[2], argv[4], std::cout,
                                          argv[6], argv[8]) ? 0 : 1;
    }
    if (argc == 5 && std::string(argv[1]) == "--admit-generated-kernel" &&
        std::string(argv[3]) == "--feedback-output") {
      return runGeneratedKernelAdmission(runtime, argv[2], argv[4], std::cout)
                 ? 0
                 : 1;
    }
    if (argc != 1) {
      std::cerr << "Usage: " << argv[0]
                << " [--import-pytorch <graph-manifest>]"
                   " [--import-pytorch <graph-manifest> --emit-advisor-request <json>]"
                   " [--import-pytorch <graph-manifest> --advisor-response <json>]"
                   " [--kv-cache <cache-manifest>]"
                   " [--transformer-decode <decoder-manifest>]"
                   " [--decoder-llm <decoder-manifest> [--kernel-library <directory>]]"
                   " [--emit-kernel-contract <json> [--pattern <pattern>]]"
                   " [--admit-generated-kernel <json> --feedback-output <json>"
                   " --pattern <pattern> [--artifact-output <json>]]\n";
      return 1;
    }

    const auto pipeline = runtime.createComputePipeline(kVectorAddShader, "vector_add");
    if (!printPipelineResult(pipeline)) {
      return 1;
    }

    const bool addAligned = runVectorAddCase(runtime, 4096, pipeline.threadExecutionWidth);
    const bool addTail = runVectorAddCase(runtime, 4097, pipeline.threadExecutionWidth);
    const bool graphsPassed = runTensorGraphExamples(runtime, std::cout);
    return addAligned && addTail && graphsPassed ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "TensorMetalCompiler error: " << error.what() << '\n';
    return 1;
  }
}
