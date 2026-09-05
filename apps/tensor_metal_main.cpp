#include "backend/metal/MetalRuntime.hpp"
#include "planner/RMSNormTuner.hpp"
#include "tensor/TensorIR.hpp"
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

bool runRMSNormCase(tensor::metal::MetalRuntime &runtime,
                    std::size_t rows, std::size_t width) {
  const tensor::RMSNormOp op{
      {{rows, width}, tensor::DType::Float32},
      {{width}, tensor::DType::Float32},
      {{rows, width}, tensor::DType::Float32},
      1e-5f};

  std::cout << "RMSNorm shape=[" << rows << ", " << width << "]"
            << ", dtype=fp32, epsilon=" << op.epsilon << '\n';
  std::vector<float> input(op.input.elementCount());
  std::vector<float> weight(op.weight.elementCount());
  for (std::size_t column = 0; column < width; ++column) {
    weight[column] = 0.5f + static_cast<float>(column % 31) / 32.0f;
  }
  for (std::size_t row = 0; row < rows; ++row) {
    // The multi-row case covers zero, epsilon-dominated, and ordinary inputs.
    const float scale = rows == 1 ? 1.0f : (row == 0 ? 0.0f : (row == 1 ? 1e-4f : 1.0f));
    for (std::size_t column = 0; column < width; ++column) {
      const int centered = static_cast<int>((column * 17 + row * 13) % 257) - 128;
      input[row * width + column] = static_cast<float>(centered) / 64.0f * scale;
    }
  }

  const auto result = tensor::planner::tuneRMSNorm(runtime, op, input, weight, std::cout);
  if (!result.success) {
    std::cerr << "RMSNorm tuning failed: " << result.errorMessage << '\n';
    return false;
  }
  if (result.finalExecution.gpuExecutionTimeUs) {
    std::cout << "Final GPU command buffer time (us): "
              << *result.finalExecution.gpuExecutionTimeUs << '\n';
  } else {
    std::cout << "Final GPU command buffer time (us): Unavailable\n";
  }
  std::cout << "Final CPU submit-to-completion time (us): "
            << result.finalExecution.cpuSubmitToCompletionTimeUs << '\n';
  return true;
}

} // namespace

int main() {
  try {
    tensor::metal::MetalRuntime runtime;
    const std::string deviceName = runtime.deviceName();
    std::cout << "Metal device: "
              << (deviceName.empty() ? "Unavailable" : deviceName) << '\n';

    const auto pipeline = runtime.createComputePipeline(kVectorAddShader, "vector_add");
    if (!printPipelineResult(pipeline)) {
      return 1;
    }

    const bool addAligned = runVectorAddCase(runtime, 4096, pipeline.threadExecutionWidth);
    const bool addTail = runVectorAddCase(runtime, 4097, pipeline.threadExecutionWidth);
    const bool normAligned = runRMSNormCase(runtime, 1, 4096);
    const bool normTail = runRMSNormCase(runtime, 3, 4097);
    return addAligned && addTail && normAligned && normTail ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "TensorMetalCompiler error: " << error.what() << '\n';
    return 1;
  }
}
