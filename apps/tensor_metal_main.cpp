#include "backend/metal/MetalRuntime.hpp"
#include "backend/metal/MetalEmitter.hpp"
#include "tensor/TensorIR.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
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

  if (result.output.size() != reference.size()) {
    std::cerr << "Numerical validation: FAIL (unexpected output length)\n";
    return false;
  }

  double maxAbsoluteError = 0.0;
  std::size_t mismatchCount = 0;
  std::size_t firstMismatch = reference.size();
  for (std::size_t i = 0; i < reference.size(); ++i) {
    const double actual = result.output[i];
    const double expected = reference[i];
    const bool finite = std::isfinite(actual) && std::isfinite(expected);
    const double error = finite
                            ? std::abs(actual - expected)
                            : std::numeric_limits<double>::infinity();
    maxAbsoluteError = std::max(maxAbsoluteError, error);
    if (!finite || error > absoluteTolerance + relativeTolerance * std::abs(expected)) {
      if (mismatchCount == 0) {
        firstMismatch = i;
      }
      ++mismatchCount;
    }
  }

  std::cout << "Tolerance: atol=" << absoluteTolerance
            << ", rtol=" << relativeTolerance << '\n';
  std::cout << "Max absolute error: " << maxAbsoluteError << '\n';
  std::cout << "Numerical validation: " << passFail(mismatchCount == 0) << '\n';
  if (mismatchCount != 0) {
    std::cerr << "Mismatches: " << mismatchCount
              << "; first index: " << firstMismatch
              << "; expected: " << reference[firstMismatch]
              << "; actual: " << result.output[firstMismatch] << '\n';
  }
  return mismatchCount == 0;
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

std::vector<double> rmsNormReference(const tensor::RMSNormOp &op,
                                     const std::vector<float> &input,
                                     const std::vector<float> &weight) {
  const std::size_t width = op.input.shape.back();
  const std::size_t rows = op.input.elementCount() / width;
  std::vector<double> reference(input.size());
  for (std::size_t row = 0; row < rows; ++row) {
    double sumSquares = 0.0;
    for (std::size_t column = 0; column < width; ++column) {
      const double x = input[row * width + column];
      sumSquares += x * x;
    }
    const double inverseRms =
        1.0 / std::sqrt(sumSquares / static_cast<double>(width) + op.epsilon);
    for (std::size_t column = 0; column < width; ++column) {
      reference[row * width + column] =
          static_cast<double>(input[row * width + column]) * inverseRms *
          static_cast<double>(weight[column]);
    }
  }
  return reference;
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
  const auto kernel = tensor::metal::emitRMSNorm(op);
  std::cout << "TensorIR -> MSL: PASS\n";
  std::cout << "Threadgroups: " << kernel.threadgroupCount
            << ", threads per threadgroup: " << kernel.threadsPerThreadgroup << '\n';
  if (!printPipelineResult(runtime.createComputePipeline(kernel.source,
                                                        kernel.functionName))) {
    return false;
  }

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

  const auto reference = rmsNormReference(op, input, weight);
  const auto result = runtime.run(
      {{input.data(), input.size()}, {weight.data(), weight.size()}},
      op.output.elementCount(),
      {kernel.threadgroupCount, kernel.threadsPerThreadgroup});
  return validateOutput(result, reference, 1e-5, 1e-4);
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
