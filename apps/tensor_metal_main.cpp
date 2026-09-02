#include "backend/metal/MetalRuntime.hpp"

#include <iostream>
#include <string>

namespace {

constexpr const char *kVectorAddShader = R"metal(
#include <metal_stdlib>
using namespace metal;

kernel void vector_add(device const float *lhs [[buffer(0)]],
                       device const float *rhs [[buffer(1)]],
                       device float *output [[buffer(2)]],
                       uint index [[thread_position_in_grid]]) {
  output[index] = lhs[index] + rhs[index];
}
)metal";

const char *passFail(bool passed) {
  return passed ? "PASS" : "FAIL";
}

} // namespace

int main() {
  tensor::metal::MetalRuntime runtime;

  const std::string deviceName = runtime.deviceName();
  std::cout << "Metal device: "
            << (deviceName.empty() ? "Unavailable" : deviceName) << '\n';

  const tensor::metal::ComputePipelineResult result =
      runtime.createComputePipeline(kVectorAddShader, "vector_add");

  std::cout << "Library compile: " << passFail(result.libraryCompilePassed)
            << '\n';
  std::cout << "Kernel lookup: " << passFail(result.kernelLookupPassed) << '\n';
  std::cout << "Pipeline creation: "
            << passFail(result.pipelineCreationPassed) << '\n';
  std::cout << "threadExecutionWidth: " << result.threadExecutionWidth << '\n';
  std::cout << "maxTotalThreadsPerThreadgroup: "
            << result.maxTotalThreadsPerThreadgroup << '\n';

  if (!result.errorMessage.empty()) {
    std::cerr << "Metal error: " << result.errorMessage << '\n';
  }

  return runtime.isAvailable() && result.libraryCompilePassed &&
                 result.kernelLookupPassed && result.pipelineCreationPassed
             ? 0
             : 1;
}
