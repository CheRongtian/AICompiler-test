#include "planner/DecodeGEMVTuner.hpp"

#include "benchmark/Benchmark.hpp"
#include "validation/Validator.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <utility>

namespace tensor::planner {
namespace {

constexpr std::size_t kWarmup = 5;
constexpr std::size_t kSamples = 31;
constexpr double kMinimumSpeedup = 1.05;

std::vector<double> reference(std::size_t batch, std::size_t inputSize,
                              std::size_t outputSize,
                              const std::vector<float> &input,
                              const std::vector<float> &weight) {
  std::vector<double> output(batch * outputSize);
  for (std::size_t row = 0; row < batch; ++row) {
    for (std::size_t feature = 0; feature < outputSize; ++feature) {
      double sum = 0.0;
      for (std::size_t inner = 0; inner < inputSize; ++inner) {
        sum += static_cast<double>(input[row * inputSize + inner]) *
               static_cast<double>(weight[feature * inputSize + inner]);
      }
      output[row * outputSize + feature] = sum;
    }
  }
  return output;
}

std::unique_ptr<metal::PreparedExecution> prepare(
    metal::MetalRuntime &runtime, const metal::GeneratedKernel &kernel,
    const std::vector<float> &input, const std::vector<float> &weight,
    std::size_t outputCount, const std::vector<double> &expected,
    std::string &error) {
  const auto pipeline =
      runtime.createComputePipeline(kernel.source, kernel.functionName);
  if (!pipeline.pipelineCreationPassed) {
    error = pipeline.errorMessage;
    return {};
  }
  error = metal::checkBufferInterface(
      pipeline, {metal::ElementType::Float32, metal::ElementType::Float32},
      metal::ElementType::Float32);
  if (!error.empty()) return {};
  auto prepared = runtime.prepare(
      {{input.data(), input.size()}, {weight.data(), weight.size()}}, outputCount,
      {kernel.threadgroupCount, kernel.threadsPerThreadgroup});
  if (!prepared.execution) {
    error = prepared.errorMessage;
    return {};
  }
  const auto execution = prepared.execution->run();
  if (!execution.executionPassed) {
    error = execution.errorMessage;
    return {};
  }
  const auto validation =
      validation::compare(execution.output, expected, 2e-4, 2e-3);
  if (!validation.passed) {
    error = validation.errorMessage;
    return {};
  }
  return std::move(prepared.execution);
}

} // namespace

DecodeGEMVSelection tuneDecodeGEMV(
    metal::MetalRuntime &runtime, std::size_t batch, std::size_t inputSize,
    std::size_t outputSize, const std::vector<float> &input,
    const std::vector<float> &weight, std::ostream &log) {
  DecodeGEMVSelection result;
  if (input.size() != batch * inputSize ||
      weight.size() != outputSize * inputSize) {
    result.errorMessage = "Decode GEMV tuning inputs do not match the shape.";
    return result;
  }
  const auto expected = reference(batch, inputSize, outputSize, input, weight);
  const auto baselineKernel = metal::emitLinearBaseline(
      batch, inputSize, outputSize, 128, "decode_gemv_baseline");
  std::string error;
  auto baseline = prepare(runtime, baselineKernel, input, weight,
                          batch * outputSize, expected, error);
  if (!baseline) {
    result.errorMessage = "Decode GEMV baseline failed: " + error;
    return result;
  }
  const auto warmup = benchmark::warmup(*baseline, kWarmup);
  if (!warmup.empty()) {
    result.errorMessage = "Decode GEMV baseline warmup failed: " + warmup;
    return result;
  }
  log << "Decode GEMV shape M=" << batch << ", K=" << inputSize
      << ", N=" << outputSize
      << " | baseline validation/warmup: PASS\n";

  const std::vector<metal::DecodeGEMVConfig> candidates{
      {32, 1}, {64, 1}, {128, 1}, {32, 4}, {64, 4}, {128, 4}};
  double bestSpeedup = 1.0;
  for (const auto candidateConfig : candidates) {
    log << "  candidate threads=" << candidateConfig.threads
        << ", vector_width=" << candidateConfig.vectorWidth;
    const auto hardware = runtime.hardwareInfo();
    if (candidateConfig.threads > hardware.maxThreadsPerThreadgroup ||
        candidateConfig.threads * sizeof(float) >
            hardware.maxThreadgroupMemoryLength ||
        inputSize % candidateConfig.vectorWidth != 0) {
      log << " | Hardware Filter: FAIL\n";
      continue;
    }
    metal::GeneratedKernel kernel;
    try {
      kernel = metal::emitDecodeGEMV(
          batch, inputSize, outputSize, candidateConfig,
          "decode_gemv_" + std::to_string(candidateConfig.threads) + "_v" +
              std::to_string(candidateConfig.vectorWidth));
    } catch (const std::exception &exception) {
      log << " | Emit: FAIL: " << exception.what() << '\n';
      continue;
    }
    auto candidate = prepare(runtime, kernel, input, weight,
                             batch * outputSize, expected, error);
    if (!candidate) {
      log << " | Compile/interface/numerical: FAIL: " << error << '\n';
      continue;
    }
    const auto candidateWarmup = benchmark::warmup(*candidate, kWarmup);
    if (!candidateWarmup.empty()) {
      log << " | Warmup: FAIL: " << candidateWarmup << '\n';
      continue;
    }
    const auto first = benchmark::measurePair(*baseline, *candidate, kSamples);
    if (!first.passed) {
      log << " | Benchmark: FAIL: " << first.errorMessage << '\n';
      continue;
    }
    log << " | validation: PASS, speedup=" << first.speedup << 'x';
    if (first.speedup < kMinimumSpeedup) {
      log << " -> fallback\n";
      continue;
    }
    const auto confirmation =
        benchmark::measurePair(*baseline, *candidate, kSamples);
    if (!confirmation.passed) {
      log << ", confirmation: FAIL: " << confirmation.errorMessage << '\n';
      continue;
    }
    const auto conservative = std::min(first.speedup, confirmation.speedup);
    log << ", confirmation=" << confirmation.speedup << 'x';
    if (conservative < kMinimumSpeedup) {
      log << " -> fallback\n";
      continue;
    }
    log << " -> admitted\n";
    if (conservative > bestSpeedup) {
      bestSpeedup = conservative;
      result.useBaseline = false;
      result.config = candidateConfig;
      result.conservativeSpeedup = conservative;
    }
  }
  result.success = true;
  if (result.useBaseline) {
    result.config = {128, 1};
    log << "Decode GEMV selected: generic Linear fallback\n";
  } else {
    log << "Decode GEMV selected: threads=" << result.config.threads
        << ", vector_width=" << result.config.vectorWidth
        << ", conservative speedup=" << result.conservativeSpeedup << "x\n";
  }
  return result;
}

} // namespace tensor::planner
