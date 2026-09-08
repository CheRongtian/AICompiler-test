#pragma once

#include "backend/metal/MetalRuntime.hpp"
#include "benchmark/Benchmark.hpp"
#include "tensor/DecoderLLM.hpp"

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace tensor::runtime {

struct ServingRequestSpec {
  std::string id;
  std::vector<float> promptTokenIds;
  std::size_t decodeTokenCount = 0;
  std::size_t arrivalStep = 0;
};

struct ServingConfig {
  std::size_t pageSize = 4;
  std::size_t physicalPagesPerLayer = 8;
  std::size_t prefillChunkSize = 3;
  std::string kernelLibrary;
};

struct ServingRequestOutput {
  std::string id;
  std::vector<float> prefillLogits;
  std::vector<std::vector<float>> decodeLogits;
  std::vector<float> generatedTokenIds;
  double timeToFirstTokenUs = 0.0;
};

struct ServingMetrics {
  benchmark::Statistics prefillGpu;
  benchmark::Statistics prefillEndToEnd;
  benchmark::Statistics decodeGpu;
  benchmark::Statistics decodeEndToEnd;
  double totalTimeUs = 0.0;
  double tokensPerSecond = 0.0;
  double schedulerOverheadUs = 0.0;
  double kvManagementOverheadUs = 0.0;
  double preemptionOverheadUs = 0.0;
  double averageActiveBatchSize = 0.0;
  std::size_t maximumActiveBatchSize = 0;
  std::size_t preemptionCount = 0;
  std::size_t resumeCount = 0;
  std::size_t peakPagesPerLayer = 0;
  std::size_t pageAllocationCount = 0;
  std::size_t pageReleaseCount = 0;
  std::size_t decodeCommandSubmissions = 0;
  std::size_t decodedTokens = 0;
};

struct ServingExecutionResult {
  bool passed = false;
  std::vector<ServingRequestOutput> requests;
  ServingMetrics metrics;
  std::string errorMessage;
};

[[nodiscard]] ServingExecutionResult runContinuousBatch(
    metal::MetalRuntime &runtime, const DecoderLLMWorkload &baseWorkload,
    const std::vector<ServingRequestSpec> &requests,
    const ServingConfig &config, std::ostream &log);

} // namespace tensor::runtime
