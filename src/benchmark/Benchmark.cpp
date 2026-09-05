#include "benchmark/Benchmark.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace tensor::benchmark {
namespace {

Statistics summarize(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t count = values.size();
  const double median = count % 2 == 0
                            ? (values[count / 2 - 1] + values[count / 2]) / 2.0
                            : values[count / 2];
  return {count, median, values.front(), values.back()};
}

std::string sample(const metal::PreparedExecution &execution,
                    std::vector<double> &values) {
  const auto result = execution.measure();
  if (!result.executionPassed) {
    return "GPU execution failed: " + result.errorMessage;
  }
  if (!result.gpuExecutionTimeUs || !std::isfinite(*result.gpuExecutionTimeUs) ||
      *result.gpuExecutionTimeUs <= 0.0) {
    return "GPU timestamps unavailable; no performance comparison is possible.";
  }
  values.push_back(*result.gpuExecutionTimeUs);
  return {};
}

} // namespace

std::string warmup(const metal::PreparedExecution &execution, std::size_t iterations) {
  for (std::size_t i = 0; i < iterations; ++i) {
    const auto result = execution.measure();
    if (!result.executionPassed) {
      return result.errorMessage;
    }
  }
  return {};
}

Result measure(const metal::PreparedExecution &execution, std::size_t samples) {
  Result result;
  if (samples == 0) {
    result.errorMessage = "Benchmark requires at least one sample.";
    return result;
  }
  std::vector<double> values;
  values.reserve(samples);
  for (std::size_t i = 0; i < samples; ++i) {
    result.errorMessage = sample(execution, values);
    if (!result.errorMessage.empty()) {
      return result;
    }
  }
  result.stats = summarize(values);
  result.passed = true;
  return result;
}

PairedResult measurePair(const metal::PreparedExecution &baseline,
                         const metal::PreparedExecution &candidate,
                         std::size_t samples) {
  PairedResult result;
  if (samples == 0) {
    result.errorMessage = "Benchmark requires at least one sample.";
    return result;
  }
  std::vector<double> baselineTimes;
  std::vector<double> candidateTimes;
  baselineTimes.reserve(samples);
  candidateTimes.reserve(samples);
  for (std::size_t i = 0; i < samples; ++i) {
    for (std::size_t position = 0; position < 2; ++position) {
      const bool runBaseline = (i + position) % 2 == 0;
      const std::string error = runBaseline ? sample(baseline, baselineTimes)
                                            : sample(candidate, candidateTimes);
      if (!error.empty()) {
        result.errorMessage = (runBaseline ? "Baseline: " : "Candidate: ") + error;
        return result;
      }
    }
  }
  result.baseline = summarize(baselineTimes);
  result.candidate = summarize(candidateTimes);
  result.speedup = result.baseline.medianUs / result.candidate.medianUs;
  result.passed = true;
  return result;
}

} // namespace tensor::benchmark
