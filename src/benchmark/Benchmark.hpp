#pragma once

#include "backend/metal/MetalRuntime.hpp"

#include <cstddef>
#include <string>

namespace tensor::benchmark {

struct Statistics {
  std::size_t samples = 0;
  double medianUs = 0.0;
  double minUs = 0.0;
  double maxUs = 0.0;
};

struct Result {
  bool passed = false;
  Statistics stats;
  std::string errorMessage;
};

struct PairedResult {
  bool passed = false;
  Statistics baseline;
  Statistics candidate;
  double speedup = 0.0;
  std::string errorMessage;
};

// Empty string means every warmup submission completed successfully.
[[nodiscard]] std::string warmup(const metal::PreparedExecution &execution,
                                 std::size_t iterations);
[[nodiscard]] Result measure(const metal::PreparedExecution &execution,
                              std::size_t samples);
// Alternate baseline/candidate submission order to reduce order and clock drift.
[[nodiscard]] PairedResult measurePair(const metal::PreparedExecution &baseline,
                                      const metal::PreparedExecution &candidate,
                                      std::size_t samples);

} // namespace tensor::benchmark
