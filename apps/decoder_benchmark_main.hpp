#pragma once

#include "backend/metal/MetalRuntime.hpp"

#include <cstddef>
#include <iosfwd>
#include <string>

struct DecoderBenchmarkOptions {
  std::string kernelLibrary;
  std::size_t warmupRuns = 2;
  std::size_t measuredRuns = 10;
  bool tokenOnly = false;
  bool compareFusions = false;
};

[[nodiscard]] bool runDecoderLLMBenchmark(
    tensor::metal::MetalRuntime &runtime, const std::string &manifestPath,
    std::ostream &log, const DecoderBenchmarkOptions &options);
