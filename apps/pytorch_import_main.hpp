#pragma once

#include "backend/metal/MetalRuntime.hpp"

#include <cstddef>
#include <iosfwd>
#include <string>

struct PyTorchAdvisorOptions {
  std::string requestOutputPath;
  std::string responsePath;
  std::size_t benchmarkWarmupRuns = 0;
  std::size_t benchmarkMeasuredRuns = 0;
  std::string benchmarkLabel;
};

bool runImportedPyTorchGraph(tensor::metal::MetalRuntime &runtime,
                             const std::string &manifestPath,
                             std::ostream &log,
                             const PyTorchAdvisorOptions &advisor = {});
