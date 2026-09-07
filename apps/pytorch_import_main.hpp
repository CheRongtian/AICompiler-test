#pragma once

#include "backend/metal/MetalRuntime.hpp"

#include <iosfwd>
#include <string>

struct PyTorchAdvisorOptions {
  std::string requestOutputPath;
  std::string responsePath;
};

bool runImportedPyTorchGraph(tensor::metal::MetalRuntime &runtime,
                             const std::string &manifestPath,
                             std::ostream &log,
                             const PyTorchAdvisorOptions &advisor = {});
