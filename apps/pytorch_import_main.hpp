#pragma once

#include "backend/metal/MetalRuntime.hpp"

#include <iosfwd>
#include <string>

bool runImportedPyTorchGraph(tensor::metal::MetalRuntime &runtime,
                             const std::string &manifestPath,
                             std::ostream &log);
