#pragma once

#include "backend/metal/MetalRuntime.hpp"

#include <iosfwd>
#include <string>

namespace tensor::runtime {

[[nodiscard]] bool admitGeneratedKernel(
    metal::MetalRuntime &runtime, const std::string &pattern, const std::string &responsePath,
    const std::string &feedbackPath, std::ostream &log,
    const std::string &artifactPath = {});

} // namespace tensor::runtime
