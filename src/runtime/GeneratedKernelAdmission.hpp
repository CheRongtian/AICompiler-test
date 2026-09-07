#pragma once

#include "backend/metal/MetalRuntime.hpp"

#include <iosfwd>
#include <string>

namespace tensor::runtime {

[[nodiscard]] bool admitGeneratedSiLUMulKernel(
    metal::MetalRuntime &runtime, const std::string &responsePath,
    const std::string &feedbackPath, std::ostream &log);

} // namespace tensor::runtime
