#pragma once

#include "backend/metal/MetalRuntime.hpp"

#include <iosfwd>
#include <string>

[[nodiscard]] bool runDecoderLLMWorkload(
    tensor::metal::MetalRuntime &runtime, const std::string &manifestPath,
    std::ostream &log);
