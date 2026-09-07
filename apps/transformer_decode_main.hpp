#pragma once

#include "backend/metal/MetalRuntime.hpp"

#include <iosfwd>
#include <string>

[[nodiscard]] bool runTransformerDecodeWorkload(
    tensor::metal::MetalRuntime &runtime, const std::string &manifestPath,
    std::ostream &log);
