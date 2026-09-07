#pragma once

#include "backend/metal/MetalRuntime.hpp"

#include <string>

namespace tensor::llm {

[[nodiscard]] std::string
writeSiLUMulKernelContract(const metal::MetalRuntime &runtime,
                          const std::string &path);

} // namespace tensor::llm
