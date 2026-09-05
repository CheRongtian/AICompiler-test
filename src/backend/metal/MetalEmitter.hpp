#pragma once

#include "tensor/TensorIR.hpp"

#include <cstddef>
#include <string>

namespace tensor::metal {

struct GeneratedKernel {
  std::string source;
  std::string functionName;
  std::size_t threadgroupCount = 0;
  std::size_t threadsPerThreadgroup = 0;
};

// Buffer interface: 0 = input, 1 = weight, 2 = output (all fp32).
// Shape and epsilon are specialized into the source. No runtime constants.
[[nodiscard]] GeneratedKernel emitRMSNorm(const RMSNormOp &op);

} // namespace tensor::metal
