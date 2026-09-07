#pragma once

#include "tensor/TensorIR.hpp"

#include <cstddef>
#include <string>

namespace tensor::planner { struct KernelPlan; }
namespace tensor::analyzer { struct Region; struct AnalyzedGraph; }

namespace tensor::metal {

struct GeneratedKernel {
  std::string source;
  std::string functionName;
  std::size_t threadgroupCount = 0;
  std::size_t threadsPerThreadgroup = 0;
};

// Buffer interface: 0 = input, 1 = weight, 2 = output (all fp32).
// Shape and epsilon are specialized into the source. No runtime constants.
[[nodiscard]] GeneratedKernel emitRMSNorm(const RMSNormOp &op,
                                        std::size_t threadsPerThreadgroup);

// Frozen V0b algorithm, always 256 threads. Performance fallback for V1.
[[nodiscard]] GeneratedKernel emitRMSNormBaseline(const RMSNormOp &op);

[[nodiscard]] GeneratedKernel emitKernel(const planner::KernelPlan &plan);
[[nodiscard]] GeneratedKernel emitFusion(const analyzer::Region &region,
                                         const analyzer::AnalyzedGraph &graph,
                                         std::size_t threadsPerThreadgroup);

} // namespace tensor::metal
