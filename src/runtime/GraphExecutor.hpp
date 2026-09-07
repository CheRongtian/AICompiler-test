#pragma once

#include "backend/metal/MetalRuntime.hpp"
#include "planner/KernelPlan.hpp"

#include <iosfwd>

namespace tensor::runtime {

struct GraphExecutionResult {
  bool passed = false;
  std::vector<std::vector<float>> outputs;
  // Sum of per-node GPU command-buffer durations, excluding gaps between nodes.
  std::optional<double> gpuExecutionTimeUs;
  std::string errorMessage;
};

class CompiledGraph {
public:
  CompiledGraph(planner::GraphPlan plan, std::vector<metal::BufferHandle> buffers,
                std::vector<std::unique_ptr<metal::PreparedExecution>> steps);
  [[nodiscard]] GraphExecutionResult run() const;

private:
  planner::GraphPlan plan_;
  std::vector<metal::BufferHandle> buffers_;
  std::vector<std::unique_ptr<metal::PreparedExecution>> steps_;
};

struct GraphCompilation {
  std::unique_ptr<CompiledGraph> executable;
  std::vector<TensorType> outputTypes;
  std::vector<std::vector<double>> referenceOutputs;
  std::string errorMessage;
};

// Compiles for these static input tensors. RMSNorm uses V1 autotuning on CPU
// reference intermediates; graph execution binds the actual GPU intermediates.
[[nodiscard]] GraphCompilation compileGraph(metal::MetalRuntime &runtime,
                                            const TensorGraph &graph,
                                            const GraphInputs &inputs,
                                            std::ostream &log);

} // namespace tensor::runtime
