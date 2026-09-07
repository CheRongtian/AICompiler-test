#pragma once

#include "backend/metal/MetalRuntime.hpp"
#include "planner/RegionPlan.hpp"

#include <iosfwd>

namespace tensor::runtime {

struct GraphExecutionResult {
  bool passed = false;
  std::vector<std::vector<float>> outputs;
  std::optional<double> gpuExecutionTimeUs;
  std::string errorMessage;
};

class CompiledGraph {
public:
  struct ExecutionUnit {
    std::string name;
    std::unique_ptr<metal::PreparedSequence> execution;
  };

  CompiledGraph(planner::ProgramPlan plan, std::vector<metal::BufferHandle> buffers,
                std::vector<ExecutionUnit> units);
  [[nodiscard]] GraphExecutionResult run() const;

private:
  planner::ProgramPlan plan_;
  std::vector<metal::BufferHandle> buffers_;
  std::vector<ExecutionUnit> units_;
};

struct GraphCompilation {
  std::unique_ptr<CompiledGraph> executable;
  std::vector<TensorType> outputTypes;
  std::vector<std::vector<double>> referenceOutputs;
  std::string errorMessage;
};

[[nodiscard]] GraphCompilation compileGraph(metal::MetalRuntime &runtime,
                                            const TensorGraph &graph,
                                            const GraphInputs &inputs,
                                            std::ostream &log);

} // namespace tensor::runtime
