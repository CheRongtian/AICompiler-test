#pragma once

#include "analyzer/GraphAnalyzer.hpp"

namespace tensor {

namespace validation {

struct GraphReference {
  // fp32 intermediates are also the representative inputs for RMSNorm autotuning.
  std::vector<std::vector<float>> values;
  std::vector<std::vector<double>> outputs;
};

[[nodiscard]] GraphReference evaluateGraph(const analyzer::AnalyzedGraph &graph,
                                           const GraphInputs &inputs);

} // namespace validation
} // namespace tensor
