#pragma once

#include "tensor/TensorIR.hpp"

namespace tensor::analyzer {

struct AnalyzedGraph {
  TensorGraph graph;
  std::vector<TensorType> types; // Indexed by ValueId.
  std::vector<std::size_t> order; // Topological node indices.
};

[[nodiscard]] std::vector<std::size_t>
broadcastShape(const std::vector<std::size_t> &a,
               const std::vector<std::size_t> &b);
[[nodiscard]] AnalyzedGraph analyze(const TensorGraph &graph);

} // namespace tensor::analyzer
