#pragma once

#include "tensor/TensorIR.hpp"

namespace tensor::analyzer {

enum class MemoryEffect { ReadOnly, View, MayAlias, Writes, Unknown };

struct AnalyzedGraph {
  TensorGraph graph;
  std::vector<TensorType> types;
  std::vector<std::size_t> order;
  std::vector<MemoryEffect> effects; // Indexed by node index.
  std::vector<std::optional<ValueId>> aliasOf; // Indexed by ValueId.
};

[[nodiscard]] std::vector<std::size_t>
broadcastShape(const std::vector<std::size_t> &a,
               const std::vector<std::size_t> &b);
[[nodiscard]] std::vector<std::size_t>
normalizeAxes(const std::vector<std::int64_t> &axes, std::size_t rank);
[[nodiscard]] AnalyzedGraph analyze(const TensorGraph &graph);

} // namespace tensor::analyzer
