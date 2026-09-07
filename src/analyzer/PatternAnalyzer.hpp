#pragma once

#include "analyzer/GraphAnalyzer.hpp"

namespace tensor::analyzer {

enum class FusionPattern { None, AddRMSNorm, AddLayerNorm, SiLUMul, LinearReLU };
[[nodiscard]] const char *fusionName(FusionPattern pattern);

struct Region {
  std::size_t id = 0;
  std::vector<std::size_t> nodes;
  std::vector<ValueId> inputs;
  std::vector<ValueId> outputs;
  FusionPattern fusion = FusionPattern::None;
};

struct RegionGraph {
  AnalyzedGraph analyzed;
  std::vector<Region> regions;
};

[[nodiscard]] RegionGraph formRegions(AnalyzedGraph graph);

} // namespace tensor::analyzer
