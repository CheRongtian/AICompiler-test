#pragma once

#include "analyzer/PatternAnalyzer.hpp"
#include "planner/KernelPlan.hpp"

namespace tensor::planner {

struct RegionPlan {
  analyzer::Region region;
  std::vector<KernelPlan> baseline;
  std::vector<std::size_t> candidateThreads;
};

struct ProgramPlan {
  GraphPlan graph;
  std::vector<RegionPlan> regions;
};

[[nodiscard]] ProgramPlan planRegions(GraphPlan graph);

} // namespace tensor::planner
