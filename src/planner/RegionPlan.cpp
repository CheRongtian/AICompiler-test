#include "planner/RegionPlan.hpp"

#include <stdexcept>
#include <utility>

namespace tensor::planner {

ProgramPlan planRegions(GraphPlan graph) {
  ProgramPlan result;
  auto regions = analyzer::formRegions(graph.analyzed);
  result.graph = std::move(graph);
  for (const auto &region : regions.regions) {
    RegionPlan plan;
    plan.region = region;
    for (auto nodeIndex : region.nodes) {
      for (const auto &kernel : result.graph.kernels) {
        if (kernel.nodeIndex == nodeIndex) plan.baseline.push_back(kernel);
      }
    }
    if (region.fusion != analyzer::FusionPattern::None) {
      plan.candidateThreads = {64, 128, 256};
      if (plan.baseline.size() != region.nodes.size()) {
        throw std::invalid_argument("A fused region cannot contain a view-only operation.");
      }
    }
    result.regions.push_back(std::move(plan));
  }
  return result;
}

} // namespace tensor::planner
