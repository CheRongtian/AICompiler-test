#include "analyzer/PatternAnalyzer.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tensor::analyzer {
namespace {

bool contains(const std::vector<ValueId> &values, ValueId value) {
  return std::find(values.begin(), values.end(), value) != values.end();
}

void appendUnique(std::vector<ValueId> &values, ValueId value) {
  if (!contains(values, value)) values.push_back(value);
}

} // namespace

const char *fusionName(FusionPattern pattern) {
  switch (pattern) {
  case FusionPattern::None: return "Unfused";
  case FusionPattern::AddRMSNorm: return "Add + RMSNorm";
  case FusionPattern::AddLayerNorm: return "Residual Add + LayerNorm";
  case FusionPattern::SiLUMul: return "SiLU + Mul";
  case FusionPattern::LinearReLU: return "Linear + ReLU";
  }
  throw std::invalid_argument("Unknown fusion pattern.");
}

RegionGraph formRegions(AnalyzedGraph graph) {
  RegionGraph result;
  result.analyzed = std::move(graph);
  const auto &ir = result.analyzed.graph;
  const auto missing = std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> producer(ir.values.size(), missing);
  std::vector<std::vector<std::size_t>> consumers(ir.values.size());
  for (std::size_t i = 0; i < ir.nodes.size(); ++i) {
    for (auto output : ir.nodes[i].outputs) producer[output] = i;
    for (auto input : ir.nodes[i].inputs) consumers[input].push_back(i);
  }
  std::vector<bool> assigned(ir.nodes.size(), false);
  auto exclusiveEdge = [&](ValueId value, std::size_t consumer) {
    return consumers[value].size() == 1 && consumers[value].front() == consumer &&
           !contains(ir.outputs, value) &&
           result.analyzed.effects[producer[value]] == MemoryEffect::ReadOnly;
  };

  for (auto nodeIndex : result.analyzed.order) {
    if (assigned[nodeIndex]) continue;
    Region region;
    region.id = result.regions.size();
    region.nodes = {nodeIndex};
    const auto &first = ir.nodes[nodeIndex];

    if (first.op == OpType::MatMul && first.outputs.size() == 1 &&
        consumers[first.outputs[0]].size() == 1) {
      const auto addIndex = consumers[first.outputs[0]].front();
      const auto &add = ir.nodes[addIndex];
      if (!assigned[addIndex] && add.op == OpType::Add && add.outputs.size() == 1 &&
          exclusiveEdge(first.outputs[0], addIndex) && consumers[add.outputs[0]].size() == 1) {
        const auto reluIndex = consumers[add.outputs[0]].front();
        if (!assigned[reluIndex] && ir.nodes[reluIndex].op == OpType::ReLU &&
            exclusiveEdge(add.outputs[0], reluIndex)) {
          region.nodes = {nodeIndex, addIndex, reluIndex};
          region.fusion = FusionPattern::LinearReLU;
        }
      }
    }
    if (region.fusion == FusionPattern::None && first.op == OpType::Add &&
        first.outputs.size() == 1 && consumers[first.outputs[0]].size() == 1) {
      const auto nextIndex = consumers[first.outputs[0]].front();
      const auto nextOp = ir.nodes[nextIndex].op;
      if (!assigned[nextIndex] && exclusiveEdge(first.outputs[0], nextIndex) &&
          !ir.nodes[nextIndex].inputs.empty() &&
          ir.nodes[nextIndex].inputs.front() == first.outputs[0] &&
          (nextOp == OpType::RMSNorm || nextOp == OpType::LayerNorm)) {
        region.nodes = {nodeIndex, nextIndex};
        region.fusion = nextOp == OpType::RMSNorm ? FusionPattern::AddRMSNorm
                                                  : FusionPattern::AddLayerNorm;
      }
    }
    if (region.fusion == FusionPattern::None && first.op == OpType::SiLU &&
        first.outputs.size() == 1 && consumers[first.outputs[0]].size() == 1) {
      const auto mulIndex = consumers[first.outputs[0]].front();
      if (!assigned[mulIndex] && ir.nodes[mulIndex].op == OpType::Mul &&
          exclusiveEdge(first.outputs[0], mulIndex)) {
        region.nodes = {nodeIndex, mulIndex};
        region.fusion = FusionPattern::SiLUMul;
      }
    }

    if (region.fusion != FusionPattern::None) {
      bool contiguous = true;
      for (auto candidateNode : region.nodes) {
        for (auto input : ir.nodes[candidateNode].inputs) {
          if (!result.analyzed.types[input].isContiguous()) contiguous = false;
        }
      }
      if (!contiguous) {
        region.nodes = {nodeIndex};
        region.fusion = FusionPattern::None;
      }
    }

    for (auto index : region.nodes) assigned[index] = true;
    for (auto index : region.nodes) {
      for (auto input : ir.nodes[index].inputs) {
        const auto source = producer[input];
        if (source == missing || std::find(region.nodes.begin(), region.nodes.end(), source) == region.nodes.end()) {
          appendUnique(region.inputs, input);
        }
      }
    }
    for (auto index : region.nodes) {
      for (auto output : ir.nodes[index].outputs) {
        bool escapes = contains(ir.outputs, output);
        for (auto consumer : consumers[output]) {
          if (std::find(region.nodes.begin(), region.nodes.end(), consumer) == region.nodes.end()) escapes = true;
        }
        if (escapes) appendUnique(region.outputs, output);
      }
    }
    if (region.outputs.empty()) region.outputs.push_back(ir.nodes[region.nodes.back()].outputs.front());
    if (region.fusion != FusionPattern::None && region.outputs.size() != 1) {
      throw std::invalid_argument("A fused region must have one external result.");
    }
    result.regions.push_back(std::move(region));
  }
  return result;
}

} // namespace tensor::analyzer
