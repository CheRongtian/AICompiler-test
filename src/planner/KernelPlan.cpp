#include "planner/KernelPlan.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tensor::planner {

GraphPlan planGraph(analyzer::AnalyzedGraph graph) {
  GraphPlan result;
  for (ValueId id = 0; id < graph.types.size(); ++id) {
    const auto count = graph.types[id].elementCount();
    // Leave headroom for padding the 1D grid; generated indices use uint.
    if (count > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
        graph.types[id].storageElementCount() >
            static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument("V2 tensor exceeds the supported 32-bit index range.");
    }
    BufferRole role = BufferRole::Intermediate;
    if (std::find(graph.graph.inputs.begin(), graph.graph.inputs.end(), id) != graph.graph.inputs.end()) {
      role = BufferRole::Input;
    } else if (std::find(graph.graph.outputs.begin(), graph.graph.outputs.end(), id) != graph.graph.outputs.end()) {
      role = BufferRole::Output;
    }
    result.buffers.push_back({id, count, graph.types[id].storageElementCount(),
                              graph.types[id], role, graph.aliasOf[id]});
  }
  for (auto index : graph.order) {
    const auto &node = graph.graph.nodes[index];
    if (isViewOp(node.op)) continue;
    KernelPlan plan;
    plan.nodeIndex = index;
    plan.op = node.op;
    plan.inputs = node.inputs;
    plan.output = node.outputs.front();
    plan.outputType = graph.types[plan.output];
    plan.attributes = node.attributes;
    for (auto id : node.inputs) plan.inputTypes.push_back(graph.types[id]);
    auto workItems = plan.outputType.elementCount();
    if (node.op == OpType::RMSNorm) {
      plan.threadsPerThreadgroup = 256;
      plan.threadgroupCount = workItems / plan.outputType.shape.back();
    } else if (node.op == OpType::LayerNorm) {
      plan.threadsPerThreadgroup = 256;
      std::size_t normalized = 1;
      const auto &attributes = std::get<LayerNormAttributes>(plan.attributes);
      for (auto axis : attributes.axes) normalized *= plan.outputType.shape[axis];
      plan.threadgroupCount = workItems / normalized;
    } else {
      if (node.op == OpType::RoPE) workItems /= 2;
      plan.threadgroupCount = (workItems + plan.threadsPerThreadgroup - 1) / plan.threadsPerThreadgroup;
    }
    result.kernels.push_back(std::move(plan));
  }
  result.analyzed = std::move(graph);
  return result;
}

} // namespace tensor::planner
