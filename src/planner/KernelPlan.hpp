#pragma once

#include "analyzer/GraphAnalyzer.hpp"

namespace tensor::planner {

struct KernelPlan {
  std::size_t nodeIndex = 0;
  OpType op = OpType::Add;
  std::vector<ValueId> inputs;
  ValueId output = 0;
  std::vector<TensorType> inputTypes;
  TensorType outputType;
  OpAttributes attributes;
  std::size_t threadsPerThreadgroup = 128;
  std::size_t threadgroupCount = 0;
  bool useRMSNormBaseline = true;
};

enum class BufferRole { Input, Intermediate, Output };
struct BufferPlan {
  ValueId value = 0;
  std::size_t elementCount = 0;
  std::size_t storageElementCount = 0;
  TensorType type;
  BufferRole role = BufferRole::Intermediate;
  std::optional<ValueId> aliasOf;
};

struct GraphPlan {
  analyzer::AnalyzedGraph analyzed;
  std::vector<BufferPlan> buffers;
  std::vector<KernelPlan> kernels;
};

[[nodiscard]] GraphPlan planGraph(analyzer::AnalyzedGraph graph);

} // namespace tensor::planner
