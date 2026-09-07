#include "analyzer/GraphAnalyzer.hpp"

#include <algorithm>
#include <limits>
#include <queue>
#include <stdexcept>

namespace tensor::analyzer {
namespace {

void validateType(const TensorType &type) {
  if (type.dtype != DType::Float32 || type.layout != Layout::Contiguous) {
    throw std::invalid_argument("V2 supports fp32 contiguous tensors only.");
  }
  (void)type.elementCount();
}

TensorType infer(Node &node, const std::vector<TensorType> &types) {
  std::size_t arity = 2;
  if (node.op == OpType::SiLU) arity = 1;
  if (node.op == OpType::RoPE) arity = 3;
  if (node.inputs.size() != arity) {
    throw std::invalid_argument(std::string(opName(node.op)) + ": invalid input count.");
  }
  if (node.op != OpType::RMSNorm &&
      !std::holds_alternative<std::monostate>(node.attributes)) {
    throw std::invalid_argument("Unexpected attributes for this operation.");
  }
  const auto &a = types.at(node.inputs[0]);
  switch (node.op) {
  case OpType::Add:
  case OpType::Mul:
    return {broadcastShape(a.shape, types.at(node.inputs[1]).shape)};
  case OpType::SiLU:
    return a;
  case OpType::MatMul: {
    const auto &b = types.at(node.inputs[1]);
    if (a.shape.size() < 2 || b.shape.size() < 2 ||
        a.shape.back() != b.shape[b.shape.size() - 2]) {
      throw std::invalid_argument("MatMul requires [..., M, K] and [..., K, N].");
    }
    std::vector<std::size_t> batchA(a.shape.begin(), a.shape.end() - 2);
    std::vector<std::size_t> batchB(b.shape.begin(), b.shape.end() - 2);
    auto shape = broadcastShape(batchA, batchB);
    shape.push_back(a.shape[a.shape.size() - 2]);
    shape.push_back(b.shape.back());
    return {shape};
  }
  case OpType::RMSNorm: {
    if (std::holds_alternative<std::monostate>(node.attributes)) {
      node.attributes = RMSNormAttributes{};
    }
    auto &attributes = std::get<RMSNormAttributes>(node.attributes);
    const auto rank = static_cast<std::int64_t>(a.shape.size());
    const auto axis = attributes.axis < 0 ? attributes.axis + rank : attributes.axis;
    if (rank == 0 || axis != rank - 1) {
      throw std::invalid_argument("V2 RMSNorm reduces only the last axis.");
    }
    RMSNormOp{a, types.at(node.inputs[1]), a, attributes.epsilon}.validate();
    attributes.axis = axis;
    return a;
  }
  case OpType::RoPE: {
    if (a.shape.empty() || a.shape.back() % 2 != 0) {
      throw std::invalid_argument("RoPE requires an even, nonempty last dimension.");
    }
    auto pairs = a.shape;
    pairs.back() /= 2;
    for (std::size_t i = 1; i < 3; ++i) {
      if (broadcastShape(pairs, types.at(node.inputs[i]).shape) != pairs) {
        throw std::invalid_argument("RoPE cos/sin must broadcast to [..., D/2].");
      }
    }
    return a;
  }
  }
  throw std::invalid_argument("Unsupported tensor operation.");
}

} // namespace

std::vector<std::size_t> broadcastShape(const std::vector<std::size_t> &a,
                                       const std::vector<std::size_t> &b) {
  const auto rank = std::max(a.size(), b.size());
  std::vector<std::size_t> result(rank, 1);
  for (std::size_t offset = 0; offset < rank; ++offset) {
    const auto x = offset < a.size() ? a[a.size() - 1 - offset] : 1;
    const auto y = offset < b.size() ? b[b.size() - 1 - offset] : 1;
    if (x != y && x != 1 && y != 1) {
      throw std::invalid_argument("Incompatible broadcasting dimensions.");
    }
    result[rank - 1 - offset] = std::max(x, y);
  }
  return result;
}

AnalyzedGraph analyze(const TensorGraph &graph) {
  if (graph.values.empty() || graph.outputs.empty()) {
    throw std::invalid_argument("Graph must contain values and declared outputs.");
  }
  AnalyzedGraph result;
  result.graph = graph;
  result.types.resize(graph.values.size());
  const auto missing = std::numeric_limits<std::size_t>::max();
  std::vector<std::size_t> producer(graph.values.size(), missing);
  std::vector<bool> isInput(graph.values.size(), false);
  auto checkId = [&](ValueId id) {
    if (id >= graph.values.size()) throw std::invalid_argument("Invalid graph ValueId.");
  };
  for (const auto id : graph.inputs) {
    checkId(id);
    if (isInput[id] || !graph.values[id].type) {
      throw std::invalid_argument("Graph inputs must be unique and typed.");
    }
    isInput[id] = true;
    validateType(*graph.values[id].type);
    result.types[id] = *graph.values[id].type;
  }
  for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
    const auto &node = graph.nodes[i];
    checkId(node.output);
    if (isInput[node.output] || producer[node.output] != missing) {
      throw std::invalid_argument("A graph value must have one definition; mutation is unsupported.");
    }
    producer[node.output] = i;
    for (auto id : node.inputs) checkId(id);
  }
  for (ValueId id = 0; id < graph.values.size(); ++id) {
    if (!isInput[id] && producer[id] == missing) {
      throw std::invalid_argument("Graph contains an undefined value.");
    }
  }
  for (auto id : graph.outputs) checkId(id);

  std::vector<std::size_t> indegree(graph.nodes.size(), 0);
  std::vector<std::vector<std::size_t>> consumers(graph.nodes.size());
  for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
    for (auto id : graph.nodes[i].inputs) {
      if (!isInput[id]) {
        ++indegree[i];
        consumers[producer[id]].push_back(i);
      }
    }
  }
  std::queue<std::size_t> ready;
  for (std::size_t i = 0; i < indegree.size(); ++i) {
    if (indegree[i] == 0) ready.push(i);
  }
  while (!ready.empty()) {
    const auto i = ready.front();
    ready.pop();
    auto &node = result.graph.nodes[i];
    auto type = infer(node, result.types);
    validateType(type);
    const auto &declared = graph.values[node.output].type;
    if (declared && (declared->shape != type.shape || declared->dtype != type.dtype ||
                     declared->layout != type.layout)) {
      throw std::invalid_argument("Declared output type disagrees with inferred type.");
    }
    result.types[node.output] = type;
    result.graph.values[node.output].type = type;
    result.order.push_back(i);
    for (auto next : consumers[i]) {
      if (--indegree[next] == 0) ready.push(next);
    }
  }
  if (result.order.size() != graph.nodes.size()) {
    throw std::invalid_argument("Tensor graph contains a dependency cycle.");
  }
  return result;
}

} // namespace tensor::analyzer
