#include "analyzer/GraphAnalyzer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>

namespace tensor::analyzer {
namespace {

bool isFloat(DType dtype) {
  return dtype == DType::Float16 || dtype == DType::Float32;
}

void validateType(const TensorType &type) {
  (void)type.elementCount();
  (void)type.storageElementCount();
}

TensorType dense(std::vector<std::size_t> shape, DType dtype) {
  return {std::move(shape), dtype, Layout::Contiguous, {}};
}

void requireFloatMatch(const TensorType &a, const TensorType &b,
                       const char *operation) {
  if (!isFloat(a.dtype) || a.dtype != b.dtype) {
    throw std::invalid_argument(std::string(operation) +
                                " requires matching floating-point dtypes.");
  }
}

std::size_t normalizeAxis(std::int64_t axis, std::size_t rank) {
  const auto signedRank = static_cast<std::int64_t>(rank);
  const auto value = axis < 0 ? axis + signedRank : axis;
  if (value < 0 || value >= signedRank) throw std::invalid_argument("Axis is out of range.");
  return static_cast<std::size_t>(value);
}

void requireSingleOutput(const Node &node) {
  if (node.outputs.size() != 1) {
    throw std::invalid_argument(std::string(opName(node.op)) +
                                " currently defines exactly one result.");
  }
}

TensorType infer(Node &node, const std::vector<TensorType> &types) {
  requireSingleOutput(node);
  auto requireArity = [&](std::size_t count) {
    if (node.inputs.size() != count) {
      throw std::invalid_argument(std::string(opName(node.op)) + ": invalid input count.");
    }
  };
  const auto &a = types.at(node.inputs.at(0));
  switch (node.op) {
  case OpType::Add:
  case OpType::Mul: {
    requireArity(2);
    const auto &b = types.at(node.inputs[1]);
    requireFloatMatch(a, b, opName(node.op));
    return dense(broadcastShape(a.shape, b.shape), a.dtype);
  }
  case OpType::ReLU:
  case OpType::SiLU:
  case OpType::Contiguous:
    requireArity(1);
    if (!isFloat(a.dtype)) throw std::invalid_argument("Operation requires a floating tensor.");
    return dense(a.shape, a.dtype);
  case OpType::MatMul: {
    requireArity(2);
    const auto &b = types.at(node.inputs[1]);
    requireFloatMatch(a, b, "MatMul");
    if (a.shape.size() < 2 || b.shape.size() < 2 ||
        a.shape.back() != b.shape[b.shape.size() - 2]) {
      throw std::invalid_argument("MatMul requires [..., M, K] and [..., K, N].");
    }
    std::vector<std::size_t> batchA(a.shape.begin(), a.shape.end() - 2);
    std::vector<std::size_t> batchB(b.shape.begin(), b.shape.end() - 2);
    auto shape = broadcastShape(batchA, batchB);
    shape.push_back(a.shape[a.shape.size() - 2]);
    shape.push_back(b.shape.back());
    return dense(std::move(shape), a.dtype);
  }
  case OpType::RMSNorm: {
    requireArity(2);
    if (std::holds_alternative<std::monostate>(node.attributes)) {
      node.attributes = RMSNormAttributes{};
    }
    auto &attributes = std::get<RMSNormAttributes>(node.attributes);
    if (normalizeAxis(attributes.axis, a.shape.size()) != a.shape.size() - 1) {
      throw std::invalid_argument("RMSNorm reduces only the last axis.");
    }
    attributes.axis = static_cast<std::int64_t>(a.shape.size() - 1);
    const auto output = dense(a.shape, a.dtype);
    RMSNormOp{a, types.at(node.inputs[1]), output, attributes.epsilon}.validate();
    return output;
  }
  case OpType::LayerNorm: {
    requireArity(3);
    if (std::holds_alternative<std::monostate>(node.attributes)) {
      node.attributes = LayerNormAttributes{};
    }
    auto &attributes = std::get<LayerNormAttributes>(node.attributes);
    const auto axes = normalizeAxes(attributes.axes, a.shape.size());
    if (!a.isContiguous()) {
      throw std::invalid_argument("LayerNorm requires a contiguous input; insert Contiguous after a view.");
    }
    if (!std::isfinite(attributes.epsilon) || attributes.epsilon <= 0.0f) {
      throw std::invalid_argument("LayerNorm epsilon must be finite and positive.");
    }
    const auto first = axes.front();
    for (std::size_t i = 0; i < axes.size(); ++i) {
      if (axes[i] != first + i || first + axes.size() != a.shape.size()) {
        throw std::invalid_argument("LayerNorm axes must form a trailing suffix.");
      }
    }
    std::vector<std::size_t> parameterShape(a.shape.begin() + first, a.shape.end());
    const auto &weight = types.at(node.inputs[1]);
    const auto &bias = types.at(node.inputs[2]);
    requireFloatMatch(a, weight, "LayerNorm");
    requireFloatMatch(a, bias, "LayerNorm");
    if (weight.shape != parameterShape || bias.shape != parameterShape ||
        !weight.isContiguous() || !bias.isContiguous()) {
      throw std::invalid_argument("LayerNorm weight and bias must match normalized axes.");
    }
    attributes.axes.assign(axes.begin(), axes.end());
    return dense(a.shape, a.dtype);
  }
  case OpType::Softmax: {
    requireArity(1);
    if (!isFloat(a.dtype)) throw std::invalid_argument("Softmax requires a floating tensor.");
    if (std::holds_alternative<std::monostate>(node.attributes)) {
      node.attributes = SoftmaxAttributes{};
    }
    auto &attributes = std::get<SoftmaxAttributes>(node.attributes);
    attributes.axis = static_cast<std::int64_t>(normalizeAxis(attributes.axis, a.shape.size()));
    return dense(a.shape, a.dtype);
  }
  case OpType::RoPE: {
    requireArity(3);
    if (!isFloat(a.dtype) || a.shape.empty() || a.shape.back() % 2 != 0) {
      throw std::invalid_argument("RoPE requires a floating tensor with an even last dimension.");
    }
    auto pairs = a.shape;
    pairs.back() /= 2;
    for (std::size_t i = 1; i < 3; ++i) {
      requireFloatMatch(a, types.at(node.inputs[i]), "RoPE");
      if (broadcastShape(pairs, types.at(node.inputs[i]).shape) != pairs) {
        throw std::invalid_argument("RoPE cos/sin must broadcast to [..., D/2].");
      }
    }
    return dense(a.shape, a.dtype);
  }
  case OpType::ReduceSum:
  case OpType::ReduceMean: {
    requireArity(1);
    if (!isFloat(a.dtype)) throw std::invalid_argument("Reduction requires a floating tensor.");
    auto &attributes = std::get<ReductionAttributes>(node.attributes);
    const auto axes = normalizeAxes(attributes.axes, a.shape.size());
    std::vector<std::size_t> shape;
    for (std::size_t i = 0; i < a.shape.size(); ++i) {
      const bool reduced = std::binary_search(axes.begin(), axes.end(), i);
      if (!reduced) shape.push_back(a.shape[i]);
      else if (attributes.keepDimensions) shape.push_back(1);
    }
    attributes.axes.assign(axes.begin(), axes.end());
    return dense(std::move(shape), a.dtype);
  }
  case OpType::Reshape:
  case OpType::View: {
    requireArity(1);
    const auto &attributes = std::get<ReshapeAttributes>(node.attributes);
    TensorType output = dense(attributes.shape, a.dtype);
    if (!a.isContiguous() || output.elementCount() != a.elementCount()) {
      throw std::invalid_argument("Reshape/View requires contiguous storage and equal element counts.");
    }
    return output;
  }
  case OpType::Transpose: {
    requireArity(1);
    auto &attributes = std::get<TransposeAttributes>(node.attributes);
    const auto first = normalizeAxis(attributes.first, a.shape.size());
    const auto second = normalizeAxis(attributes.second, a.shape.size());
    auto shape = a.shape;
    auto strides = a.strides();
    std::swap(shape[first], shape[second]);
    std::swap(strides[first], strides[second]);
    attributes.first = static_cast<std::int64_t>(first);
    attributes.second = static_cast<std::int64_t>(second);
    return {std::move(shape), a.dtype, Layout::Strided, std::move(strides)};
  }
  case OpType::MaskedFill: {
    requireArity(2);
    if (!isFloat(a.dtype)) throw std::invalid_argument("MaskedFill requires floating data.");
    (void)std::get<MaskedFillAttributes>(node.attributes);
    return dense(broadcastShape(a.shape, types.at(node.inputs[1]).shape), a.dtype);
  }
  case OpType::Slice: {
    requireArity(1);
    const auto &attributes = std::get<SliceAttributes>(node.attributes);
    if (attributes.starts.size() != a.shape.size() ||
        attributes.sizes.size() != a.shape.size() ||
        attributes.steps.size() != a.shape.size()) {
      throw std::invalid_argument("Slice attributes must match input rank.");
    }
    for (std::size_t i = 0; i < a.shape.size(); ++i) {
      if (attributes.steps[i] == 0 || attributes.sizes[i] == 0 ||
          attributes.starts[i] + (attributes.sizes[i] - 1) * attributes.steps[i] >= a.shape[i]) {
        throw std::invalid_argument("Slice range is outside the input tensor.");
      }
    }
    return dense(attributes.sizes, a.dtype);
  }
  case OpType::Embedding: {
    requireArity(2);
    const auto &indices = types.at(node.inputs[1]);
    if (a.shape.size() != 2 || !a.isContiguous() || !isFloat(a.dtype) ||
        indices.dtype != DType::Int32) {
      throw std::invalid_argument("Embedding requires a rank-2 table and int32 indices.");
    }
    auto shape = indices.shape;
    shape.push_back(a.shape[1]);
    return dense(std::move(shape), a.dtype);
  }
  case OpType::Gather: {
    requireArity(2);
    const auto &indices = types.at(node.inputs[1]);
    if (indices.dtype != DType::Int32 || a.shape.empty()) {
      throw std::invalid_argument("Gather requires ranked data and int32 indices.");
    }
    if (std::holds_alternative<std::monostate>(node.attributes)) {
      node.attributes = GatherAttributes{};
    }
    auto &attributes = std::get<GatherAttributes>(node.attributes);
    const auto axis = normalizeAxis(attributes.axis, a.shape.size());
    attributes.axis = static_cast<std::int64_t>(axis);
    std::vector<std::size_t> shape(a.shape.begin(), a.shape.begin() + axis);
    shape.insert(shape.end(), indices.shape.begin(), indices.shape.end());
    shape.insert(shape.end(), a.shape.begin() + axis + 1, a.shape.end());
    return dense(std::move(shape), a.dtype);
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

std::vector<std::size_t> normalizeAxes(const std::vector<std::int64_t> &axes,
                                       std::size_t rank) {
  if (axes.empty() || rank == 0) throw std::invalid_argument("Reduction axes cannot be empty.");
  std::vector<std::size_t> result;
  result.reserve(axes.size());
  for (auto axis : axes) result.push_back(normalizeAxis(axis, rank));
  std::sort(result.begin(), result.end());
  if (std::adjacent_find(result.begin(), result.end()) != result.end()) {
    throw std::invalid_argument("Reduction axes must be unique.");
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
  result.effects.resize(graph.nodes.size(), MemoryEffect::ReadOnly);
  result.aliasOf.resize(graph.values.size());
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
    if (!graph.values[id].type->isContiguous()) {
      throw std::invalid_argument("V2 external graph inputs must use contiguous storage.");
    }
    result.types[id] = *graph.values[id].type;
  }
  for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
    const auto &node = graph.nodes[i];
    if (node.outputs.empty()) throw std::invalid_argument("Graph node has no results.");
    for (auto output : node.outputs) {
      checkId(output);
      if (isInput[output] || producer[output] != missing) {
        throw std::invalid_argument("A graph value must have one definition; mutation is unsupported.");
      }
      producer[output] = i;
    }
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
  for (std::size_t i = 0; i < indegree.size(); ++i) if (indegree[i] == 0) ready.push(i);
  while (!ready.empty()) {
    const auto i = ready.front();
    ready.pop();
    auto &node = result.graph.nodes[i];
    const auto type = infer(node, result.types);
    validateType(type);
    const auto output = node.outputs.front();
    const auto &declared = graph.values[output].type;
    if (declared && (declared->shape != type.shape || declared->dtype != type.dtype ||
                     declared->layout != type.layout ||
                     declared->explicitStrides != type.explicitStrides)) {
      throw std::invalid_argument("Declared output type disagrees with inferred type.");
    }
    result.types[output] = type;
    result.graph.values[output].type = type;
    if (isViewOp(node.op)) {
      result.effects[i] = MemoryEffect::View;
      result.aliasOf[output] = node.inputs.front();
    }
    result.order.push_back(i);
    for (auto next : consumers[i]) if (--indegree[next] == 0) ready.push(next);
  }
  if (result.order.size() != graph.nodes.size()) {
    throw std::invalid_argument("Tensor graph contains a dependency cycle.");
  }
  return result;
}

} // namespace tensor::analyzer
