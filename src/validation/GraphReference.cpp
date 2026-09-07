#include "validation/GraphReference.hpp"
#include "validation/Validator.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace tensor::validation {
namespace {

// CPU coordinates are decoded from the right, independently of the MSL emitter.
std::size_t inputIndex(std::size_t index, const std::vector<std::size_t> &outputShape,
                       const std::vector<std::size_t> &inputShape) {
  std::size_t result = 0;
  std::size_t stride = 1;
  for (std::size_t offset = 0; offset < outputShape.size(); ++offset) {
    const auto coordinate = index % outputShape[outputShape.size() - 1 - offset];
    index /= outputShape[outputShape.size() - 1 - offset];
    if (offset < inputShape.size()) {
      const auto dimension = inputShape[inputShape.size() - 1 - offset];
      if (dimension != 1) result += coordinate * stride;
      stride *= dimension;
    }
  }
  return result;
}

std::vector<std::size_t> batchShape(const TensorType &type) {
  return {type.shape.begin(), type.shape.end() - 2};
}

} // namespace

GraphReference evaluateGraph(const analyzer::AnalyzedGraph &analyzed,
                              const GraphInputs &inputs) {
  const auto &graph = analyzed.graph;
  if (inputs.size() != graph.inputs.size()) {
    throw std::invalid_argument("Host input count does not match the graph.");
  }
  GraphReference result;
  result.values.resize(graph.values.size());
  std::vector<std::vector<double>> precise(graph.values.size());
  for (auto id : graph.inputs) {
    const auto found = inputs.find(id);
    if (found == inputs.end() || found->second.size() != analyzed.types[id].elementCount()) {
      throw std::invalid_argument("Missing graph input or incorrect input element count.");
    }
    result.values[id] = found->second;
    precise[id].assign(found->second.begin(), found->second.end());
  }
  for (auto index : analyzed.order) {
    const auto &node = graph.nodes[index];
    const auto &type = analyzed.types[node.output];
    const auto &aType = analyzed.types[node.inputs[0]];
    const auto &a = result.values[node.inputs[0]];
    auto &output = precise[node.output];
    output.resize(type.elementCount());
    switch (node.op) {
    case OpType::Add:
    case OpType::Mul: {
      const auto &bType = analyzed.types[node.inputs[1]];
      const auto &b = result.values[node.inputs[1]];
      for (std::size_t i = 0; i < output.size(); ++i) {
        const double x = a[inputIndex(i, type.shape, aType.shape)];
        const double y = b[inputIndex(i, type.shape, bType.shape)];
        output[i] = node.op == OpType::Add ? x + y : x * y;
      }
      break;
    }
    case OpType::SiLU:
      for (std::size_t i = 0; i < output.size(); ++i) {
        const double x = a[i];
        const double e = std::exp(x >= 0.0 ? -x : x);
        output[i] = x >= 0.0 ? x / (1.0 + e) : x * e / (1.0 + e);
      }
      break;
    case OpType::MatMul: {
      const auto &bType = analyzed.types[node.inputs[1]];
      const auto &b = result.values[node.inputs[1]];
      const auto m = aType.shape[aType.shape.size() - 2];
      const auto k = aType.shape.back();
      const auto n = bType.shape.back();
      for (std::size_t batch = 0; batch < output.size() / (m * n); ++batch) {
        const auto aBase = inputIndex(batch, batchShape(type), batchShape(aType)) * m * k;
        const auto bBase = inputIndex(batch, batchShape(type), batchShape(bType)) * k * n;
        for (std::size_t row = 0; row < m; ++row) {
          for (std::size_t column = 0; column < n; ++column) {
            double sum = 0.0;
            for (std::size_t inner = 0; inner < k; ++inner) {
              sum += static_cast<double>(a[aBase + row * k + inner]) *
                     static_cast<double>(b[bBase + inner * n + column]);
            }
            output[(batch * m + row) * n + column] = sum;
          }
        }
      }
      break;
    }
    case OpType::RMSNorm: {
      const auto &attributes = std::get<RMSNormAttributes>(node.attributes);
      output = rmsNormReference(
          {aType, analyzed.types[node.inputs[1]], type, attributes.epsilon},
          a, result.values[node.inputs[1]]);
      break;
    }
    case OpType::RoPE: {
      auto pairShape = type.shape;
      pairShape.back() /= 2;
      for (std::size_t pair = 0; pair < output.size() / 2; ++pair) {
        const double c = result.values[node.inputs[1]][inputIndex(
            pair, pairShape, analyzed.types[node.inputs[1]].shape)];
        const double s = result.values[node.inputs[2]][inputIndex(
            pair, pairShape, analyzed.types[node.inputs[2]].shape)];
        const double even = a[pair * 2];
        const double odd = a[pair * 2 + 1];
        output[pair * 2] = even * c - odd * s;
        output[pair * 2 + 1] = even * s + odd * c;
      }
      break;
    }
    }
    result.values[node.output].assign(output.begin(), output.end());
  }
  for (auto id : graph.outputs) result.outputs.push_back(precise[id]);
  return result;
}

} // namespace tensor::validation
