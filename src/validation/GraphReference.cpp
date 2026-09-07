#include "validation/GraphReference.hpp"
#include "validation/Validator.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tensor::validation {
namespace {

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

std::vector<std::size_t> denseStrides(const std::vector<std::size_t> &shape) {
  return TensorType{shape}.strides();
}

std::uint16_t floatToHalf(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t sign = (bits >> 16) & 0x8000u;
  const std::uint32_t exponent = (bits >> 23) & 0xffu;
  const std::uint32_t mantissa = bits & 0x7fffffu;
  if (exponent == 0xffu) return static_cast<std::uint16_t>(sign | (mantissa ? 0x7e00u : 0x7c00u));
  const int halfExponent = static_cast<int>(exponent) - 112;
  if (halfExponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
  if (halfExponent <= 0) {
    if (halfExponent < -10) return static_cast<std::uint16_t>(sign);
    const std::uint32_t normalized = mantissa | 0x800000u;
    const int shift = 14 - halfExponent;
    return static_cast<std::uint16_t>(sign | ((normalized + (1u << (shift - 1))) >> shift));
  }
  const std::uint32_t rounded = mantissa + 0x1000u;
  if (rounded & 0x800000u) {
    if (halfExponent + 1 >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    return static_cast<std::uint16_t>(sign | ((halfExponent + 1) << 10));
  }
  return static_cast<std::uint16_t>(sign | (halfExponent << 10) | (rounded >> 13));
}

float halfToFloat(std::uint16_t value) {
  const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000u) << 16;
  std::uint32_t exponent = (value >> 10) & 0x1fu;
  std::uint32_t mantissa = value & 0x3ffu;
  std::uint32_t bits;
  if (exponent == 0) {
    if (mantissa == 0) bits = sign;
    else {
      int shift = 0;
      while ((mantissa & 0x400u) == 0) { mantissa <<= 1; ++shift; }
      mantissa &= 0x3ffu;
      bits = sign | static_cast<std::uint32_t>(113 - shift) << 23 | mantissa << 13;
    }
  } else if (exponent == 31) bits = sign | 0x7f800000u | mantissa << 13;
  else bits = sign | (exponent + 112) << 23 | mantissa << 13;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

float storeValue(double value, DType dtype) {
  const auto asFloat = static_cast<float>(value);
  if (dtype == DType::Float16) return halfToFloat(floatToHalf(asFloat));
  if (dtype == DType::Int32) return static_cast<float>(static_cast<std::int32_t>(asFloat));
  return asFloat;
}

std::size_t replaceAxisIndex(std::size_t linear, const std::vector<std::size_t> &shape,
                             std::size_t axis, std::size_t coordinate) {
  const auto strides = denseStrides(shape);
  const auto current = (linear / strides[axis]) % shape[axis];
  return linear - current * strides[axis] + coordinate * strides[axis];
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
    result.values[id].resize(found->second.size());
    precise[id].resize(found->second.size());
    for (std::size_t i = 0; i < found->second.size(); ++i) {
      result.values[id][i] = storeValue(found->second[i], analyzed.types[id].dtype);
      precise[id][i] = result.values[id][i];
    }
  }

  for (auto index : analyzed.order) {
    const auto &node = graph.nodes[index];
    const auto outputId = node.outputs.front();
    const auto &type = analyzed.types[outputId];
    const auto &aType = analyzed.types[node.inputs[0]];
    const auto &a = result.values[node.inputs[0]];
    auto &output = precise[outputId];
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
    case OpType::ReLU:
      for (std::size_t i = 0; i < output.size(); ++i) output[i] = std::max(0.0, double(a[i]));
      break;
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
              sum += double(a[aBase + row * k + inner]) * double(b[bBase + inner * n + column]);
            }
            output[(batch * m + row) * n + column] = sum;
          }
        }
      }
      break;
    }
    case OpType::RMSNorm: {
      const auto &attributes = std::get<RMSNormAttributes>(node.attributes);
      output = rmsNormReference({aType, analyzed.types[node.inputs[1]], type, attributes.epsilon},
                                a, result.values[node.inputs[1]]);
      break;
    }
    case OpType::LayerNorm: {
      const auto &attributes = std::get<LayerNormAttributes>(node.attributes);
      std::size_t normalized = 1;
      for (auto axis : attributes.axes) normalized *= type.shape[axis];
      const auto &weight = result.values[node.inputs[1]];
      const auto &bias = result.values[node.inputs[2]];
      for (std::size_t base = 0; base < output.size(); base += normalized) {
        double sum = 0.0;
        double squares = 0.0;
        for (std::size_t i = 0; i < normalized; ++i) {
          sum += a[base + i];
          squares += double(a[base + i]) * a[base + i];
        }
        const double mean = sum / normalized;
        const double variance = std::max(0.0, squares / normalized - mean * mean);
        const double inverse = 1.0 / std::sqrt(variance + attributes.epsilon);
        for (std::size_t i = 0; i < normalized; ++i) {
          output[base + i] = (a[base + i] - mean) * inverse * weight[i] + bias[i];
        }
      }
      break;
    }
    case OpType::Softmax: {
      const auto axis = std::size_t(std::get<SoftmaxAttributes>(node.attributes).axis);
      const auto width = aType.shape[axis];
      for (std::size_t i = 0; i < output.size(); ++i) {
        double maximum = -std::numeric_limits<double>::infinity();
        for (std::size_t j = 0; j < width; ++j) maximum = std::max(maximum, double(a[replaceAxisIndex(i, aType.shape, axis, j)]));
        double denominator = 0.0;
        for (std::size_t j = 0; j < width; ++j) denominator += std::exp(a[replaceAxisIndex(i, aType.shape, axis, j)] - maximum);
        output[i] = std::exp(a[i] - maximum) / denominator;
      }
      break;
    }
    case OpType::RoPE: {
      auto pairShape = type.shape;
      pairShape.back() /= 2;
      for (std::size_t pair = 0; pair < output.size() / 2; ++pair) {
        const double c = result.values[node.inputs[1]][inputIndex(pair, pairShape, analyzed.types[node.inputs[1]].shape)];
        const double s = result.values[node.inputs[2]][inputIndex(pair, pairShape, analyzed.types[node.inputs[2]].shape)];
        output[pair * 2] = a[pair * 2] * c - a[pair * 2 + 1] * s;
        output[pair * 2 + 1] = a[pair * 2] * s + a[pair * 2 + 1] * c;
      }
      break;
    }
    case OpType::ReduceSum:
    case OpType::ReduceMean: {
      const auto &attributes = std::get<ReductionAttributes>(node.attributes);
      std::vector<bool> reduced(aType.shape.size(), false);
      for (auto axis : attributes.axes) reduced[axis] = true;
      const auto outputStrides = denseStrides(type.shape);
      const auto inputStrides = denseStrides(aType.shape);
      for (std::size_t inputLinear = 0; inputLinear < a.size(); ++inputLinear) {
        std::size_t remaining = inputLinear;
        std::size_t outputLinear = 0;
        std::size_t outAxis = 0;
        for (std::size_t axis = 0; axis < aType.shape.size(); ++axis) {
          const auto coordinate = remaining / inputStrides[axis];
          remaining %= inputStrides[axis];
          if (!reduced[axis]) outputLinear += coordinate * outputStrides[outAxis++];
          else if (attributes.keepDimensions) ++outAxis;
        }
        output[outputLinear] += a[inputLinear];
      }
      if (node.op == OpType::ReduceMean) {
        std::size_t count = 1;
        for (auto axis : attributes.axes) count *= aType.shape[axis];
        for (auto &value : output) value /= count;
      }
      break;
    }
    case OpType::Reshape:
    case OpType::View:
    case OpType::Contiguous:
      output.assign(a.begin(), a.end());
      break;
    case OpType::Transpose: {
      const auto &attributes = std::get<TransposeAttributes>(node.attributes);
      const auto outStrides = denseStrides(type.shape);
      const auto inStrides = denseStrides(aType.shape);
      for (std::size_t i = 0; i < output.size(); ++i) {
        std::size_t remaining = i;
        std::size_t inputLinear = 0;
        for (std::size_t axis = 0; axis < type.shape.size(); ++axis) {
          const auto coordinate = remaining / outStrides[axis];
          remaining %= outStrides[axis];
          std::size_t inputAxis = axis;
          if (axis == std::size_t(attributes.first)) inputAxis = attributes.second;
          else if (axis == std::size_t(attributes.second)) inputAxis = attributes.first;
          inputLinear += coordinate * inStrides[inputAxis];
        }
        output[i] = a[inputLinear];
      }
      break;
    }
    case OpType::MaskedFill: {
      const auto &maskType = analyzed.types[node.inputs[1]];
      const auto &mask = result.values[node.inputs[1]];
      const auto fill = std::get<MaskedFillAttributes>(node.attributes).value;
      for (std::size_t i = 0; i < output.size(); ++i) {
        output[i] = mask[inputIndex(i, type.shape, maskType.shape)] != 0.0f
                        ? fill : a[inputIndex(i, type.shape, aType.shape)];
      }
      break;
    }
    case OpType::Slice: {
      const auto &attributes = std::get<SliceAttributes>(node.attributes);
      const auto outStrides = denseStrides(type.shape);
      const auto inStrides = denseStrides(aType.shape);
      for (std::size_t i = 0; i < output.size(); ++i) {
        std::size_t remaining = i;
        std::size_t source = 0;
        for (std::size_t axis = 0; axis < type.shape.size(); ++axis) {
          const auto coordinate = remaining / outStrides[axis];
          remaining %= outStrides[axis];
          source += (attributes.starts[axis] + coordinate * attributes.steps[axis]) * inStrides[axis];
        }
        output[i] = a[source];
      }
      break;
    }
    case OpType::Embedding: {
      const auto &indices = result.values[node.inputs[1]];
      const auto width = aType.shape[1];
      for (std::size_t i = 0; i < output.size(); ++i) {
        const auto row = static_cast<std::int64_t>(indices[i / width]);
        if (row < 0 || std::size_t(row) >= aType.shape[0]) throw std::invalid_argument("Embedding index is out of range.");
        output[i] = a[std::size_t(row) * width + i % width];
      }
      break;
    }
    case OpType::Gather: {
      const auto axis = std::size_t(std::get<GatherAttributes>(node.attributes).axis);
      const auto &indicesType = analyzed.types[node.inputs[1]];
      const auto &indices = result.values[node.inputs[1]];
      const auto outStrides = denseStrides(type.shape);
      const auto dataStrides = denseStrides(aType.shape);
      const auto indexStrides = denseStrides(indicesType.shape);
      for (std::size_t i = 0; i < output.size(); ++i) {
        std::size_t indexLinear = 0;
        for (std::size_t j = 0; j < indicesType.shape.size(); ++j) {
          indexLinear += ((i / outStrides[axis + j]) % type.shape[axis + j]) * indexStrides[j];
        }
        const auto selected = static_cast<std::int64_t>(indices[indexLinear]);
        if (selected < 0 || std::size_t(selected) >= aType.shape[axis]) throw std::invalid_argument("Gather index is out of range.");
        std::size_t dataLinear = std::size_t(selected) * dataStrides[axis];
        for (std::size_t j = 0; j < aType.shape.size(); ++j) {
          if (j == axis) continue;
          const auto outAxis = j < axis ? j : j + indicesType.shape.size() - 1;
          dataLinear += ((i / outStrides[outAxis]) % type.shape[outAxis]) * dataStrides[j];
        }
        output[i] = a[dataLinear];
      }
      break;
    }
    }
    auto &stored = result.values[outputId];
    stored.resize(output.size());
    for (std::size_t i = 0; i < output.size(); ++i) stored[i] = storeValue(output[i], type.dtype);
  }
  for (auto id : graph.outputs) result.outputs.push_back(precise[id]);
  return result;
}

} // namespace tensor::validation
