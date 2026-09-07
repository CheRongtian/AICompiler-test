#include "tensor/TensorIR.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tensor {

std::size_t TensorType::elementCount() const {
  std::size_t count = 1;
  for (const std::size_t dimension : shape) {
    if (dimension == 0) {
      throw std::invalid_argument("Tensor dimensions must be positive.");
    }
    if (count > std::numeric_limits<std::size_t>::max() / dimension) {
      throw std::overflow_error("Tensor element count exceeds size_t capacity.");
    }
    count *= dimension;
  }
  return count;
}

std::vector<std::size_t> TensorType::strides() const {
  (void)elementCount();
  std::vector<std::size_t> result(shape.size());
  std::size_t stride = 1;
  for (std::size_t i = shape.size(); i > 0; --i) {
    result[i - 1] = stride;
    stride *= shape[i - 1];
  }
  return result;
}

void RMSNormOp::validate() const {
  if (input.shape.empty()) {
    throw std::invalid_argument("RMSNorm input must have at least one dimension.");
  }
  if (input.dtype != DType::Float32 || weight.dtype != DType::Float32 ||
      output.dtype != DType::Float32) {
    throw std::invalid_argument("V0b RMSNorm supports only fp32 tensors.");
  }
  if (input.layout != Layout::Contiguous || weight.layout != Layout::Contiguous ||
      output.layout != Layout::Contiguous) {
    throw std::invalid_argument("RMSNorm requires contiguous tensors.");
  }
  (void)input.elementCount();
  (void)weight.elementCount();
  (void)output.elementCount();
  if (output.shape != input.shape) {
    throw std::invalid_argument("RMSNorm output shape must match input shape.");
  }
  if (weight.shape.size() != 1 || weight.shape.front() != input.shape.back()) {
    throw std::invalid_argument("RMSNorm weight must match the last input dimension.");
  }
  if (!std::isfinite(epsilon) || epsilon <= 0.0f) {
    throw std::invalid_argument("RMSNorm epsilon must be finite and positive.");
  }
}

const char *opName(OpType op) {
  switch (op) {
  case OpType::Add: return "Add";
  case OpType::Mul: return "Mul";
  case OpType::MatMul: return "MatMul";
  case OpType::RMSNorm: return "RMSNorm";
  case OpType::SiLU: return "SiLU";
  case OpType::RoPE: return "RoPE";
  }
  throw std::invalid_argument("Unknown tensor operation.");
}

ValueId TensorGraph::addInput(std::string name, TensorType type) {
  const ValueId id = values.size();
  values.push_back({std::move(name), std::move(type)});
  inputs.push_back(id);
  return id;
}

ValueId TensorGraph::addNode(OpType op, std::vector<ValueId> operands,
                              OpAttributes attributes, std::string name) {
  const ValueId id = values.size();
  if (name.empty()) {
    name = std::string(opName(op)) + "_" + std::to_string(nodes.size());
  }
  values.push_back({std::move(name), std::nullopt});
  nodes.push_back({op, std::move(operands), id, std::move(attributes)});
  return id;
}

} // namespace tensor
