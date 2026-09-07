#include "tensor/TensorIR.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tensor {
namespace {

std::vector<std::size_t> contiguousStrides(const std::vector<std::size_t> &shape) {
  std::vector<std::size_t> result(shape.size());
  std::size_t stride = 1;
  for (std::size_t i = shape.size(); i > 0; --i) {
    result[i - 1] = stride;
    if (shape[i - 1] != 0 && stride > std::numeric_limits<std::size_t>::max() / shape[i - 1]) {
      throw std::overflow_error("Tensor strides exceed size_t capacity.");
    }
    stride *= shape[i - 1];
  }
  return result;
}

bool isFloat(DType dtype) {
  return dtype == DType::Float16 || dtype == DType::Float32;
}

} // namespace

std::size_t TensorType::elementCount() const {
  std::size_t count = 1;
  for (const std::size_t dimension : shape) {
    if (dimension == 0) throw std::invalid_argument("Tensor dimensions must be positive.");
    if (count > std::numeric_limits<std::size_t>::max() / dimension) {
      throw std::overflow_error("Tensor element count exceeds size_t capacity.");
    }
    count *= dimension;
  }
  return count;
}

std::vector<std::size_t> TensorType::strides() const {
  (void)elementCount();
  if (layout == Layout::Contiguous) {
    if (!explicitStrides.empty()) {
      throw std::invalid_argument("Contiguous tensors must not carry explicit strides.");
    }
    return contiguousStrides(shape);
  }
  if (explicitStrides.size() != shape.size()) {
    throw std::invalid_argument("Strided tensors require one stride per dimension.");
  }
  return explicitStrides;
}

std::size_t TensorType::storageElementCount() const {
  const auto tensorStrides = strides();
  if (shape.empty()) return 1;
  std::size_t maximumOffset = 0;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    const auto extent = shape[i] - 1;
    if (extent != 0 && tensorStrides[i] >
                           (std::numeric_limits<std::size_t>::max() - maximumOffset) / extent) {
      throw std::overflow_error("Tensor storage span exceeds size_t capacity.");
    }
    maximumOffset += extent * tensorStrides[i];
  }
  return maximumOffset + 1;
}

bool TensorType::isContiguous() const {
  if (layout == Layout::Contiguous) return explicitStrides.empty();
  return explicitStrides == contiguousStrides(shape);
}

void RMSNormOp::validate() const {
  if (input.shape.empty()) throw std::invalid_argument("RMSNorm input must be ranked.");
  (void)input.storageElementCount();
  (void)weight.storageElementCount();
  (void)output.storageElementCount();
  if (!isFloat(input.dtype) || weight.dtype != input.dtype || output.dtype != input.dtype) {
    throw std::invalid_argument("RMSNorm requires matching fp16 or fp32 tensors.");
  }
  if (!input.isContiguous() || !weight.isContiguous() || !output.isContiguous()) {
    throw std::invalid_argument("RMSNorm requires contiguous tensors.");
  }
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
  case OpType::LayerNorm: return "LayerNorm";
  case OpType::ReLU: return "ReLU";
  case OpType::SiLU: return "SiLU";
  case OpType::Softmax: return "Softmax";
  case OpType::RoPE: return "RoPE";
  case OpType::ReduceSum: return "ReduceSum";
  case OpType::ReduceMean: return "ReduceMean";
  case OpType::Reshape: return "Reshape";
  case OpType::View: return "View";
  case OpType::Transpose: return "Transpose";
  case OpType::Contiguous: return "Contiguous";
  case OpType::MaskedFill: return "MaskedFill";
  case OpType::Slice: return "Slice";
  case OpType::Embedding: return "Embedding";
  case OpType::Gather: return "Gather";
  }
  throw std::invalid_argument("Unknown tensor operation.");
}

bool isViewOp(OpType op) {
  return op == OpType::Reshape || op == OpType::View || op == OpType::Transpose;
}

ValueId TensorGraph::addInput(std::string name, TensorType type) {
  const ValueId id = values.size();
  values.push_back({std::move(name), std::move(type)});
  inputs.push_back(id);
  return id;
}

ValueId TensorGraph::addNode(OpType op, std::vector<ValueId> operands,
                             OpAttributes attributes, std::string name) {
  if (name.empty()) name = std::string(opName(op)) + "_" + std::to_string(nodes.size());
  return addNodeResults(op, std::move(operands), {std::move(name)},
                        std::move(attributes)).front();
}

std::vector<ValueId> TensorGraph::addNodeResults(OpType op,
                                                 std::vector<ValueId> operands,
                                                 std::vector<std::string> names,
                                                 OpAttributes attributes) {
  if (names.empty()) throw std::invalid_argument("A tensor node must define an output.");
  std::vector<ValueId> ids;
  ids.reserve(names.size());
  for (auto &name : names) {
    if (name.empty()) name = std::string(opName(op)) + "_" + std::to_string(values.size());
    ids.push_back(values.size());
    values.push_back({std::move(name), std::nullopt});
  }
  nodes.push_back({op, std::move(operands), ids, std::move(attributes)});
  return ids;
}

} // namespace tensor
