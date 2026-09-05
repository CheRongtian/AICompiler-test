#pragma once

#include <cstddef>
#include <vector>

namespace tensor {

enum class DType { Float32 };

// V0b tensors are static, dense, row-major, and own distinct storage.
struct TensorType {
  std::vector<std::size_t> shape;
  DType dtype = DType::Float32;

  [[nodiscard]] std::size_t elementCount() const;
};

// Normalize along the last dimension; weight has shape [input.shape.back()].
// y = x * rsqrt(mean(x * x, last axis) + epsilon) * weight
struct RMSNormOp {
  TensorType input;
  TensorType weight;
  TensorType output;
  float epsilon = 1e-5f;

  // Throws std::invalid_argument or std::overflow_error for unsupported IR.
  void validate() const;
};

} // namespace tensor
