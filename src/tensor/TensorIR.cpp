#include "tensor/TensorIR.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace tensor {

std::size_t TensorType::elementCount() const {
  if (shape.empty()) {
    throw std::invalid_argument("Tensor shape must contain at least one dimension.");
  }
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

void RMSNormOp::validate() const {
  if (input.dtype != DType::Float32 || weight.dtype != DType::Float32 ||
      output.dtype != DType::Float32) {
    throw std::invalid_argument("V0b RMSNorm supports only fp32 tensors.");
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

} // namespace tensor
