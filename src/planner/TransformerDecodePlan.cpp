#include "planner/TransformerDecodePlan.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace tensor::planner {
namespace {

std::size_t checkedMultiply(std::size_t left, std::size_t right) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error("Transformer decode tensor size exceeds size_t capacity.");
  }
  return left * right;
}

} // namespace

std::size_t TransformerDecodePlan::modelDimension() const {
  return selfAttention.modelDimension();
}

std::size_t
TransformerDecodePlan::hiddenElementCount(std::size_t sequenceLength) const {
  return selfAttention.inputElementCount(sequenceLength);
}

std::size_t
TransformerDecodePlan::logitsElementCount(std::size_t sequenceLength) const {
  return checkedMultiply(checkedMultiply(selfAttention.batch, sequenceLength),
                         vocabularySize);
}

void TransformerDecodePlan::validate() const {
  selfAttention.validate();
  if (layerCount == 0 || feedForwardDimension == 0 || vocabularySize == 0 ||
      sourceLength == 0) {
    throw std::invalid_argument("Transformer decode dimensions must be nonzero.");
  }
  if (!std::isfinite(layerNormEpsilon) || layerNormEpsilon <= 0.0f) {
    throw std::invalid_argument("Transformer decode LayerNorm epsilon must be positive.");
  }
  (void)checkedMultiply(vocabularySize, modelDimension());
  (void)checkedMultiply(feedForwardDimension, modelDimension());
  (void)hiddenElementCount(sourceLength);
  (void)logitsElementCount(selfAttention.prefillLength);
}

} // namespace tensor::planner
