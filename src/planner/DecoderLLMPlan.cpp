#include "planner/DecoderLLMPlan.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace tensor::planner {
namespace {

std::size_t checkedMultiply(std::size_t left, std::size_t right) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error("Decoder-only tensor size exceeds size_t capacity.");
  }
  return left * right;
}

} // namespace

std::size_t DecoderLLMPlan::hiddenSize() const {
  return attention.modelDimension();
}

std::size_t
DecoderLLMPlan::hiddenElementCount(std::size_t sequenceLength) const {
  return attention.inputElementCount(sequenceLength);
}

std::size_t
DecoderLLMPlan::intermediateElementCount(std::size_t sequenceLength) const {
  return checkedMultiply(checkedMultiply(attention.batch, sequenceLength),
                         intermediateSize);
}

std::size_t
DecoderLLMPlan::logitsElementCount(std::size_t sequenceLength) const {
  return checkedMultiply(checkedMultiply(attention.batch, sequenceLength),
                         vocabularySize);
}

void DecoderLLMPlan::validate() const {
  attention.validate();
  if (layerCount == 0 || intermediateSize == 0 || vocabularySize == 0) {
    throw std::invalid_argument("Decoder-only dimensions must be nonzero.");
  }
  if (attention.headDimension % 2 != 0) {
    throw std::invalid_argument("Decoder-only RoPE requires an even head dimension.");
  }
  if (!std::isfinite(rmsNormEpsilon) || rmsNormEpsilon <= 0.0f) {
    throw std::invalid_argument("Decoder-only RMSNorm epsilon must be positive.");
  }
  if (accumulationDtype != DType::Float32) {
    throw std::invalid_argument(
        "Decoder-only reductions currently require fp32 accumulation.");
  }
  (void)checkedMultiply(vocabularySize, hiddenSize());
  (void)checkedMultiply(intermediateSize, hiddenSize());
  (void)logitsElementCount(attention.prefillLength);
}

} // namespace tensor::planner
