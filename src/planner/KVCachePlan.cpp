#include "planner/KVCachePlan.hpp"

#include <limits>
#include <stdexcept>

namespace tensor::planner {
namespace {

std::size_t checkedMultiply(std::size_t left, std::size_t right) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error("KV cache tensor size exceeds size_t capacity.");
  }
  return left * right;
}

} // namespace

std::size_t KVCachePlan::modelDimension() const {
  return checkedMultiply(heads, headDimension);
}

std::size_t KVCachePlan::cacheElementCount() const {
  return checkedMultiply(checkedMultiply(batch, heads),
                         checkedMultiply(capacity, headDimension));
}

std::size_t KVCachePlan::inputElementCount(std::size_t sequenceLength) const {
  return checkedMultiply(checkedMultiply(batch, sequenceLength), modelDimension());
}

void KVCachePlan::validate() const {
  if (batch == 0 || heads == 0 || headDimension == 0 || capacity == 0 ||
      prefillLength == 0 || prefillLength > capacity) {
    throw std::invalid_argument("KV cache dimensions and prefill length must be valid.");
  }
  if (dtype != DType::Float16 && dtype != DType::Float32 &&
      dtype != DType::BFloat16) {
    throw std::invalid_argument("KV cache storage dtype is unsupported.");
  }
  if (threadsPerThreadgroup == 0 || threadsPerThreadgroup > 1024) {
    throw std::invalid_argument("KV cache threadgroup size must be in [1, 1024].");
  }
  if ((threadsPerThreadgroup & (threadsPerThreadgroup - 1)) != 0) {
    throw std::invalid_argument(
        "KV cache threadgroup size must be a power of two.");
  }
  if (headDimension > threadsPerThreadgroup) {
    throw std::invalid_argument(
        "KV attention requires head dimension <= threadgroup size.");
  }
  (void)modelDimension();
  (void)cacheElementCount();
  (void)inputElementCount(prefillLength);
}

} // namespace tensor::planner
