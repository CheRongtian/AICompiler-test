#pragma once

#include "tensor/TensorIR.hpp"

#include <cstddef>

namespace tensor::planner {

struct KVCachePlan {
  std::size_t batch = 1;
  std::size_t heads = 1;
  std::size_t headDimension = 1;
  std::size_t capacity = 1;
  std::size_t prefillLength = 1;
  DType dtype = DType::Float32;
  std::size_t threadsPerThreadgroup = 128;

  [[nodiscard]] std::size_t modelDimension() const;
  [[nodiscard]] std::size_t cacheElementCount() const;
  [[nodiscard]] std::size_t inputElementCount(std::size_t sequenceLength) const;
  void validate() const;
};

} // namespace tensor::planner
