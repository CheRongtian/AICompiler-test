#pragma once

#include "planner/KVCachePlan.hpp"

#include <cstddef>

namespace tensor::planner {

struct TransformerDecodePlan {
  KVCachePlan selfAttention;
  std::size_t layerCount = 1;
  std::size_t feedForwardDimension = 1;
  std::size_t vocabularySize = 1;
  std::size_t sourceLength = 1;
  float layerNormEpsilon = 1e-5f;

  [[nodiscard]] std::size_t modelDimension() const;
  [[nodiscard]] std::size_t hiddenElementCount(std::size_t sequenceLength) const;
  [[nodiscard]] std::size_t logitsElementCount(std::size_t sequenceLength) const;
  void validate() const;
};

} // namespace tensor::planner
