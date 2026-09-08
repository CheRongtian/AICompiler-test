#pragma once

#include "planner/KVCachePlan.hpp"

#include <cstddef>

namespace tensor::planner {

struct DecoderLLMPlan {
  KVCachePlan attention;
  std::size_t layerCount = 1;
  std::size_t intermediateSize = 1;
  std::size_t vocabularySize = 1;
  float rmsNormEpsilon = 1e-5f;

  [[nodiscard]] std::size_t hiddenSize() const;
  [[nodiscard]] std::size_t hiddenElementCount(std::size_t sequenceLength) const;
  [[nodiscard]] std::size_t intermediateElementCount(std::size_t sequenceLength) const;
  [[nodiscard]] std::size_t logitsElementCount(std::size_t sequenceLength) const;
  void validate() const;
};

} // namespace tensor::planner
