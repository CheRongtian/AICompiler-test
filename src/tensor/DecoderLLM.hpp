#pragma once

#include "planner/DecoderLLMPlan.hpp"

#include <string>
#include <vector>

namespace tensor {

struct DecoderLLMLayerParameters {
  std::vector<float> inputNormWeight;
  std::vector<float> queryWeight;
  std::vector<float> keyWeight;
  std::vector<float> valueWeight;
  std::vector<float> outputWeight;
  std::vector<float> postAttentionNormWeight;
  std::vector<float> gateWeight;
  std::vector<float> upWeight;
  std::vector<float> downWeight;
};

struct DecoderLLMReference {
  std::vector<float> tokenIds;
  std::vector<double> logits;
  std::vector<double> nextTokenIds;
  std::vector<std::vector<double>> keyCaches;
  std::vector<std::vector<double>> valueCaches;
};

struct DecoderLLMWorkload {
  std::string modelName;
  planner::DecoderLLMPlan plan;
  DType requestedStorageDtype = DType::Float32;
  std::size_t decodeCount = 0;
  std::vector<float> embeddingWeight;
  std::vector<float> finalNormWeight;
  std::vector<float> languageModelHeadWeight;
  std::vector<float> ropeCosine;
  std::vector<float> ropeSine;
  std::vector<DecoderLLMLayerParameters> layers;
  DecoderLLMReference prefill;
  std::vector<DecoderLLMReference> decodeSteps;
};

} // namespace tensor
