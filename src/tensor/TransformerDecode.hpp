#pragma once

#include "planner/TransformerDecodePlan.hpp"

#include <string>
#include <vector>

namespace tensor {

struct DecoderLayerParameters {
  std::vector<float> selfQueryWeight;
  std::vector<float> selfKeyWeight;
  std::vector<float> selfValueWeight;
  std::vector<float> selfOutputWeight;
  std::vector<float> selfNormWeight;
  std::vector<float> selfNormBias;

  std::vector<float> crossQueryWeight;
  std::vector<float> crossQueryBias;
  std::vector<float> crossKeyWeight;
  std::vector<float> crossKeyBias;
  std::vector<float> crossValueWeight;
  std::vector<float> crossValueBias;
  std::vector<float> crossOutputWeight;
  std::vector<float> crossOutputBias;
  std::vector<float> crossNormWeight;
  std::vector<float> crossNormBias;

  std::vector<float> feedForwardInputWeight;
  std::vector<float> feedForwardInputBias;
  std::vector<float> feedForwardOutputWeight;
  std::vector<float> feedForwardOutputBias;
  std::vector<float> feedForwardNormWeight;
  std::vector<float> feedForwardNormBias;
};

struct TransformerDecodeReference {
  std::vector<float> tokenIds;
  std::vector<double> logits;
  std::vector<double> nextTokenIds;
  std::vector<std::vector<double>> keyCaches;
  std::vector<std::vector<double>> valueCaches;
};

struct TransformerDecodeWorkload {
  std::string modelName;
  planner::TransformerDecodePlan plan;
  std::vector<float> encoderMemory;
  std::vector<float> embeddingWeight;
  std::vector<float> positionalEncoding;
  std::vector<DecoderLayerParameters> layers;
  std::vector<float> languageModelHeadWeight;
  std::vector<float> languageModelHeadBias;
  TransformerDecodeReference prefill;
  std::vector<TransformerDecodeReference> decodeSteps;
};

} // namespace tensor
