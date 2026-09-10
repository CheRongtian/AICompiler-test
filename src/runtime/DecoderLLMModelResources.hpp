#pragma once

#include "backend/metal/MetalRuntime.hpp"
#include "tensor/TensorIR.hpp"

#include <cstddef>
#include <vector>

namespace tensor::runtime {

struct DecoderLLMModelLayerBuffers {
  metal::BufferHandle inputNormWeight;
  metal::BufferHandle queryWeight;
  metal::BufferHandle keyWeight;
  metal::BufferHandle valueWeight;
  metal::BufferHandle outputWeight;
  metal::BufferHandle postAttentionNormWeight;
  metal::BufferHandle gateWeight;
  metal::BufferHandle upWeight;
  metal::BufferHandle downWeight;
};

// Immutable GPU model storage shared by independent request executors.
// Stateful KV buffers and activation scratch remain request-owned.
struct DecoderLLMModelResources {
  DType storageDtype = DType::Float32;
  std::size_t hiddenSize = 0;
  std::size_t headCount = 0;
  std::size_t headDimension = 0;
  std::size_t cacheCapacity = 0;
  std::size_t intermediateSize = 0;
  std::size_t vocabularySize = 0;
  std::size_t layerCount = 0;
  std::size_t storageBytes = 0;
  metal::BufferHandle embeddingWeight;
  metal::BufferHandle finalNormWeight;
  metal::BufferHandle languageModelHeadWeight;
  metal::BufferHandle ropeCosine;
  metal::BufferHandle ropeSine;
  std::vector<DecoderLLMModelLayerBuffers> layers;
};

} // namespace tensor::runtime
