#include "runtime/TransformerDecodeExecutor.hpp"

#include "backend/metal/KVCacheMetalEmitter.hpp"
#include "backend/metal/TransformerDecodeMetalEmitter.hpp"
#include "runtime/KVCacheState.hpp"

#include <ostream>
#include <stdexcept>
#include <utility>

namespace tensor::runtime {
namespace {

using Buffer = metal::BufferHandle;

Buffer allocate(metal::MetalRuntime &runtime, std::size_t count,
                const float *data = nullptr,
                metal::ElementType type = metal::ElementType::Float32) {
  auto result = runtime.createBuffer(count, data, type);
  if (!result.buffer) throw std::runtime_error(result.errorMessage);
  return std::move(result.buffer);
}

std::unique_ptr<metal::PreparedExecution> prepare(
    metal::MetalRuntime &runtime, const metal::GeneratedKernel &kernel,
    const std::vector<Buffer> &inputs,
    const std::vector<metal::ElementType> &inputTypes, const Buffer &output,
    metal::ElementType outputType, std::ostream &log) {
  const auto pipeline =
      runtime.createComputePipeline(kernel.source, kernel.functionName);
  if (!pipeline.pipelineCreationPassed) {
    throw std::runtime_error(kernel.functionName + ": " + pipeline.errorMessage);
  }
  const auto interfaceError =
      metal::checkBufferInterface(pipeline, inputTypes, outputType);
  if (!interfaceError.empty()) {
    throw std::runtime_error(kernel.functionName + ": " + interfaceError);
  }
  auto result = runtime.prepareBuffers(
      inputs, output,
      {kernel.threadgroupCount, kernel.threadsPerThreadgroup});
  if (!result.execution) {
    throw std::runtime_error(kernel.functionName + ": " + result.errorMessage);
  }
  log << "Transformer kernel " << kernel.functionName << ": PASS\n";
  return std::move(result.execution);
}

std::unique_ptr<metal::PreparedSequence> makeSequence(
    metal::MetalRuntime &runtime,
    const std::vector<std::unique_ptr<metal::PreparedExecution>> &steps) {
  std::vector<const metal::PreparedExecution *> pointers;
  pointers.reserve(steps.size());
  for (const auto &step : steps) pointers.push_back(step.get());
  auto result = runtime.prepareSequence(pointers);
  if (!result.execution) throw std::runtime_error(result.errorMessage);
  return std::move(result.execution);
}

struct LayerBuffers {
  Buffer selfQueryWeight;
  Buffer selfKeyWeight;
  Buffer selfValueWeight;
  Buffer selfOutputWeight;
  Buffer selfNormWeight;
  Buffer selfNormBias;

  Buffer crossQueryWeight;
  Buffer crossQueryBias;
  Buffer crossKeyWeight;
  Buffer crossKeyBias;
  Buffer crossValueWeight;
  Buffer crossValueBias;
  Buffer crossOutputWeight;
  Buffer crossOutputBias;
  Buffer crossNormWeight;
  Buffer crossNormBias;
  Buffer projectedMemoryKey;
  Buffer projectedMemoryValue;

  Buffer feedForwardInputWeight;
  Buffer feedForwardInputBias;
  Buffer feedForwardOutputWeight;
  Buffer feedForwardOutputBias;
  Buffer feedForwardNormWeight;
  Buffer feedForwardNormBias;
};

LayerBuffers allocateLayer(metal::MetalRuntime &runtime,
                           const TransformerDecodeWorkload &workload,
                           const DecoderLayerParameters &parameters) {
  const auto dimension = workload.plan.modelDimension();
  const auto memoryCount = workload.plan.selfAttention.batch *
                           workload.plan.sourceLength * dimension;
  LayerBuffers layer;
  auto weight = [&](const std::vector<float> &values) {
    return allocate(runtime, values.size(), values.data());
  };
  layer.selfQueryWeight = weight(parameters.selfQueryWeight);
  layer.selfKeyWeight = weight(parameters.selfKeyWeight);
  layer.selfValueWeight = weight(parameters.selfValueWeight);
  layer.selfOutputWeight = weight(parameters.selfOutputWeight);
  layer.selfNormWeight = weight(parameters.selfNormWeight);
  layer.selfNormBias = weight(parameters.selfNormBias);
  layer.crossQueryWeight = weight(parameters.crossQueryWeight);
  layer.crossQueryBias = weight(parameters.crossQueryBias);
  layer.crossKeyWeight = weight(parameters.crossKeyWeight);
  layer.crossKeyBias = weight(parameters.crossKeyBias);
  layer.crossValueWeight = weight(parameters.crossValueWeight);
  layer.crossValueBias = weight(parameters.crossValueBias);
  layer.crossOutputWeight = weight(parameters.crossOutputWeight);
  layer.crossOutputBias = weight(parameters.crossOutputBias);
  layer.crossNormWeight = weight(parameters.crossNormWeight);
  layer.crossNormBias = weight(parameters.crossNormBias);
  layer.projectedMemoryKey = allocate(runtime, memoryCount);
  layer.projectedMemoryValue = allocate(runtime, memoryCount);
  layer.feedForwardInputWeight = weight(parameters.feedForwardInputWeight);
  layer.feedForwardInputBias = weight(parameters.feedForwardInputBias);
  layer.feedForwardOutputWeight = weight(parameters.feedForwardOutputWeight);
  layer.feedForwardOutputBias = weight(parameters.feedForwardOutputBias);
  layer.feedForwardNormWeight = weight(parameters.feedForwardNormWeight);
  layer.feedForwardNormBias = weight(parameters.feedForwardNormBias);
  return layer;
}

struct DecodeStage {
  std::size_t sequenceLength = 0;
  Buffer tokenIds;
  Buffer logits;
  Buffer allNextTokenIds;
  std::vector<Buffer> intermediates;
  std::unique_ptr<metal::PreparedSequence> sequence;
};

Buffer intermediate(metal::MetalRuntime &runtime, DecodeStage &stage,
                    std::size_t count,
                    metal::ElementType type = metal::ElementType::Float32) {
  auto buffer = allocate(runtime, count, nullptr, type);
  stage.intermediates.push_back(buffer);
  return buffer;
}

DecodeStage buildStage(
    metal::MetalRuntime &runtime, const TransformerDecodeWorkload &workload,
    std::size_t sequenceLength, const Buffer &embeddingWeight,
    const Buffer &positionTable, const Buffer &languageModelHeadWeight,
    const Buffer &languageModelHeadBias, const std::vector<LayerBuffers> &layers,
    const std::vector<std::unique_ptr<KVCacheState>> &states,
    std::ostream &log) {
  DecodeStage stage;
  stage.sequenceLength = sequenceLength;
  const auto &plan = workload.plan;
  const auto dimension = plan.modelDimension();
  const auto rows = plan.selfAttention.batch * sequenceLength;
  const auto hiddenCount = plan.hiddenElementCount(sequenceLength);
  const auto threads = plan.selfAttention.threadsPerThreadgroup;
  stage.tokenIds = allocate(runtime, rows, nullptr, metal::ElementType::Int32);
  stage.logits = allocate(runtime, plan.logitsElementCount(sequenceLength));
  stage.allNextTokenIds =
      allocate(runtime, rows, nullptr, metal::ElementType::Int32);

  std::vector<std::unique_ptr<metal::PreparedExecution>> steps;
  Buffer current = intermediate(runtime, stage, hiddenCount);
  steps.push_back(prepare(
      runtime,
      metal::emitTokenEmbedding(
          plan, sequenceLength,
          sequenceLength == 1 ? "transformer_embedding_decode"
                              : "transformer_embedding_prefill"),
      {stage.tokenIds, embeddingWeight, positionTable,
       states.front()->lengthBuffer()},
      {metal::ElementType::Int32, metal::ElementType::Float32,
       metal::ElementType::Float32, metal::ElementType::Int32},
      current, metal::ElementType::Float32, log));

  for (std::size_t index = 0; index < layers.size(); ++index) {
    const auto prefix = "transformer_" +
                        std::string(sequenceLength == 1 ? "decode_l" : "prefill_l") +
                        std::to_string(index) + "_";
    const auto &layer = layers[index];
    auto projectedQuery = intermediate(runtime, stage, hiddenCount);
    auto selfContext = intermediate(runtime, stage, hiddenCount);
    auto selfOutput = intermediate(runtime, stage, hiddenCount);
    auto selfNormalized = intermediate(runtime, stage, hiddenCount);
    auto crossQuery = intermediate(runtime, stage, hiddenCount);
    auto crossContext = intermediate(runtime, stage, hiddenCount);
    auto crossOutput = intermediate(runtime, stage, hiddenCount);
    auto crossNormalized = intermediate(runtime, stage, hiddenCount);
    auto feedForwardHidden = intermediate(
        runtime, stage, rows * plan.feedForwardDimension);
    auto feedForwardOutput = intermediate(runtime, stage, hiddenCount);
    auto layerOutput = intermediate(runtime, stage, hiddenCount);

    steps.push_back(prepare(
        runtime, metal::emitKVQueryProjection(plan.selfAttention, sequenceLength),
        {current, layer.selfQueryWeight},
        {metal::ElementType::Float32, metal::ElementType::Float32},
        projectedQuery, metal::ElementType::Float32, log));
    steps.push_back(prepare(
        runtime,
        metal::emitKVCacheProjection(plan.selfAttention, sequenceLength, true),
        {current, layer.selfKeyWeight, states[index]->lengthBuffer()},
        {metal::ElementType::Float32, metal::ElementType::Float32,
         metal::ElementType::Int32},
        states[index]->keyBuffer(), metal::ElementType::Float32, log));
    steps.push_back(prepare(
        runtime,
        metal::emitKVCacheProjection(plan.selfAttention, sequenceLength, false),
        {current, layer.selfValueWeight, states[index]->lengthBuffer()},
        {metal::ElementType::Float32, metal::ElementType::Float32,
         metal::ElementType::Int32},
        states[index]->valueBuffer(), metal::ElementType::Float32, log));
    steps.push_back(prepare(
        runtime,
        metal::emitKVAttention(plan.selfAttention, sequenceLength, true),
        {projectedQuery, states[index]->keyBuffer(), states[index]->valueBuffer(),
         states[index]->lengthBuffer()},
        {metal::ElementType::Float32, metal::ElementType::Float32,
         metal::ElementType::Float32, metal::ElementType::Int32},
        selfContext, metal::ElementType::Float32, log));
    steps.push_back(prepare(
        runtime,
        metal::emitKVOutputProjection(plan.selfAttention, sequenceLength),
        {selfContext, layer.selfOutputWeight},
        {metal::ElementType::Float32, metal::ElementType::Float32},
        selfOutput, metal::ElementType::Float32, log));
    steps.push_back(prepare(
        runtime,
        metal::emitResidualLayerNorm(rows, dimension, plan.layerNormEpsilon,
                                     threads, prefix + "self_norm"),
        {current, selfOutput, layer.selfNormWeight, layer.selfNormBias},
        {metal::ElementType::Float32, metal::ElementType::Float32,
         metal::ElementType::Float32, metal::ElementType::Float32},
        selfNormalized, metal::ElementType::Float32, log));

    steps.push_back(prepare(
        runtime,
        metal::emitBiasedLinear(rows, dimension, dimension, false, threads,
                                prefix + "cross_query"),
        {selfNormalized, layer.crossQueryWeight, layer.crossQueryBias},
        {metal::ElementType::Float32, metal::ElementType::Float32,
         metal::ElementType::Float32},
        crossQuery, metal::ElementType::Float32, log));
    steps.push_back(prepare(
        runtime,
        metal::emitCrossAttention(plan, sequenceLength,
                                  prefix + "cross_attention"),
        {crossQuery, layer.projectedMemoryKey, layer.projectedMemoryValue},
        {metal::ElementType::Float32, metal::ElementType::Float32,
         metal::ElementType::Float32},
        crossContext, metal::ElementType::Float32, log));
    steps.push_back(prepare(
        runtime,
        metal::emitBiasedLinear(rows, dimension, dimension, false, threads,
                                prefix + "cross_output"),
        {crossContext, layer.crossOutputWeight, layer.crossOutputBias},
        {metal::ElementType::Float32, metal::ElementType::Float32,
         metal::ElementType::Float32},
        crossOutput, metal::ElementType::Float32, log));
    steps.push_back(prepare(
        runtime,
        metal::emitResidualLayerNorm(rows, dimension, plan.layerNormEpsilon,
                                     threads, prefix + "cross_norm"),
        {selfNormalized, crossOutput, layer.crossNormWeight, layer.crossNormBias},
        {metal::ElementType::Float32, metal::ElementType::Float32,
         metal::ElementType::Float32, metal::ElementType::Float32},
        crossNormalized, metal::ElementType::Float32, log));

    steps.push_back(prepare(
        runtime,
        metal::emitBiasedLinear(rows, dimension, plan.feedForwardDimension,
                                true, threads, prefix + "ffn_input"),
        {crossNormalized, layer.feedForwardInputWeight,
         layer.feedForwardInputBias},
        {metal::ElementType::Float32, metal::ElementType::Float32,
         metal::ElementType::Float32},
        feedForwardHidden, metal::ElementType::Float32, log));
    steps.push_back(prepare(
        runtime,
        metal::emitBiasedLinear(rows, plan.feedForwardDimension, dimension,
                                false, threads, prefix + "ffn_output"),
        {feedForwardHidden, layer.feedForwardOutputWeight,
         layer.feedForwardOutputBias},
        {metal::ElementType::Float32, metal::ElementType::Float32,
         metal::ElementType::Float32},
        feedForwardOutput, metal::ElementType::Float32, log));
    steps.push_back(prepare(
        runtime,
        metal::emitResidualLayerNorm(rows, dimension, plan.layerNormEpsilon,
                                     threads, prefix + "ffn_norm"),
        {crossNormalized, feedForwardOutput, layer.feedForwardNormWeight,
         layer.feedForwardNormBias},
        {metal::ElementType::Float32, metal::ElementType::Float32,
         metal::ElementType::Float32, metal::ElementType::Float32},
        layerOutput, metal::ElementType::Float32, log));
    current = layerOutput;
  }

  steps.push_back(prepare(
      runtime,
      metal::emitBiasedLinear(
          rows, dimension, plan.vocabularySize, false, threads,
          sequenceLength == 1 ? "transformer_lm_head_decode"
                              : "transformer_lm_head_prefill"),
      {current, languageModelHeadWeight, languageModelHeadBias},
      {metal::ElementType::Float32, metal::ElementType::Float32,
       metal::ElementType::Float32},
      stage.logits, metal::ElementType::Float32, log));
  steps.push_back(prepare(
      runtime,
      metal::emitTokenArgmax(
          rows, plan.vocabularySize, threads,
          sequenceLength == 1 ? "transformer_argmax_decode"
                              : "transformer_argmax_prefill"),
      {stage.logits}, {metal::ElementType::Float32}, stage.allNextTokenIds,
      metal::ElementType::Int32, log));
  stage.sequence = makeSequence(runtime, steps);
  return stage;
}

} // namespace

class CompiledTransformerDecoder::Impl {
public:
  metal::MetalRuntime *runtime = nullptr;
  planner::TransformerDecodePlan plan;
  std::vector<std::unique_ptr<KVCacheState>> states;
  Buffer embeddingWeight;
  Buffer positionTable;
  Buffer languageModelHeadWeight;
  Buffer languageModelHeadBias;
  Buffer encoderMemory;
  std::vector<LayerBuffers> layers;
  DecodeStage prefillStage;
  DecodeStage decodeStage;

  TransformerDecodeRunResult run(DecodeStage &stage, std::size_t nextLength,
                                 const std::vector<float> &tokenIds) {
    TransformerDecodeRunResult result;
    if (tokenIds.size() != stage.tokenIds->elementCount()) {
      result.errorMessage = "Transformer token input shape does not match the compiled stage.";
      return result;
    }
    auto error = runtime->writeBuffer(stage.tokenIds, tokenIds.data(), tokenIds.size());
    if (!error.empty()) {
      result.errorMessage = error;
      return result;
    }
    const auto previousLength = currentLength();
    for (auto &state : states) {
      error = state->stageLength(*runtime, nextLength);
      if (!error.empty()) {
        result.errorMessage = error;
        return result;
      }
    }
    const auto execution = stage.sequence->execute();
    if (!execution.executionPassed) {
      const float previous = static_cast<float>(previousLength);
      for (auto &state : states) {
        (void)runtime->writeBuffer(state->lengthBuffer(), &previous, 1);
      }
      result.errorMessage = execution.errorMessage;
      return result;
    }
    for (auto &state : states) state->commitLength(nextLength);
    result.logits = stage.logits->read();
    const auto allTokens = stage.allNextTokenIds->read();
    result.nextTokenIds.reserve(plan.selfAttention.batch);
    for (std::size_t batch = 0; batch < plan.selfAttention.batch; ++batch) {
      const auto row = batch * stage.sequenceLength + stage.sequenceLength - 1;
      result.nextTokenIds.push_back(allTokens[row]);
    }
    result.gpuExecutionTimeUs = execution.gpuExecutionTimeUs;
    result.cpuSubmitToCompletionTimeUs = execution.cpuSubmitToCompletionTimeUs;
    result.passed = true;
    return result;
  }

  [[nodiscard]] std::size_t currentLength() const noexcept {
    return states.empty() ? 0 : states.front()->currentLength();
  }
};

CompiledTransformerDecoder::CompiledTransformerDecoder(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
CompiledTransformerDecoder::~CompiledTransformerDecoder() = default;

std::string CompiledTransformerDecoder::reset() {
  for (auto &state : impl_->states) {
    const auto error = state->reset(*impl_->runtime);
    if (!error.empty()) return error;
  }
  return {};
}

TransformerDecodeRunResult CompiledTransformerDecoder::prefill(
    const std::vector<float> &tokenIds) {
  if (currentLength() != 0) {
    return {false, {}, {}, {}, 0.0,
            "Transformer prefill requires an empty cache; call reset first."};
  }
  return impl_->run(impl_->prefillStage, impl_->plan.selfAttention.prefillLength,
                    tokenIds);
}

TransformerDecodeRunResult CompiledTransformerDecoder::decode(
    const std::vector<float> &tokenIds) {
  const auto current = currentLength();
  if (current == 0) {
    return {false, {}, {}, {}, 0.0,
            "Transformer decode requires a completed prefill."};
  }
  if (current >= impl_->plan.selfAttention.capacity) {
    return {false, {}, {}, {}, 0.0,
            "Transformer KV cache capacity has been reached."};
  }
  return impl_->run(impl_->decodeStage, current + 1, tokenIds);
}

std::size_t CompiledTransformerDecoder::currentLength() const noexcept {
  return impl_->currentLength();
}

std::vector<float>
CompiledTransformerDecoder::readKeyPrefix(std::size_t layer) const {
  if (layer >= impl_->states.size()) {
    throw std::out_of_range("Transformer decoder layer index is out of range.");
  }
  return impl_->states[layer]->readKeyPrefix();
}

std::vector<float>
CompiledTransformerDecoder::readValuePrefix(std::size_t layer) const {
  if (layer >= impl_->states.size()) {
    throw std::out_of_range("Transformer decoder layer index is out of range.");
  }
  return impl_->states[layer]->readValuePrefix();
}

bool CompiledTransformerDecoder::cacheStorageReused() const noexcept {
  for (const auto &state : impl_->states) {
    if (!state->storageReused()) return false;
  }
  return true;
}

TransformerDecodeCompilation compileTransformerDecoder(
    metal::MetalRuntime &runtime, const TransformerDecodeWorkload &workload,
    std::ostream &log) {
  TransformerDecodeCompilation result;
  try {
    workload.plan.validate();
    if (!runtime.isAvailable()) {
      throw std::runtime_error(runtime.initializationError());
    }
    if (workload.layers.size() != workload.plan.layerCount) {
      throw std::invalid_argument("Transformer layer parameters do not match the plan.");
    }
    auto impl = std::make_unique<CompiledTransformerDecoder::Impl>();
    impl->runtime = &runtime;
    impl->plan = workload.plan;
    impl->encoderMemory = allocate(runtime, workload.encoderMemory.size(),
                                   workload.encoderMemory.data());
    impl->embeddingWeight = allocate(runtime, workload.embeddingWeight.size(),
                                     workload.embeddingWeight.data());
    impl->positionTable = allocate(runtime, workload.positionalEncoding.size(),
                                   workload.positionalEncoding.data());
    impl->languageModelHeadWeight = allocate(
        runtime, workload.languageModelHeadWeight.size(),
        workload.languageModelHeadWeight.data());
    impl->languageModelHeadBias = allocate(
        runtime, workload.languageModelHeadBias.size(),
        workload.languageModelHeadBias.data());

    for (const auto &parameters : workload.layers) {
      impl->layers.push_back(allocateLayer(runtime, workload, parameters));
      auto state = createKVCacheState(runtime, workload.plan.selfAttention);
      if (!state.state) throw std::runtime_error(state.errorMessage);
      impl->states.push_back(std::move(state.state));
    }

    const auto dimension = workload.plan.modelDimension();
    const auto memoryRows = workload.plan.selfAttention.batch *
                            workload.plan.sourceLength;
    const auto threads = workload.plan.selfAttention.threadsPerThreadgroup;
    std::vector<std::unique_ptr<metal::PreparedExecution>> setupSteps;
    for (std::size_t index = 0; index < impl->layers.size(); ++index) {
      auto &layer = impl->layers[index];
      const auto prefix = "transformer_setup_l" + std::to_string(index);
      setupSteps.push_back(prepare(
          runtime,
          metal::emitBiasedLinear(memoryRows, dimension, dimension, false,
                                  threads, prefix + "_cross_key"),
          {impl->encoderMemory, layer.crossKeyWeight, layer.crossKeyBias},
          {metal::ElementType::Float32, metal::ElementType::Float32,
           metal::ElementType::Float32},
          layer.projectedMemoryKey, metal::ElementType::Float32, log));
      setupSteps.push_back(prepare(
          runtime,
          metal::emitBiasedLinear(memoryRows, dimension, dimension, false,
                                  threads, prefix + "_cross_value"),
          {impl->encoderMemory, layer.crossValueWeight, layer.crossValueBias},
          {metal::ElementType::Float32, metal::ElementType::Float32,
           metal::ElementType::Float32},
          layer.projectedMemoryValue, metal::ElementType::Float32, log));
    }
    auto setup = makeSequence(runtime, setupSteps);
    const auto setupRun = setup->execute();
    if (!setupRun.executionPassed) {
      throw std::runtime_error("Transformer cross-attention setup failed: " +
                               setupRun.errorMessage);
    }
    log << "Transformer cross-attention memory setup: PASS\n";

    impl->prefillStage = buildStage(
        runtime, workload, workload.plan.selfAttention.prefillLength,
        impl->embeddingWeight, impl->positionTable,
        impl->languageModelHeadWeight, impl->languageModelHeadBias,
        impl->layers, impl->states, log);
    impl->decodeStage = buildStage(
        runtime, workload, 1, impl->embeddingWeight, impl->positionTable,
        impl->languageModelHeadWeight, impl->languageModelHeadBias,
        impl->layers, impl->states, log);
    result.executable = std::unique_ptr<CompiledTransformerDecoder>(
        new CompiledTransformerDecoder(std::move(impl)));
    log << "Transformer stateful compilation: PASS\n";
  } catch (const std::exception &error) {
    result.errorMessage = error.what();
  }
  return result;
}

} // namespace tensor::runtime
