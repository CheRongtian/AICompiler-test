#include "runtime/DecoderLLMExecutor.hpp"

#include "backend/metal/DecodeGEMVMetalEmitter.hpp"
#include "backend/metal/DecoderLLMMetalEmitter.hpp"
#include "backend/metal/KVCacheMetalEmitter.hpp"
#include "backend/metal/TransformerDecodeMetalEmitter.hpp"
#include "planner/DecodeGEMVTuner.hpp"
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
  log << "Decoder kernel " << kernel.functionName << ": PASS\n";
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
  Buffer inputNormWeight;
  Buffer queryWeight;
  Buffer keyWeight;
  Buffer valueWeight;
  Buffer outputWeight;
  Buffer postAttentionNormWeight;
  Buffer gateWeight;
  Buffer upWeight;
  Buffer downWeight;
};

LayerBuffers allocateLayer(metal::MetalRuntime &runtime,
                           const DecoderLLMLayerParameters &parameters) {
  auto weight = [&](const std::vector<float> &values) {
    return allocate(runtime, values.size(), values.data());
  };
  return {weight(parameters.inputNormWeight),
          weight(parameters.queryWeight),
          weight(parameters.keyWeight),
          weight(parameters.valueWeight),
          weight(parameters.outputWeight),
          weight(parameters.postAttentionNormWeight),
          weight(parameters.gateWeight),
          weight(parameters.upWeight),
          weight(parameters.downWeight)};
}

struct LinearChoice {
  bool useBaseline = true;
  metal::DecodeGEMVConfig config{128, 1};
};

LinearChoice tuneLinear(metal::MetalRuntime &runtime, std::size_t batch,
                        std::size_t inputSize, std::size_t outputSize,
                        const std::vector<float> &input,
                        const std::vector<float> &weight,
                        std::ostream &log) {
  const auto result = planner::tuneDecodeGEMV(
      runtime, batch, inputSize, outputSize, input, weight, log);
  if (!result.success) throw std::runtime_error(result.errorMessage);
  return {result.useBaseline, result.config};
}

metal::GeneratedKernel emitLinear(std::size_t rows, std::size_t inputSize,
                                  std::size_t outputSize,
                                  const LinearChoice &choice,
                                  bool decodeStage,
                                  const std::string &functionName) {
  if (decodeStage && !choice.useBaseline) {
    return metal::emitDecodeGEMV(rows, inputSize, outputSize, choice.config,
                                 functionName);
  }
  return metal::emitLinearBaseline(rows, inputSize, outputSize, 128,
                                   functionName);
}

struct LinearChoices {
  LinearChoice hiddenToHidden;
  LinearChoice hiddenToIntermediate;
  LinearChoice intermediateToHidden;
  LinearChoice hiddenToVocabulary;
};

struct DecoderStage {
  std::size_t sequenceLength = 0;
  Buffer tokenIds;
  Buffer logits;
  Buffer allNextTokenIds;
  std::vector<Buffer> intermediates;
  std::unique_ptr<metal::PreparedSequence> sequence;
};

Buffer intermediate(metal::MetalRuntime &runtime, DecoderStage &stage,
                    std::size_t count,
                    metal::ElementType type = metal::ElementType::Float32) {
  auto buffer = allocate(runtime, count, nullptr, type);
  stage.intermediates.push_back(buffer);
  return buffer;
}

DecoderStage buildStage(
    metal::MetalRuntime &runtime, const DecoderLLMWorkload &workload,
    std::size_t sequenceLength, const Buffer &embeddingWeight,
    const Buffer &finalNormWeight, const Buffer &languageModelHeadWeight,
    const Buffer &ropeCosine, const Buffer &ropeSine,
    const std::vector<LayerBuffers> &layers,
    const std::vector<std::unique_ptr<KVCacheState>> &states,
    const LinearChoices &choices, std::ostream &log) {
  DecoderStage stage;
  stage.sequenceLength = sequenceLength;
  const auto &plan = workload.plan;
  const auto batch = plan.attention.batch;
  const auto rows = batch * sequenceLength;
  const auto hidden = plan.hiddenSize();
  const auto hiddenCount = plan.hiddenElementCount(sequenceLength);
  const auto intermediateCount =
      plan.intermediateElementCount(sequenceLength);
  const auto threads = plan.attention.threadsPerThreadgroup;
  const bool decodeStage = sequenceLength == 1;
  const std::string stageName = decodeStage ? "decode" : "prefill";

  stage.tokenIds =
      allocate(runtime, rows, nullptr, metal::ElementType::Int32);
  stage.logits = allocate(runtime, plan.logitsElementCount(sequenceLength));
  stage.allNextTokenIds =
      allocate(runtime, rows, nullptr, metal::ElementType::Int32);

  std::vector<std::unique_ptr<metal::PreparedExecution>> steps;
  Buffer current = intermediate(runtime, stage, hiddenCount);
  steps.push_back(prepare(
      runtime,
      metal::emitDecoderEmbedding(plan, sequenceLength,
                                  "decoder_" + stageName + "_embedding"),
      {stage.tokenIds, embeddingWeight},
      {metal::ElementType::Int32, metal::ElementType::Float32}, current,
      metal::ElementType::Float32, log));

  for (std::size_t index = 0; index < layers.size(); ++index) {
    const auto prefix = "decoder_" + stageName + "_l" +
                        std::to_string(index) + "_";
    const auto &layer = layers[index];
    auto inputNormalized = intermediate(runtime, stage, hiddenCount);
    auto query = intermediate(runtime, stage, hiddenCount);
    auto key = intermediate(runtime, stage, hiddenCount);
    auto value = intermediate(runtime, stage, hiddenCount);
    auto rotatedQuery = intermediate(runtime, stage, hiddenCount);
    auto rotatedKey = intermediate(runtime, stage, hiddenCount);
    auto context = intermediate(runtime, stage, hiddenCount);
    auto attentionOutput = intermediate(runtime, stage, hiddenCount);
    auto afterAttention = intermediate(runtime, stage, hiddenCount);
    auto postAttentionNormalized = intermediate(runtime, stage, hiddenCount);
    auto gate = intermediate(runtime, stage, intermediateCount);
    auto up = intermediate(runtime, stage, intermediateCount);
    auto gated = intermediate(runtime, stage, intermediateCount);
    auto mlpOutput = intermediate(runtime, stage, hiddenCount);
    auto layerOutput = intermediate(runtime, stage, hiddenCount);

    steps.push_back(prepare(
        runtime,
        metal::emitDecoderRMSNorm(rows, hidden, plan.rmsNormEpsilon, threads,
                                  prefix + "input_norm"),
        {current, layer.inputNormWeight},
        {metal::ElementType::Float32, metal::ElementType::Float32},
        inputNormalized, metal::ElementType::Float32, log));
    steps.push_back(prepare(
        runtime,
        emitLinear(rows, hidden, hidden, choices.hiddenToHidden, decodeStage,
                   prefix + "query"),
        {inputNormalized, layer.queryWeight},
        {metal::ElementType::Float32, metal::ElementType::Float32}, query,
        metal::ElementType::Float32, log));
    steps.push_back(prepare(
        runtime,
        emitLinear(rows, hidden, hidden, choices.hiddenToHidden, decodeStage,
                   prefix + "key"),
        {inputNormalized, layer.keyWeight},
        {metal::ElementType::Float32, metal::ElementType::Float32}, key,
        metal::ElementType::Float32, log));
    steps.push_back(prepare(
        runtime,
        emitLinear(rows, hidden, hidden, choices.hiddenToHidden, decodeStage,
                   prefix + "value"),
        {inputNormalized, layer.valueWeight},
        {metal::ElementType::Float32, metal::ElementType::Float32}, value,
        metal::ElementType::Float32, log));
    steps.push_back(prepare(
        runtime, metal::emitDecoderRoPE(plan, sequenceLength, prefix + "q_rope"),
        {query, ropeCosine, ropeSine, states[index]->lengthBuffer()},
        {metal::ElementType::Float32, metal::ElementType::Float32,
         metal::ElementType::Float32, metal::ElementType::Int32},
        rotatedQuery, metal::ElementType::Float32, log));
    steps.push_back(prepare(
        runtime, metal::emitDecoderRoPE(plan, sequenceLength, prefix + "k_rope"),
        {key, ropeCosine, ropeSine, states[index]->lengthBuffer()},
        {metal::ElementType::Float32, metal::ElementType::Float32,
         metal::ElementType::Float32, metal::ElementType::Int32},
        rotatedKey, metal::ElementType::Float32, log));
    steps.push_back(prepare(
        runtime,
        metal::emitDecoderCacheAppend(plan, sequenceLength, true,
                                      prefix + "key_cache_append"),
        {rotatedKey, states[index]->lengthBuffer()},
        {metal::ElementType::Float32, metal::ElementType::Int32},
        states[index]->keyBuffer(), metal::ElementType::Float32, log));
    steps.push_back(prepare(
        runtime,
        metal::emitDecoderCacheAppend(plan, sequenceLength, false,
                                      prefix + "value_cache_append"),
        {value, states[index]->lengthBuffer()},
        {metal::ElementType::Float32, metal::ElementType::Int32},
        states[index]->valueBuffer(), metal::ElementType::Float32, log));
    steps.push_back(prepare(
        runtime, metal::emitKVAttention(plan.attention, sequenceLength, true),
        {rotatedQuery, states[index]->keyBuffer(), states[index]->valueBuffer(),
         states[index]->lengthBuffer()},
        {metal::ElementType::Float32, metal::ElementType::Float32,
         metal::ElementType::Float32, metal::ElementType::Int32},
        context, metal::ElementType::Float32, log));
    steps.push_back(prepare(
        runtime,
        emitLinear(rows, hidden, hidden, choices.hiddenToHidden, decodeStage,
                   prefix + "attention_output"),
        {context, layer.outputWeight},
        {metal::ElementType::Float32, metal::ElementType::Float32},
        attentionOutput, metal::ElementType::Float32, log));
    steps.push_back(prepare(
        runtime, metal::emitDecoderAdd(hiddenCount, threads,
                                       prefix + "attention_residual"),
        {current, attentionOutput},
        {metal::ElementType::Float32, metal::ElementType::Float32},
        afterAttention, metal::ElementType::Float32, log));
    steps.push_back(prepare(
        runtime,
        metal::emitDecoderRMSNorm(rows, hidden, plan.rmsNormEpsilon, threads,
                                  prefix + "post_attention_norm"),
        {afterAttention, layer.postAttentionNormWeight},
        {metal::ElementType::Float32, metal::ElementType::Float32},
        postAttentionNormalized, metal::ElementType::Float32, log));
    steps.push_back(prepare(
        runtime,
        emitLinear(rows, hidden, plan.intermediateSize,
                   choices.hiddenToIntermediate, decodeStage, prefix + "gate"),
        {postAttentionNormalized, layer.gateWeight},
        {metal::ElementType::Float32, metal::ElementType::Float32}, gate,
        metal::ElementType::Float32, log));
    steps.push_back(prepare(
        runtime,
        emitLinear(rows, hidden, plan.intermediateSize,
                   choices.hiddenToIntermediate, decodeStage, prefix + "up"),
        {postAttentionNormalized, layer.upWeight},
        {metal::ElementType::Float32, metal::ElementType::Float32}, up,
        metal::ElementType::Float32, log));
    steps.push_back(prepare(
        runtime, metal::emitDecoderSiLUMul(intermediateCount, threads,
                                           prefix + "silu_mul"),
        {gate, up},
        {metal::ElementType::Float32, metal::ElementType::Float32}, gated,
        metal::ElementType::Float32, log));
    steps.push_back(prepare(
        runtime,
        emitLinear(rows, plan.intermediateSize, hidden,
                   choices.intermediateToHidden, decodeStage, prefix + "down"),
        {gated, layer.downWeight},
        {metal::ElementType::Float32, metal::ElementType::Float32}, mlpOutput,
        metal::ElementType::Float32, log));
    steps.push_back(prepare(
        runtime,
        metal::emitDecoderAdd(hiddenCount, threads, prefix + "mlp_residual"),
        {afterAttention, mlpOutput},
        {metal::ElementType::Float32, metal::ElementType::Float32}, layerOutput,
        metal::ElementType::Float32, log));
    current = layerOutput;
  }

  auto normalized = intermediate(runtime, stage, hiddenCount);
  steps.push_back(prepare(
      runtime,
      metal::emitDecoderRMSNorm(rows, hidden, plan.rmsNormEpsilon, threads,
                                "decoder_" + stageName + "_final_norm"),
      {current, finalNormWeight},
      {metal::ElementType::Float32, metal::ElementType::Float32}, normalized,
      metal::ElementType::Float32, log));
  steps.push_back(prepare(
      runtime,
      emitLinear(rows, hidden, plan.vocabularySize,
                 choices.hiddenToVocabulary, decodeStage,
                 "decoder_" + stageName + "_lm_head"),
      {normalized, languageModelHeadWeight},
      {metal::ElementType::Float32, metal::ElementType::Float32}, stage.logits,
      metal::ElementType::Float32, log));
  steps.push_back(prepare(
      runtime,
      metal::emitTokenArgmax(rows, plan.vocabularySize, threads,
                             "decoder_" + stageName + "_argmax"),
      {stage.logits}, {metal::ElementType::Float32}, stage.allNextTokenIds,
      metal::ElementType::Int32, log));
  stage.sequence = makeSequence(runtime, steps);
  return stage;
}

} // namespace

class CompiledDecoderLLM::Impl {
public:
  metal::MetalRuntime *runtime = nullptr;
  planner::DecoderLLMPlan plan;
  std::vector<std::unique_ptr<KVCacheState>> states;
  Buffer embeddingWeight;
  Buffer finalNormWeight;
  Buffer languageModelHeadWeight;
  Buffer ropeCosine;
  Buffer ropeSine;
  std::vector<LayerBuffers> layers;
  DecoderStage prefillStage;
  DecoderStage decodeStage;

  [[nodiscard]] std::size_t currentLength() const noexcept {
    return states.empty() ? 0 : states.front()->currentLength();
  }

  DecoderLLMRunResult run(DecoderStage &stage, std::size_t nextLength,
                          const std::vector<float> &tokenIds) {
    DecoderLLMRunResult result;
    if (tokenIds.size() != stage.tokenIds->elementCount()) {
      result.errorMessage =
          "Decoder token input shape does not match the compiled stage.";
      return result;
    }
    auto error =
        runtime->writeBuffer(stage.tokenIds, tokenIds.data(), tokenIds.size());
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
    result.nextTokenIds.reserve(plan.attention.batch);
    for (std::size_t batch = 0; batch < plan.attention.batch; ++batch) {
      const auto row = batch * stage.sequenceLength + stage.sequenceLength - 1;
      result.nextTokenIds.push_back(allTokens[row]);
    }
    result.gpuExecutionTimeUs = execution.gpuExecutionTimeUs;
    result.cpuSubmitToCompletionTimeUs = execution.cpuSubmitToCompletionTimeUs;
    result.passed = true;
    return result;
  }
};

CompiledDecoderLLM::CompiledDecoderLLM(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
CompiledDecoderLLM::~CompiledDecoderLLM() = default;

std::string CompiledDecoderLLM::reset() {
  for (auto &state : impl_->states) {
    const auto error = state->reset(*impl_->runtime);
    if (!error.empty()) return error;
  }
  return {};
}

DecoderLLMRunResult
CompiledDecoderLLM::prefill(const std::vector<float> &tokenIds) {
  if (currentLength() != 0) {
    return {false, {}, {}, {}, 0.0,
            "Decoder prefill requires an empty cache; call reset first."};
  }
  return impl_->run(impl_->prefillStage, impl_->plan.attention.prefillLength,
                    tokenIds);
}

DecoderLLMRunResult
CompiledDecoderLLM::decode(const std::vector<float> &tokenIds) {
  const auto current = currentLength();
  if (current == 0) {
    return {false, {}, {}, {}, 0.0,
            "Decoder decode requires a completed prefill."};
  }
  if (current >= impl_->plan.attention.capacity) {
    return {false, {}, {}, {}, 0.0,
            "Decoder KV cache capacity has been reached."};
  }
  return impl_->run(impl_->decodeStage, current + 1, tokenIds);
}

std::size_t CompiledDecoderLLM::currentLength() const noexcept {
  return impl_->currentLength();
}

std::vector<float>
CompiledDecoderLLM::readKeyPrefix(std::size_t layer) const {
  if (layer >= impl_->states.size()) {
    throw std::out_of_range("Decoder layer index is out of range.");
  }
  return impl_->states[layer]->readKeyPrefix();
}

std::vector<float>
CompiledDecoderLLM::readValuePrefix(std::size_t layer) const {
  if (layer >= impl_->states.size()) {
    throw std::out_of_range("Decoder layer index is out of range.");
  }
  return impl_->states[layer]->readValuePrefix();
}

bool CompiledDecoderLLM::cacheStorageReused() const noexcept {
  for (const auto &state : impl_->states) {
    if (!state->storageReused()) return false;
  }
  return true;
}

DecoderLLMCompilation compileDecoderLLM(
    metal::MetalRuntime &runtime, const DecoderLLMWorkload &workload,
    std::ostream &log) {
  DecoderLLMCompilation result;
  try {
    workload.plan.validate();
    if (!runtime.isAvailable()) {
      throw std::runtime_error(runtime.initializationError());
    }
    if (workload.layers.size() != workload.plan.layerCount) {
      throw std::invalid_argument(
          "Decoder layer parameters do not match the plan.");
    }

    const auto &plan = workload.plan;
    const auto hidden = plan.hiddenSize();
    const auto &firstLayer = workload.layers.front();
    std::vector<float> hiddenInput(
        workload.embeddingWeight.begin(),
        workload.embeddingWeight.begin() + static_cast<std::ptrdiff_t>(hidden));
    std::vector<float> intermediateInput(plan.intermediateSize);
    for (std::size_t index = 0; index < intermediateInput.size(); ++index) {
      intermediateInput[index] =
          static_cast<float>(static_cast<int>(index % 17) - 8) / 17.0f;
    }

    log << "Decoder Decode-GEMV admission:\n";
    LinearChoices choices;
    choices.hiddenToHidden = tuneLinear(
        runtime, plan.attention.batch, hidden, hidden, hiddenInput,
        firstLayer.queryWeight, log);
    choices.hiddenToIntermediate = tuneLinear(
        runtime, plan.attention.batch, hidden, plan.intermediateSize,
        hiddenInput, firstLayer.gateWeight, log);
    choices.intermediateToHidden = tuneLinear(
        runtime, plan.attention.batch, plan.intermediateSize, hidden,
        intermediateInput, firstLayer.downWeight, log);
    choices.hiddenToVocabulary = tuneLinear(
        runtime, plan.attention.batch, hidden, plan.vocabularySize,
        hiddenInput, workload.languageModelHeadWeight, log);

    auto impl = std::make_unique<CompiledDecoderLLM::Impl>();
    impl->runtime = &runtime;
    impl->plan = plan;
    impl->embeddingWeight = allocate(runtime, workload.embeddingWeight.size(),
                                     workload.embeddingWeight.data());
    impl->finalNormWeight = allocate(runtime, workload.finalNormWeight.size(),
                                     workload.finalNormWeight.data());
    impl->languageModelHeadWeight = allocate(
        runtime, workload.languageModelHeadWeight.size(),
        workload.languageModelHeadWeight.data());
    impl->ropeCosine = allocate(runtime, workload.ropeCosine.size(),
                                workload.ropeCosine.data());
    impl->ropeSine = allocate(runtime, workload.ropeSine.size(),
                              workload.ropeSine.data());
    for (const auto &parameters : workload.layers) {
      impl->layers.push_back(allocateLayer(runtime, parameters));
      auto state = createKVCacheState(runtime, plan.attention);
      if (!state.state) throw std::runtime_error(state.errorMessage);
      impl->states.push_back(std::move(state.state));
    }

    impl->prefillStage = buildStage(
        runtime, workload, plan.attention.prefillLength, impl->embeddingWeight,
        impl->finalNormWeight, impl->languageModelHeadWeight, impl->ropeCosine,
        impl->ropeSine, impl->layers, impl->states, choices, log);
    impl->decodeStage = buildStage(
        runtime, workload, 1, impl->embeddingWeight, impl->finalNormWeight,
        impl->languageModelHeadWeight, impl->ropeCosine, impl->ropeSine,
        impl->layers, impl->states, choices, log);
    result.executable = std::unique_ptr<CompiledDecoderLLM>(
        new CompiledDecoderLLM(std::move(impl)));
    log << "Decoder-only stateful compilation: PASS\n";
  } catch (const std::exception &error) {
    result.errorMessage = error.what();
  }
  return result;
}

} // namespace tensor::runtime
