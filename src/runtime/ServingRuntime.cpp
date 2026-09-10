#include "runtime/ServingRuntime.hpp"

#include "backend/metal/DecodeGEMVMetalEmitter.hpp"
#include "backend/metal/DecoderLLMMetalEmitter.hpp"
#include "backend/metal/KVCacheMetalEmitter.hpp"
#include "backend/metal/TransformerDecodeMetalEmitter.hpp"
#include "runtime/DecoderLLMExecutor.hpp"
#include "runtime/KVCacheState.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace tensor::runtime {
namespace {

using Clock = std::chrono::steady_clock;
using Buffer = metal::BufferHandle;

double elapsedUs(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::micro>(end - start).count();
}

std::size_t pagesFor(std::size_t tokens, std::size_t pageSize) {
  return tokens == 0 ? 0 : (tokens + pageSize - 1) / pageSize;
}

Buffer allocate(metal::MetalRuntime &runtime, std::size_t count,
                const float *data = nullptr,
                metal::ElementType type = metal::ElementType::Float32) {
  auto result = runtime.createBuffer(count, data, type);
  if (!result.buffer) throw std::runtime_error(result.errorMessage);
  return std::move(result.buffer);
}

metal::ElementType metalStorageType(DType dtype) {
  switch (dtype) {
  case DType::Float16: return metal::ElementType::Float16;
  case DType::Float32: return metal::ElementType::Float32;
  case DType::BFloat16: return metal::ElementType::BFloat16;
  case DType::Int32: return metal::ElementType::Int32;
  }
  throw std::invalid_argument("Unsupported serving storage dtype.");
}

std::unique_ptr<metal::PreparedExecution> prepare(
    metal::MetalRuntime &runtime, const metal::GeneratedKernel &kernel,
    const std::vector<Buffer> &inputs,
    const std::vector<metal::ElementType> &inputTypes,
    const std::vector<Buffer> &outputs,
    const std::vector<metal::ElementType> &outputTypes) {
  const auto pipeline =
      runtime.createComputePipeline(kernel.source, kernel.functionName);
  if (!pipeline.pipelineCreationPassed) {
    throw std::runtime_error(kernel.functionName + ": " +
                             pipeline.errorMessage);
  }
  const auto interfaceError =
      metal::checkBufferInterface(pipeline, inputTypes, outputTypes);
  if (!interfaceError.empty()) {
    throw std::runtime_error(kernel.functionName + ": " + interfaceError);
  }
  auto result = runtime.prepareBuffers(
      inputs, outputs,
      {kernel.threadgroupCount, kernel.threadsPerThreadgroup});
  if (!result.execution) {
    throw std::runtime_error(kernel.functionName + ": " + result.errorMessage);
  }
  return std::move(result.execution);
}

std::unique_ptr<metal::PreparedExecution> prepare(
    metal::MetalRuntime &runtime, const metal::GeneratedKernel &kernel,
    const std::vector<Buffer> &inputs,
    const std::vector<metal::ElementType> &inputTypes, const Buffer &output,
    metal::ElementType outputType) {
  return prepare(runtime, kernel, inputs, inputTypes,
                 std::vector<Buffer>{output},
                 std::vector<metal::ElementType>{outputType});
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

enum class RequestState { Waiting, Decode, Preempted, Finished };

struct Request {
  ServingRequestSpec spec;
  DecoderLLMWorkload workload;
  std::unique_ptr<CompiledDecoderLLM> decoder;
  RequestState state = RequestState::Waiting;
  ServingRequestOutput output;
  std::vector<float> nextTokenIds;
  std::optional<DecoderKVSnapshot> snapshot;
  std::optional<Clock::time_point> arrivalTime;
  std::size_t decoded = 0;
};

struct BatchedDecodeRun {
  bool passed = false;
  std::vector<float> logits;
  std::vector<float> nextTokenIds;
  std::optional<double> gpuExecutionTimeUs;
  double cpuSubmitToCompletionTimeUs = 0.0;
  std::string errorMessage;
};

class BatchedDecodeExecutor {
public:
  BatchedDecodeExecutor(
      metal::MetalRuntime &runtime, const DecoderLLMWorkload &workload,
      const std::vector<std::shared_ptr<KVPagePool>> &pools,
      std::size_t maximumBatchSize, std::size_t pageSize,
      std::shared_ptr<DecoderLLMModelResources> modelResources)
      : runtime_(runtime), plan_(workload.plan), pools_(pools),
        maximumBatchSize_(maximumBatchSize), pageSize_(pageSize),
        storageType_(metalStorageType(workload.plan.attention.dtype)),
        modelResources_(std::move(modelResources)) {
    if (maximumBatchSize_ == 0 || pageSize_ == 0 ||
        pools_.size() != plan_.layerCount ||
        workload.layers.size() != plan_.layerCount || !modelResources_ ||
        modelResources_->storageDtype != workload.plan.attention.dtype ||
        modelResources_->layerCount != plan_.layerCount ||
        modelResources_->hiddenSize != plan_.hiddenSize() ||
        modelResources_->headCount != plan_.attention.heads ||
        modelResources_->headDimension != plan_.attention.headDimension ||
        modelResources_->cacheCapacity != plan_.attention.capacity ||
        modelResources_->intermediateSize != plan_.intermediateSize ||
        modelResources_->vocabularySize != plan_.vocabularySize) {
      throw std::invalid_argument(
          "Batched serving decoder configuration is invalid.");
    }
    maximumPages_ = pagesFor(plan_.attention.capacity, pageSize_);
    plan_.attention.batch = maximumBatchSize_;
    plan_.attention.prefillLength = 1;
    plan_.validate();
    build();
  }

  [[nodiscard]] BatchedDecodeRun run(
      const std::vector<float> &tokenIds,
      const std::vector<float> &validLengths,
      const std::vector<std::vector<float>> &blockTables) {
    BatchedDecodeRun result;
    if (tokenIds.empty() || tokenIds.size() > maximumBatchSize_ ||
        tokenIds.size() != validLengths.size() ||
        blockTables.size() != plan_.layerCount) {
      result.errorMessage = "Batched decode metadata shape is invalid.";
      return result;
    }
    for (const auto &table : blockTables) {
      if (table.size() != tokenIds.size() * maximumPages_) {
        result.errorMessage = "Batched decode block-table shape is invalid.";
        return result;
      }
    }

    std::vector<float> paddedTokens(maximumBatchSize_, 0.0f);
    std::vector<float> paddedLengths(maximumBatchSize_, 0.0f);
    std::copy(tokenIds.begin(), tokenIds.end(), paddedTokens.begin());
    std::copy(validLengths.begin(), validLengths.end(), paddedLengths.begin());
    if (auto error = runtime_.writeBuffer(tokenIds_, paddedTokens.data(),
                                          paddedTokens.size());
        !error.empty()) {
      result.errorMessage = error;
      return result;
    }
    if (auto error = runtime_.writeBuffer(validLengths_, paddedLengths.data(),
                                          paddedLengths.size());
        !error.empty()) {
      result.errorMessage = error;
      return result;
    }
    const float activeCount = static_cast<float>(tokenIds.size());
    if (auto error = runtime_.writeBuffer(activeCount_, &activeCount, 1);
        !error.empty()) {
      result.errorMessage = error;
      return result;
    }
    for (std::size_t layer = 0; layer < blockTables.size(); ++layer) {
      std::vector<float> packed(maximumBatchSize_ * maximumPages_, -1.0f);
      std::copy(blockTables[layer].begin(), blockTables[layer].end(),
                packed.begin());
      if (auto error = runtime_.writeBuffer(
              blockTables_[layer], packed.data(), packed.size());
          !error.empty()) {
        result.errorMessage = error;
        return result;
      }
    }

    const auto execution = sequence_->execute();
    if (!execution.executionPassed) {
      result.errorMessage = execution.errorMessage;
      return result;
    }
    const auto allLogits = logits_->read();
    const auto allTokens = nextTokenIds_->read();
    const auto activeLogitCount = tokenIds.size() * plan_.vocabularySize;
    result.logits.assign(
        allLogits.begin(),
        allLogits.begin() + static_cast<std::ptrdiff_t>(activeLogitCount));
    result.nextTokenIds.assign(
        allTokens.begin(),
        allTokens.begin() + static_cast<std::ptrdiff_t>(tokenIds.size()));
    result.gpuExecutionTimeUs = execution.gpuExecutionTimeUs;
    result.cpuSubmitToCompletionTimeUs =
        execution.cpuSubmitToCompletionTimeUs;
    result.passed = true;
    return result;
  }

  [[nodiscard]] std::size_t activationStorageBytes() const noexcept {
    return activationStorageBytes_;
  }

private:
  void build() {
    const auto rows = maximumBatchSize_;
    const auto hidden = plan_.hiddenSize();
    const auto hiddenCount = rows * hidden;
    const auto intermediateCount = rows * plan_.intermediateSize;
    const auto threads = plan_.attention.threadsPerThreadgroup;
    const auto storageBytes = dtypeStorageBytes(plan_.attention.dtype);

    activationStorageBytes_ =
        (3 * rows + 1 + plan_.layerCount * rows * maximumPages_) *
            sizeof(std::int32_t) +
        rows * plan_.vocabularySize * sizeof(float) +
        (13 * hiddenCount + intermediateCount) * storageBytes;

    tokenIds_ = allocate(runtime_, rows, nullptr, metal::ElementType::Int32);
    validLengths_ =
        allocate(runtime_, rows, nullptr, metal::ElementType::Int32);
    activeCount_ = allocate(runtime_, 1, nullptr, metal::ElementType::Int32);
    logits_ = allocate(runtime_, rows * plan_.vocabularySize, nullptr,
                       metal::ElementType::Float32);
    nextTokenIds_ =
        allocate(runtime_, rows, nullptr, metal::ElementType::Int32);
    blockTables_.reserve(plan_.layerCount);
    for (std::size_t layer = 0; layer < plan_.layerCount; ++layer) {
      blockTables_.push_back(allocate(runtime_, rows * maximumPages_, nullptr,
                                      metal::ElementType::Int32));
    }

    const auto &embeddingWeight = modelResources_->embeddingWeight;
    const auto &finalNormWeight = modelResources_->finalNormWeight;
    const auto &languageModelHeadWeight =
        modelResources_->languageModelHeadWeight;
    const auto &ropeCosine = modelResources_->ropeCosine;
    const auto &ropeSine = modelResources_->ropeSine;
    const auto &layers = modelResources_->layers;

    std::vector<std::unique_ptr<metal::PreparedExecution>> steps;
    auto intermediate = [&](std::size_t count) {
      return allocate(runtime_, count, nullptr, storageType_);
    };
    auto add = [&](const metal::GeneratedKernel &kernel,
                   const std::vector<Buffer> &inputs,
                   const std::vector<metal::ElementType> &inputTypes,
                   const Buffer &output,
                   metal::ElementType outputType = metal::ElementType::Float32) {
      steps.push_back(prepare(runtime_, kernel, inputs, inputTypes, output,
                              outputType));
    };
    auto addLinear = [&](std::size_t inputSize, std::size_t outputSize,
                         const std::string &name, const Buffer &input,
                         const Buffer &matrix, const Buffer &output) {
      const auto outputType = output->elementType();
      const metal::DecodeGEMVConfig configuration{
          128, inputSize % 4 == 0 ? std::size_t{4} : std::size_t{1}};
      add(metal::emitDecodeGEMV(rows, inputSize, outputSize, configuration,
                                name, storageType_, outputType),
          {input, matrix},
          {storageType_, storageType_}, output, outputType);
    };

    auto current = intermediate(hiddenCount);
    auto nextLayerOutput = intermediate(hiddenCount);
    std::vector<Buffer> hiddenScratch;
    hiddenScratch.reserve(11);
    for (std::size_t slot = 0; slot < 11; ++slot) {
      hiddenScratch.push_back(intermediate(hiddenCount));
    }
    const auto gated = intermediate(intermediateCount);
    add(metal::emitDecoderEmbedding(plan_, 1, "serving_decode_embedding",
                                    storageType_),
        {tokenIds_, embeddingWeight},
        {metal::ElementType::Int32, storageType_}, current, storageType_);

    for (std::size_t index = 0; index < layers.size(); ++index) {
      const auto prefix = "serving_decode_l" + std::to_string(index) + "_";
      const auto &layer = layers[index];
      const auto &inputNormalized = hiddenScratch[0];
      const auto &query = hiddenScratch[1];
      const auto &key = hiddenScratch[2];
      const auto &value = hiddenScratch[3];
      const auto &rotatedQuery = hiddenScratch[4];
      const auto &rotatedKey = hiddenScratch[5];
      const auto &context = hiddenScratch[6];
      const auto &attentionOutput = hiddenScratch[7];
      const auto &afterAttention = hiddenScratch[8];
      const auto &postAttentionNormalized = hiddenScratch[9];
      const auto &mlpOutput = hiddenScratch[10];
      const auto &layerOutput = nextLayerOutput;

      add(metal::emitDecoderRMSNorm(rows, hidden, plan_.rmsNormEpsilon,
                                    threads, prefix + "input_norm", storageType_),
          {current, layer.inputNormWeight},
          {storageType_, storageType_}, inputNormalized, storageType_);
      addLinear(hidden, hidden, prefix + "query", inputNormalized,
                layer.queryWeight, query);
      addLinear(hidden, hidden, prefix + "key", inputNormalized,
                layer.keyWeight, key);
      addLinear(hidden, hidden, prefix + "value", inputNormalized,
                layer.valueWeight, value);
      add(metal::emitServingRoPE(plan_, maximumBatchSize_,
                                 prefix + "query_rope", storageType_),
          {query, ropeCosine, ropeSine, validLengths_, activeCount_},
          {storageType_, metal::ElementType::Float32,
           metal::ElementType::Float32, metal::ElementType::Int32,
           metal::ElementType::Int32},
          rotatedQuery, storageType_);
      add(metal::emitServingRoPE(plan_, maximumBatchSize_,
                                 prefix + "key_rope", storageType_),
          {key, ropeCosine, ropeSine, validLengths_, activeCount_},
          {storageType_, metal::ElementType::Float32,
           metal::ElementType::Float32, metal::ElementType::Int32,
           metal::ElementType::Int32},
          rotatedKey, storageType_);
      add(metal::emitServingPagedCacheAppend(
              plan_, maximumBatchSize_, true, pageSize_,
              prefix + "key_cache_append", storageType_),
          {rotatedKey, validLengths_, blockTables_[index], activeCount_},
          {storageType_, metal::ElementType::Int32,
           metal::ElementType::Int32, metal::ElementType::Int32},
          pools_[index]->keyBuffer(), storageType_);
      add(metal::emitServingPagedCacheAppend(
              plan_, maximumBatchSize_, false, pageSize_,
              prefix + "value_cache_append", storageType_),
          {value, validLengths_, blockTables_[index], activeCount_},
          {storageType_, metal::ElementType::Int32,
           metal::ElementType::Int32, metal::ElementType::Int32},
          pools_[index]->valueBuffer(), storageType_);
      add(metal::emitServingPagedKVAttention(
              plan_.attention, maximumBatchSize_, pageSize_,
              prefix + "paged_attention"),
          {rotatedQuery, pools_[index]->keyBuffer(),
           pools_[index]->valueBuffer(), validLengths_, blockTables_[index],
           activeCount_},
          {storageType_, storageType_, storageType_, metal::ElementType::Int32,
           metal::ElementType::Int32, metal::ElementType::Int32},
          context, storageType_);
      addLinear(hidden, hidden, prefix + "attention_output", context,
                layer.outputWeight, attentionOutput);
      const auto residualNorm = metal::emitDecoderResidualRMSNorm(
          rows, hidden, plan_.rmsNormEpsilon, threads,
          prefix + "residual_rmsnorm", storageType_);
      steps.push_back(prepare(
          runtime_, residualNorm,
          {current, attentionOutput, layer.postAttentionNormWeight},
          {storageType_, storageType_, storageType_},
          {afterAttention, postAttentionNormalized},
          {storageType_, storageType_}));
      add(metal::emitDecoderGatedMLP(
              rows, hidden, plan_.intermediateSize, threads,
              prefix + "gated_mlp", storageType_),
          {postAttentionNormalized, layer.gateWeight, layer.upWeight},
          {storageType_, storageType_, storageType_}, gated, storageType_);
      addLinear(plan_.intermediateSize, hidden, prefix + "down", gated,
                layer.downWeight, mlpOutput);
      add(metal::emitDecoderAdd(hiddenCount, threads,
                                prefix + "mlp_residual", storageType_),
          {afterAttention, mlpOutput},
          {storageType_, storageType_}, layerOutput, storageType_);
      std::swap(current, nextLayerOutput);
    }

    const auto &normalized = hiddenScratch[0];
    add(metal::emitDecoderRMSNorm(rows, hidden, plan_.rmsNormEpsilon, threads,
                                  "serving_decode_final_norm", storageType_),
        {current, finalNormWeight},
        {storageType_, storageType_}, normalized, storageType_);
    addLinear(hidden, plan_.vocabularySize, "serving_decode_lm_head",
              normalized, languageModelHeadWeight, logits_);
    add(metal::emitTokenArgmax(rows, plan_.vocabularySize, threads,
                               "serving_decode_argmax",
                               metal::ElementType::Float32),
        {logits_}, {metal::ElementType::Float32}, nextTokenIds_,
        metal::ElementType::Int32);
    sequence_ = makeSequence(runtime_, steps);
  }

  metal::MetalRuntime &runtime_;
  planner::DecoderLLMPlan plan_;
  std::vector<std::shared_ptr<KVPagePool>> pools_;
  std::size_t maximumBatchSize_ = 0;
  std::size_t pageSize_ = 0;
  metal::ElementType storageType_ = metal::ElementType::Float32;
  std::shared_ptr<DecoderLLMModelResources> modelResources_;
  std::size_t maximumPages_ = 0;
  std::size_t activationStorageBytes_ = 0;
  Buffer tokenIds_;
  Buffer validLengths_;
  Buffer activeCount_;
  std::vector<Buffer> blockTables_;
  Buffer logits_;
  Buffer nextTokenIds_;
  std::unique_ptr<metal::PreparedSequence> sequence_;
};

std::size_t availablePages(
    const std::vector<std::shared_ptr<KVPagePool>> &pools) {
  std::size_t available = std::numeric_limits<std::size_t>::max();
  for (const auto &pool : pools) {
    available = std::min(available, pool->availablePages());
  }
  return pools.empty() ? 0 : available;
}

bool auditPageIsolation(const std::vector<Request> &requests,
                        std::size_t layerCount, std::string &error) {
  for (std::size_t layer = 0; layer < layerCount; ++layer) {
    std::set<std::int32_t> owners;
    for (const auto &request : requests) {
      if (request.state != RequestState::Decode) continue;
      for (const auto page : request.decoder->kvBlockTable(layer)) {
        if (!owners.insert(page).second) {
          error = "Two active requests own the same physical KV page in layer " +
                  std::to_string(layer) + ".";
          return false;
        }
      }
    }
  }
  return true;
}

std::optional<std::size_t> selectPreemptionVictim(
    const std::vector<Request> &requests) {
  std::optional<std::size_t> selected;
  std::size_t selectedPages = 0;
  for (std::size_t index = 0; index < requests.size(); ++index) {
    if (requests[index].state != RequestState::Decode) continue;
    const auto pages = requests[index].decoder->allocatedKVPageCount();
    if (!selected || pages > selectedPages) {
      selected = index;
      selectedPages = pages;
    }
  }
  return selected;
}

std::string preempt(Request &request, ServingMetrics &metrics) {
  const auto start = Clock::now();
  request.snapshot = request.decoder->snapshotKVCache();
  const auto error = request.decoder->releaseKVCache();
  metrics.preemptionOverheadUs += elapsedUs(start, Clock::now());
  if (!error.empty()) return error;
  request.state = RequestState::Preempted;
  ++metrics.preemptionCount;
  return {};
}

std::string resume(Request &request, ServingMetrics &metrics) {
  if (!request.snapshot) return "Preempted request has no KV snapshot.";
  const auto start = Clock::now();
  const auto error = request.decoder->restoreKVCache(*request.snapshot);
  metrics.preemptionOverheadUs += elapsedUs(start, Clock::now());
  if (!error.empty()) return error;
  request.snapshot.reset();
  request.state = RequestState::Decode;
  ++metrics.resumeCount;
  return {};
}

std::vector<std::size_t> activeRequests(const std::vector<Request> &requests) {
  std::vector<std::size_t> active;
  for (std::size_t index = 0; index < requests.size(); ++index) {
    if (requests[index].state == RequestState::Decode) active.push_back(index);
  }
  return active;
}

std::size_t additionalPagesForNextToken(
    const std::vector<Request> &requests,
    const std::vector<std::size_t> &active, std::size_t pageSize) {
  std::size_t required = 0;
  for (const auto index : active) {
    const auto length = requests[index].decoder->currentLength();
    required += pagesFor(length + 1, pageSize) - pagesFor(length, pageSize);
  }
  return required;
}

} // namespace

ServingExecutionResult runContinuousBatch(
    metal::MetalRuntime &runtime, const DecoderLLMWorkload &baseWorkload,
    const std::vector<ServingRequestSpec> &requestSpecs,
    const ServingConfig &config, std::ostream &log) {
  ServingExecutionResult result;
  try {
    if (requestSpecs.empty() || config.pageSize == 0 ||
        config.physicalPagesPerLayer == 0 || config.prefillChunkSize == 0) {
      throw std::invalid_argument(
          "Serving configuration and requests must be nonempty.");
    }
    baseWorkload.plan.validate();
    DecoderLLMWorkload effectiveWorkload = baseWorkload;
    const auto requestedDtype =
        config.requestedStorageDtype.value_or(config.storageDtype);
    DType effectiveDtype = config.storageDtype;
    bool precisionFallback = requestedDtype != effectiveDtype;
    if (effectiveDtype == DType::BFloat16 && !runtime.hardwareInfo().supportsBFloat16) {
      if (!config.allowPrecisionFallback) {
        throw std::invalid_argument(
            "Serving bf16 storage is unavailable on this Metal backend; "
            "enable allowPrecisionFallback to use fp16.");
      }
      effectiveDtype = DType::Float16;
      precisionFallback = true;
    }
    if (effectiveDtype != DType::Float16 && effectiveDtype != DType::Float32 &&
        effectiveDtype != DType::BFloat16) {
      throw std::invalid_argument(
          "Serving storage dtype must be fp32, fp16, or bf16.");
    }
    effectiveWorkload.plan.attention.dtype = effectiveDtype;
    effectiveWorkload.plan.validate();
    log << "Serving precision: requested=" << dtypeName(requestedDtype)
        << ", effective=" << dtypeName(effectiveDtype)
        << ", fallback=" << (precisionFallback ? "true" : "false") << '\n';
    std::size_t largestRequestPages = 0;
    for (const auto &spec : requestSpecs) {
      if (spec.id.empty() || spec.promptTokenIds.size() < 2 ||
          spec.promptTokenIds.size() >=
              effectiveWorkload.plan.attention.capacity ||
          spec.decodeTokenCount >
              effectiveWorkload.plan.attention.capacity -
                  spec.promptTokenIds.size()) {
        throw std::invalid_argument("Serving request dimensions are invalid.");
      }
      largestRequestPages = std::max(
          largestRequestPages,
          pagesFor(spec.promptTokenIds.size() + spec.decodeTokenCount,
                   config.pageSize));
    }
    if (config.physicalPagesPerLayer < largestRequestPages) {
      throw std::invalid_argument(
          "Shared KV pool cannot hold the largest configured request.");
    }

    std::vector<std::shared_ptr<KVPagePool>> pools;
    pools.reserve(effectiveWorkload.plan.layerCount);
    for (std::size_t layer = 0; layer < effectiveWorkload.plan.layerCount; ++layer) {
      auto creation = createKVPagePool(
          runtime, effectiveWorkload.plan.attention, config.pageSize,
          config.physicalPagesPerLayer);
      if (!creation.pool) throw std::runtime_error(creation.errorMessage);
      pools.push_back(std::move(creation.pool));
    }

    std::vector<Request> requests;
    requests.reserve(requestSpecs.size());
    std::shared_ptr<DecoderLLMModelResources> sharedModelResources;
    for (const auto &spec : requestSpecs) {
      Request request;
      request.spec = spec;
      request.workload = effectiveWorkload;
      request.workload.plan.attention.prefillLength =
          spec.promptTokenIds.size();
      request.workload.decodeCount = spec.decodeTokenCount;
      DecoderLLMCompileOptions options;
      options.kernelLibrary = config.kernelLibrary;
      options.kvPageSize = config.pageSize;
      options.prefillChunkSize =
          std::min(config.prefillChunkSize, spec.promptTokenIds.size() - 1);
      options.sharedKVPools = pools;
      options.storageDtype = effectiveDtype;
      options.requestedStorageDtype = requestedDtype;
      options.allowPrecisionFallback = false;
      options.sharedModelResources = sharedModelResources;
      std::ostringstream compilationLog;
      auto compilation = compileDecoderLLM(runtime, request.workload,
                                           compilationLog, options);
      if (!compilation.executable) {
        throw std::runtime_error("Request " + spec.id +
                                 " compilation failed: " +
                                 compilation.errorMessage);
      }
      request.decoder = std::move(compilation.executable);
      if (!sharedModelResources) {
        sharedModelResources = request.decoder->sharedModelResources();
      }
      request.output.id = spec.id;
      requests.push_back(std::move(request));
      log << "Serving request " << spec.id << ": compiled, prompt="
          << spec.promptTokenIds.size() << ", decode="
          << spec.decodeTokenCount << ", arrival_step=" << spec.arrivalStep
          << '\n';
    }

    BatchedDecodeExecutor batchedDecoder(
        runtime, effectiveWorkload, pools, requestSpecs.size(), config.pageSize,
        sharedModelResources);
    result.metrics.modelStorageBytes = sharedModelResources->storageBytes;
    result.metrics.kvPoolStorageBytes =
        effectiveWorkload.plan.layerCount * 2 *
        config.physicalPagesPerLayer * config.pageSize *
        effectiveWorkload.plan.hiddenSize() * dtypeStorageBytes(effectiveDtype);
    for (const auto &request : requests) {
      result.metrics.requestActivationStorageBytes +=
          request.decoder->activationStorageBytes();
    }
    result.metrics.batchedActivationStorageBytes =
        batchedDecoder.activationStorageBytes();
    log << "Shared batched decode pipeline: PASS, maximum_batch="
        << requestSpecs.size() << '\n';

    std::vector<double> prefillGpuTimes;
    std::vector<double> prefillEndToEndTimes;
    std::vector<double> decodeGpuTimes;
    std::vector<double> decodeEndToEndTimes;
    std::size_t completed = 0;
    std::size_t tick = 0;
    std::size_t activeBatchSamples = 0;
    std::size_t activeBatchSum = 0;
    const auto runStart = Clock::now();

    while (completed < requests.size()) {
      if (tick > 10000) {
        throw std::runtime_error(
            "Serving scheduler exceeded its progress limit.");
      }
      bool progress = false;

      for (auto &request : requests) {
        const auto schedulerStart = Clock::now();
        const bool eligible = request.state == RequestState::Preempted &&
                              request.snapshot.has_value();
        const auto required = eligible
                                  ? pagesFor(request.snapshot->length,
                                             config.pageSize)
                                  : std::size_t{0};
        const bool fits = eligible && availablePages(pools) >= required;
        result.metrics.schedulerOverheadUs +=
            elapsedUs(schedulerStart, Clock::now());
        if (!fits) continue;
        const auto error = resume(request, result.metrics);
        if (!error.empty()) {
          throw std::runtime_error("Request " + request.spec.id +
                                   " resume failed: " + error);
        }
        log << "Serving request " << request.spec.id
            << ": resumed at length=" << request.decoder->currentLength()
            << '\n';
        progress = true;
      }

      for (auto &request : requests) {
        const auto schedulerStart = Clock::now();
        const bool eligible = request.state == RequestState::Waiting &&
                              request.spec.arrivalStep <= tick;
        if (eligible && !request.arrivalTime) {
          request.arrivalTime = Clock::now();
        }
        const auto required =
            eligible ? pagesFor(request.spec.promptTokenIds.size(),
                                config.pageSize)
                     : std::size_t{0};
        const bool fits = eligible && availablePages(pools) >= required;
        result.metrics.schedulerOverheadUs +=
            elapsedUs(schedulerStart, Clock::now());
        if (!fits) continue;
        const auto prefillStart = Clock::now();
        const auto prefill =
            request.decoder->prefillChunked(request.spec.promptTokenIds);
        const auto prefillEnd = Clock::now();
        prefillEndToEndTimes.push_back(elapsedUs(prefillStart, prefillEnd));
        if (prefill.gpuExecutionTimeUs) {
          prefillGpuTimes.push_back(*prefill.gpuExecutionTimeUs);
        }
        if (!prefill.passed) {
          throw std::runtime_error("Request " + request.spec.id +
                                   " prefill failed: " +
                                   prefill.errorMessage);
        }
        request.output.prefillLogits = prefill.logits;
        request.output.generatedTokenIds.insert(
            request.output.generatedTokenIds.end(),
            prefill.nextTokenIds.begin(), prefill.nextTokenIds.end());
        request.nextTokenIds = prefill.nextTokenIds;
        request.output.timeToFirstTokenUs =
            elapsedUs(*request.arrivalTime, Clock::now());
        request.state = RequestState::Decode;
        log << "Serving request " << request.spec.id
            << ": prefill complete, pages="
            << request.decoder->allocatedKVPageCount() << ", ttft(us)="
            << request.output.timeToFirstTokenUs << '\n';
        progress = true;
        if (request.spec.decodeTokenCount == 0) {
          const auto kvStart = Clock::now();
          const auto error = request.decoder->releaseKVCache();
          result.metrics.kvManagementOverheadUs +=
              elapsedUs(kvStart, Clock::now());
          if (!error.empty()) throw std::runtime_error(error);
          request.state = RequestState::Finished;
          ++completed;
        }
      }

      const auto selectionStart = Clock::now();
      auto active = activeRequests(requests);
      auto requiredPages =
          additionalPagesForNextToken(requests, active, config.pageSize);
      result.metrics.schedulerOverheadUs +=
          elapsedUs(selectionStart, Clock::now());
      while (!active.empty() && availablePages(pools) < requiredPages) {
        const auto victimStart = Clock::now();
        const auto victim = selectPreemptionVictim(requests);
        result.metrics.schedulerOverheadUs +=
            elapsedUs(victimStart, Clock::now());
        if (!victim) {
          throw std::runtime_error(
              "Serving scheduler cannot free pages for a decode batch.");
        }
        const auto error = preempt(requests[*victim], result.metrics);
        if (!error.empty()) {
          throw std::runtime_error("Request " + requests[*victim].spec.id +
                                   " preemption failed: " + error);
        }
        log << "Serving request " << requests[*victim].spec.id
            << ": preempted for page pressure\n";
        progress = true;
        const auto rebuildStart = Clock::now();
        active = activeRequests(requests);
        requiredPages =
            additionalPagesForNextToken(requests, active, config.pageSize);
        result.metrics.schedulerOverheadUs +=
            elapsedUs(rebuildStart, Clock::now());
      }

      if (!active.empty()) {
        std::vector<std::size_t> previousLengths;
        previousLengths.reserve(active.size());
        const auto stageStart = Clock::now();
        for (const auto index : active) {
          const auto previous = requests[index].decoder->currentLength();
          previousLengths.push_back(previous);
          const auto error =
              requests[index].decoder->stageKVLength(previous + 1);
          if (!error.empty()) {
            for (std::size_t staged = 0; staged < previousLengths.size();
                 ++staged) {
              (void)requests[active[staged]].decoder->rollbackKVLength(
                  previousLengths[staged]);
            }
            throw std::runtime_error("Request " + requests[index].spec.id +
                                     " KV stage failed: " + error);
          }
        }
        result.metrics.kvManagementOverheadUs +=
            elapsedUs(stageStart, Clock::now());

        std::vector<float> tokenIds;
        std::vector<float> validLengths;
        std::vector<std::vector<float>> blockTables(
            effectiveWorkload.plan.layerCount);
        tokenIds.reserve(active.size());
        validLengths.reserve(active.size());
        const auto maximumPages =
            pagesFor(effectiveWorkload.plan.attention.capacity, config.pageSize);
        const auto packingStart = Clock::now();
        for (std::size_t row = 0; row < active.size(); ++row) {
          const auto index = active[row];
          tokenIds.push_back(requests[index].nextTokenIds.front());
          validLengths.push_back(static_cast<float>(previousLengths[row] + 1));
          for (std::size_t layer = 0; layer < blockTables.size(); ++layer) {
            const auto table = requests[index].decoder->kvBlockTable(layer);
            blockTables[layer].insert(blockTables[layer].end(), table.begin(),
                                      table.end());
            blockTables[layer].resize(
                blockTables[layer].size() + maximumPages - table.size(),
                -1.0f);
          }
        }
        result.metrics.schedulerOverheadUs +=
            elapsedUs(packingStart, Clock::now());

        const auto decodeStart = Clock::now();
        const auto decoded =
            batchedDecoder.run(tokenIds, validLengths, blockTables);
        const auto decodeEnd = Clock::now();
        decodeEndToEndTimes.push_back(elapsedUs(decodeStart, decodeEnd));
        ++result.metrics.decodeCommandSubmissions;
        if (!decoded.passed) {
          const auto rollbackStart = Clock::now();
          std::string rollbackError;
          for (std::size_t row = 0; row < active.size(); ++row) {
            const auto error = requests[active[row]].decoder->rollbackKVLength(
                previousLengths[row]);
            if (rollbackError.empty()) rollbackError = error;
          }
          result.metrics.kvManagementOverheadUs +=
              elapsedUs(rollbackStart, Clock::now());
          throw std::runtime_error(
              "Batched decode failed: " + decoded.errorMessage +
              (rollbackError.empty() ? std::string{}
                                     : "; rollback failed: " + rollbackError));
        }
        if (decoded.gpuExecutionTimeUs) {
          decodeGpuTimes.push_back(*decoded.gpuExecutionTimeUs);
        }

        const auto commitStart = Clock::now();
        for (std::size_t row = 0; row < active.size(); ++row) {
          auto &request = requests[active[row]];
          request.decoder->commitKVLength(previousLengths[row] + 1);
          const auto logitsBegin =
              decoded.logits.begin() + static_cast<std::ptrdiff_t>(
              row * effectiveWorkload.plan.vocabularySize);
          const auto logitsEnd =
              logitsBegin + static_cast<std::ptrdiff_t>(
                                effectiveWorkload.plan.vocabularySize);
          request.output.decodeLogits.emplace_back(logitsBegin, logitsEnd);
          request.nextTokenIds = {decoded.nextTokenIds[row]};
          request.output.generatedTokenIds.push_back(
              decoded.nextTokenIds[row]);
          ++request.decoded;
          ++result.metrics.decodedTokens;
          progress = true;
          if (request.decoded == request.spec.decodeTokenCount) {
            const auto error = request.decoder->releaseKVCache();
            if (!error.empty()) {
              throw std::runtime_error("Request " + request.spec.id +
                                       " completion release failed: " +
                                       error);
            }
            request.state = RequestState::Finished;
            ++completed;
            log << "Serving request " << request.spec.id << ": finished\n";
          }
        }
        result.metrics.kvManagementOverheadUs +=
            elapsedUs(commitStart, Clock::now());
        ++activeBatchSamples;
        activeBatchSum += active.size();
        result.metrics.maximumActiveBatchSize =
            std::max(result.metrics.maximumActiveBatchSize, active.size());
      }

      std::string isolationError;
      if (!auditPageIsolation(requests, effectiveWorkload.plan.layerCount,
                              isolationError)) {
        throw std::runtime_error(isolationError);
      }
      if (!progress) {
        const bool futureArrival = std::any_of(
            requests.begin(), requests.end(), [&](const Request &request) {
              return request.state == RequestState::Waiting &&
                     request.spec.arrivalStep > tick;
            });
        if (!futureArrival) {
          throw std::runtime_error("Serving scheduler made no progress.");
        }
      }
      ++tick;
    }

    result.metrics.totalTimeUs = elapsedUs(runStart, Clock::now());
    for (const auto &request : requests) {
      result.requests.push_back(request.output);
    }
    if (result.metrics.totalTimeUs > 0.0) {
      result.metrics.tokensPerSecond =
          static_cast<double>(result.metrics.decodedTokens) * 1.0e6 /
          result.metrics.totalTimeUs;
    }
    if (activeBatchSamples != 0) {
      result.metrics.averageActiveBatchSize =
          static_cast<double>(activeBatchSum) /
          static_cast<double>(activeBatchSamples);
    }
    if (const auto stats = benchmark::summarizeTimings(prefillGpuTimes)) {
      result.metrics.prefillGpu = *stats;
    }
    if (const auto stats = benchmark::summarizeTimings(prefillEndToEndTimes)) {
      result.metrics.prefillEndToEnd = *stats;
    }
    if (const auto stats = benchmark::summarizeTimings(decodeGpuTimes)) {
      result.metrics.decodeGpu = *stats;
    }
    if (const auto stats = benchmark::summarizeTimings(decodeEndToEndTimes)) {
      result.metrics.decodeEndToEnd = *stats;
    }
    result.metrics.peakPagesPerLayer = pools.front()->peakUsedPages();
    result.metrics.pageAllocationCount = pools.front()->allocationCount();
    result.metrics.pageReleaseCount = pools.front()->releaseCount();
    for (const auto &pool : pools) {
      if (pool->usedPages() != 0) {
        throw std::runtime_error("Serving completed with unreleased KV pages.");
      }
      if (pool->allocationCount() != result.metrics.pageAllocationCount ||
          pool->releaseCount() != result.metrics.pageReleaseCount) {
        throw std::runtime_error(
            "Per-layer KV page-pool lifecycle counts diverged.");
      }
    }
    result.passed = true;
  } catch (const std::exception &error) {
    result.errorMessage = error.what();
  }
  return result;
}

} // namespace tensor::runtime
