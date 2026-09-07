#include "runtime/StatefulExecutor.hpp"

#include "backend/metal/KVCacheMetalEmitter.hpp"
#include "runtime/KVCacheState.hpp"

#include <ostream>
#include <stdexcept>
#include <utility>

namespace tensor::runtime {
namespace {

metal::BufferHandle allocate(metal::MetalRuntime &runtime, std::size_t count,
                             const float *data = nullptr,
                             metal::ElementType type = metal::ElementType::Float32) {
  auto result = runtime.createBuffer(count, data, type);
  if (!result.buffer) throw std::runtime_error(result.errorMessage);
  return std::move(result.buffer);
}

std::unique_ptr<metal::PreparedExecution>
prepareKernel(metal::MetalRuntime &runtime, const metal::GeneratedKernel &kernel,
              const std::vector<metal::BufferHandle> &inputs,
              const std::vector<metal::ElementType> &inputTypes,
              const metal::BufferHandle &output, std::ostream &log) {
  const auto pipeline = runtime.createComputePipeline(kernel.source, kernel.functionName);
  if (!pipeline.pipelineCreationPassed) {
    throw std::runtime_error(kernel.functionName + ": " + pipeline.errorMessage);
  }
  const auto interfaceError = metal::checkBufferInterface(
      pipeline, inputTypes, metal::ElementType::Float32);
  if (!interfaceError.empty()) {
    throw std::runtime_error(kernel.functionName + ": " + interfaceError);
  }
  auto prepared = runtime.prepareBuffers(
      inputs, output, {kernel.threadgroupCount, kernel.threadsPerThreadgroup});
  if (!prepared.execution) {
    throw std::runtime_error(kernel.functionName + ": " + prepared.errorMessage);
  }
  log << "KV kernel " << kernel.functionName << ": PASS\n";
  return std::move(prepared.execution);
}

std::unique_ptr<metal::PreparedSequence>
makeSequence(metal::MetalRuntime &runtime,
             const std::vector<std::unique_ptr<metal::PreparedExecution>> &steps) {
  std::vector<const metal::PreparedExecution *> pointers;
  pointers.reserve(steps.size());
  for (const auto &step : steps) pointers.push_back(step.get());
  auto result = runtime.prepareSequence(pointers);
  if (!result.execution) throw std::runtime_error(result.errorMessage);
  return std::move(result.execution);
}

struct StageBuffers {
  metal::BufferHandle queryInput;
  metal::BufferHandle keyInput;
  metal::BufferHandle valueInput;
  metal::BufferHandle projectedQuery;
  metal::BufferHandle context;
  metal::BufferHandle output;
  std::unique_ptr<metal::PreparedSequence> sequence;
};

StageBuffers buildStage(metal::MetalRuntime &runtime,
                        const planner::KVCachePlan &plan,
                        std::size_t queryLength,
                        const metal::BufferHandle &queryWeight,
                        const metal::BufferHandle &keyWeight,
                        const metal::BufferHandle &valueWeight,
                        const metal::BufferHandle &outputWeight,
                        KVCacheState &state, std::ostream &log) {
  StageBuffers stage;
  const auto count = plan.inputElementCount(queryLength);
  stage.queryInput = allocate(runtime, count);
  stage.keyInput = allocate(runtime, count);
  stage.valueInput = allocate(runtime, count);
  stage.projectedQuery = allocate(runtime, count);
  stage.context = allocate(runtime, count);
  stage.output = allocate(runtime, count);

  std::vector<std::unique_ptr<metal::PreparedExecution>> steps;
  steps.push_back(prepareKernel(
      runtime, metal::emitKVQueryProjection(plan, queryLength),
      {stage.queryInput, queryWeight},
      {metal::ElementType::Float32, metal::ElementType::Float32},
      stage.projectedQuery, log));
  steps.push_back(prepareKernel(
      runtime, metal::emitKVCacheProjection(plan, queryLength, true),
      {stage.keyInput, keyWeight, state.lengthBuffer()},
      {metal::ElementType::Float32, metal::ElementType::Float32,
       metal::ElementType::Int32},
      state.keyBuffer(), log));
  steps.push_back(prepareKernel(
      runtime, metal::emitKVCacheProjection(plan, queryLength, false),
      {stage.valueInput, valueWeight, state.lengthBuffer()},
      {metal::ElementType::Float32, metal::ElementType::Float32,
       metal::ElementType::Int32},
      state.valueBuffer(), log));
  steps.push_back(prepareKernel(
      runtime, metal::emitKVAttention(plan, queryLength),
      {stage.projectedQuery, state.keyBuffer(), state.valueBuffer(),
       state.lengthBuffer()},
      {metal::ElementType::Float32, metal::ElementType::Float32,
       metal::ElementType::Float32, metal::ElementType::Int32},
      stage.context, log));
  steps.push_back(prepareKernel(
      runtime, metal::emitKVOutputProjection(plan, queryLength),
      {stage.context, outputWeight},
      {metal::ElementType::Float32, metal::ElementType::Float32},
      stage.output, log));
  stage.sequence = makeSequence(runtime, steps);
  return stage;
}

std::string updateInputs(metal::MetalRuntime &runtime, StageBuffers &stage,
                         const std::vector<float> &query,
                         const std::vector<float> &key,
                         const std::vector<float> &value) {
  if (query.size() != stage.queryInput->elementCount() ||
      key.size() != stage.keyInput->elementCount() ||
      value.size() != stage.valueInput->elementCount()) {
    return "KV cache invocation input shape does not match its compiled plan.";
  }
  auto error = runtime.writeBuffer(stage.queryInput, query.data(), query.size());
  if (!error.empty()) return error;
  error = runtime.writeBuffer(stage.keyInput, key.data(), key.size());
  if (!error.empty()) return error;
  return runtime.writeBuffer(stage.valueInput, value.data(), value.size());
}

} // namespace

class CompiledKVCacheAttention::Impl {
public:
  metal::MetalRuntime *runtime = nullptr;
  planner::KVCachePlan plan;
  std::unique_ptr<KVCacheState> state;
  metal::BufferHandle queryWeight;
  metal::BufferHandle keyWeight;
  metal::BufferHandle valueWeight;
  metal::BufferHandle outputWeight;
  StageBuffers prefill;
  StageBuffers decode;

  KVCacheRunResult run(StageBuffers &stage, std::size_t nextLength,
                       const std::vector<float> &query,
                       const std::vector<float> &key,
                       const std::vector<float> &value) {
    KVCacheRunResult result;
    auto error = updateInputs(*runtime, stage, query, key, value);
    if (!error.empty()) {
      result.errorMessage = error;
      return result;
    }
    error = state->stageLength(*runtime, nextLength);
    if (!error.empty()) {
      result.errorMessage = error;
      return result;
    }
    const auto execution = stage.sequence->execute();
    if (!execution.executionPassed) {
      const float previous = static_cast<float>(state->currentLength());
      (void)runtime->writeBuffer(state->lengthBuffer(), &previous, 1);
      result.errorMessage = execution.errorMessage;
      return result;
    }
    state->commitLength(nextLength);
    result.output = stage.output->read();
    result.gpuExecutionTimeUs = execution.gpuExecutionTimeUs;
    result.cpuSubmitToCompletionTimeUs = execution.cpuSubmitToCompletionTimeUs;
    result.passed = true;
    return result;
  }
};

CompiledKVCacheAttention::CompiledKVCacheAttention(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
CompiledKVCacheAttention::~CompiledKVCacheAttention() = default;

std::string CompiledKVCacheAttention::reset() {
  return impl_->state->reset(*impl_->runtime);
}

KVCacheRunResult CompiledKVCacheAttention::prefill(
    const std::vector<float> &query, const std::vector<float> &key,
    const std::vector<float> &value) {
  if (impl_->state->currentLength() != 0) {
    return {false, {}, {}, 0.0,
            "Prefill requires an empty KV cache; call reset first."};
  }
  return impl_->run(impl_->prefill, impl_->plan.prefillLength,
                    query, key, value);
}

KVCacheRunResult CompiledKVCacheAttention::decode(
    const std::vector<float> &query, const std::vector<float> &key,
    const std::vector<float> &value) {
  const auto current = impl_->state->currentLength();
  if (current == 0) {
    return {false, {}, {}, 0.0, "Decode requires a completed prefill."};
  }
  if (current >= impl_->plan.capacity) {
    return {false, {}, {}, 0.0, "KV cache capacity has been reached."};
  }
  return impl_->run(impl_->decode, current + 1, query, key, value);
}

std::size_t CompiledKVCacheAttention::currentLength() const noexcept {
  return impl_->state->currentLength();
}

std::vector<float> CompiledKVCacheAttention::readKeyPrefix() const {
  return impl_->state->readKeyPrefix();
}

std::vector<float> CompiledKVCacheAttention::readValuePrefix() const {
  return impl_->state->readValuePrefix();
}

bool CompiledKVCacheAttention::cacheStorageReused() const noexcept {
  return impl_->state->storageReused();
}

KVCacheCompilation compileKVCacheAttention(
    metal::MetalRuntime &runtime, const planner::KVCachePlan &plan,
    const std::vector<float> &queryWeight,
    const std::vector<float> &keyWeight,
    const std::vector<float> &valueWeight,
    const std::vector<float> &outputWeight, std::ostream &log) {
  KVCacheCompilation result;
  try {
    plan.validate();
    if (!runtime.isAvailable()) throw std::runtime_error(runtime.initializationError());
    const auto weightCount = plan.modelDimension() * plan.modelDimension();
    if (queryWeight.size() != weightCount || keyWeight.size() != weightCount ||
        valueWeight.size() != weightCount || outputWeight.size() != weightCount) {
      throw std::invalid_argument("KV cache projection weight shape is invalid.");
    }
    auto impl = std::make_unique<CompiledKVCacheAttention::Impl>();
    impl->runtime = &runtime;
    impl->plan = plan;
    auto state = createKVCacheState(runtime, plan);
    if (!state.state) throw std::runtime_error(state.errorMessage);
    impl->state = std::move(state.state);
    impl->queryWeight = allocate(runtime, weightCount, queryWeight.data());
    impl->keyWeight = allocate(runtime, weightCount, keyWeight.data());
    impl->valueWeight = allocate(runtime, weightCount, valueWeight.data());
    impl->outputWeight = allocate(runtime, weightCount, outputWeight.data());
    impl->prefill = buildStage(runtime, plan, plan.prefillLength,
                               impl->queryWeight, impl->keyWeight,
                               impl->valueWeight, impl->outputWeight,
                               *impl->state, log);
    impl->decode = buildStage(runtime, plan, 1,
                              impl->queryWeight, impl->keyWeight,
                              impl->valueWeight, impl->outputWeight,
                              *impl->state, log);
    result.executable = std::unique_ptr<CompiledKVCacheAttention>(
        new CompiledKVCacheAttention(std::move(impl)));
    log << "KV cache stateful compilation: PASS\n";
  } catch (const std::exception &error) {
    result.errorMessage = error.what();
  }
  return result;
}

} // namespace tensor::runtime
