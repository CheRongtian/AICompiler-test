#include "runtime/DecoderLLMExecutor.hpp"

#include "backend/metal/DecodeGEMVMetalEmitter.hpp"
#include "backend/metal/DecoderLLMMetalEmitter.hpp"
#include "backend/metal/KVCacheMetalEmitter.hpp"
#include "backend/metal/TransformerDecodeMetalEmitter.hpp"
#include "planner/DecodeGEMVTuner.hpp"
#include "runtime/KVCacheState.hpp"
#include "runtime/KernelRegistry.hpp"
#include "benchmark/Benchmark.hpp"
#include "validation/Validator.hpp"
#include <functional>
#include <algorithm>

#include <ostream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace tensor::runtime {
namespace {

using Buffer = metal::BufferHandle;
using FusionDecisionCache = std::unordered_map<std::string, bool>;

metal::ElementType metalStorageType(DType dtype) {
  switch (dtype) {
  case DType::Float16: return metal::ElementType::Float16;
  case DType::Float32: return metal::ElementType::Float32;
  case DType::BFloat16: return metal::ElementType::BFloat16;
  case DType::Int32: return metal::ElementType::Int32;
  }
  throw std::invalid_argument("Unsupported decoder storage dtype.");
}

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
    const std::vector<metal::ElementType> &inputTypes,
    const std::vector<Buffer> &outputs,
    const std::vector<metal::ElementType> &outputTypes, std::ostream &log) {
  const auto pipeline =
      runtime.createComputePipeline(kernel.source, kernel.functionName);
  if (!pipeline.pipelineCreationPassed) {
    throw std::runtime_error(kernel.functionName + ": " + pipeline.errorMessage);
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
  log << "Decoder kernel " << kernel.functionName << ": PASS\n";
  return std::move(result.execution);
}

std::unique_ptr<metal::PreparedExecution> prepare(
    metal::MetalRuntime &runtime, const metal::GeneratedKernel &kernel,
    const std::vector<Buffer> &inputs,
    const std::vector<metal::ElementType> &inputTypes, const Buffer &output,
    metal::ElementType outputType, std::ostream &log) {
  return prepare(runtime, kernel, inputs, inputTypes,
                 std::vector<Buffer>{output},
                 std::vector<metal::ElementType>{outputType}, log);
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

bool admitFusion(metal::MetalRuntime &runtime,
    std::vector<std::unique_ptr<metal::PreparedExecution>> &steps,
    std::size_t begin, const std::function<std::vector<std::unique_ptr<metal::PreparedExecution>>()>& build,
    const std::vector<Buffer> &testInputs,const std::vector<Buffer> &outputs,
    const std::string &name, const std::string &decisionKey,
    FusionDecisionCache &decisions, std::ostream &log) {
  try {
    const auto cached = decisions.find(decisionKey);
    if (cached != decisions.end()) {
      if (!cached->second) {
        log << "Fusion " << name << ": cached FALLBACK\n";
        return false;
      }
      auto candidates = build();
      steps.resize(begin);
      for (auto &step : candidates) steps.push_back(std::move(step));
      log << "Fusion " << name << ": cached ADMITTED\n";
      return true;
    }
    auto candidates=build();
    auto candidate=makeSequence(runtime,candidates);
    std::vector<const metal::PreparedExecution*> baselineSteps;
    for(std::size_t i=begin;i<steps.size();++i)baselineSteps.push_back(steps[i].get());
    auto preparation=runtime.prepareSequence(baselineSteps);
    if(!preparation.execution)throw std::runtime_error(preparation.errorMessage);
    auto &baseline=*preparation.execution;
    for(const auto &input:testInputs) {
      std::vector<float> data(input->elementCount());
      for(std::size_t i=0;i<data.size();++i)data[i]=float(int(i%31)-15)/31.0f;
      const auto error=runtime.writeBuffer(input,data.data(),data.size());
      if(!error.empty())throw std::runtime_error(error);
    }
    auto baselineResult=baseline.execute();
    if(!baselineResult.executionPassed)throw std::runtime_error(baselineResult.errorMessage);
    std::vector<std::vector<double>> references;
    for(const auto &output:outputs) {
      const auto values=output->read();references.emplace_back(values.begin(),values.end());
    }
    auto validateCandidate=[&]() {
      const auto run=candidate->execute();
      if(!run.executionPassed)throw std::runtime_error(run.errorMessage);
      for(std::size_t i=0;i<outputs.size();++i) {
        const double tolerance=outputs[i]->elementType()==metal::ElementType::Float32?2e-3:2e-2;
        const auto comparison=validation::compare(outputs[i]->read(),references[i],tolerance,tolerance);
        if(!comparison.passed)throw std::runtime_error(comparison.errorMessage);
      }
    };
    validateCandidate();
    for(const auto *sequence:{&baseline,candidate.get()}) {
      const auto error=benchmark::warmup(*sequence,5);
      if(!error.empty())throw std::runtime_error(error);
    }
    const auto first=benchmark::measurePair(baseline,*candidate,31);
    const auto second=benchmark::measurePair(baseline,*candidate,31);
    if(!first.passed || !second.passed)throw std::runtime_error("Fusion benchmark failed.");
    validateCandidate();
    const auto speedup=std::min(first.speedup,second.speedup);
    const bool admitted=speedup>=1.05;
    log << "Fusion " << name << ": compile/interface/template-comparison PASS, speedup="
        << speedup << "x, " << (admitted?"ADMITTED":"FALLBACK") << '\n';
    if(admitted) {
      steps.resize(begin);
      for(auto &step:candidates)steps.push_back(std::move(step));
    }
    decisions.emplace(decisionKey, admitted);
    return admitted;
  } catch(const std::exception &error) {
    decisions[decisionKey] = false;
    log << "Fusion " << name << ": FALLBACK: " << error.what() << '\n';
    return false;
  }
}

void requireWeightCount(const std::string &name, const std::vector<float> &values,
                        std::size_t expected) {
  if (values.size() != expected) {
    throw std::invalid_argument(name + " has " + std::to_string(values.size()) +
                                " elements; expected " +
                                std::to_string(expected) + ".");
  }
}

void validateWeights(const DecoderLLMWorkload &workload,
                     const planner::DecoderLLMPlan &plan) {
  const auto hidden = plan.hiddenSize();
  requireWeightCount("embedding.weight", workload.embeddingWeight,
                     plan.vocabularySize * hidden);
  requireWeightCount("final_norm.weight", workload.finalNormWeight, hidden);
  requireWeightCount("lm_head.weight", workload.languageModelHeadWeight,
                     plan.vocabularySize * hidden);
  const auto ropeCount = plan.attention.capacity * plan.attention.headDimension / 2;
  requireWeightCount("rope.cosine", workload.ropeCosine, ropeCount);
  requireWeightCount("rope.sine", workload.ropeSine, ropeCount);
  const auto hiddenMatrix = hidden * hidden;
  for (std::size_t index = 0; index < workload.layers.size(); ++index) {
    const auto &layer = workload.layers[index];
    const auto prefix = "layer" + std::to_string(index) + ".";
    requireWeightCount(prefix + "input_norm.weight", layer.inputNormWeight, hidden);
    requireWeightCount(prefix + "q.weight", layer.queryWeight, hiddenMatrix);
    requireWeightCount(prefix + "k.weight", layer.keyWeight, hiddenMatrix);
    requireWeightCount(prefix + "v.weight", layer.valueWeight, hiddenMatrix);
    requireWeightCount(prefix + "o.weight", layer.outputWeight, hiddenMatrix);
    requireWeightCount(prefix + "post_norm.weight", layer.postAttentionNormWeight,
                       hidden);
    requireWeightCount(prefix + "gate.weight", layer.gateWeight,
                       plan.intermediateSize * hidden);
    requireWeightCount(prefix + "up.weight", layer.upWeight,
                       plan.intermediateSize * hidden);
    requireWeightCount(prefix + "down.weight", layer.downWeight,
                       hidden * plan.intermediateSize);
  }
}

std::size_t modelStorageBytes(const DecoderLLMWorkload &workload,
                              DType storageDtype) {
  std::size_t elements = workload.embeddingWeight.size() +
                         workload.finalNormWeight.size() +
                         workload.languageModelHeadWeight.size();
  for (const auto &layer : workload.layers) {
    elements += layer.inputNormWeight.size() + layer.queryWeight.size() +
                layer.keyWeight.size() + layer.valueWeight.size() +
                layer.outputWeight.size() +
                layer.postAttentionNormWeight.size() +
                layer.gateWeight.size() + layer.upWeight.size() +
                layer.downWeight.size();
  }
  return elements * dtypeStorageBytes(storageDtype) +
         (workload.ropeCosine.size() + workload.ropeSine.size()) *
             sizeof(float);
}

std::size_t bufferBytes(const Buffer &buffer) {
  return buffer->elementCount() *
         (buffer->elementType() == metal::ElementType::Float16 ||
                  buffer->elementType() == metal::ElementType::BFloat16
              ? 2u
              : 4u);
}

DecoderLLMModelLayerBuffers allocateLayer(
    metal::MetalRuntime &runtime,
    const DecoderLLMLayerParameters &parameters,
    metal::ElementType storageType) {
  auto weight = [&](const std::vector<float> &values) {
    return allocate(runtime, values.size(), values.data(), storageType);
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
                        std::ostream &log, DType storageDtype, bool prefill = false) {
  const auto result = planner::tuneDecodeGEMV(
      runtime, batch, inputSize, outputSize, input, weight, log,
      storageDtype, prefill);
  if (!result.success) throw std::runtime_error(result.errorMessage);
  return {result.useBaseline, result.config};
}

metal::GeneratedKernel emitLinear(std::size_t rows, std::size_t inputSize,
                                  std::size_t outputSize,
                                  const LinearChoice &choice,
                                  bool decodeStage,
                                  const std::string &functionName,
                                  metal::ElementType storageType,
                                  metal::ElementType outputType) {
  if (!choice.useBaseline && choice.config.tileSize)
    return metal::emitTiledGEMM(rows,inputSize,outputSize,choice.config.tileSize,
                                functionName,storageType,outputType);
  if (decodeStage && !choice.useBaseline) {
    return metal::emitDecodeGEMV(rows, inputSize, outputSize, choice.config,
                                 functionName, storageType, outputType);
  }
  const auto vectorWidth = std::size_t{1};
  return metal::emitLinearBaseline(rows, inputSize, outputSize,
                                   choice.config.threads, functionName,
                                   storageType, vectorWidth, outputType);
}

struct LinearChoices {
  LinearChoice hiddenToHidden;
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
  std::vector<std::size_t> uses;
  std::unique_ptr<metal::PreparedSequence> tokenSequence;
  std::vector<std::size_t> tokenUses;
  bool tokenAdmissionChecked = false;
  bool tokenAdmitted = false;
  double tokenSpeedup = 0.0;
};

std::size_t stageBytes(const DecoderStage &stage) {
  std::size_t bytes = bufferBytes(stage.tokenIds) + bufferBytes(stage.logits) +
                      bufferBytes(stage.allNextTokenIds);
  for (const auto &buffer : stage.intermediates) bytes += bufferBytes(buffer);
  return bytes;
}

Buffer intermediate(metal::MetalRuntime &runtime, DecoderStage &stage,
                    std::size_t count,
                    metal::ElementType type = metal::ElementType::Float32) {
  auto buffer = allocate(runtime, count, nullptr, type);
  stage.intermediates.push_back(buffer);
  return buffer;
}

DecoderStage buildStage(
    metal::MetalRuntime &runtime, const DecoderLLMWorkload &workload,
    std::size_t sequenceLength, const std::string &stageName,
    const Buffer &embeddingWeight,
    const Buffer &finalNormWeight, const Buffer &languageModelHeadWeight,
    const Buffer &ropeCosine, const Buffer &ropeSine,
    const std::vector<DecoderLLMModelLayerBuffers> &layers,
    const std::vector<std::unique_ptr<KVCacheState>> &states,
    LinearChoices choices, KernelRegistry &registry, std::ostream &log,
    metal::ElementType storageType, bool enableFusion,
    FusionDecisionCache &fusionDecisions) {
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
  const auto fusionDecisionKey = [&](const std::string &pattern,
                                     std::size_t variant = 0) {
    return pattern + ":rows=" + std::to_string(rows) +
           ":hidden=" + std::to_string(hidden) +
           ":intermediate=" + std::to_string(plan.intermediateSize) +
           ":variant=" + std::to_string(variant) +
           ":dtype=" + std::to_string(static_cast<int>(storageType));
  };
  if (!decodeStage) {
    std::vector<float> hiddenInput(rows * hidden);
    std::vector<float> intermediateInput(rows * plan.intermediateSize);
    for (std::size_t i=0;i<hiddenInput.size();++i)
      hiddenInput[i]=float(int(i%17)-8)/17.0f;
    for (std::size_t i=0;i<intermediateInput.size();++i)
      intermediateInput[i]=float(int(i%19)-9)/19.0f;
    log << "Prefill GEMM admission: rows=" << rows << '\n';
    choices.hiddenToHidden=tuneLinear(runtime,rows,hidden,hidden,hiddenInput,
        workload.layers.front().queryWeight,log,plan.attention.dtype,true);
    choices.intermediateToHidden=tuneLinear(runtime,rows,plan.intermediateSize,
        hidden,intermediateInput,workload.layers.front().downWeight,log,
        plan.attention.dtype,true);
    choices.hiddenToVocabulary=tuneLinear(runtime,rows,hidden,plan.vocabularySize,
        hiddenInput,workload.languageModelHeadWeight,log,plan.attention.dtype,true);
  }

  stage.tokenIds =
      allocate(runtime, rows, nullptr, metal::ElementType::Int32);
  stage.logits = allocate(runtime, plan.logitsElementCount(sequenceLength),
                          nullptr, metal::ElementType::Float32);
  stage.allNextTokenIds =
      allocate(runtime, rows, nullptr, metal::ElementType::Int32);

  std::vector<std::unique_ptr<metal::PreparedExecution>> steps;
  auto track = [&](const std::string &node, const std::string &pattern,
                   const std::string &implementation, const std::string &functions,
                   std::size_t dispatches) {
    stage.uses.push_back(registry.track({node,pattern,implementation,functions,dispatches,0}));
    log << "Decoder selection " << node << ": " << implementation
        << ", pattern=" << pattern << ", kernel=" << functions << '\n';
  };
  auto generated = [&](const std::string &pattern, const std::string &node,
                       const std::vector<Buffer> &inputs,
                       const std::vector<Buffer> &outputs) {
    if (!decodeStage || (!enableFusion && pattern.rfind("decoder_gemv_",0)!=0)) return false;
    const auto typedPattern=llm::typedKernelPattern(pattern,storageType);
    const auto *entry = registry.find(typedPattern);
    if (!entry) return false;
    try {
      const auto contract = llm::makeKernelContract(typedPattern);
      const auto &data = contract.cases.front();
      if (inputs.size()!=data.inputs.size() || outputs.size()!=data.references.size())
        throw std::runtime_error("Binding count differs from admitted contract.");
      for (std::size_t i=0;i<inputs.size();++i)
        if (inputs[i]->elementCount()!=data.inputs[i].size() ||
            inputs[i]->elementType()!=contract.inputTypes[i])
          throw std::runtime_error("Input shape/dtype differs from admitted contract.");
      for (std::size_t i=0;i<outputs.size();++i)
        if (outputs[i]->elementCount()!=data.references[i].size() ||
            outputs[i]->elementType()!=storageType)
          throw std::runtime_error("Output shape/dtype differs from admitted contract.");
      if (std::find(contract.workgroupSizes.begin(),contract.workgroupSizes.end(),
                    entry->workgroupSize)==contract.workgroupSizes.end())
        throw std::runtime_error("Illegal admitted workgroup size.");
      const auto pipeline=runtime.createComputePipeline(entry->source,entry->functionName);
      if (!pipeline.pipelineCreationPassed) throw std::runtime_error(pipeline.errorMessage);
      const auto hardware=runtime.hardwareInfo();
      if (entry->workgroupSize>pipeline.maxTotalThreadsPerThreadgroup ||
          entry->workgroupSize>hardware.maxThreadsPerThreadgroup ||
          pipeline.staticThreadgroupMemoryLength>hardware.maxThreadgroupMemoryLength)
        throw std::runtime_error("Admitted kernel exceeds current Metal hardware limits.");
      const auto error=metal::checkBufferInterface(
          pipeline,contract.inputTypes,
          std::vector<metal::ElementType>(outputs.size(),storageType),true);
      if (!error.empty()) throw std::runtime_error(error);
      auto prepared=runtime.prepareBuffers(inputs,outputs,
          llm::contractDispatch(contract,data.workItems,entry->workgroupSize),data.constants);
      if (!prepared.execution) throw std::runtime_error(prepared.errorMessage);
      steps.push_back(std::move(prepared.execution));
      track(node,typedPattern,"generated",entry->functionName,1);
      return true;
    } catch (const std::exception &error) {
      log << "Decoder generated fallback " << node << ": " << error.what() << '\n';
      return false;
    }
  };
  auto addLinear = [&](std::size_t inputSize, std::size_t outputSize,
                       const LinearChoice &choice, const std::string &name,
                       const Buffer &input, const Buffer &weight, const Buffer &output) {
    const auto pattern="decoder_gemv_"+std::to_string(inputSize)+"_"+std::to_string(outputSize);
    const auto outputType = output->elementType();
    if (batch==1 && outputType==storageType &&
        generated(pattern,name,{input,weight},{output})) return;
    const auto kernel=emitLinear(rows,inputSize,outputSize,choice,decodeStage,name,
                                 storageType,outputType);
    steps.push_back(prepare(runtime,kernel,{input,weight},
        {storageType,storageType}, output,outputType,log));
    track(name,pattern,"template",kernel.functionName,1);
  };
  auto makeIntermediate = [&](std::size_t count) {
    return intermediate(runtime, stage, count, storageType);
  };
  // The execution sequence is strictly ordered, so layers can share a fixed
  // scratch arena.  Two hidden buffers ping-pong the residual stream while
  // the remaining slots cover values that are simultaneously live inside one
  // decoder layer.
  Buffer current = makeIntermediate(hiddenCount);
  Buffer nextLayerOutput = makeIntermediate(hiddenCount);
  std::vector<Buffer> hiddenScratch;
  hiddenScratch.reserve(11);
  for (std::size_t slot = 0; slot < 11; ++slot) {
    hiddenScratch.push_back(makeIntermediate(hiddenCount));
  }
  const auto gated = makeIntermediate(intermediateCount);
  steps.push_back(prepare(
      runtime,
      metal::emitDecoderEmbedding(plan, sequenceLength,
                                  "decoder_" + stageName + "_embedding",
                                  storageType),
      {stage.tokenIds, embeddingWeight},
      {metal::ElementType::Int32, storageType}, current, storageType, log));
  track("decoder_" + stageName + "_embedding", "embedding", "template",
        "decoder_" + stageName + "_embedding", 1);

  for (std::size_t index = 0; index < layers.size(); ++index) {
    const auto prefix = "decoder_" + stageName + "_l" +
                        std::to_string(index) + "_";
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
    const auto qkvBegin=steps.size(), qkvUses=stage.uses.size();

    steps.push_back(prepare(
        runtime,
        metal::emitDecoderRMSNorm(rows, hidden, plan.rmsNormEpsilon, threads,
                                  prefix + "input_norm", storageType),
        {current, layer.inputNormWeight},
        {storageType, storageType}, inputNormalized, storageType, log));
    track(prefix + "input_norm", "rmsnorm", "template",
          prefix + "input_norm", 1);
    addLinear(hidden, hidden, choices.hiddenToHidden, prefix + "query",
              inputNormalized, layer.queryWeight, query);
    addLinear(hidden, hidden, choices.hiddenToHidden, prefix + "key",
              inputNormalized, layer.keyWeight, key);
    addLinear(hidden, hidden, choices.hiddenToHidden, prefix + "value",
              inputNormalized, layer.valueWeight, value);
    if (!(batch==1 && hidden==64 && plan.attention.heads==4 &&
          plan.attention.headDimension==16 && plan.attention.capacity==32 &&
          generated("decoder_rope",prefix+"qk_rope",
                    {query,key,ropeCosine,ropeSine,states[index]->lengthBuffer()},
                    {rotatedQuery,rotatedKey}))) {
    steps.push_back(prepare(
        runtime, metal::emitDecoderRoPE(plan, sequenceLength, prefix + "q_rope",
                                        storageType),
        {query, ropeCosine, ropeSine, states[index]->lengthBuffer()},
        {storageType, metal::ElementType::Float32,
         metal::ElementType::Float32, metal::ElementType::Int32},
        rotatedQuery, storageType, log));
    steps.push_back(prepare(
        runtime, metal::emitDecoderRoPE(plan, sequenceLength, prefix + "k_rope",
                                        storageType),
        {key, ropeCosine, ropeSine, states[index]->lengthBuffer()},
        {storageType, metal::ElementType::Float32,
         metal::ElementType::Float32, metal::ElementType::Int32},
        rotatedKey, storageType, log));
      track(prefix+"qk_rope","decoder_rope","template",prefix+"q_rope,"+prefix+"k_rope",2);
    }
    if(enableFusion) {
      const auto lengthBuffer=states[index]->lengthBuffer();
      const auto previousLength=lengthBuffer->read();
      const float length=static_cast<float>(sequenceLength);
      const auto lengthError=runtime.writeBuffer(lengthBuffer,&length,1);
      if(!lengthError.empty())throw std::runtime_error(lengthError);
      for(bool normalize:{true,false}) {
        const auto name=prefix+(normalize?"rmsnorm_qkv":"qkv_rope");
        const auto requiredThreadgroupBytes =
            (normalize ? hidden + threads : 0u) * sizeof(float);
        if (requiredThreadgroupBytes >
            runtime.hardwareInfo().maxThreadgroupMemoryLength) {
          log << "Fusion " << name
              << ": Hardware Filter: FALLBACK: threadgroup memory requirement="
              << requiredThreadgroupBytes << " bytes\n";
          continue;
        }
        auto build=[&]() {
          std::vector<std::unique_ptr<metal::PreparedExecution>> candidate;
          if(normalize) {
            candidate.push_back(prepare(runtime,metal::emitDecoderQKVFusion(plan,
                sequenceLength,true,name,storageType),
                {current,layer.queryWeight,layer.keyWeight,layer.valueWeight,layer.inputNormWeight},
                {storageType,storageType,storageType,storageType,storageType},
                {query,key,value},{storageType,storageType,storageType},log));
            for(bool q:{true,false})
              candidate.push_back(prepare(runtime,metal::emitDecoderRoPE(plan,sequenceLength,
                  name+(q?"_q_rope":"_k_rope"),storageType),
                  {q?query:key,ropeCosine,ropeSine,lengthBuffer},
                  {storageType,metal::ElementType::Float32,metal::ElementType::Float32,metal::ElementType::Int32},
                  q?rotatedQuery:rotatedKey,storageType,log));
          } else {
            candidate.push_back(prepare(runtime,metal::emitDecoderRMSNorm(rows,hidden,
                plan.rmsNormEpsilon,threads,name+"_norm",storageType),{current,layer.inputNormWeight},
                {storageType,storageType},inputNormalized,storageType,log));
            candidate.push_back(prepare(runtime,metal::emitDecoderQKVFusion(plan,
                sequenceLength,false,name,storageType),
                {inputNormalized,layer.queryWeight,layer.keyWeight,layer.valueWeight,ropeCosine,ropeSine,lengthBuffer},
                {storageType,storageType,storageType,storageType,metal::ElementType::Float32,
                 metal::ElementType::Float32,metal::ElementType::Int32},
                {rotatedQuery,rotatedKey,value},{storageType,storageType,storageType},log));
          }
          return candidate;
        };
        if(admitFusion(runtime,steps,qkvBegin,build,{current},
                       {rotatedQuery,rotatedKey,value},name,
                       fusionDecisionKey(normalize ? "rmsnorm_qkv"
                                                   : "qkv_rope"),
                       fusionDecisions,log)) {
          stage.uses.resize(qkvUses);
          track(name,normalize?"rmsnorm_qkv":"qkv_rope","template",name,normalize?3:2);
        }
      }
      const auto restoreError=runtime.writeBuffer(lengthBuffer,previousLength.data(),previousLength.size());
      if(!restoreError.empty())throw std::runtime_error(restoreError);
    }
    std::size_t pagedAttentionBegin = 0;
    std::size_t pagedAttentionUses = 0;
    if (states[index]->isPaged()) {
      steps.push_back(prepare(
          runtime,
          metal::emitDecoderPagedCacheAppend(
              plan, sequenceLength, true, states[index]->pageSize(),
              prefix + "key_cache_append", storageType),
          {rotatedKey, states[index]->lengthBuffer(),
           states[index]->blockTableBuffer()},
          {storageType, metal::ElementType::Int32,
           metal::ElementType::Int32},
          states[index]->keyBuffer(), storageType, log));
      steps.push_back(prepare(
          runtime,
          metal::emitDecoderPagedCacheAppend(
              plan, sequenceLength, false, states[index]->pageSize(),
              prefix + "value_cache_append", storageType),
          {value, states[index]->lengthBuffer(),
           states[index]->blockTableBuffer()},
          {storageType, metal::ElementType::Int32,
           metal::ElementType::Int32},
          states[index]->valueBuffer(), storageType, log));
      track(prefix + "cache_append", "paged_kv_append", "template",
            prefix + "key_cache_append," + prefix + "value_cache_append", 2);
      pagedAttentionBegin = steps.size();
      pagedAttentionUses = stage.uses.size();
      steps.push_back(prepare(
          runtime,
          metal::emitPagedKVAttention(plan.attention, sequenceLength,
                                      states[index]->pageSize(), true),
          {rotatedQuery, states[index]->keyBuffer(),
           states[index]->valueBuffer(), states[index]->lengthBuffer(),
           states[index]->blockTableBuffer()},
          {storageType, storageType, storageType, metal::ElementType::Int32,
           metal::ElementType::Int32},
          context, storageType, log));
      track(prefix + "attention", "paged_attention", "template",
            sequenceLength == 1 ? "paged_kv_attention_decode"
                                : "paged_kv_attention_prefill",
            1);
    } else {
      steps.push_back(prepare(
          runtime,
          metal::emitDecoderCacheAppend(plan, sequenceLength, true,
                                        prefix + "key_cache_append", storageType),
          {rotatedKey, states[index]->lengthBuffer()},
          {storageType, metal::ElementType::Int32}, states[index]->keyBuffer(),
          storageType, log));
      steps.push_back(prepare(
          runtime,
          metal::emitDecoderCacheAppend(plan, sequenceLength, false,
                                        prefix + "value_cache_append", storageType),
          {value, states[index]->lengthBuffer()},
          {storageType, metal::ElementType::Int32}, states[index]->valueBuffer(),
          storageType, log));
      steps.push_back(prepare(
          runtime, metal::emitKVAttention(plan.attention, sequenceLength, true),
          {rotatedQuery, states[index]->keyBuffer(),
           states[index]->valueBuffer(), states[index]->lengthBuffer()},
          {storageType, storageType, storageType, metal::ElementType::Int32},
          context, storageType, log));
      track(prefix + "cache_append", "kv_append", "template",
            prefix + "key_cache_append," + prefix + "value_cache_append", 2);
      track(prefix + "attention", "attention", "template",
            sequenceLength == 1 ? "kv_attention_decode"
                                : "kv_attention_prefill",
            1);
    }
    const auto outputBegin=steps.size(), outputUses=stage.uses.size();
    addLinear(hidden, hidden, choices.hiddenToHidden, prefix + "attention_output",
              context, layer.outputWeight, attentionOutput);
    bool pagedAttentionOutputAdmitted = false;
    if (enableFusion && states[index]->isPaged()) {
      const auto requiredThreadgroupBytes =
          (hidden + threads + 4u) * sizeof(float);
      const auto name = prefix + "paged_attention_output";
      if (requiredThreadgroupBytes >
          runtime.hardwareInfo().maxThreadgroupMemoryLength) {
        log << "Fusion " << name
            << ": Hardware Filter: FALLBACK: threadgroup memory requirement="
            << requiredThreadgroupBytes << " bytes\n";
      } else {
        const auto lengthBuffer = states[index]->lengthBuffer();
        const auto blockTable = states[index]->blockTableBuffer();
        const auto previousLength = lengthBuffer->read();
        const auto previousTable = blockTable->read();
        const float validLength = static_cast<float>(sequenceLength);
        std::vector<float> identityTable(blockTable->elementCount(), -1.0f);
        for (std::size_t page = 0; page < identityTable.size(); ++page)
          identityTable[page] = static_cast<float>(page);
        auto writeError = runtime.writeBuffer(lengthBuffer, &validLength, 1);
        if (writeError.empty()) {
          writeError = runtime.writeBuffer(blockTable, identityTable.data(),
                                           identityTable.size());
        }
        if (!writeError.empty()) throw std::runtime_error(writeError);
        auto build = [&]() {
          std::vector<std::unique_ptr<metal::PreparedExecution>> candidate;
          candidate.push_back(prepare(
              runtime,
              metal::emitPagedAttentionOutputProjection(
                  plan, sequenceLength, states[index]->pageSize(), name,
                  storageType),
              {rotatedQuery, states[index]->keyBuffer(),
               states[index]->valueBuffer(), lengthBuffer, blockTable,
               layer.outputWeight},
              {storageType, storageType, storageType,
               metal::ElementType::Int32, metal::ElementType::Int32,
               storageType},
              attentionOutput, storageType, log));
          return candidate;
        };
        pagedAttentionOutputAdmitted = admitFusion(
            runtime, steps, pagedAttentionBegin, build,
            {rotatedQuery, states[index]->keyBuffer(),
             states[index]->valueBuffer()},
            {attentionOutput}, name,
            fusionDecisionKey("paged_attention_output_projection",
                              states[index]->pageSize()),
            fusionDecisions, log);
        const auto lengthRestore = runtime.writeBuffer(
            lengthBuffer, previousLength.data(), previousLength.size());
        const auto tableRestore = runtime.writeBuffer(
            blockTable, previousTable.data(), previousTable.size());
        if (!lengthRestore.empty()) throw std::runtime_error(lengthRestore);
        if (!tableRestore.empty()) throw std::runtime_error(tableRestore);
        if (pagedAttentionOutputAdmitted) {
          stage.uses.resize(pagedAttentionUses);
          track(name, "paged_attention_output_projection", "template", name,
                1);
        }
      }
    }
    if (!(batch==1 && hidden==64 && plan.rmsNormEpsilon==1e-5f &&
          generated("decoder_residual_rmsnorm",prefix+"residual_rmsnorm",
                    {current,attentionOutput,layer.postAttentionNormWeight},
                    {afterAttention,postAttentionNormalized}))) {
    if(!enableFusion) {
      steps.push_back(prepare(runtime,metal::emitDecoderAdd(hiddenCount,threads,prefix+"attention_add",storageType),
          {current,attentionOutput},{storageType,storageType},afterAttention,storageType,log));
      steps.push_back(prepare(runtime,metal::emitDecoderRMSNorm(rows,hidden,plan.rmsNormEpsilon,
          threads,prefix+"post_norm",storageType),{afterAttention,layer.postAttentionNormWeight},
          {storageType,storageType},postAttentionNormalized,storageType,log));
      track(prefix+"residual_norm","residual_rmsnorm","template","add,norm",2);
    } else {
    const auto kernel = metal::emitDecoderResidualRMSNorm(
        rows, hidden, plan.rmsNormEpsilon, threads,
        prefix + "residual_rmsnorm", storageType);
    steps.push_back(prepare(
        runtime, kernel,
        {current, attentionOutput, layer.postAttentionNormWeight},
        {storageType, storageType, storageType},
        {afterAttention, postAttentionNormalized},
        {storageType, storageType}, log));
      track(prefix+"residual_rmsnorm","decoder_residual_rmsnorm","template",
            kernel.functionName,1);
    }
    }
    if(enableFusion && !pagedAttentionOutputAdmitted) {
      const auto name=prefix+"output_projection_residual";
      auto build=[&]() {
        std::vector<std::unique_ptr<metal::PreparedExecution>> candidate;
        candidate.push_back(prepare(runtime,metal::emitLinearResidual(rows,hidden,hidden,threads,name,storageType),
            {context,layer.outputWeight,current},{storageType,storageType,storageType},afterAttention,storageType,log));
        candidate.push_back(prepare(runtime,metal::emitDecoderRMSNorm(rows,hidden,plan.rmsNormEpsilon,
            threads,name+"_norm",storageType),{afterAttention,layer.postAttentionNormWeight},
            {storageType,storageType},postAttentionNormalized,storageType,log));
        return candidate;
      };
      if(admitFusion(runtime,steps,outputBegin,build,{context,current},
                     {afterAttention,postAttentionNormalized},name,
                     fusionDecisionKey("output_projection_residual"),
                     fusionDecisions,log)) {
        stage.uses.resize(outputUses);track(name,"output_projection_residual","template",name,2);
      }
    }
    if (!(batch==1 && hidden==64 && plan.intermediateSize==128 &&
          generated("decoder_gated_mlp",prefix+"gated_mlp",
                    {postAttentionNormalized,layer.gateWeight,layer.upWeight},{gated}))) {
    if(!enableFusion) {
      const auto gate=makeIntermediate(intermediateCount),up=makeIntermediate(intermediateCount);
      for(bool gateProjection:{true,false})
        steps.push_back(prepare(runtime,metal::emitLinearBaseline(rows,hidden,plan.intermediateSize,
            threads,prefix+(gateProjection?"gate":"up"),storageType,1,
            storageType),
            {postAttentionNormalized,gateProjection?layer.gateWeight:layer.upWeight},
            {storageType,storageType},gateProjection?gate:up,storageType,log));
      steps.push_back(prepare(runtime,metal::emitDecoderSiLUMul(intermediateCount,threads,
          prefix+"silu_mul",storageType),{gate,up},{storageType,storageType},gated,storageType,log));
      track(prefix+"gated_mlp","gated_mlp","template","gate,up,silu_mul",3);
    } else {
    const auto kernel = metal::emitDecoderGatedMLP(
        rows, hidden, plan.intermediateSize, threads,
        prefix + "gated_mlp", storageType);
    steps.push_back(prepare(
        runtime, kernel,
        {postAttentionNormalized, layer.gateWeight, layer.upWeight},
        {storageType, storageType, storageType}, gated, storageType, log));
      track(prefix+"gated_mlp","decoder_gated_mlp","template",
            kernel.functionName,1);
    }
    }
    const auto downBegin=steps.size(), downUses=stage.uses.size();
    addLinear(plan.intermediateSize, hidden, choices.intermediateToHidden, prefix + "down",
              gated, layer.downWeight, mlpOutput);
    steps.push_back(prepare(
        runtime,
        metal::emitDecoderAdd(hiddenCount, threads, prefix + "mlp_residual",
                              storageType),
        {afterAttention, mlpOutput},
        {storageType, storageType}, layerOutput, storageType, log));
    track(prefix + "mlp_residual", "residual_add", "template",
          prefix + "mlp_residual", 1);
    if(enableFusion) {
      const auto name=prefix+"down_projection_residual";
      auto build=[&]() {
        std::vector<std::unique_ptr<metal::PreparedExecution>> candidate;
        candidate.push_back(prepare(runtime,metal::emitLinearResidual(rows,plan.intermediateSize,
            hidden,threads,name,storageType),{gated,layer.downWeight,afterAttention},
            {storageType,storageType,storageType},layerOutput,storageType,log));
        return candidate;
      };
      if(admitFusion(runtime,steps,downBegin,build,{gated,afterAttention},
                     {layerOutput},name,
                     fusionDecisionKey("down_projection_residual"),
                     fusionDecisions,log)) {
        stage.uses.resize(downUses);track(name,"down_projection_residual","template",name,1);
      }
    }
    std::swap(current, nextLayerOutput);
  }

  const auto &normalized = hiddenScratch[0];
  steps.push_back(prepare(
      runtime,
      metal::emitDecoderRMSNorm(rows, hidden, plan.rmsNormEpsilon, threads,
                                "decoder_" + stageName + "_final_norm", storageType),
      {current, finalNormWeight},
      {storageType, storageType}, normalized, storageType, log));
  track("decoder_" + stageName + "_final_norm", "rmsnorm", "template",
        "decoder_" + stageName + "_final_norm", 1);
  const auto trunkStepCount=steps.size();
  stage.tokenUses=stage.uses;
  addLinear(hidden, plan.vocabularySize, choices.hiddenToVocabulary,
            "decoder_" + stageName + "_lm_head", normalized,
            languageModelHeadWeight, stage.logits);
  steps.push_back(prepare(
      runtime,
      metal::emitTokenArgmax(rows, plan.vocabularySize, threads,
                             "decoder_" + stageName + "_argmax",
                             metal::ElementType::Float32),
      {stage.logits}, {metal::ElementType::Float32}, stage.allNextTokenIds,
      metal::ElementType::Int32, log));
  track("decoder_" + stageName + "_argmax", "lm_head_argmax", "template",
        "decoder_" + stageName + "_argmax", 1);
  stage.sequence = makeSequence(runtime, steps);
  steps.resize(trunkStepCount);
  const auto chunks=(plan.vocabularySize+threads-1)/threads;
  const auto partialValues=intermediate(runtime,stage,rows*chunks,metal::ElementType::Float32);
  const auto partialTokens=intermediate(runtime,stage,rows*chunks,metal::ElementType::Int32);
  steps.push_back(prepare(runtime,metal::emitLMHeadPartialArgmax(rows,hidden,
      plan.vocabularySize,threads,"decoder_"+stageName+"_head_partial_argmax",storageType),
      {normalized,languageModelHeadWeight},{storageType,storageType},
      {partialValues,partialTokens},{metal::ElementType::Float32,metal::ElementType::Int32},log));
  steps.push_back(prepare(runtime,metal::emitLMHeadFinalArgmax(rows,chunks,threads,
      "decoder_"+stageName+"_head_final_argmax"),{partialValues,partialTokens},
      {metal::ElementType::Float32,metal::ElementType::Int32},stage.allNextTokenIds,
      metal::ElementType::Int32,log));
  stage.tokenSequence=makeSequence(runtime,steps);
  stage.tokenUses.push_back(registry.track({"decoder_"+stageName+"_head_argmax",
      "lm_head_argmax","template","partial_argmax,final_argmax",2,0}));
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
  std::vector<DecoderLLMModelLayerBuffers> layers;
  std::shared_ptr<DecoderLLMModelResources> modelResources;
  KernelRegistry registry;
  DecoderStage prefillStage;
  DecoderStage decodeStage;
  std::unique_ptr<DecoderStage> prefillChunkStage;
  std::unique_ptr<DecoderStage> prefillTailStage;
  std::size_t configuredChunkSize = 0;
  DType storageDtype = DType::Float32;
  bool precisionFallback = false;
  std::size_t modelBytes = 0;
  std::size_t kvBytes = 0;
  std::size_t activationBytes = 0;

  [[nodiscard]] std::size_t currentLength() const noexcept {
    return states.empty() ? 0 : states.front()->currentLength();
  }

  DecoderLLMRunResult run(DecoderStage &stage, std::size_t nextLength,
                          const std::vector<float> &tokenIds, bool readLogits = true) {
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
        for (auto &stagedState : states) {
          (void)stagedState->rollbackLength(*runtime, previousLength);
        }
        result.errorMessage = error;
        return result;
      }
    }
    if(!readLogits && !stage.tokenAdmissionChecked) {
      // Both sequences overwrite exactly the same staged KV append range.
      // Replaying at this uncommitted length is safe and does not append tokens.
      stage.tokenAdmissionChecked=true;
      const auto baseline=stage.sequence->execute();
      const auto expected=stage.allNextTokenIds->read();
      const auto candidate=stage.tokenSequence->execute();
      const auto actual=stage.allNextTokenIds->read();
      if(baseline.executionPassed && candidate.executionPassed && actual==expected) {
        const auto warmup=benchmark::warmup(*stage.tokenSequence,5);
        if(warmup.empty()) {
          const auto first=benchmark::measurePair(*stage.sequence,*stage.tokenSequence,31);
          const auto second=benchmark::measurePair(*stage.sequence,*stage.tokenSequence,31);
          stage.tokenAdmitted=first.passed && second.passed &&
              std::min(first.speedup,second.speedup)>=1.05;
          if(first.passed && second.passed)
            stage.tokenSpeedup=std::min(first.speedup,second.speedup);
          const auto final=stage.tokenSequence->execute();
          stage.tokenAdmitted &= final.executionPassed &&
                                 stage.allNextTokenIds->read()==expected;
        }
      }
    }
    const bool tokenOnly=!readLogits && stage.tokenAdmitted;
    const auto execution = (tokenOnly?stage.tokenSequence:stage.sequence)->execute();
    if (!execution.executionPassed) {
      std::string rollbackError;
      for (auto &state : states) {
        const auto stateError = state->rollbackLength(*runtime, previousLength);
        if (rollbackError.empty()) rollbackError = stateError;
      }
      result.errorMessage = execution.errorMessage;
      if (!rollbackError.empty()) {
        result.errorMessage += " KV cache rollback failed: " + rollbackError;
      }
      return result;
    }
    for (auto &state : states) state->commitLength(nextLength);
    registry.completed(tokenOnly?stage.tokenUses:stage.uses);
    if(readLogits) result.logits = stage.logits->read();
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
CompiledDecoderLLM::prefill(const std::vector<float> &tokenIds, bool readLogits) {
  if (currentLength() != 0) {
    return {false, {}, {}, {}, 0.0,
            "Decoder prefill requires an empty cache; call reset first."};
  }
  return impl_->run(impl_->prefillStage, impl_->plan.attention.prefillLength,
                    tokenIds,readLogits);
}

DecoderLLMRunResult
CompiledDecoderLLM::prefillChunked(const std::vector<float> &tokenIds) {
  DecoderLLMRunResult result;
  if (currentLength() != 0) {
    result.errorMessage =
        "Chunked decoder prefill requires an empty cache; call reset first.";
    return result;
  }
  if (!impl_->prefillChunkStage || impl_->configuredChunkSize == 0) {
    result.errorMessage = "Chunked decoder prefill was not compiled.";
    return result;
  }
  const auto expected = impl_->plan.attention.inputElementCount(
      impl_->plan.attention.prefillLength) / impl_->plan.hiddenSize();
  if (tokenIds.size() != expected) {
    result.errorMessage =
        "Chunked decoder token input shape does not match the prefill plan.";
    return result;
  }

  bool hasGpuTiming = true;
  double gpuTime = 0.0;
  for (std::size_t offset = 0; offset < tokenIds.size();) {
    const auto count = std::min(impl_->configuredChunkSize,
                                tokenIds.size() - offset);
    DecoderStage *stage = impl_->prefillChunkStage.get();
    if (count != impl_->configuredChunkSize) {
      stage = impl_->prefillTailStage.get();
    }
    if (stage == nullptr || stage->sequenceLength != count) {
      result.errorMessage = "Missing compiled stage for the final prefill chunk.";
      return result;
    }
    const std::vector<float> chunk(
        tokenIds.begin() + static_cast<std::ptrdiff_t>(offset),
        tokenIds.begin() + static_cast<std::ptrdiff_t>(offset + count));
    auto chunkResult = impl_->run(*stage, currentLength() + count, chunk);
    if (!chunkResult.passed) return chunkResult;
    result.logits.insert(result.logits.end(), chunkResult.logits.begin(),
                         chunkResult.logits.end());
    result.nextTokenIds = std::move(chunkResult.nextTokenIds);
    result.cpuSubmitToCompletionTimeUs +=
        chunkResult.cpuSubmitToCompletionTimeUs;
    if (chunkResult.gpuExecutionTimeUs) {
      gpuTime += *chunkResult.gpuExecutionTimeUs;
    } else {
      hasGpuTiming = false;
    }
    offset += count;
  }
  if (hasGpuTiming) result.gpuExecutionTimeUs = gpuTime;
  result.passed = true;
  return result;
}

DecoderLLMRunResult
CompiledDecoderLLM::decode(const std::vector<float> &tokenIds, bool readLogits) {
  const auto current = currentLength();
  if (current == 0) {
    return {false, {}, {}, {}, 0.0,
            "Decoder decode requires a completed prefill."};
  }
  if (current >= impl_->plan.attention.capacity) {
    return {false, {}, {}, {}, 0.0,
            "Decoder KV cache capacity has been reached."};
  }
  return impl_->run(impl_->decodeStage, current + 1, tokenIds,readLogits);
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

bool CompiledDecoderLLM::usesPagedKVCache() const noexcept {
  return !impl_->states.empty() && impl_->states.front()->isPaged();
}

std::size_t CompiledDecoderLLM::kvPageSize() const noexcept {
  return usesPagedKVCache() ? impl_->states.front()->pageSize() : 0;
}

std::size_t CompiledDecoderLLM::prefillChunkSize() const noexcept {
  return impl_->configuredChunkSize;
}

std::size_t CompiledDecoderLLM::allocatedKVPageCount() const noexcept {
  return usesPagedKVCache() ? impl_->states.front()->allocatedPageCount() : 0;
}

DType CompiledDecoderLLM::storageDtype() const noexcept {
  return impl_->storageDtype;
}

bool CompiledDecoderLLM::precisionFallbackUsed() const noexcept {
  return impl_->precisionFallback;
}

std::size_t CompiledDecoderLLM::modelStorageBytes() const noexcept {
  return impl_->modelBytes;
}

std::size_t CompiledDecoderLLM::kvStorageBytes() const noexcept {
  return impl_->kvBytes;
}

std::size_t CompiledDecoderLLM::activationStorageBytes() const noexcept {
  return impl_->activationBytes;
}

std::shared_ptr<DecoderLLMModelResources>
CompiledDecoderLLM::sharedModelResources() const noexcept {
  return impl_->modelResources;
}

std::vector<std::int32_t>
CompiledDecoderLLM::kvBlockTable(std::size_t layer) const {
  if (layer >= impl_->states.size()) {
    throw std::out_of_range("Decoder layer index is out of range.");
  }
  return impl_->states[layer]->pageTable();
}

DecoderKVSnapshot CompiledDecoderLLM::snapshotKVCache() const {
  DecoderKVSnapshot snapshot;
  snapshot.length = currentLength();
  snapshot.keyCaches.reserve(impl_->states.size());
  snapshot.valueCaches.reserve(impl_->states.size());
  for (const auto &state : impl_->states) {
    snapshot.keyCaches.push_back(state->readKeyPrefix());
    snapshot.valueCaches.push_back(state->readValuePrefix());
  }
  return snapshot;
}

std::string CompiledDecoderLLM::releaseKVCache() { return reset(); }

std::string
CompiledDecoderLLM::restoreKVCache(const DecoderKVSnapshot &snapshot) {
  if (currentLength() != 0) {
    return "Decoder KV restore requires an empty cache.";
  }
  if (snapshot.length == 0 ||
      snapshot.keyCaches.size() != impl_->states.size() ||
      snapshot.valueCaches.size() != impl_->states.size()) {
    return "Decoder KV snapshot does not match the compiled model.";
  }
  for (std::size_t layer = 0; layer < impl_->states.size(); ++layer) {
    const auto error = impl_->states[layer]->restore(
        *impl_->runtime, snapshot.length, snapshot.keyCaches[layer],
        snapshot.valueCaches[layer]);
    if (!error.empty()) {
      for (auto &state : impl_->states) {
        (void)state->reset(*impl_->runtime);
      }
      return error;
    }
  }
  return {};
}

std::string CompiledDecoderLLM::stageKVLength(std::size_t length) {
  const auto previous = currentLength();
  for (auto &state : impl_->states) {
    const auto error = state->stageLength(*impl_->runtime, length);
    if (!error.empty()) {
      for (auto &staged : impl_->states) {
        (void)staged->rollbackLength(*impl_->runtime, previous);
      }
      return error;
    }
  }
  return {};
}

void CompiledDecoderLLM::commitKVLength(std::size_t length) {
  for (auto &state : impl_->states) state->commitLength(length);
}

std::string CompiledDecoderLLM::rollbackKVLength(std::size_t length) {
  std::string firstError;
  for (auto &state : impl_->states) {
    const auto error = state->rollbackLength(*impl_->runtime, length);
    if (firstError.empty()) firstError = error;
  }
  return firstError;
}

void CompiledDecoderLLM::resetKernelUsage() noexcept {
  impl_->registry.resetUsage();
}

void CompiledDecoderLLM::reportKernelUsage(std::ostream &log) const {
  impl_->registry.report(log);
  for(const auto *stage:{&impl_->prefillStage,&impl_->decodeStage})
    log << "LM Head + Argmax sequence_length=" << stage->sequenceLength
        << ": " << (!stage->tokenAdmissionChecked?"not requested":
                      stage->tokenAdmitted?"admitted":"fallback")
        << ", speedup=" << stage->tokenSpeedup << "x\n";
}

DecoderLLMCompilation compileDecoderLLM(
    metal::MetalRuntime &runtime, const DecoderLLMWorkload &workload,
    std::ostream &log, const DecoderLLMCompileOptions &options) {
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
    planner::DecoderLLMPlan plan = workload.plan;
    const auto requestedDtype =
        options.requestedStorageDtype.value_or(options.storageDtype);
    bool precisionFallback = requestedDtype != options.storageDtype;
    DType effectiveDtype = options.storageDtype;
    if (effectiveDtype == DType::BFloat16 &&
        !runtime.hardwareInfo().supportsBFloat16) {
      if (!options.allowPrecisionFallback) {
        throw std::invalid_argument(
            "Decoder bf16 storage is unavailable on this Metal backend; "
            "enable allowPrecisionFallback to use fp16.");
      }
      effectiveDtype = DType::Float16;
      precisionFallback = true;
      log << "Decoder precision: requested=" << dtypeName(requestedDtype)
          << ", effective=fp16, fallback=PASS\n";
    }
    if (effectiveDtype != DType::Float16 && effectiveDtype != DType::Float32 &&
        effectiveDtype != DType::BFloat16) {
      throw std::invalid_argument(
          "Decoder storage dtype must be fp32, fp16, or bf16.");
    }
    plan.attention.dtype = effectiveDtype;
    plan.validate();
    const auto hardware = runtime.hardwareInfo();
    if (plan.attention.threadsPerThreadgroup >
            hardware.maxThreadsPerThreadgroup ||
        plan.attention.threadsPerThreadgroup * sizeof(float) +
                4 * sizeof(float) >
            hardware.maxThreadgroupMemoryLength) {
      throw std::invalid_argument(
          "Decoder attention threadgroup configuration exceeds Metal limits.");
    }
    const auto storageType = metalStorageType(effectiveDtype);
    DecoderLLMWorkload effectiveWorkload = workload;
    effectiveWorkload.plan = plan;
    validateWeights(effectiveWorkload, plan);

    if (options.kvPageSize != 0 && plan.attention.batch != 1) {
      throw std::invalid_argument(
          "Paged decoder execution currently supports one request.");
    }
    if (options.prefillChunkSize != 0) {
      if (options.kvPageSize == 0) {
        throw std::invalid_argument(
            "Chunked prefill requires paged KV cache execution.");
      }
      if (options.prefillChunkSize >= plan.attention.prefillLength) {
        throw std::invalid_argument(
            "Prefill chunk size must be smaller than the full prefill length.");
      }
    }
    if (!options.sharedKVPools.empty()) {
      if (options.kvPageSize == 0 ||
          options.sharedKVPools.size() != plan.layerCount) {
        throw std::invalid_argument(
            "Shared KV pools require a page size and one pool per layer.");
      }
      for (const auto &pool : options.sharedKVPools) {
        if (!pool || pool->pageSize() != options.kvPageSize) {
          throw std::invalid_argument(
              "Shared KV pool page size differs from decoder options.");
        }
      }
    }

    const auto hidden = plan.hiddenSize();
    const auto &firstLayer = effectiveWorkload.layers.front();
    std::vector<float> hiddenInput(plan.attention.batch * hidden);
    for (std::size_t row = 0; row < plan.attention.batch; ++row) {
      std::copy(effectiveWorkload.embeddingWeight.begin(),
                effectiveWorkload.embeddingWeight.begin() +
                    static_cast<std::ptrdiff_t>(hidden),
                hiddenInput.begin() + static_cast<std::ptrdiff_t>(row * hidden));
    }
    std::vector<float> intermediateInput(plan.attention.batch * plan.intermediateSize);
    for (std::size_t index = 0; index < intermediateInput.size(); ++index) {
      intermediateInput[index] =
          static_cast<float>(static_cast<int>(index % 17) - 8) / 17.0f;
    }

    log << "Decoder precision: storage=" << dtypeName(effectiveDtype)
        << ", accumulation=" << dtypeName(plan.accumulationDtype)
        << ", fallback=" << (precisionFallback ? "true" : "false") << '\n';
    LinearChoices choices;
    log << "Decoder Decode-GEMV admission:\n";
    choices.hiddenToHidden = tuneLinear(
        runtime, plan.attention.batch, hidden, hidden, hiddenInput,
        firstLayer.queryWeight, log, effectiveDtype);
    choices.intermediateToHidden = tuneLinear(
        runtime, plan.attention.batch, plan.intermediateSize, hidden,
        intermediateInput, firstLayer.downWeight, log, effectiveDtype);
    choices.hiddenToVocabulary = tuneLinear(
        runtime, plan.attention.batch, hidden, plan.vocabularySize,
        hiddenInput, effectiveWorkload.languageModelHeadWeight, log,
        effectiveDtype);

    auto impl = std::make_unique<CompiledDecoderLLM::Impl>();
    impl->runtime = &runtime;
    impl->plan = plan;
    impl->storageDtype = effectiveDtype;
    impl->precisionFallback = precisionFallback;
    impl->registry.load(runtime, options.kernelLibrary, log);
    auto modelResources = options.sharedModelResources;
    if (modelResources) {
      if (modelResources->storageDtype != effectiveDtype ||
          modelResources->hiddenSize != hidden ||
          modelResources->headCount != plan.attention.heads ||
          modelResources->headDimension != plan.attention.headDimension ||
          modelResources->cacheCapacity != plan.attention.capacity ||
          modelResources->intermediateSize != plan.intermediateSize ||
          modelResources->vocabularySize != plan.vocabularySize ||
          modelResources->layerCount != plan.layerCount ||
          modelResources->layers.size() != plan.layerCount) {
        throw std::invalid_argument(
            "Shared decoder model resources do not match this plan.");
      }
      log << "Decoder model storage: reused shared GPU weights\n";
    } else {
      modelResources = std::make_shared<DecoderLLMModelResources>();
      modelResources->storageDtype = effectiveDtype;
      modelResources->hiddenSize = hidden;
      modelResources->headCount = plan.attention.heads;
      modelResources->headDimension = plan.attention.headDimension;
      modelResources->cacheCapacity = plan.attention.capacity;
      modelResources->intermediateSize = plan.intermediateSize;
      modelResources->vocabularySize = plan.vocabularySize;
      modelResources->layerCount = plan.layerCount;
      modelResources->storageBytes =
          modelStorageBytes(effectiveWorkload, effectiveDtype);
      modelResources->embeddingWeight = allocate(
          runtime, effectiveWorkload.embeddingWeight.size(),
          effectiveWorkload.embeddingWeight.data(), storageType);
      modelResources->finalNormWeight = allocate(
          runtime, effectiveWorkload.finalNormWeight.size(),
          effectiveWorkload.finalNormWeight.data(), storageType);
      modelResources->languageModelHeadWeight = allocate(
          runtime, effectiveWorkload.languageModelHeadWeight.size(),
          effectiveWorkload.languageModelHeadWeight.data(), storageType);
      // Rotary tables retain fp32 storage while activations use the selected
      // low-precision type.
      modelResources->ropeCosine = allocate(
          runtime, effectiveWorkload.ropeCosine.size(),
          effectiveWorkload.ropeCosine.data(), metal::ElementType::Float32);
      modelResources->ropeSine = allocate(
          runtime, effectiveWorkload.ropeSine.size(),
          effectiveWorkload.ropeSine.data(), metal::ElementType::Float32);
      for (const auto &layer : effectiveWorkload.layers) {
        modelResources->layers.push_back(
            allocateLayer(runtime, layer, storageType));
      }
      log << "Decoder model storage: allocated shared GPU weights\n";
    }
    impl->modelResources = modelResources;
    impl->embeddingWeight = modelResources->embeddingWeight;
    impl->finalNormWeight = modelResources->finalNormWeight;
    impl->languageModelHeadWeight =
        modelResources->languageModelHeadWeight;
    impl->ropeCosine = modelResources->ropeCosine;
    impl->ropeSine = modelResources->ropeSine;
    impl->layers = modelResources->layers;
    for (std::size_t layer = 0; layer < effectiveWorkload.layers.size(); ++layer) {
      auto state = options.sharedKVPools.empty()
                       ? createKVCacheState(runtime, plan.attention,
                                            options.kvPageSize)
                       : createKVCacheState(runtime, plan.attention,
                                            options.sharedKVPools[layer]);
      if (!state.state) throw std::runtime_error(state.errorMessage);
      impl->states.push_back(std::move(state.state));
    }

    FusionDecisionCache fusionDecisions;
    impl->prefillStage = buildStage(
        runtime, effectiveWorkload, plan.attention.prefillLength, "prefill",
        impl->embeddingWeight, impl->finalNormWeight,
        impl->languageModelHeadWeight, impl->ropeCosine, impl->ropeSine,
        impl->layers, impl->states, choices, impl->registry, log, storageType,
        options.enableFusion, fusionDecisions);
    impl->decodeStage = buildStage(
        runtime, effectiveWorkload, 1, "decode", impl->embeddingWeight,
        impl->finalNormWeight, impl->languageModelHeadWeight, impl->ropeCosine,
        impl->ropeSine, impl->layers, impl->states, choices, impl->registry, log,
        storageType,options.enableFusion,fusionDecisions);
    if (options.prefillChunkSize != 0) {
      impl->configuredChunkSize = options.prefillChunkSize;
      impl->prefillChunkStage = std::make_unique<DecoderStage>(buildStage(
          runtime, effectiveWorkload, options.prefillChunkSize,
          "prefill_chunk_" + std::to_string(options.prefillChunkSize),
          impl->embeddingWeight, impl->finalNormWeight,
          impl->languageModelHeadWeight, impl->ropeCosine, impl->ropeSine,
          impl->layers, impl->states, choices, impl->registry, log,
          storageType,options.enableFusion,fusionDecisions));
      const auto tail =
          plan.attention.prefillLength % options.prefillChunkSize;
      if (tail != 0) {
        impl->prefillTailStage = std::make_unique<DecoderStage>(buildStage(
            runtime, effectiveWorkload, tail,
            "prefill_chunk_" + std::to_string(tail), impl->embeddingWeight,
            impl->finalNormWeight, impl->languageModelHeadWeight,
            impl->ropeCosine, impl->ropeSine, impl->layers, impl->states,
            choices, impl->registry, log, storageType,options.enableFusion,
            fusionDecisions));
      }
    }
    impl->modelBytes = modelResources->storageBytes;
    std::unordered_set<const metal::MetalBuffer *> counted;
    for (const auto &state : impl->states) {
      for (const auto &buffer : {state->keyBuffer(), state->valueBuffer()}) {
        if (buffer && counted.insert(buffer.get()).second) impl->kvBytes += bufferBytes(buffer);
      }
    }
    impl->activationBytes = stageBytes(impl->prefillStage) +
                            stageBytes(impl->decodeStage);
    if (impl->prefillChunkStage) impl->activationBytes += stageBytes(*impl->prefillChunkStage);
    if (impl->prefillTailStage) impl->activationBytes += stageBytes(*impl->prefillTailStage);
    result.executable = std::unique_ptr<CompiledDecoderLLM>(
        new CompiledDecoderLLM(std::move(impl)));
    log << "Decoder-only stateful compilation: PASS\n";
  } catch (const std::exception &error) {
    result.errorMessage = error.what();
  }
  return result;
}

} // namespace tensor::runtime
