#include "backend/metal/DecoderLLMMetalEmitter.hpp"

#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace tensor::metal {
namespace {

const char *storageTypeName(ElementType type) {
  if (type == ElementType::Float16) return "half";
  if (type == ElementType::Float32) return "float";
  if (type == ElementType::BFloat16) return "bfloat";
  throw std::invalid_argument(
      "Decoder Metal emitters support fp16, bf16, or fp32 storage.");
}

GeneratedKernel beginKernel(std::size_t workItems, std::size_t threads,
                            const std::string &name) {
  if (workItems == 0 || threads == 0 || name.empty()) {
    throw std::invalid_argument("Decoder-only kernel configuration is invalid.");
  }
  GeneratedKernel kernel;
  kernel.functionName = name;
  kernel.threadsPerThreadgroup = threads;
  kernel.threadgroupCount = (workItems + threads - 1) / threads;
  return kernel;
}

} // namespace

GeneratedKernel emitDecoderEmbedding(
    const planner::DecoderLLMPlan &plan, std::size_t sequenceLength,
    const std::string &functionName, ElementType storageType) {
  plan.validate();
  const auto hidden = plan.hiddenSize();
  const auto *type = storageTypeName(storageType);
  const auto workItems = plan.hiddenElementCount(sequenceLength);
  auto kernel = beginKernel(workItems, plan.attention.threadsPerThreadgroup,
                            functionName);
  std::ostringstream source;
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << functionName << "(\n"
         << "  device const int *tokenIds [[buffer(0)]],\n"
         << "  device const " << type << " *embedding [[buffer(1)]],\n"
         << "  device " << type << " *output [[buffer(2)]],\n"
         << "  uint gid [[thread_position_in_grid]]) {\n"
         << "  if (gid >= " << workItems << "u) return;\n"
         << "  const uint row = gid / " << hidden << "u;\n"
         << "  const uint feature = gid % " << hidden << "u;\n"
         << "  const int token = tokenIds[row];\n"
         << "  if (token < 0 || uint(token) >= " << plan.vocabularySize << "u) return;\n"
         << "  output[gid] = " << type << "(float(embedding[uint(token) * " << hidden
         << "u + feature]));\n"
         << "}\n";
  kernel.source = source.str();
  return kernel;
}

GeneratedKernel emitDecoderRMSNorm(
    std::size_t rows, std::size_t width, float epsilon,
    std::size_t threads, const std::string &functionName,
    ElementType storageType) {
  if (rows == 0 || width == 0 || threads == 0 ||
      (threads & (threads - 1)) != 0 || functionName.empty() ||
      !std::isfinite(epsilon) || epsilon <= 0.0f) {
    throw std::invalid_argument("Decoder RMSNorm configuration is invalid.");
  }
  GeneratedKernel kernel;
  kernel.functionName = functionName;
  kernel.threadsPerThreadgroup = threads;
  kernel.threadgroupCount = rows;
  std::ostringstream source;
  const auto *type = storageTypeName(storageType);
  source.imbue(std::locale::classic());
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << functionName << "(\n"
         << "  device const " << type << " *input [[buffer(0)]],\n"
         << "  device const " << type << " *weight [[buffer(1)]],\n"
         << "  device " << type << " *output [[buffer(2)]],\n"
         << "  uint tid [[thread_index_in_threadgroup]],\n"
         << "  uint row [[threadgroup_position_in_grid]]) {\n"
         << "  threadgroup float partial[" << threads << "];\n"
         << "  const uint base = row * " << width << "u;\n"
         << "  float squareSum = 0.0f;\n"
         << "  for (uint i = tid; i < " << width << "u; i += " << threads
         << "u) { const float x = float(input[base + i]); squareSum += x * x; }\n"
         << "  partial[tid] = squareSum;\n"
         << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         << "  for (uint offset = " << threads / 2
         << "u; offset > 0u; offset >>= 1u) {\n"
         << "    if (tid < offset) partial[tid] += partial[tid + offset];\n"
         << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         << "  }\n"
         << "  const float inverse = rsqrt(partial[0] / " << width << ".0f + "
         << std::scientific
         << std::setprecision(std::numeric_limits<float>::max_digits10) << epsilon
         << "f);\n"
         << "  for (uint i = tid; i < " << width << "u; i += " << threads
         << "u) output[base + i] = " << type
         << "(float(input[base + i]) * inverse * float(weight[i]));\n"
         << "}\n";
  kernel.source = source.str();
  return kernel;
}

GeneratedKernel emitDecoderRoPE(
    const planner::DecoderLLMPlan &plan, std::size_t sequenceLength,
    const std::string &functionName, ElementType storageType) {
  plan.validate();
  const auto pairs = plan.attention.headDimension / 2;
  const auto workItems = plan.attention.batch * plan.attention.heads *
                         sequenceLength * pairs;
  auto kernel = beginKernel(workItems, plan.attention.threadsPerThreadgroup,
                            functionName);
  const auto hidden = plan.hiddenSize();
  const auto *type = storageTypeName(storageType);
  std::ostringstream source;
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << functionName << "(\n"
         << "  device const " << type << " *input [[buffer(0)]],\n"
         << "  device const float *cosine [[buffer(1)]],\n"
         << "  device const float *sine [[buffer(2)]],\n"
         << "  device const int *validLength [[buffer(3)]],\n"
         << "  device " << type << " *output [[buffer(4)]],\n"
         << "  uint gid [[thread_position_in_grid]]) {\n"
         << "  if (gid >= " << workItems << "u) return;\n"
         << "  const uint pair = gid % " << pairs << "u;\n"
         << "  const uint token = (gid / " << pairs << "u) % "
         << sequenceLength << "u;\n"
         << "  const uint head = (gid / " << pairs * sequenceLength
         << "u) % " << plan.attention.heads << "u;\n"
         << "  const uint batch = gid / "
         << pairs * sequenceLength * plan.attention.heads << "u;\n"
         << "  const uint valid = uint(validLength[0]);\n"
         << "  if (valid < " << sequenceLength << "u || valid > "
         << plan.attention.capacity << "u) return;\n"
         << "  const uint position = valid - " << sequenceLength << "u + token;\n"
         << "  const uint inputBase = (batch * " << sequenceLength
         << "u + token) * " << hidden << "u + head * "
         << plan.attention.headDimension << "u + pair * 2u;\n"
         << "  const uint outputBase = ((batch * " << plan.attention.heads
         << "u + head) * " << sequenceLength << "u + token) * "
         << plan.attention.headDimension << "u + pair * 2u;\n"
         << "  const float c = cosine[position * " << pairs << "u + pair];\n"
         << "  const float s = sine[position * " << pairs << "u + pair];\n"
         << "  const float even = float(input[inputBase]);\n"
         << "  const float odd = float(input[inputBase + 1u]);\n"
         << "  output[outputBase] = " << type << "(even * c - odd * s);\n"
         << "  output[outputBase + 1u] = " << type
         << "(even * s + odd * c);\n"
         << "}\n";
  kernel.source = source.str();
  return kernel;
}

GeneratedKernel emitDecoderCacheAppend(
    const planner::DecoderLLMPlan &plan, std::size_t sequenceLength,
    bool inputIsHeadMajor, const std::string &functionName,
    ElementType storageType) {
  plan.validate();
  const auto workItems = plan.hiddenElementCount(sequenceLength);
  const auto hidden = plan.hiddenSize();
  const auto *type = storageTypeName(storageType);
  auto kernel = beginKernel(workItems, plan.attention.threadsPerThreadgroup,
                            functionName);
  std::ostringstream source;
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << functionName << "(\n"
         << "  device const " << type << " *input [[buffer(0)]],\n"
         << "  device const int *validLength [[buffer(1)]],\n"
         << "  device " << type << " *cache [[buffer(2)]],\n"
         << "  uint gid [[thread_position_in_grid]]) {\n"
         << "  if (gid >= " << workItems << "u) return;\n"
         << "  const uint feature = gid % " << hidden << "u;\n"
         << "  const uint token = (gid / " << hidden << "u) % "
         << sequenceLength << "u;\n"
         << "  const uint batch = gid / " << hidden * sequenceLength << "u;\n"
         << "  const uint head = feature / " << plan.attention.headDimension << "u;\n"
         << "  const uint component = feature % " << plan.attention.headDimension << "u;\n"
         << "  const uint valid = uint(validLength[0]);\n"
         << "  if (valid < " << sequenceLength << "u || valid > "
         << plan.attention.capacity << "u) return;\n"
         << "  const uint position = valid - " << sequenceLength << "u + token;\n"
         << "  const uint cacheIndex = ((batch * " << plan.attention.heads
         << "u + head) * " << plan.attention.capacity << "u + position) * "
         << plan.attention.headDimension << "u + component;\n";
  if (inputIsHeadMajor) {
    source << "  const uint inputIndex = ((batch * " << plan.attention.heads
           << "u + head) * " << sequenceLength << "u + token) * "
           << plan.attention.headDimension << "u + component;\n";
  } else {
    source << "  const uint inputIndex = (batch * " << sequenceLength
           << "u + token) * " << hidden << "u + feature;\n";
  }
  source << "  cache[cacheIndex] = input[inputIndex];\n"
         << "}\n";
  kernel.source = source.str();
  return kernel;
}

GeneratedKernel emitDecoderPagedCacheAppend(
    const planner::DecoderLLMPlan &plan, std::size_t sequenceLength,
    bool inputIsHeadMajor, std::size_t pageSize,
    const std::string &functionName, ElementType storageType) {
  plan.validate();
  if (pageSize == 0 || pageSize > plan.attention.capacity) {
    throw std::invalid_argument("Decoder paged-cache page size is invalid.");
  }
  const auto workItems = plan.hiddenElementCount(sequenceLength);
  const auto hidden = plan.hiddenSize();
  const auto *type = storageTypeName(storageType);
  auto kernel = beginKernel(workItems, plan.attention.threadsPerThreadgroup,
                            functionName);
  std::ostringstream source;
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << functionName << "(\n"
         << "  device const " << type << " *input [[buffer(0)]],\n"
         << "  device const int *validLength [[buffer(1)]],\n"
         << "  device const int *blockTable [[buffer(2)]],\n"
         << "  device " << type << " *cache [[buffer(3)]],\n"
         << "  uint gid [[thread_position_in_grid]]) {\n"
         << "  if (gid >= " << workItems << "u) return;\n"
         << "  const uint feature = gid % " << hidden << "u;\n"
         << "  const uint token = (gid / " << hidden << "u) % "
         << sequenceLength << "u;\n"
         << "  const uint batch = gid / " << hidden * sequenceLength << "u;\n"
         << "  const uint head = feature / " << plan.attention.headDimension << "u;\n"
         << "  const uint component = feature % " << plan.attention.headDimension << "u;\n"
         << "  const uint valid = uint(validLength[0]);\n"
         << "  if (valid < " << sequenceLength << "u || valid > "
         << plan.attention.capacity << "u) return;\n"
         << "  const uint position = valid - " << sequenceLength << "u + token;\n"
         << "  const uint logicalPage = position / " << pageSize << "u;\n"
         << "  const uint pageOffset = position % " << pageSize << "u;\n"
         << "  const int physicalPageValue = blockTable[logicalPage];\n"
         << "  if (physicalPageValue < 0) return;\n"
         << "  const uint physicalPage = uint(physicalPageValue);\n"
         << "  const uint cacheIndex = (((physicalPage * "
         << plan.attention.batch << "u + batch) * " << plan.attention.heads
         << "u + head) * " << pageSize << "u + pageOffset) * "
         << plan.attention.headDimension << "u + component;\n";
  if (inputIsHeadMajor) {
    source << "  const uint inputIndex = ((batch * " << plan.attention.heads
           << "u + head) * " << sequenceLength << "u + token) * "
           << plan.attention.headDimension << "u + component;\n";
  } else {
    source << "  const uint inputIndex = (batch * " << sequenceLength
           << "u + token) * " << hidden << "u + feature;\n";
  }
  source << "  cache[cacheIndex] = input[inputIndex];\n"
         << "}\n";
  kernel.source = source.str();
  return kernel;
}

GeneratedKernel emitServingRoPE(
    const planner::DecoderLLMPlan &plan, std::size_t maximumBatchSize,
    const std::string &functionName, ElementType storageType) {
  plan.validate();
  if (maximumBatchSize == 0) {
    throw std::invalid_argument("Serving RoPE batch size is invalid.");
  }
  const auto pairs = plan.attention.headDimension / 2;
  const auto perRequest = plan.attention.heads * pairs;
  const auto workItems = maximumBatchSize * perRequest;
  auto kernel = beginKernel(workItems, plan.attention.threadsPerThreadgroup,
                            functionName);
  const auto hidden = plan.hiddenSize();
  const auto *type = storageTypeName(storageType);
  std::ostringstream source;
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << functionName << "(\n"
         << "  device const " << type << " *input [[buffer(0)]],\n"
         << "  device const float *cosine [[buffer(1)]],\n"
         << "  device const float *sine [[buffer(2)]],\n"
         << "  device const int *validLengths [[buffer(3)]],\n"
         << "  device const int *activeCount [[buffer(4)]],\n"
         << "  device " << type << " *output [[buffer(5)]],\n"
         << "  uint gid [[thread_position_in_grid]]) {\n"
         << "  const uint request = gid / " << perRequest << "u;\n"
         << "  if (gid >= " << workItems
         << "u || request >= uint(activeCount[0])) return;\n"
         << "  const uint pair = gid % " << pairs << "u;\n"
         << "  const uint head = (gid / " << pairs << "u) % "
         << plan.attention.heads << "u;\n"
         << "  const int validValue = validLengths[request];\n"
         << "  if (validValue <= 0 || uint(validValue) > "
         << plan.attention.capacity << "u) return;\n"
         << "  const uint position = uint(validValue - 1);\n"
         << "  const uint inputBase = request * " << hidden
         << "u + head * " << plan.attention.headDimension
         << "u + pair * 2u;\n"
         << "  const uint outputBase = (request * "
         << plan.attention.heads << "u + head) * "
         << plan.attention.headDimension << "u + pair * 2u;\n"
         << "  const float c = cosine[position * " << pairs
         << "u + pair];\n"
         << "  const float s = sine[position * " << pairs
         << "u + pair];\n"
         << "  const float even = float(input[inputBase]);\n"
         << "  const float odd = float(input[inputBase + 1u]);\n"
         << "  output[outputBase] = " << type << "(even * c - odd * s);\n"
         << "  output[outputBase + 1u] = " << type
         << "(even * s + odd * c);\n"
         << "}\n";
  kernel.source = source.str();
  return kernel;
}

GeneratedKernel emitServingPagedCacheAppend(
    const planner::DecoderLLMPlan &plan, std::size_t maximumBatchSize,
    bool inputIsHeadMajor, std::size_t pageSize,
    const std::string &functionName, ElementType storageType) {
  plan.validate();
  if (maximumBatchSize == 0 || pageSize == 0 ||
      pageSize > plan.attention.capacity) {
    throw std::invalid_argument("Serving paged-cache configuration is invalid.");
  }
  const auto hidden = plan.hiddenSize();
  const auto *type = storageTypeName(storageType);
  const auto maximumPages =
      (plan.attention.capacity + pageSize - 1) / pageSize;
  const auto workItems = maximumBatchSize * hidden;
  auto kernel = beginKernel(workItems, plan.attention.threadsPerThreadgroup,
                            functionName);
  std::ostringstream source;
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << functionName << "(\n"
         << "  device const " << type << " *input [[buffer(0)]],\n"
         << "  device const int *validLengths [[buffer(1)]],\n"
         << "  device const int *blockTables [[buffer(2)]],\n"
         << "  device const int *activeCount [[buffer(3)]],\n"
         << "  device " << type << " *cache [[buffer(4)]],\n"
         << "  uint gid [[thread_position_in_grid]]) {\n"
         << "  const uint request = gid / " << hidden << "u;\n"
         << "  if (gid >= " << workItems
         << "u || request >= uint(activeCount[0])) return;\n"
         << "  const uint feature = gid % " << hidden << "u;\n"
         << "  const uint head = feature / "
         << plan.attention.headDimension << "u;\n"
         << "  const uint component = feature % "
         << plan.attention.headDimension << "u;\n"
         << "  const int validValue = validLengths[request];\n"
         << "  if (validValue <= 0 || uint(validValue) > "
         << plan.attention.capacity << "u) return;\n"
         << "  const uint position = uint(validValue - 1);\n"
         << "  const uint logicalPage = position / " << pageSize << "u;\n"
         << "  const uint pageOffset = position % " << pageSize << "u;\n"
         << "  const int physicalPageValue = blockTables[request * "
         << maximumPages << "u + logicalPage];\n"
         << "  if (physicalPageValue < 0) return;\n"
         << "  const uint physicalPage = uint(physicalPageValue);\n"
         << "  const uint cacheIndex = ((physicalPage * "
         << plan.attention.heads << "u + head) * " << pageSize
         << "u + pageOffset) * " << plan.attention.headDimension
         << "u + component;\n";
  if (inputIsHeadMajor) {
    source << "  const uint inputIndex = (request * "
           << plan.attention.heads << "u + head) * "
           << plan.attention.headDimension << "u + component;\n";
  } else {
    source << "  const uint inputIndex = request * " << hidden
           << "u + feature;\n";
  }
  source << "  cache[cacheIndex] = input[inputIndex];\n"
         << "}\n";
  kernel.source = source.str();
  return kernel;
}

GeneratedKernel emitDecoderAdd(std::size_t elementCount, std::size_t threads,
                               const std::string &functionName,
                               ElementType storageType) {
  auto kernel = beginKernel(elementCount, threads, functionName);
  const auto *type = storageTypeName(storageType);
  std::ostringstream source;
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << functionName << "(\n"
         << "  device const " << type << " *left [[buffer(0)]],\n"
         << "  device const " << type << " *right [[buffer(1)]],\n"
         << "  device " << type << " *output [[buffer(2)]],\n"
         << "  uint gid [[thread_position_in_grid]]) {\n"
         << "  if (gid >= " << elementCount << "u) return;\n"
         << "  output[gid] = " << type
         << "(float(left[gid]) + float(right[gid]));\n"
         << "}\n";
  kernel.source = source.str();
  return kernel;
}

GeneratedKernel emitDecoderSiLUMul(std::size_t elementCount,
                                   std::size_t threads,
                                   const std::string &functionName,
                                   ElementType storageType) {
  auto kernel = beginKernel(elementCount, threads, functionName);
  const auto *type = storageTypeName(storageType);
  std::ostringstream source;
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << functionName << "(\n"
         << "  device const " << type << " *gate [[buffer(0)]],\n"
         << "  device const " << type << " *up [[buffer(1)]],\n"
         << "  device " << type << " *output [[buffer(2)]],\n"
         << "  uint gid [[thread_position_in_grid]]) {\n"
         << "  if (gid >= " << elementCount << "u) return;\n"
         << "  const float x = float(gate[gid]);\n"
         << "  const float activated = float(" << type << "(x / (1.0f + exp(-x))));\n"
         << "  output[gid] = " << type << "(activated * float(up[gid]));\n"
         << "}\n";
  kernel.source = source.str();
  return kernel;
}

GeneratedKernel emitDecoderResidualRMSNorm(
    std::size_t rows, std::size_t width, float epsilon,
    std::size_t threads, const std::string &functionName,
    ElementType storageType) {
  if (width == 0 || !std::isfinite(epsilon) || epsilon <= 0.0f) {
    throw std::invalid_argument(
        "Decoder residual RMSNorm configuration is invalid.");
  }
  if (rows == 0 || threads == 0 || (threads & (threads - 1)) != 0 ||
      functionName.empty()) {
    throw std::invalid_argument(
        "Decoder residual RMSNorm dispatch configuration is invalid.");
  }
  GeneratedKernel kernel;
  kernel.functionName = functionName;
  kernel.threadsPerThreadgroup = threads;
  kernel.threadgroupCount = rows;
  const auto *type = storageTypeName(storageType);
  std::ostringstream source;
  source.imbue(std::locale::classic());
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << functionName << "(\n"
         << "  device const " << type << " *residual [[buffer(0)]],\n"
         << "  device const " << type << " *update [[buffer(1)]],\n"
         << "  device const " << type << " *weight [[buffer(2)]],\n"
         << "  device " << type << " *sumOutput [[buffer(3)]],\n"
         << "  device " << type << " *normalizedOutput [[buffer(4)]],\n"
         << "  uint tid [[thread_index_in_threadgroup]],\n"
         << "  uint row [[threadgroup_position_in_grid]]) {\n"
         << "  threadgroup float partial[" << threads << "];\n"
         << "  const uint base = row * " << width << "u;\n"
         << "  float squareSum = 0.0f;\n"
         << "  for (uint feature = tid; feature < " << width
         << "u; feature += " << threads << "u) {\n"
         << "    const float value = float(" << type
         << "(float(residual[base + feature]) + float(update[base + feature])));\n"
         << "    sumOutput[base + feature] = " << type << "(value);\n"
         << "    squareSum += value * value;\n"
         << "  }\n"
         << "  partial[tid] = squareSum;\n"
         << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         << "  for (uint offset = " << threads / 2
         << "u; offset > 0u; offset >>= 1u) {\n"
         << "    if (tid < offset) partial[tid] += partial[tid + offset];\n"
         << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         << "  }\n"
         << "  const float inverse = rsqrt(partial[0] / " << width << ".0f + "
         << std::scientific
         << std::setprecision(std::numeric_limits<float>::max_digits10)
         << epsilon << "f);\n"
         << "  for (uint feature = tid; feature < " << width
         << "u; feature += " << threads << "u) {\n"
         << "    normalizedOutput[base + feature] = " << type
         << "(float(" << type
         << "(float(residual[base + feature]) + float(update[base + feature])))"
         << " * inverse * float(weight[feature]));\n"
         << "  }\n"
         << "}\n";
  kernel.source = source.str();
  return kernel;
}

GeneratedKernel emitDecoderGatedMLP(
    std::size_t rows, std::size_t hiddenSize, std::size_t intermediateSize,
    std::size_t threads, const std::string &functionName,
    ElementType storageType) {
  if (hiddenSize == 0 || intermediateSize == 0) {
    throw std::invalid_argument("Decoder gated MLP configuration is invalid.");
  }
  auto kernel = beginKernel(rows * intermediateSize, threads, functionName);
  const auto *type = storageTypeName(storageType);
  std::ostringstream source;
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << functionName << "(\n"
         << "  device const " << type << " *input [[buffer(0)]],\n"
         << "  device const " << type << " *gateWeight [[buffer(1)]],\n"
         << "  device const " << type << " *upWeight [[buffer(2)]],\n"
         << "  device " << type << " *output [[buffer(3)]],\n"
         << "  uint gid [[thread_position_in_grid]]) {\n"
         << "  if (gid >= " << rows * intermediateSize << "u) return;\n"
         << "  const uint row = gid / " << intermediateSize << "u;\n"
         << "  const uint feature = gid % " << intermediateSize << "u;\n"
         << "  float gate = 0.0f;\n"
         << "  float up = 0.0f;\n";
  if (hiddenSize % 4 == 0) {
    source << "  device const " << type
           << "4 *input4 = reinterpret_cast<device const " << type
           << "4 *>(input + row * " << hiddenSize << "u);\n"
           << "  device const " << type
           << "4 *gate4 = reinterpret_cast<device const " << type
           << "4 *>(gateWeight + feature * " << hiddenSize << "u);\n"
           << "  device const " << type
           << "4 *up4 = reinterpret_cast<device const " << type
           << "4 *>(upWeight + feature * " << hiddenSize << "u);\n"
           << "  for (uint inner = 0; inner < " << hiddenSize / 4
           << "u; ++inner) {\n"
           << "    const float4 value = float4(input4[inner]);\n"
           << "    gate += dot(value, float4(gate4[inner]));\n"
           << "    up += dot(value, float4(up4[inner]));\n"
           << "  }\n";
  } else {
    source << "  for (uint inner = 0; inner < " << hiddenSize
           << "u; ++inner) {\n"
           << "    const float value = float(input[row * " << hiddenSize
           << "u + inner]);\n"
           << "    gate += value * float(gateWeight[feature * " << hiddenSize
           << "u + inner]);\n"
           << "    up += value * float(upWeight[feature * " << hiddenSize
           << "u + inner]);\n"
           << "  }\n";
  }
  source << "  gate = float(" << type << "(gate));\n"
         << "  up = float(" << type << "(up));\n"
         << "  const float activated = float(" << type
         << "(gate / (1.0f + exp(-gate))));\n"
         << "  output[gid] = " << type << "(activated * up);\n"
         << "}\n";
  kernel.source = source.str();
  return kernel;
}

GeneratedKernel emitDecoderQKVFusion(
    const planner::DecoderLLMPlan &plan, std::size_t sequenceLength,
    bool normalize, const std::string &name, ElementType storageType) {
  plan.validate();
  const auto hidden=plan.hiddenSize(), threads=plan.attention.threadsPerThreadgroup;
  const auto rows=plan.attention.batch*sequenceLength;
  const auto tiles=(hidden/2+threads-1)/threads;
  const auto *type=storageTypeName(storageType);
  GeneratedKernel kernel{{},name,rows*tiles,threads};
  std::ostringstream s;
  s.imbue(std::locale::classic());
  s << "#include <metal_stdlib>\nusing namespace metal;\n"
    << "kernel void " << name << "(device const " << type << "* x [[buffer(0)]],"
    << "device const " << type << "* qw [[buffer(1)]],device const " << type
    << "* kw [[buffer(2)]],device const " << type << "* vw [[buffer(3)]],";
  if(normalize) s << "device const " << type << "* nw [[buffer(4)]],";
  else s << "device const float* cosine [[buffer(4)]],device const float* sine [[buffer(5)]],"
            "device const int* length [[buffer(6)]],";
  const auto outputBinding=normalize?5:7;
  s << "device " << type << "* q [[buffer(" << outputBinding << ")]],device " << type
    << "* k [[buffer(" << outputBinding+1 << ")]],device " << type << "* v [[buffer("
    << outputBinding+2 << ")]],uint tid [[thread_index_in_threadgroup]],"
       "uint group [[threadgroup_position_in_grid]]){\n"
    << "const uint row=group/" << tiles << "u, feature=2u*((group%" << tiles << "u)*"
    << threads << "u+tid);\n";
  if(normalize) {
    s << "threadgroup float sums[" << threads << "], normalized[" << hidden << "];\n"
      << "float square=0.0f;for(uint i=tid;i<" << hidden << "u;i+=" << threads
      << "u){float a=float(x[row*" << hidden << "u+i]);square+=a*a;}\n"
      << "sums[tid]=square;threadgroup_barrier(mem_flags::mem_threadgroup);\n"
      << "for(uint off=" << threads/2 << "u;off;off>>=1u){if(tid<off)sums[tid]+=sums[tid+off];"
         "threadgroup_barrier(mem_flags::mem_threadgroup);}\n"
      << "float inv=rsqrt(sums[0]/" << hidden << ".0f+" << std::scientific
      << std::setprecision(9) << plan.rmsNormEpsilon << "f);\n"
      << "for(uint i=tid;i<" << hidden << "u;i+=" << threads << "u) normalized[i]=float("
      << type << "(float(x[row*" << hidden << "u+i])*inv*float(nw[i])));\n"
      << "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  }
  s << "if(feature>=" << hidden << "u)return;\nfloat qe=0,qo=0,ke=0,ko=0,ve=0,vo=0;\n"
    << "for(uint i=0;i<" << hidden << "u;++i){float a=";
  if(normalize) s << "normalized[i];\n";
  else s << "float(x[row*" << hidden << "u+i]);\n";
  s << "uint e=feature*" << hidden << "u+i,o=e+" << hidden << "u;"
       "qe+=a*float(qw[e]);qo+=a*float(qw[o]);ke+=a*float(kw[e]);ko+=a*float(kw[o]);"
       "ve+=a*float(vw[e]);vo+=a*float(vw[o]);}\n"
    << "uint flat=row*" << hidden << "u+feature;v[flat]=" << type
    << "(ve);v[flat+1]=" << type << "(vo);\n";
  if(normalize) s << "q[flat]=" << type << "(qe);q[flat+1]=" << type
                 << "(qo);k[flat]=" << type << "(ke);k[flat+1]=" << type << "(ko);\n";
  else {
    s << "qe=float(" << type << "(qe));qo=float(" << type << "(qo));ke=float("
      << type << "(ke));ko=float(" << type << "(ko));\n"
      << "uint token=row%" << sequenceLength << "u,batch=row/" << sequenceLength
      << "u,head=feature/" << plan.attention.headDimension << "u,component=feature%"
      << plan.attention.headDimension << "u;\n"
      << "uint position=uint(length[0])-" << sequenceLength << "u+token;\n"
      << "uint table=position*" << plan.attention.headDimension/2 << "u+component/2u;\n"
      << "uint dest=((batch*" << plan.attention.heads << "u+head)*" << sequenceLength
      << "u+token)*" << plan.attention.headDimension << "u+component;\n"
      << "float c=cosine[table],s=sine[table];q[dest]=" << type
      << "(qe*c-qo*s);q[dest+1]=" << type << "(qe*s+qo*c);k[dest]=" << type
      << "(ke*c-ko*s);k[dest+1]=" << type << "(ke*s+ko*c);\n";
  }
  s << "}\n";kernel.source=s.str();return kernel;
}

GeneratedKernel emitLinearResidual(
    std::size_t rows, std::size_t inputSize, std::size_t outputSize,
    std::size_t threads, const std::string &name, ElementType storageType) {
  if(!rows || !inputSize || !outputSize) throw std::invalid_argument("Invalid residual projection shape.");
  auto kernel=beginKernel(rows*outputSize,threads,name);
  const auto *type=storageTypeName(storageType);
  std::ostringstream s;
  s << "#include <metal_stdlib>\nusing namespace metal;\nkernel void " << name
    << "(device const " << type << "* x [[buffer(0)]],device const " << type
    << "* w [[buffer(1)]],device const " << type << "* residual [[buffer(2)]],device "
    << type << "* y [[buffer(3)]],uint gid [[thread_position_in_grid]]){\n"
    << "if(gid>=" << rows*outputSize << "u)return;uint row=gid/" << outputSize
    << "u,col=gid%" << outputSize << "u;float sum=0.0f;\n"
    << "for(uint i=0;i<" << inputSize << "u;++i)sum+=float(x[row*" << inputSize
    << "u+i])*float(w[col*" << inputSize << "u+i]);\n"
    << "y[gid]=" << type << "(float(" << type << "(sum))+float(residual[gid]));\n}\n";
  kernel.source=s.str();return kernel;
}

GeneratedKernel emitPagedAttentionOutputProjection(
    const planner::DecoderLLMPlan &plan, std::size_t queryLength,
    std::size_t pageSize, const std::string &name, ElementType storageType) {
  plan.validate();
  if (!queryLength || queryLength > plan.attention.capacity || !pageSize ||
      pageSize > plan.attention.capacity || name.empty()) {
    throw std::invalid_argument("Invalid paged attention/output fusion shape.");
  }
  const auto &attention = plan.attention;
  const auto hidden = plan.hiddenSize();
  const auto threads = attention.threadsPerThreadgroup;
  const auto rows = attention.batch * queryLength;
  const auto *type = storageTypeName(storageType);
  const auto scale = static_cast<float>(
      1.0 / std::sqrt(static_cast<double>(attention.headDimension)));
  GeneratedKernel kernel{{}, name, rows, threads};
  std::ostringstream source;
  source.imbue(std::locale::classic());
  source << "#include <metal_stdlib>\nusing namespace metal;\n"
         << "kernel void " << name << "(device const " << type
         << "* query [[buffer(0)]],device const " << type
         << "* keyCache [[buffer(1)]],device const " << type
         << "* valueCache [[buffer(2)]],device const int* validLength [[buffer(3)]],"
            "device const int* blockTable [[buffer(4)]],device const "
         << type << "* weight [[buffer(5)]],device " << type
         << "* output [[buffer(6)]],uint tid [[thread_index_in_threadgroup]],"
            "uint row [[threadgroup_position_in_grid]]){\n"
         << "threadgroup float reduction[" << threads << "],context[" << hidden
         << "],maximum,denominator,probability,rescale;\n"
         << "const uint valid=uint(validLength[0]);if(valid<" << queryLength
         << "u || valid>" << attention.capacity << "u)return;\n"
         << "const uint queryPosition=row%" << queryLength
         << "u,batch=row/" << queryLength << "u;\n"
         << "const uint attended=valid-" << queryLength
         << "u+queryPosition+1u;\n"
         << "for(uint head=0;head<" << attention.heads << "u;++head){\n"
         << "const uint queryBase=((batch*" << attention.heads
         << "u+head)*" << queryLength << "u+queryPosition)*"
         << attention.headDimension
         << "u;if(tid==0){maximum=-INFINITY;denominator=0.0f;}"
            "float weighted=0.0f;threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         << "for(uint position=0;position<attended;++position){uint logical=position/"
         << pageSize << "u,offset=position%" << pageSize
         << "u;int physicalValue=blockTable[logical];if(physicalValue<0)return;"
            "uint physical=uint(physicalValue);uint base=(((physical*"
         << attention.batch << "u+batch)*" << attention.heads
         << "u+head)*" << pageSize << "u+offset)*"
         << attention.headDimension << "u;float score=0.0f;"
            "for(uint inner=tid;inner<"
         << attention.headDimension << "u;inner+=" << threads
         << "u)score+=float(query[queryBase+inner])*float(keyCache[base+inner]);"
            "reduction[tid]=score;threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         << "for(uint delta=" << threads
         << "u/2u;delta;delta>>=1u){if(tid<delta)reduction[tid]+=reduction[tid+delta];"
            "threadgroup_barrier(mem_flags::mem_threadgroup);}if(tid==0){"
            "float scaled=reduction[0]*"
         << std::scientific << std::setprecision(9) << scale
         << "f;float nextMaximum=max(maximum,scaled);"
            "rescale=position==0u?0.0f:exp(maximum-nextMaximum);"
            "probability=exp(scaled-nextMaximum);"
            "denominator=denominator*rescale+probability;maximum=nextMaximum;}"
            "threadgroup_barrier(mem_flags::mem_threadgroup);if(tid<"
         << attention.headDimension
         << "u)weighted=weighted*rescale+probability*float(valueCache[base+tid]);"
            "threadgroup_barrier(mem_flags::mem_threadgroup);}\n"
         << "if(tid<" << attention.headDimension << "u)context[head*"
         << attention.headDimension
         << "u+tid]=weighted/denominator;threadgroup_barrier(mem_flags::mem_threadgroup);}\n"
         << "for(uint feature=tid;feature<" << hidden << "u;feature+=" << threads
         << "u){float sum=0.0f;for(uint inner=0;inner<" << hidden
         << "u;++inner)sum+=context[inner]*float(weight[feature*" << hidden
         << "u+inner]);output[row*" << hidden << "u+feature]=" << type
         << "(sum);}\n}\n";
  kernel.source = source.str();
  return kernel;
}

GeneratedKernel emitLMHeadPartialArgmax(
    std::size_t rows, std::size_t hidden, std::size_t vocabulary,
    std::size_t threads, const std::string &name, ElementType storageType) {
  if(!rows || !hidden || !vocabulary || !threads || (threads&(threads-1)))
    throw std::invalid_argument("Invalid LM Head argmax dimensions.");
  const auto chunks=(vocabulary+threads-1)/threads;
  const auto *type=storageTypeName(storageType);
  GeneratedKernel kernel{{},name,rows*chunks,threads};
  std::ostringstream s;
  s << "#include <metal_stdlib>\nusing namespace metal;\n"
    << "kernel void " << name << "(device const " << type << "* x [[buffer(0)]],"
    << "device const " << type << "* w [[buffer(1)]],"
    << "device float* values [[buffer(2)]],device int* indices [[buffer(3)]],"
    << "uint tid [[thread_index_in_threadgroup]],uint group [[threadgroup_position_in_grid]]){\n"
    << "const uint row=group/" << chunks << "u, token=(group%" << chunks << "u)*" << threads << "u+tid;\n"
    << "float sum=0.0f; if(token<" << vocabulary << "u){\n"
    << "for(uint k=0;k<" << hidden << "u;++k) sum+=float(x[row*" << hidden
    << "u+k])*float(w[token*" << hidden << "u+k]);\n}\n"
    << "threadgroup float v[" << threads << "]; threadgroup int ix[" << threads << "];\n"
    << "v[tid]=token<" << vocabulary << "u?sum:-INFINITY;\n"
    << "ix[tid]=token<" << vocabulary << "u?int(token):2147483647;\n"
    << "threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    << "for(uint off=" << threads/2 << "u;off;off>>=1u){\n"
    << "if(tid<off && (v[tid+off]>v[tid] || (v[tid+off]==v[tid] && ix[tid+off]<ix[tid])))"
       "{v[tid]=v[tid+off];ix[tid]=ix[tid+off];}\n"
    << "threadgroup_barrier(mem_flags::mem_threadgroup);}\n"
    << "if(tid==0){values[group]=v[0];indices[group]=ix[0];}\n}\n";
  kernel.source=s.str(); return kernel;
}

GeneratedKernel emitLMHeadFinalArgmax(
    std::size_t rows, std::size_t chunks, std::size_t threads,
    const std::string &name) {
  if(!rows || !chunks || !threads || (threads&(threads-1)))
    throw std::invalid_argument("Invalid LM Head final argmax dimensions.");
  GeneratedKernel kernel{{},name,rows,threads};
  std::ostringstream s;
  s << "#include <metal_stdlib>\nusing namespace metal;\n"
    << "kernel void " << name << "(device const float* values [[buffer(0)]],"
       "device const int* indices [[buffer(1)]],device int* output [[buffer(2)]],"
       "uint tid [[thread_index_in_threadgroup]],uint row [[threadgroup_position_in_grid]]){\n"
    << "float best=-INFINITY; int chosen=2147483647;\n"
    << "for(uint c=tid;c<" << chunks << "u;c+=" << threads << "u){uint i=row*" << chunks
    << "u+c;float v=values[i];int ix=indices[i];if(v>best || (v==best && ix<chosen))"
       "{best=v;chosen=ix;}}\n"
    << "threadgroup float v[" << threads << "];threadgroup int ix[" << threads << "];\n"
    << "v[tid]=best;ix[tid]=chosen;threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    << "for(uint off=" << threads/2 << "u;off;off>>=1u){\n"
    << "if(tid<off && (v[tid+off]>v[tid] || (v[tid+off]==v[tid] && ix[tid+off]<ix[tid])))"
       "{v[tid]=v[tid+off];ix[tid]=ix[tid+off];}\n"
    << "threadgroup_barrier(mem_flags::mem_threadgroup);}\n"
    << "if(tid==0)output[row]=ix[0];}\n";
  kernel.source=s.str();return kernel;
}

} // namespace tensor::metal
