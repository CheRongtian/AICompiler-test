#include "backend/metal/DecoderLLMMetalEmitter.hpp"

#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace tensor::metal {
namespace {

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
    const std::string &functionName) {
  plan.validate();
  const auto hidden = plan.hiddenSize();
  const auto workItems = plan.hiddenElementCount(sequenceLength);
  auto kernel = beginKernel(workItems, plan.attention.threadsPerThreadgroup,
                            functionName);
  std::ostringstream source;
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << functionName << "(\n"
         << "  device const int *tokenIds [[buffer(0)]],\n"
         << "  device const float *embedding [[buffer(1)]],\n"
         << "  device float *output [[buffer(2)]],\n"
         << "  uint gid [[thread_position_in_grid]]) {\n"
         << "  if (gid >= " << workItems << "u) return;\n"
         << "  const uint row = gid / " << hidden << "u;\n"
         << "  const uint feature = gid % " << hidden << "u;\n"
         << "  const int token = tokenIds[row];\n"
         << "  if (token < 0 || uint(token) >= " << plan.vocabularySize << "u) return;\n"
         << "  output[gid] = embedding[uint(token) * " << hidden
         << "u + feature];\n"
         << "}\n";
  kernel.source = source.str();
  return kernel;
}

GeneratedKernel emitDecoderRMSNorm(
    std::size_t rows, std::size_t width, float epsilon,
    std::size_t threads, const std::string &functionName) {
  auto kernel = beginKernel(rows, threads, functionName);
  std::ostringstream source;
  source.imbue(std::locale::classic());
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << functionName << "(\n"
         << "  device const float *input [[buffer(0)]],\n"
         << "  device const float *weight [[buffer(1)]],\n"
         << "  device float *output [[buffer(2)]],\n"
         << "  uint row [[thread_position_in_grid]]) {\n"
         << "  if (row >= " << rows << "u) return;\n"
         << "  const uint base = row * " << width << "u;\n"
         << "  float squareSum = 0.0f;\n"
         << "  for (uint i = 0; i < " << width
         << "u; ++i) squareSum += input[base + i] * input[base + i];\n"
         << "  const float inverse = rsqrt(squareSum / " << width << ".0f + "
         << std::setprecision(std::numeric_limits<float>::max_digits10) << epsilon
         << "f);\n"
         << "  for (uint i = 0; i < " << width
         << "u; ++i) output[base + i] = input[base + i] * inverse * weight[i];\n"
         << "}\n";
  kernel.source = source.str();
  return kernel;
}

GeneratedKernel emitDecoderRoPE(
    const planner::DecoderLLMPlan &plan, std::size_t sequenceLength,
    const std::string &functionName) {
  plan.validate();
  const auto pairs = plan.attention.headDimension / 2;
  const auto workItems = plan.attention.batch * plan.attention.heads *
                         sequenceLength * pairs;
  auto kernel = beginKernel(workItems, plan.attention.threadsPerThreadgroup,
                            functionName);
  const auto hidden = plan.hiddenSize();
  std::ostringstream source;
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << functionName << "(\n"
         << "  device const float *input [[buffer(0)]],\n"
         << "  device const float *cosine [[buffer(1)]],\n"
         << "  device const float *sine [[buffer(2)]],\n"
         << "  device const int *validLength [[buffer(3)]],\n"
         << "  device float *output [[buffer(4)]],\n"
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
         << "  const float even = input[inputBase];\n"
         << "  const float odd = input[inputBase + 1u];\n"
         << "  output[outputBase] = even * c - odd * s;\n"
         << "  output[outputBase + 1u] = even * s + odd * c;\n"
         << "}\n";
  kernel.source = source.str();
  return kernel;
}

GeneratedKernel emitDecoderCacheAppend(
    const planner::DecoderLLMPlan &plan, std::size_t sequenceLength,
    bool inputIsHeadMajor, const std::string &functionName) {
  plan.validate();
  const auto workItems = plan.hiddenElementCount(sequenceLength);
  const auto hidden = plan.hiddenSize();
  auto kernel = beginKernel(workItems, plan.attention.threadsPerThreadgroup,
                            functionName);
  std::ostringstream source;
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << functionName << "(\n"
         << "  device const float *input [[buffer(0)]],\n"
         << "  device const int *validLength [[buffer(1)]],\n"
         << "  device float *cache [[buffer(2)]],\n"
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
    const std::string &functionName) {
  plan.validate();
  if (pageSize == 0 || pageSize > plan.attention.capacity) {
    throw std::invalid_argument("Decoder paged-cache page size is invalid.");
  }
  const auto workItems = plan.hiddenElementCount(sequenceLength);
  const auto hidden = plan.hiddenSize();
  auto kernel = beginKernel(workItems, plan.attention.threadsPerThreadgroup,
                            functionName);
  std::ostringstream source;
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << functionName << "(\n"
         << "  device const float *input [[buffer(0)]],\n"
         << "  device const int *validLength [[buffer(1)]],\n"
         << "  device const int *blockTable [[buffer(2)]],\n"
         << "  device float *cache [[buffer(3)]],\n"
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
    const std::string &functionName) {
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
  std::ostringstream source;
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << functionName << "(\n"
         << "  device const float *input [[buffer(0)]],\n"
         << "  device const float *cosine [[buffer(1)]],\n"
         << "  device const float *sine [[buffer(2)]],\n"
         << "  device const int *validLengths [[buffer(3)]],\n"
         << "  device const int *activeCount [[buffer(4)]],\n"
         << "  device float *output [[buffer(5)]],\n"
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
         << "  const float even = input[inputBase];\n"
         << "  const float odd = input[inputBase + 1u];\n"
         << "  output[outputBase] = even * c - odd * s;\n"
         << "  output[outputBase + 1u] = even * s + odd * c;\n"
         << "}\n";
  kernel.source = source.str();
  return kernel;
}

GeneratedKernel emitServingPagedCacheAppend(
    const planner::DecoderLLMPlan &plan, std::size_t maximumBatchSize,
    bool inputIsHeadMajor, std::size_t pageSize,
    const std::string &functionName) {
  plan.validate();
  if (maximumBatchSize == 0 || pageSize == 0 ||
      pageSize > plan.attention.capacity) {
    throw std::invalid_argument("Serving paged-cache configuration is invalid.");
  }
  const auto hidden = plan.hiddenSize();
  const auto maximumPages =
      (plan.attention.capacity + pageSize - 1) / pageSize;
  const auto workItems = maximumBatchSize * hidden;
  auto kernel = beginKernel(workItems, plan.attention.threadsPerThreadgroup,
                            functionName);
  std::ostringstream source;
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << functionName << "(\n"
         << "  device const float *input [[buffer(0)]],\n"
         << "  device const int *validLengths [[buffer(1)]],\n"
         << "  device const int *blockTables [[buffer(2)]],\n"
         << "  device const int *activeCount [[buffer(3)]],\n"
         << "  device float *cache [[buffer(4)]],\n"
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
                               const std::string &functionName) {
  auto kernel = beginKernel(elementCount, threads, functionName);
  std::ostringstream source;
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << functionName << "(\n"
         << "  device const float *left [[buffer(0)]],\n"
         << "  device const float *right [[buffer(1)]],\n"
         << "  device float *output [[buffer(2)]],\n"
         << "  uint gid [[thread_position_in_grid]]) {\n"
         << "  if (gid >= " << elementCount << "u) return;\n"
         << "  output[gid] = left[gid] + right[gid];\n"
         << "}\n";
  kernel.source = source.str();
  return kernel;
}

GeneratedKernel emitDecoderSiLUMul(std::size_t elementCount,
                                   std::size_t threads,
                                   const std::string &functionName) {
  auto kernel = beginKernel(elementCount, threads, functionName);
  std::ostringstream source;
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << functionName << "(\n"
         << "  device const float *gate [[buffer(0)]],\n"
         << "  device const float *up [[buffer(1)]],\n"
         << "  device float *output [[buffer(2)]],\n"
         << "  uint gid [[thread_position_in_grid]]) {\n"
         << "  if (gid >= " << elementCount << "u) return;\n"
         << "  const float x = gate[gid];\n"
         << "  output[gid] = (x / (1.0f + exp(-x))) * up[gid];\n"
         << "}\n";
  kernel.source = source.str();
  return kernel;
}

} // namespace tensor::metal
