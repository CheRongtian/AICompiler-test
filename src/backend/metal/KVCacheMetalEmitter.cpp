#include "backend/metal/KVCacheMetalEmitter.hpp"

#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace tensor::metal {
namespace {

const char *storageTypeName(DType type) {
  if (type == DType::Float16) return "half";
  if (type == DType::Float32) return "float";
  if (type == DType::BFloat16) return "bfloat";
  throw std::invalid_argument(
      "KV cache Metal emitters support fp16, bf16, or fp32 storage.");
}

GeneratedKernel beginKernel(const planner::KVCachePlan &plan,
                            std::size_t workItems,
                            const char *functionName) {
  plan.validate();
  if (workItems == 0) throw std::invalid_argument("KV cache kernel has no work items.");
  GeneratedKernel kernel;
  kernel.functionName = functionName;
  kernel.threadsPerThreadgroup = plan.threadsPerThreadgroup;
  kernel.threadgroupCount =
      (workItems + plan.threadsPerThreadgroup - 1) / plan.threadsPerThreadgroup;
  return kernel;
}

void requireQueryLength(const planner::KVCachePlan &plan,
                        std::size_t queryLength) {
  plan.validate();
  if (queryLength == 0 || queryLength > plan.capacity) {
    throw std::invalid_argument("KV cache query length is outside cache capacity.");
  }
}

} // namespace

GeneratedKernel emitKVQueryProjection(const planner::KVCachePlan &plan,
                                      std::size_t queryLength) {
  requireQueryLength(plan, queryLength);
  const auto dimension = plan.modelDimension();
  const auto *type = storageTypeName(plan.dtype);
  const auto workItems = plan.inputElementCount(queryLength);
  auto kernel = beginKernel(plan, workItems,
                            queryLength == 1 ? "kv_query_decode" : "kv_query_prefill");
  std::ostringstream source;
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << kernel.functionName << "(\n"
         << "  device const " << type << " *input [[buffer(0)]],\n"
         << "  device const " << type << " *weight [[buffer(1)]],\n"
         << "  device " << type << " *output [[buffer(2)]],\n"
         << "  uint gid [[thread_position_in_grid]]) {\n"
         << "  if (gid >= " << workItems << "u) return;\n"
         << "  const uint feature = gid % " << dimension << "u;\n"
         << "  const uint token = (gid / " << dimension << "u) % "
         << queryLength << "u;\n"
         << "  const uint batch = gid / " << queryLength * dimension << "u;\n"
         << "  float sum = 0.0f;\n"
         << "  for (uint inner = 0; inner < " << dimension
         << "u; ++inner) sum += float(input[(batch * " << queryLength
         << "u + token) * " << dimension << "u + inner]) * float(weight[feature * "
         << dimension << "u + inner]);\n"
         << "  const uint head = feature / " << plan.headDimension << "u;\n"
         << "  const uint component = feature % " << plan.headDimension << "u;\n"
         << "  output[((batch * " << plan.heads << "u + head) * " << queryLength
         << "u + token) * " << plan.headDimension << "u + component] = sum;\n"
         << "}\n";
  kernel.source = source.str();
  return kernel;
}

GeneratedKernel emitKVCacheProjection(const planner::KVCachePlan &plan,
                                      std::size_t queryLength, bool keyCache) {
  requireQueryLength(plan, queryLength);
  const auto dimension = plan.modelDimension();
  const auto *type = storageTypeName(plan.dtype);
  const auto workItems = plan.inputElementCount(queryLength);
  const char *function = keyCache
                             ? (queryLength == 1 ? "kv_key_append" : "kv_key_prefill")
                             : (queryLength == 1 ? "kv_value_append" : "kv_value_prefill");
  auto kernel = beginKernel(plan, workItems, function);
  std::ostringstream source;
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << kernel.functionName << "(\n"
         << "  device const " << type << " *input [[buffer(0)]],\n"
         << "  device const " << type << " *weight [[buffer(1)]],\n"
         << "  device const int *validLength [[buffer(2)]],\n"
         << "  device " << type << " *cache [[buffer(3)]],\n"
         << "  uint gid [[thread_position_in_grid]]) {\n"
         << "  if (gid >= " << workItems << "u) return;\n"
         << "  const uint valid = uint(validLength[0]);\n"
         << "  if (valid < " << queryLength << "u || valid > " << plan.capacity
         << "u) return;\n"
         << "  const uint feature = gid % " << dimension << "u;\n"
         << "  const uint token = (gid / " << dimension << "u) % "
         << queryLength << "u;\n"
         << "  const uint batch = gid / " << queryLength * dimension << "u;\n"
         << "  float sum = 0.0f;\n"
         << "  for (uint inner = 0; inner < " << dimension
         << "u; ++inner) sum += float(input[(batch * " << queryLength
         << "u + token) * " << dimension << "u + inner]) * float(weight[feature * "
         << dimension << "u + inner]);\n"
         << "  const uint head = feature / " << plan.headDimension << "u;\n"
         << "  const uint component = feature % " << plan.headDimension << "u;\n"
         << "  const uint position = valid - " << queryLength << "u + token;\n"
         << "  cache[((batch * " << plan.heads << "u + head) * " << plan.capacity
         << "u + position) * " << plan.headDimension << "u + component] = sum;\n"
         << "}\n";
  kernel.source = source.str();
  return kernel;
}

GeneratedKernel emitKVAttention(const planner::KVCachePlan &plan,
                                std::size_t queryLength, bool causalPrefill) {
  requireQueryLength(plan, queryLength);
  const auto dimension = plan.modelDimension();
  const auto *type = storageTypeName(plan.dtype);
  const auto scale = static_cast<float>(
      1.0 / std::sqrt(static_cast<double>(plan.headDimension)));
  GeneratedKernel kernel;
  kernel.functionName =
      queryLength == 1 ? "kv_attention_decode" : "kv_attention_prefill";
  kernel.threadsPerThreadgroup = plan.threadsPerThreadgroup;
  kernel.threadgroupCount = plan.batch * plan.heads * queryLength;
  std::ostringstream source;
  source.imbue(std::locale::classic());
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << kernel.functionName << "(\n"
         << "  device const " << type << " *query [[buffer(0)]],\n"
         << "  device const " << type << " *keyCache [[buffer(1)]],\n"
         << "  device const " << type << " *valueCache [[buffer(2)]],\n"
         << "  device const int *validLength [[buffer(3)]],\n"
         << "  device " << type << " *context [[buffer(4)]],\n"
         << "  uint tid [[thread_index_in_threadgroup]],\n"
         << "  uint group [[threadgroup_position_in_grid]]) {\n"
         << "  threadgroup float reduction[" << plan.threadsPerThreadgroup
         << "];\n"
         << "  threadgroup float maximum;\n"
         << "  threadgroup float denominator;\n"
         << "  threadgroup float probability;\n"
         << "  threadgroup float rescale;\n"
         << "  const uint valid = uint(validLength[0]);\n"
         << "  if (valid < " << queryLength << "u || valid > "
         << plan.capacity << "u) return;\n"
         << "  const uint queryPosition = group % " << queryLength << "u;\n"
         << "  const uint head = (group / " << queryLength << "u) % "
         << plan.heads << "u;\n"
         << "  const uint batch = group / " << queryLength * plan.heads
         << "u;\n"
         << "  const uint queryBase = ((batch * " << plan.heads
         << "u + head) * " << queryLength << "u + queryPosition) * "
         << plan.headDimension << "u;\n"
         << "  const uint attended = "
         << (causalPrefill ? "valid - " + std::to_string(queryLength) +
                                 "u + queryPosition + 1u"
                           : "valid")
         << ";\n"
         << "  if (tid == 0u) { maximum = -INFINITY; denominator = 0.0f; }\n"
         << "  float weighted = 0.0f;\n"
         << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         << "  for (uint position = 0; position < attended; ++position) {\n"
         << "    const uint cacheBase = ((batch * " << plan.heads
         << "u + head) * " << plan.capacity << "u + position) * "
         << plan.headDimension << "u;\n"
         << "    float score = 0.0f;\n"
         << "    for (uint inner = tid; inner < " << plan.headDimension
         << "u; inner += " << plan.threadsPerThreadgroup
         << "u) score += float(query[queryBase + inner]) * float(keyCache[cacheBase + inner]);\n"
         << "    reduction[tid] = score;\n"
         << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         << "    for (uint offset = " << plan.threadsPerThreadgroup / 2
         << "u; offset > 0u; offset >>= 1u) {\n"
         << "      if (tid < offset) reduction[tid] += reduction[tid + offset];\n"
         << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         << "    }\n"
         << "    if (tid == 0u) {\n"
         << "      const float scaled = reduction[0] * "
         << std::scientific
         << std::setprecision(std::numeric_limits<float>::max_digits10) << scale
         << "f;\n"
         << "      const float nextMaximum = max(maximum, scaled);\n"
         << "      rescale = position == 0u ? 0.0f : exp(maximum - nextMaximum);\n"
         << "      probability = exp(scaled - nextMaximum);\n"
         << "      denominator = denominator * rescale + probability;\n"
         << "      maximum = nextMaximum;\n"
         << "    }\n"
         << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         << "    if (tid < " << plan.headDimension
         << "u) weighted = weighted * rescale + probability * float(valueCache[cacheBase + tid]);\n"
         << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         << "  }\n"
         << "  if (tid < " << plan.headDimension << "u) {\n"
         << "    const uint feature = head * " << plan.headDimension
         << "u + tid;\n"
         << "    context[(batch * " << queryLength << "u + queryPosition) * "
         << dimension << "u + feature] = " << type
         << "(weighted / denominator);\n"
         << "  }\n"
         << "}\n";
  kernel.source = source.str();
  return kernel;
}

GeneratedKernel emitPagedKVAttention(const planner::KVCachePlan &plan,
                                     std::size_t queryLength,
                                     std::size_t pageSize,
                                     bool causalPrefill) {
  requireQueryLength(plan, queryLength);
  if (pageSize == 0 || pageSize > plan.capacity) {
    throw std::invalid_argument("Paged KV attention page size is invalid.");
  }
  const auto dimension = plan.modelDimension();
  const auto *type = storageTypeName(plan.dtype);
  const auto scale = static_cast<float>(
      1.0 / std::sqrt(static_cast<double>(plan.headDimension)));
  GeneratedKernel kernel;
  kernel.functionName = queryLength == 1 ? "paged_kv_attention_decode"
                                          : "paged_kv_attention_prefill";
  kernel.threadsPerThreadgroup = plan.threadsPerThreadgroup;
  kernel.threadgroupCount = plan.batch * plan.heads * queryLength;
  std::ostringstream source;
  source.imbue(std::locale::classic());
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << kernel.functionName << "(\n"
         << "  device const " << type << " *query [[buffer(0)]],\n"
         << "  device const " << type << " *keyCache [[buffer(1)]],\n"
         << "  device const " << type << " *valueCache [[buffer(2)]],\n"
         << "  device const int *validLength [[buffer(3)]],\n"
         << "  device const int *blockTable [[buffer(4)]],\n"
         << "  device " << type << " *context [[buffer(5)]],\n"
         << "  uint tid [[thread_index_in_threadgroup]],\n"
         << "  uint group [[threadgroup_position_in_grid]]) {\n"
         << "  threadgroup float reduction[" << plan.threadsPerThreadgroup
         << "];\n"
         << "  threadgroup float maximum;\n"
         << "  threadgroup float denominator;\n"
         << "  threadgroup float probability;\n"
         << "  threadgroup float rescale;\n"
         << "  const uint valid = uint(validLength[0]);\n"
         << "  if (valid < " << queryLength << "u || valid > "
         << plan.capacity << "u) return;\n"
         << "  const uint queryPosition = group % " << queryLength << "u;\n"
         << "  const uint head = (group / " << queryLength << "u) % "
         << plan.heads << "u;\n"
         << "  const uint batch = group / " << queryLength * plan.heads
         << "u;\n"
         << "  const uint queryBase = ((batch * " << plan.heads
         << "u + head) * " << queryLength << "u + queryPosition) * "
         << plan.headDimension << "u;\n"
         << "  const uint attended = "
         << (causalPrefill ? "valid - " + std::to_string(queryLength) +
                                 "u + queryPosition + 1u"
                           : "valid")
         << ";\n"
         << "  if (tid == 0u) { maximum = -INFINITY; denominator = 0.0f; }\n"
         << "  float weighted = 0.0f;\n"
         << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         << "  for (uint position = 0; position < attended; ++position) {\n"
         << "    const uint logicalPage = position / " << pageSize << "u;\n"
         << "    const uint pageOffset = position % " << pageSize << "u;\n"
         << "    const int physicalPageValue = blockTable[logicalPage];\n"
         << "    if (physicalPageValue < 0) return;\n"
         << "    const uint physicalPage = uint(physicalPageValue);\n"
         << "    const uint cacheBase = (((physicalPage * " << plan.batch
         << "u + batch) * " << plan.heads << "u + head) * " << pageSize
         << "u + pageOffset) * " << plan.headDimension << "u;\n"
         << "    float score = 0.0f;\n"
         << "    for (uint inner = tid; inner < " << plan.headDimension
         << "u; inner += " << plan.threadsPerThreadgroup
         << "u) score += float(query[queryBase + inner]) * float(keyCache[cacheBase + inner]);\n"
         << "    reduction[tid] = score;\n"
         << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         << "    for (uint offset = " << plan.threadsPerThreadgroup / 2
         << "u; offset > 0u; offset >>= 1u) {\n"
         << "      if (tid < offset) reduction[tid] += reduction[tid + offset];\n"
         << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         << "    }\n"
         << "    if (tid == 0u) {\n"
         << "      const float scaled = reduction[0] * "
         << std::scientific
         << std::setprecision(std::numeric_limits<float>::max_digits10) << scale
         << "f;\n"
         << "      const float nextMaximum = max(maximum, scaled);\n"
         << "      rescale = position == 0u ? 0.0f : exp(maximum - nextMaximum);\n"
         << "      probability = exp(scaled - nextMaximum);\n"
         << "      denominator = denominator * rescale + probability;\n"
         << "      maximum = nextMaximum;\n"
         << "    }\n"
         << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         << "    if (tid < " << plan.headDimension
         << "u) weighted = weighted * rescale + probability * float(valueCache[cacheBase + tid]);\n"
         << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         << "  }\n"
         << "  if (tid < " << plan.headDimension << "u) {\n"
         << "    const uint feature = head * " << plan.headDimension
         << "u + tid;\n"
         << "    context[(batch * " << queryLength << "u + queryPosition) * "
         << dimension << "u + feature] = " << type
         << "(weighted / denominator);\n"
         << "  }\n"
         << "}\n";
  kernel.source = source.str();
  return kernel;
}

GeneratedKernel emitServingPagedKVAttention(
    const planner::KVCachePlan &plan, std::size_t maximumBatchSize,
    std::size_t pageSize, const std::string &functionName) {
  plan.validate();
  if (maximumBatchSize == 0 || pageSize == 0 || pageSize > plan.capacity ||
      functionName.empty()) {
    throw std::invalid_argument("Serving paged-attention configuration is invalid.");
  }
  const auto dimension = plan.modelDimension();
  const auto *type = storageTypeName(plan.dtype);
  const auto maximumPages = (plan.capacity + pageSize - 1) / pageSize;
  const auto scale = static_cast<float>(
      1.0 / std::sqrt(static_cast<double>(plan.headDimension)));
  GeneratedKernel kernel;
  kernel.functionName = functionName;
  kernel.threadsPerThreadgroup = plan.threadsPerThreadgroup;
  kernel.threadgroupCount = maximumBatchSize * plan.heads;
  std::ostringstream source;
  source.imbue(std::locale::classic());
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << functionName << "(\n"
         << "  device const " << type << " *query [[buffer(0)]],\n"
         << "  device const " << type << " *keyCache [[buffer(1)]],\n"
         << "  device const " << type << " *valueCache [[buffer(2)]],\n"
         << "  device const int *validLengths [[buffer(3)]],\n"
         << "  device const int *blockTables [[buffer(4)]],\n"
         << "  device const int *activeCount [[buffer(5)]],\n"
         << "  device " << type << " *context [[buffer(6)]],\n"
         << "  uint tid [[thread_index_in_threadgroup]],\n"
         << "  uint group [[threadgroup_position_in_grid]]) {\n"
         << "  threadgroup float reduction[" << plan.threadsPerThreadgroup
         << "];\n"
         << "  threadgroup float maximum;\n"
         << "  threadgroup float denominator;\n"
         << "  threadgroup float probability;\n"
         << "  threadgroup float rescale;\n"
         << "  const uint request = group / " << plan.heads << "u;\n"
         << "  if (request >= uint(activeCount[0])) return;\n"
         << "  const uint head = group % " << plan.heads << "u;\n"
         << "  const int validValue = validLengths[request];\n"
         << "  if (validValue <= 0 || uint(validValue) > " << plan.capacity
         << "u) return;\n"
         << "  const uint valid = uint(validValue);\n"
         << "  const uint queryBase = (request * " << plan.heads
         << "u + head) * " << plan.headDimension << "u;\n"
         << "  if (tid == 0u) { maximum = -INFINITY; denominator = 0.0f; }\n"
         << "  float weighted = 0.0f;\n"
         << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         << "  for (uint position = 0; position < valid; ++position) {\n"
         << "    const uint logicalPage = position / " << pageSize << "u;\n"
         << "    const uint pageOffset = position % " << pageSize << "u;\n"
         << "    const int physicalPageValue = blockTables[request * "
         << maximumPages << "u + logicalPage];\n"
         << "    if (physicalPageValue < 0) return;\n"
         << "    const uint physicalPage = uint(physicalPageValue);\n"
         << "    const uint cacheBase = ((physicalPage * " << plan.heads
         << "u + head) * " << pageSize << "u + pageOffset) * "
         << plan.headDimension << "u;\n"
         << "    float score = 0.0f;\n"
         << "    for (uint inner = tid; inner < " << plan.headDimension
         << "u; inner += " << plan.threadsPerThreadgroup
         << "u) score += float(query[queryBase + inner]) * float(keyCache[cacheBase + inner]);\n"
         << "    reduction[tid] = score;\n"
         << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         << "    for (uint offset = " << plan.threadsPerThreadgroup / 2
         << "u; offset > 0u; offset >>= 1u) {\n"
         << "      if (tid < offset) reduction[tid] += reduction[tid + offset];\n"
         << "      threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         << "    }\n"
         << "    if (tid == 0u) {\n"
         << "      const float scaled = reduction[0] * "
         << std::scientific
         << std::setprecision(std::numeric_limits<float>::max_digits10) << scale
         << "f;\n"
         << "      const float nextMaximum = max(maximum, scaled);\n"
         << "      rescale = position == 0u ? 0.0f : exp(maximum - nextMaximum);\n"
         << "      probability = exp(scaled - nextMaximum);\n"
         << "      denominator = denominator * rescale + probability;\n"
         << "      maximum = nextMaximum;\n"
         << "    }\n"
         << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         << "    if (tid < " << plan.headDimension
         << "u) weighted = weighted * rescale + probability * float(valueCache[cacheBase + tid]);\n"
         << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         << "  }\n"
         << "  if (tid < " << plan.headDimension << "u) {\n"
         << "    const uint feature = head * " << plan.headDimension
         << "u + tid;\n"
         << "    context[request * " << dimension << "u + feature] = "
         << type << "(weighted / denominator);\n"
         << "  }\n"
         << "}\n";
  kernel.source = source.str();
  return kernel;
}

GeneratedKernel emitKVOutputProjection(const planner::KVCachePlan &plan,
                                       std::size_t queryLength) {
  requireQueryLength(plan, queryLength);
  const auto dimension = plan.modelDimension();
  const auto *type = storageTypeName(plan.dtype);
  const auto workItems = plan.inputElementCount(queryLength);
  auto kernel = beginKernel(plan, workItems,
                            queryLength == 1 ? "kv_output_decode" : "kv_output_prefill");
  std::ostringstream source;
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << kernel.functionName << "(\n"
         << "  device const " << type << " *input [[buffer(0)]],\n"
         << "  device const " << type << " *weight [[buffer(1)]],\n"
         << "  device " << type << " *output [[buffer(2)]],\n"
         << "  uint gid [[thread_position_in_grid]]) {\n"
         << "  if (gid >= " << workItems << "u) return;\n"
         << "  const uint feature = gid % " << dimension << "u;\n"
         << "  const uint row = gid / " << dimension << "u;\n"
         << "  float sum = 0.0f;\n"
         << "  for (uint inner = 0; inner < " << dimension
         << "u; ++inner) sum += float(input[row * " << dimension
         << "u + inner]) * float(weight[feature * " << dimension << "u + inner]);\n"
         << "  output[gid] = sum;\n"
         << "}\n";
  kernel.source = source.str();
  return kernel;
}

} // namespace tensor::metal
