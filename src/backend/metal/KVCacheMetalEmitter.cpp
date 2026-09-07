#include "backend/metal/KVCacheMetalEmitter.hpp"

#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace tensor::metal {
namespace {

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
  if (queryLength == 0 || queryLength > plan.capacity) {
    throw std::invalid_argument("KV cache query length is outside cache capacity.");
  }
}

} // namespace

GeneratedKernel emitKVQueryProjection(const planner::KVCachePlan &plan,
                                      std::size_t queryLength) {
  requireQueryLength(plan, queryLength);
  const auto dimension = plan.modelDimension();
  const auto workItems = plan.inputElementCount(queryLength);
  auto kernel = beginKernel(plan, workItems,
                            queryLength == 1 ? "kv_query_decode" : "kv_query_prefill");
  std::ostringstream source;
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << kernel.functionName << "(\n"
         << "  device const float *input [[buffer(0)]],\n"
         << "  device const float *weight [[buffer(1)]],\n"
         << "  device float *output [[buffer(2)]],\n"
         << "  uint gid [[thread_position_in_grid]]) {\n"
         << "  if (gid >= " << workItems << "u) return;\n"
         << "  const uint feature = gid % " << dimension << "u;\n"
         << "  const uint token = (gid / " << dimension << "u) % "
         << queryLength << "u;\n"
         << "  const uint batch = gid / " << queryLength * dimension << "u;\n"
         << "  float sum = 0.0f;\n"
         << "  for (uint inner = 0; inner < " << dimension
         << "u; ++inner) sum += input[(batch * " << queryLength
         << "u + token) * " << dimension << "u + inner] * weight[feature * "
         << dimension << "u + inner];\n"
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
  const auto workItems = plan.inputElementCount(queryLength);
  const char *function = keyCache
                             ? (queryLength == 1 ? "kv_key_append" : "kv_key_prefill")
                             : (queryLength == 1 ? "kv_value_append" : "kv_value_prefill");
  auto kernel = beginKernel(plan, workItems, function);
  std::ostringstream source;
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << kernel.functionName << "(\n"
         << "  device const float *input [[buffer(0)]],\n"
         << "  device const float *weight [[buffer(1)]],\n"
         << "  device const int *validLength [[buffer(2)]],\n"
         << "  device float *cache [[buffer(3)]],\n"
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
         << "u; ++inner) sum += input[(batch * " << queryLength
         << "u + token) * " << dimension << "u + inner] * weight[feature * "
         << dimension << "u + inner];\n"
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
                                std::size_t queryLength) {
  requireQueryLength(plan, queryLength);
  const auto dimension = plan.modelDimension();
  const auto workItems = plan.inputElementCount(queryLength);
  const auto scale = static_cast<float>(
      1.0 / std::sqrt(static_cast<double>(plan.headDimension)));
  auto kernel = beginKernel(plan, workItems,
                            queryLength == 1 ? "kv_attention_decode"
                                             : "kv_attention_prefill");
  std::ostringstream source;
  source.imbue(std::locale::classic());
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << kernel.functionName << "(\n"
         << "  device const float *query [[buffer(0)]],\n"
         << "  device const float *keyCache [[buffer(1)]],\n"
         << "  device const float *valueCache [[buffer(2)]],\n"
         << "  device const int *validLength [[buffer(3)]],\n"
         << "  device float *context [[buffer(4)]],\n"
         << "  uint gid [[thread_position_in_grid]]) {\n"
         << "  if (gid >= " << workItems << "u) return;\n"
         << "  const uint valid = uint(validLength[0]);\n"
         << "  if (valid == 0u || valid > " << plan.capacity << "u) return;\n"
         << "  const uint feature = gid % " << dimension << "u;\n"
         << "  const uint queryPosition = (gid / " << dimension << "u) % "
         << queryLength << "u;\n"
         << "  const uint batch = gid / " << queryLength * dimension << "u;\n"
         << "  const uint head = feature / " << plan.headDimension << "u;\n"
         << "  const uint component = feature % " << plan.headDimension << "u;\n"
         << "  const uint queryBase = ((batch * " << plan.heads
         << "u + head) * " << queryLength << "u + queryPosition) * "
         << plan.headDimension << "u;\n"
         << "  float maximum = -INFINITY;\n"
         << "  for (uint position = 0; position < valid; ++position) {\n"
         << "    const uint keyBase = ((batch * " << plan.heads
         << "u + head) * " << plan.capacity << "u + position) * "
         << plan.headDimension << "u;\n"
         << "    float score = 0.0f;\n"
         << "    for (uint inner = 0; inner < " << plan.headDimension
         << "u; ++inner) score += query[queryBase + inner] * keyCache[keyBase + inner];\n"
         << "    maximum = max(maximum, score * " << std::scientific
         << std::setprecision(std::numeric_limits<float>::max_digits10) << scale
         << "f);\n"
         << "  }\n"
         << "  float denominator = 0.0f;\n"
         << "  float weighted = 0.0f;\n"
         << "  for (uint position = 0; position < valid; ++position) {\n"
         << "    const uint cacheBase = ((batch * " << plan.heads
         << "u + head) * " << plan.capacity << "u + position) * "
         << plan.headDimension << "u;\n"
         << "    float score = 0.0f;\n"
         << "    for (uint inner = 0; inner < " << plan.headDimension
         << "u; ++inner) score += query[queryBase + inner] * keyCache[cacheBase + inner];\n"
         << "    const float probability = exp(score * " << std::scientific
         << std::setprecision(std::numeric_limits<float>::max_digits10) << scale
         << "f - maximum);\n"
         << "    denominator += probability;\n"
         << "    weighted += probability * valueCache[cacheBase + component];\n"
         << "  }\n"
         << "  context[(batch * " << queryLength << "u + queryPosition) * "
         << dimension << "u + feature] = weighted / denominator;\n"
         << "}\n";
  kernel.source = source.str();
  return kernel;
}

GeneratedKernel emitKVOutputProjection(const planner::KVCachePlan &plan,
                                       std::size_t queryLength) {
  requireQueryLength(plan, queryLength);
  const auto dimension = plan.modelDimension();
  const auto workItems = plan.inputElementCount(queryLength);
  auto kernel = beginKernel(plan, workItems,
                            queryLength == 1 ? "kv_output_decode" : "kv_output_prefill");
  std::ostringstream source;
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << kernel.functionName << "(\n"
         << "  device const float *input [[buffer(0)]],\n"
         << "  device const float *weight [[buffer(1)]],\n"
         << "  device float *output [[buffer(2)]],\n"
         << "  uint gid [[thread_position_in_grid]]) {\n"
         << "  if (gid >= " << workItems << "u) return;\n"
         << "  const uint feature = gid % " << dimension << "u;\n"
         << "  const uint row = gid / " << dimension << "u;\n"
         << "  float sum = 0.0f;\n"
         << "  for (uint inner = 0; inner < " << dimension
         << "u; ++inner) sum += input[row * " << dimension
         << "u + inner] * weight[feature * " << dimension << "u + inner];\n"
         << "  output[gid] = sum;\n"
         << "}\n";
  kernel.source = source.str();
  return kernel;
}

} // namespace tensor::metal
