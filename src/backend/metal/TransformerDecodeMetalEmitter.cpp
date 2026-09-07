#include "backend/metal/TransformerDecodeMetalEmitter.hpp"

#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace tensor::metal {
namespace {

GeneratedKernel beginKernel(std::size_t workItems, std::size_t threads,
                            const std::string &functionName) {
  if (workItems == 0 || threads == 0 || functionName.empty()) {
    throw std::invalid_argument("Transformer decode kernel configuration is invalid.");
  }
  GeneratedKernel kernel;
  kernel.functionName = functionName;
  kernel.threadsPerThreadgroup = threads;
  kernel.threadgroupCount = (workItems + threads - 1) / threads;
  return kernel;
}

} // namespace

GeneratedKernel emitTokenEmbedding(
    const planner::TransformerDecodePlan &plan, std::size_t sequenceLength,
    const std::string &functionName) {
  plan.validate();
  if (sequenceLength == 0 || sequenceLength > plan.selfAttention.capacity) {
    throw std::invalid_argument("Transformer embedding sequence length is invalid.");
  }
  const auto dimension = plan.modelDimension();
  const auto workItems = plan.hiddenElementCount(sequenceLength);
  auto kernel = beginKernel(workItems, plan.selfAttention.threadsPerThreadgroup,
                            functionName);
  std::ostringstream source;
  source.imbue(std::locale::classic());
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << functionName << "(\n"
         << "  device const int *tokenIds [[buffer(0)]],\n"
         << "  device const float *embedding [[buffer(1)]],\n"
         << "  device const float *positionTable [[buffer(2)]],\n"
         << "  device const int *validLength [[buffer(3)]],\n"
         << "  device float *output [[buffer(4)]],\n"
         << "  uint gid [[thread_position_in_grid]]) {\n"
         << "  if (gid >= " << workItems << "u) return;\n"
         << "  const uint feature = gid % " << dimension << "u;\n"
         << "  const uint row = gid / " << dimension << "u;\n"
         << "  const uint tokenPosition = row % " << sequenceLength << "u;\n"
         << "  const uint start = uint(validLength[0]) - " << sequenceLength << "u;\n"
         << "  const int token = tokenIds[row];\n"
         << "  if (token < 0 || uint(token) >= " << plan.vocabularySize << "u) return;\n"
         << "  output[gid] = embedding[uint(token) * " << dimension
         << "u + feature] * " << std::setprecision(std::numeric_limits<float>::max_digits10)
         << std::sqrt(static_cast<float>(dimension))
         << "f + positionTable[(start + tokenPosition) * " << dimension
         << "u + feature];\n"
         << "}\n";
  kernel.source = source.str();
  return kernel;
}

GeneratedKernel emitBiasedLinear(
    std::size_t rows, std::size_t inputDimension,
    std::size_t outputDimension, bool relu,
    std::size_t threadsPerThreadgroup, const std::string &functionName) {
  const auto workItems = rows * outputDimension;
  auto kernel = beginKernel(workItems, threadsPerThreadgroup, functionName);
  std::ostringstream source;
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << functionName << "(\n"
         << "  device const float *input [[buffer(0)]],\n"
         << "  device const float *weight [[buffer(1)]],\n"
         << "  device const float *bias [[buffer(2)]],\n"
         << "  device float *output [[buffer(3)]],\n"
         << "  uint gid [[thread_position_in_grid]]) {\n"
         << "  if (gid >= " << workItems << "u) return;\n"
         << "  const uint feature = gid % " << outputDimension << "u;\n"
         << "  const uint row = gid / " << outputDimension << "u;\n"
         << "  float sum = bias[feature];\n"
         << "  for (uint inner = 0; inner < " << inputDimension
         << "u; ++inner) sum += input[row * " << inputDimension
         << "u + inner] * weight[feature * " << inputDimension << "u + inner];\n";
  if (relu) source << "  sum = max(sum, 0.0f);\n";
  source << "  output[gid] = sum;\n}\n";
  kernel.source = source.str();
  return kernel;
}

GeneratedKernel emitResidualLayerNorm(
    std::size_t rows, std::size_t width, float epsilon,
    std::size_t threadsPerThreadgroup, const std::string &functionName) {
  if (!std::isfinite(epsilon) || epsilon <= 0.0f) {
    throw std::invalid_argument("LayerNorm epsilon must be positive.");
  }
  auto kernel = beginKernel(rows, threadsPerThreadgroup, functionName);
  std::ostringstream source;
  source.imbue(std::locale::classic());
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << functionName << "(\n"
         << "  device const float *residual [[buffer(0)]],\n"
         << "  device const float *update [[buffer(1)]],\n"
         << "  device const float *gamma [[buffer(2)]],\n"
         << "  device const float *beta [[buffer(3)]],\n"
         << "  device float *output [[buffer(4)]],\n"
         << "  uint row [[thread_position_in_grid]]) {\n"
         << "  if (row >= " << rows << "u) return;\n"
         << "  const uint base = row * " << width << "u;\n"
         << "  float mean = 0.0f;\n"
         << "  for (uint i = 0; i < " << width
         << "u; ++i) mean += residual[base + i] + update[base + i];\n"
         << "  mean /= " << width << ".0f;\n"
         << "  float variance = 0.0f;\n"
         << "  for (uint i = 0; i < " << width << "u; ++i) {\n"
         << "    const float centered = residual[base + i] + update[base + i] - mean;\n"
         << "    variance += centered * centered;\n"
         << "  }\n"
         << "  const float inverse = rsqrt(variance / " << width << ".0f + "
         << std::setprecision(std::numeric_limits<float>::max_digits10) << epsilon
         << "f);\n"
         << "  for (uint i = 0; i < " << width << "u; ++i) {\n"
         << "    const float value = residual[base + i] + update[base + i];\n"
         << "    output[base + i] = (value - mean) * inverse * gamma[i] + beta[i];\n"
         << "  }\n"
         << "}\n";
  kernel.source = source.str();
  return kernel;
}

GeneratedKernel emitCrossAttention(
    const planner::TransformerDecodePlan &plan, std::size_t queryLength,
    const std::string &functionName) {
  plan.validate();
  const auto dimension = plan.modelDimension();
  const auto workItems = plan.hiddenElementCount(queryLength);
  const auto scale = static_cast<float>(
      1.0 / std::sqrt(static_cast<double>(plan.selfAttention.headDimension)));
  auto kernel = beginKernel(workItems, plan.selfAttention.threadsPerThreadgroup,
                            functionName);
  std::ostringstream source;
  source.imbue(std::locale::classic());
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << functionName << "(\n"
         << "  device const float *query [[buffer(0)]],\n"
         << "  device const float *key [[buffer(1)]],\n"
         << "  device const float *value [[buffer(2)]],\n"
         << "  device float *context [[buffer(3)]],\n"
         << "  uint gid [[thread_position_in_grid]]) {\n"
         << "  if (gid >= " << workItems << "u) return;\n"
         << "  const uint feature = gid % " << dimension << "u;\n"
         << "  const uint row = gid / " << dimension << "u;\n"
         << "  const uint batch = row / " << queryLength << "u;\n"
         << "  const uint head = feature / " << plan.selfAttention.headDimension << "u;\n"
         << "  const uint component = feature % " << plan.selfAttention.headDimension << "u;\n"
         << "  float maximum = -INFINITY;\n"
         << "  for (uint position = 0; position < " << plan.sourceLength << "u; ++position) {\n"
         << "    float score = 0.0f;\n"
         << "    for (uint inner = 0; inner < " << plan.selfAttention.headDimension << "u; ++inner) {\n"
         << "      const uint f = head * " << plan.selfAttention.headDimension << "u + inner;\n"
         << "      score += query[row * " << dimension << "u + f] * key[(batch * "
         << plan.sourceLength << "u + position) * " << dimension << "u + f];\n"
         << "    }\n"
         << "    maximum = max(maximum, score * "
         << std::setprecision(std::numeric_limits<float>::max_digits10) << scale << "f);\n"
         << "  }\n"
         << "  float denominator = 0.0f;\n"
         << "  float weighted = 0.0f;\n"
         << "  for (uint position = 0; position < " << plan.sourceLength << "u; ++position) {\n"
         << "    float score = 0.0f;\n"
         << "    for (uint inner = 0; inner < " << plan.selfAttention.headDimension << "u; ++inner) {\n"
         << "      const uint f = head * " << plan.selfAttention.headDimension << "u + inner;\n"
         << "      score += query[row * " << dimension << "u + f] * key[(batch * "
         << plan.sourceLength << "u + position) * " << dimension << "u + f];\n"
         << "    }\n"
         << "    const float probability = exp(score * "
         << std::setprecision(std::numeric_limits<float>::max_digits10) << scale
         << "f - maximum);\n"
         << "    denominator += probability;\n"
         << "    weighted += probability * value[(batch * " << plan.sourceLength
         << "u + position) * " << dimension << "u + head * "
         << plan.selfAttention.headDimension << "u + component];\n"
         << "  }\n"
         << "  context[gid] = weighted / denominator;\n"
         << "}\n";
  kernel.source = source.str();
  return kernel;
}

GeneratedKernel emitTokenArgmax(
    std::size_t rows, std::size_t vocabularySize,
    std::size_t threadsPerThreadgroup, const std::string &functionName) {
  auto kernel = beginKernel(rows, threadsPerThreadgroup, functionName);
  std::ostringstream source;
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << functionName << "(\n"
         << "  device const float *logits [[buffer(0)]],\n"
         << "  device int *tokens [[buffer(1)]],\n"
         << "  uint row [[thread_position_in_grid]]) {\n"
         << "  if (row >= " << rows << "u) return;\n"
         << "  const uint base = row * " << vocabularySize << "u;\n"
         << "  float best = logits[base];\n"
         << "  int selected = 0;\n"
         << "  for (uint token = 1; token < " << vocabularySize << "u; ++token) {\n"
         << "    const float candidate = logits[base + token];\n"
         << "    if (candidate > best) { best = candidate; selected = int(token); }\n"
         << "  }\n"
         << "  tokens[row] = selected;\n"
         << "}\n";
  kernel.source = source.str();
  return kernel;
}

} // namespace tensor::metal
