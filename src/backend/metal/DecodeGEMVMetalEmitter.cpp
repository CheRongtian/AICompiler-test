#include "backend/metal/DecodeGEMVMetalEmitter.hpp"

#include <sstream>
#include <stdexcept>

namespace tensor::metal {

GeneratedKernel emitLinearBaseline(
    std::size_t batch, std::size_t inputSize, std::size_t outputSize,
    std::size_t threads, const std::string &functionName) {
  if (batch == 0 || inputSize == 0 || outputSize == 0 || threads == 0 ||
      functionName.empty()) {
    throw std::invalid_argument("Linear baseline configuration is invalid.");
  }
  const auto workItems = batch * outputSize;
  GeneratedKernel kernel;
  kernel.functionName = functionName;
  kernel.threadsPerThreadgroup = threads;
  kernel.threadgroupCount = (workItems + threads - 1) / threads;
  std::ostringstream source;
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << functionName << "(\n"
         << "  device const float *input [[buffer(0)]],\n"
         << "  device const float *weight [[buffer(1)]],\n"
         << "  device float *output [[buffer(2)]],\n"
         << "  uint gid [[thread_position_in_grid]]) {\n"
         << "  if (gid >= " << workItems << "u) return;\n"
         << "  const uint row = gid / " << outputSize << "u;\n"
         << "  const uint feature = gid % " << outputSize << "u;\n"
         << "  float sum = 0.0f;\n"
         << "  for (uint inner = 0; inner < " << inputSize
         << "u; ++inner) sum += input[row * " << inputSize
         << "u + inner] * weight[feature * " << inputSize << "u + inner];\n"
         << "  output[gid] = sum;\n"
         << "}\n";
  kernel.source = source.str();
  return kernel;
}

GeneratedKernel emitDecodeGEMV(
    std::size_t batch, std::size_t inputSize, std::size_t outputSize,
    DecodeGEMVConfig config, const std::string &functionName) {
  if (batch == 0 || inputSize == 0 || outputSize == 0 ||
      (config.threads != 32 && config.threads != 64 && config.threads != 128) ||
      (config.vectorWidth != 1 && config.vectorWidth != 4) ||
      inputSize % config.vectorWidth != 0 || functionName.empty()) {
    throw std::invalid_argument("Decode GEMV configuration is invalid.");
  }
  GeneratedKernel kernel;
  kernel.functionName = functionName;
  kernel.threadsPerThreadgroup = config.threads;
  kernel.threadgroupCount = batch * outputSize;
  std::ostringstream source;
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << functionName << "(\n"
         << "  device const float *input [[buffer(0)]],\n"
         << "  device const float *weight [[buffer(1)]],\n"
         << "  device float *output [[buffer(2)]],\n"
         << "  uint tid [[thread_index_in_threadgroup]],\n"
         << "  uint group [[threadgroup_position_in_grid]]) {\n"
         << "  const uint row = group / " << outputSize << "u;\n"
         << "  const uint feature = group % " << outputSize << "u;\n"
         << "  float sum = 0.0f;\n";
  if (config.vectorWidth == 4) {
    source << "  device const float4 *input4 = reinterpret_cast<device const float4 *>(input + row * "
           << inputSize << "u);\n"
           << "  device const float4 *weight4 = reinterpret_cast<device const float4 *>(weight + feature * "
           << inputSize << "u);\n"
           << "  for (uint inner = tid; inner < " << inputSize / 4
           << "u; inner += " << config.threads
           << "u) sum += dot(input4[inner], weight4[inner]);\n";
  } else {
    source << "  for (uint inner = tid; inner < " << inputSize
           << "u; inner += " << config.threads
           << "u) sum += input[row * " << inputSize
           << "u + inner] * weight[feature * " << inputSize << "u + inner];\n";
  }
  source << "  threadgroup float scratch[128];\n"
         << "  scratch[tid] = sum;\n"
         << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         << "  for (uint offset = " << config.threads / 2
         << "u; offset > 0u; offset >>= 1u) {\n"
         << "    if (tid < offset) scratch[tid] += scratch[tid + offset];\n"
         << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
         << "  }\n"
         << "  if (tid == 0u) output[row * " << outputSize
         << "u + feature] = scratch[0];\n"
         << "}\n";
  kernel.source = source.str();
  return kernel;
}

} // namespace tensor::metal
