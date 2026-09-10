#include "backend/metal/DecodeGEMVMetalEmitter.hpp"

#include <sstream>
#include <stdexcept>

namespace tensor::metal {
namespace {

const char *storageTypeName(ElementType type) {
  if (type == ElementType::Float16) return "half";
  if (type == ElementType::Float32) return "float";
  if (type == ElementType::BFloat16) return "bfloat";
  throw std::invalid_argument("Decode GEMV supports fp16, bf16, or fp32 storage.");
}

} // namespace

GeneratedKernel emitLinearBaseline(
    std::size_t batch, std::size_t inputSize, std::size_t outputSize,
    std::size_t threads, const std::string &functionName,
    ElementType storageType, std::size_t vectorWidth,
    ElementType outputType) {
  if (batch == 0 || inputSize == 0 || outputSize == 0 || threads == 0 ||
      (vectorWidth != 1 && vectorWidth != 4) ||
      inputSize % vectorWidth != 0 || functionName.empty()) {
    throw std::invalid_argument("Linear baseline configuration is invalid.");
  }
  const auto *type = storageTypeName(storageType);
  const auto *outputStorage = storageTypeName(outputType);
  const auto workItems = batch * outputSize;
  GeneratedKernel kernel;
  kernel.functionName = functionName;
  kernel.threadsPerThreadgroup = threads;
  kernel.threadgroupCount = (workItems + threads - 1) / threads;
  std::ostringstream source;
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << functionName << "(\n"
         << "  device const " << type << " *input [[buffer(0)]],\n"
         << "  device const " << type << " *weight [[buffer(1)]],\n"
         << "  device " << outputStorage << " *output [[buffer(2)]],\n"
         << "  uint gid [[thread_position_in_grid]]) {\n"
         << "  if (gid >= " << workItems << "u) return;\n"
         << "  const uint row = gid / " << outputSize << "u;\n"
         << "  const uint feature = gid % " << outputSize << "u;\n"
         << "  float sum = 0.0f;\n";
  if (vectorWidth == 4) {
    source << "  device const " << type
           << "4 *input4 = reinterpret_cast<device const " << type
           << "4 *>(input + row * " << inputSize << "u);\n"
           << "  device const " << type
           << "4 *weight4 = reinterpret_cast<device const " << type
           << "4 *>(weight + feature * " << inputSize << "u);\n"
           << "  for (uint inner = 0; inner < " << inputSize / 4
           << "u; ++inner) sum += dot(float4(input4[inner]), "
              "float4(weight4[inner]));\n";
  } else {
    source << "  for (uint inner = 0; inner < " << inputSize
           << "u; ++inner) sum += float(input[row * " << inputSize
           << "u + inner]) * float(weight[feature * " << inputSize
           << "u + inner]);\n";
  }
  source << "  output[gid] = " << outputStorage << "(sum);\n"
         << "}\n";
  kernel.source = source.str();
  return kernel;
}

GeneratedKernel emitDecodeGEMV(
    std::size_t batch, std::size_t inputSize, std::size_t outputSize,
    DecodeGEMVConfig config, const std::string &functionName,
    ElementType storageType, ElementType outputType) {
  if (batch == 0 || inputSize == 0 || outputSize == 0 ||
      (config.threads != 32 && config.threads != 64 && config.threads != 128) ||
      (config.vectorWidth != 1 && config.vectorWidth != 4) ||
      (config.simdgroupReduction && config.threads != 32) ||
      inputSize % config.vectorWidth != 0 || functionName.empty()) {
    throw std::invalid_argument("Decode GEMV configuration is invalid.");
  }
  const auto *type = storageTypeName(storageType);
  const auto *outputStorage = storageTypeName(outputType);
  GeneratedKernel kernel;
  kernel.functionName = functionName;
  kernel.threadsPerThreadgroup = config.threads;
  kernel.threadgroupCount = batch * outputSize;
  std::ostringstream source;
  source << "#include <metal_stdlib>\nusing namespace metal;\n\n"
         << "kernel void " << functionName << "(\n"
         << "  device const " << type << " *input [[buffer(0)]],\n"
         << "  device const " << type << " *weight [[buffer(1)]],\n"
         << "  device " << outputStorage << " *output [[buffer(2)]],\n"
         << "  uint tid [[thread_index_in_threadgroup]],\n"
         << "  uint group [[threadgroup_position_in_grid]]) {\n"
         << "  const uint row = group / " << outputSize << "u;\n"
         << "  const uint feature = group % " << outputSize << "u;\n"
         << "  float sum = 0.0f;\n";
  if (config.vectorWidth == 4) {
    source << "  device const " << type << "4 *input4 = reinterpret_cast<device const "
           << type << "4 *>(input + row * "
           << inputSize << "u);\n"
           << "  device const " << type << "4 *weight4 = reinterpret_cast<device const "
           << type << "4 *>(weight + feature * "
           << inputSize << "u);\n"
           << "  for (uint inner = tid; inner < " << inputSize / 4
           << "u; inner += " << config.threads
           << "u) sum += dot(float4(input4[inner]), float4(weight4[inner]));\n";
  } else {
    source << "  for (uint inner = tid; inner < " << inputSize
           << "u; inner += " << config.threads
           << "u) sum += float(input[row * " << inputSize
           << "u + inner]) * float(weight[feature * " << inputSize
           << "u + inner]);\n";
  }
  if (config.simdgroupReduction) {
    source << "  sum = simd_sum(sum);\n"
           << "  if (tid == 0u) output[row * " << outputSize
           << "u + feature] = " << outputStorage << "(sum);\n";
  } else {
    source << "  threadgroup float scratch[128];\n"
           << "  scratch[tid] = sum;\n"
           << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
           << "  for (uint offset = " << config.threads / 2
           << "u; offset > 0u; offset >>= 1u) {\n"
           << "    if (tid < offset) scratch[tid] += scratch[tid + offset];\n"
           << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n"
           << "  }\n"
           << "  if (tid == 0u) output[row * " << outputSize
           << "u + feature] = " << outputStorage << "(scratch[0]);\n";
  }
  source << "}\n";
  kernel.source = source.str();
  return kernel;
}

GeneratedKernel emitTiledGEMM(
    std::size_t rows, std::size_t inputSize, std::size_t outputSize,
    std::size_t tileSize, const std::string &functionName,
    ElementType storageType, ElementType outputType) {
  if (!rows || !inputSize || !outputSize || (tileSize != 8 && tileSize != 16))
    throw std::invalid_argument("GEMM requires nonzero dimensions and tile 8 or 16.");
  const auto *type = storageTypeName(storageType);
  const auto *outputStorage = storageTypeName(outputType);
  const auto columns = (outputSize + tileSize - 1) / tileSize;
  GeneratedKernel kernel{ {}, functionName,
      ((rows + tileSize - 1) / tileSize) * columns, tileSize * tileSize };
  std::ostringstream s;
  s << "#include <metal_stdlib>\nusing namespace metal;\n"
    << "kernel void " << functionName << "(device const " << type
    << "* x [[buffer(0)]], device const " << type
    << "* w [[buffer(1)]], device " << outputStorage
    << "* y [[buffer(2)]], uint tid [[thread_index_in_threadgroup]], "
       "uint group [[threadgroup_position_in_grid]]) {\n"
    << "const uint tx=tid%" << tileSize << "u, ty=tid/" << tileSize << "u;\n"
    << "const uint row=(group/" << columns << "u)*" << tileSize << "u+ty;\n"
    << "const uint col=(group%" << columns << "u)*" << tileSize << "u+tx;\n"
    << "threadgroup float a[" << tileSize * tileSize << "], b[" << tileSize * tileSize << "];\n"
    << "float sum=0.0f;\nfor(uint base=0;base<" << inputSize << "u;base+=" << tileSize << "u){\n"
    << "a[tid]=(row<" << rows << "u && base+tx<" << inputSize
    << "u)?float(x[row*" << inputSize << "u+base+tx]):0.0f;\n"
    << "b[tid]=(col<" << outputSize << "u && base+ty<" << inputSize
    << "u)?float(w[col*" << inputSize << "u+base+ty]):0.0f;\n"
    << "threadgroup_barrier(mem_flags::mem_threadgroup);\n"
    << "for(uint k=0;k<" << tileSize << "u;++k) sum+=a[ty*" << tileSize
    << "u+k]*b[k*" << tileSize << "u+tx];\n"
    << "threadgroup_barrier(mem_flags::mem_threadgroup);\n}\n"
    << "if(row<" << rows << "u && col<" << outputSize
    << "u) y[row*" << outputSize << "u+col]=" << outputStorage << "(sum);\n}\n";
  kernel.source=s.str();
  return kernel;
}

} // namespace tensor::metal
