#include "backend/metal/MetalEmitter.hpp"
#include "planner/KernelPlan.hpp"

#include <sstream>
#include <stdexcept>

namespace tensor::metal {
namespace {

// Shapes are checked by GraphAnalyzer; singleton and missing axes have zero stride.
std::string broadcastIndex(const TensorType &input, const TensorType &output,
                            const std::string &linearIndex) {
  const auto inputStrides = input.strides();
  const auto outputStrides = output.strides();
  const auto leading = output.shape.size() - input.shape.size();
  std::ostringstream expression;
  expression << "0u";
  for (std::size_t i = 0; i < input.shape.size(); ++i) {
    if (input.shape[i] != 1) {
      expression << " + ((" << linearIndex << " / " << outputStrides[leading + i]
                 << "u) % " << input.shape[i] << "u) * " << inputStrides[i] << "u";
    }
  }
  return "(" + expression.str() + ")";
}

TensorType batchType(const TensorType &type) {
  return {std::vector<std::size_t>(type.shape.begin(), type.shape.end() - 2)};
}

} // namespace

GeneratedKernel emitKernel(const planner::KernelPlan &plan) {
  if (plan.op == OpType::RMSNorm) {
    const auto &attributes = std::get<RMSNormAttributes>(plan.attributes);
    const RMSNormOp op{plan.inputTypes.at(0), plan.inputTypes.at(1), plan.outputType,
                       attributes.epsilon};
    return plan.useRMSNormBaseline ? emitRMSNormBaseline(op)
                                  : emitRMSNorm(op, plan.threadsPerThreadgroup);
  }
  GeneratedKernel kernel;
  kernel.functionName = std::string("tensor_") + opName(plan.op);
  kernel.threadgroupCount = plan.threadgroupCount;
  kernel.threadsPerThreadgroup = plan.threadsPerThreadgroup;
  auto workItems = plan.outputType.elementCount();
  if (plan.op == OpType::RoPE) workItems /= 2;
  std::ostringstream source;
  source << "#include <metal_stdlib>\nusing namespace metal;\n\nkernel void "
         << kernel.functionName << "(\n";
  for (std::size_t i = 0; i < plan.inputTypes.size(); ++i) {
    source << "  device const float *input" << i << " [[buffer(" << i << ")]],\n";
  }
  source << "  device float *output [[buffer(" << plan.inputTypes.size() << ")]],\n"
         << "  uint gid [[thread_position_in_grid]]) {\n"
         << "  if (gid >= " << workItems << "u) return;\n";
  switch (plan.op) {
  case OpType::Add:
  case OpType::Mul:
    source << "  output[gid] = input0["
           << broadcastIndex(plan.inputTypes[0], plan.outputType, "gid") << "] "
           << (plan.op == OpType::Add ? "+" : "*") << " input1["
           << broadcastIndex(plan.inputTypes[1], plan.outputType, "gid") << "];\n";
    break;
  case OpType::SiLU:
    source << "  const float x = input0[gid];\n"
           << "  if (x >= 0.0f) { output[gid] = x / (1.0f + exp(-x)); }\n"
           << "  else { const float e = exp(x); output[gid] = x * e / (1.0f + e); }\n";
    break;
  case OpType::MatMul: {
    const auto &a = plan.inputTypes[0];
    const auto &b = plan.inputTypes[1];
    const auto m = a.shape[a.shape.size() - 2];
    const auto k = a.shape.back();
    const auto n = b.shape.back();
    const auto outBatch = batchType(plan.outputType);
    source << "  const uint batch = gid / " << m * n << "u;\n"
           << "  const uint row = (gid / " << n << "u) % " << m << "u;\n"
           << "  const uint column = gid % " << n << "u;\n"
           << "  const uint offsetA = " << broadcastIndex(batchType(a), outBatch, "batch")
           << " * " << m * k << "u;\n"
           << "  const uint offsetB = " << broadcastIndex(batchType(b), outBatch, "batch")
           << " * " << k * n << "u;\n"
           << "  float sum = 0.0f;\n"
           << "  for (uint inner = 0; inner < " << k << "u; ++inner) {\n"
           << "    sum += input0[offsetA + row * " << k << "u + inner] * "
           << "input1[offsetB + inner * " << n << "u + column];\n"
           << "  }\n  output[gid] = sum;\n";
    break;
  }
  case OpType::RoPE: {
    auto pairType = plan.outputType;
    pairType.shape.back() /= 2;
    source << "  const float c = input1[" << broadcastIndex(plan.inputTypes[1], pairType, "gid") << "];\n"
           << "  const float s = input2[" << broadcastIndex(plan.inputTypes[2], pairType, "gid") << "];\n"
           << "  const float even = input0[2u * gid];\n"
           << "  const float odd = input0[2u * gid + 1u];\n"
           << "  output[2u * gid] = even * c - odd * s;\n"
           << "  output[2u * gid + 1u] = even * s + odd * c;\n";
    break;
  }
  default:
    throw std::invalid_argument("Unsupported operation in Metal emitter.");
  }
  source << "}\n";
  kernel.source = source.str();
  return kernel;
}

} // namespace tensor::metal
