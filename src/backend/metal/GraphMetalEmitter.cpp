#include "backend/metal/MetalEmitter.hpp"
#include "planner/KernelPlan.hpp"

#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace tensor::metal {
namespace {

const char *mslType(DType type) {
  switch (type) {
  case DType::Float16: return "half";
  case DType::Float32: return "float";
  case DType::Int32: return "int";
  }
  throw std::invalid_argument("Unsupported Metal element type.");
}

std::string logicalIndex(const TensorType &input,
                         const std::vector<std::size_t> &logicalShape,
                         const std::string &linearIndex) {
  const auto inputStrides = input.strides();
  TensorType logical{logicalShape, input.dtype};
  const auto logicalStrides = logical.strides();
  const auto leading = logicalShape.size() - input.shape.size();
  std::ostringstream expression;
  expression << "0u";
  for (std::size_t i = 0; i < input.shape.size(); ++i) {
    if (input.shape[i] != 1) {
      expression << " + ((" << '(' << linearIndex << ") / " << logicalStrides[leading + i]
                 << "u) % " << input.shape[i] << "u) * " << inputStrides[i] << "u";
    }
  }
  return "(" + expression.str() + ")";
}

std::string batchBase(const TensorType &input, const TensorType &output,
                      const std::string &batchIndex) {
  const auto inputStrides = input.strides();
  std::vector<std::size_t> outputBatch(output.shape.begin(), output.shape.end() - 2);
  TensorType batchType{outputBatch};
  const auto outputStrides = batchType.strides();
  const auto inputRank = input.shape.size() - 2;
  const auto leading = outputBatch.size() - inputRank;
  std::ostringstream expression;
  expression << "0u";
  for (std::size_t i = 0; i < inputRank; ++i) {
    if (input.shape[i] != 1) {
      expression << " + ((" << batchIndex << " / " << outputStrides[leading + i]
                 << "u) % " << input.shape[i] << "u) * " << inputStrides[i] << "u";
    }
  }
  return "(" + expression.str() + ")";
}

std::string axisIndex(const TensorType &input, const TensorType &output,
                      std::size_t axis, const std::string &linear,
                      const std::string &axisCoordinate) {
  const auto inputStrides = input.strides();
  const auto outputStrides = output.strides();
  std::ostringstream expression;
  expression << "0u";
  for (std::size_t i = 0; i < input.shape.size(); ++i) {
    if (i == axis) expression << " + (" << axisCoordinate << ") * " << inputStrides[i] << "u";
    else expression << " + ((" << linear << " / " << outputStrides[i] << "u) % "
                    << output.shape[i] << "u) * " << inputStrides[i] << "u";
  }
  return "(" + expression.str() + ")";
}

std::string reductionIndex(const TensorType &input, const TensorType &output,
                           const std::vector<std::int64_t> &axes,
                           bool keepDimensions) {
  const auto inputStrides = input.strides();
  const auto outputStrides = output.strides();
  std::vector<bool> reduced(input.shape.size(), false);
  for (auto axis : axes) reduced[axis] = true;
  std::vector<std::size_t> reductionStrides(axes.size(), 1);
  for (std::size_t i = axes.size(); i > 1; --i) {
    reductionStrides[i - 2] = reductionStrides[i - 1] * input.shape[axes[i - 1]];
  }
  std::ostringstream expression;
  expression << "0u";
  std::size_t outputAxis = 0;
  std::size_t reductionAxis = 0;
  for (std::size_t i = 0; i < input.shape.size(); ++i) {
    if (reduced[i]) {
      expression << " + ((reduce / " << reductionStrides[reductionAxis] << "u) % "
                 << input.shape[i] << "u) * " << inputStrides[i] << "u";
      ++reductionAxis;
      if (keepDimensions) ++outputAxis;
    } else {
      expression << " + ((gid / " << outputStrides[outputAxis] << "u) % "
                 << output.shape[outputAxis] << "u) * " << inputStrides[i] << "u";
      ++outputAxis;
    }
  }
  return "(" + expression.str() + ")";
}

std::string sliceIndex(const TensorType &input, const TensorType &output,
                       const SliceAttributes &attributes) {
  const auto inputStrides = input.strides();
  const auto outputStrides = output.strides();
  std::ostringstream expression;
  expression << "0u";
  for (std::size_t i = 0; i < input.shape.size(); ++i) {
    expression << " + (" << attributes.starts[i] << "u + ((gid / "
               << outputStrides[i] << "u) % " << output.shape[i] << "u) * "
               << attributes.steps[i] << "u) * " << inputStrides[i] << "u";
  }
  return "(" + expression.str() + ")";
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
  if (isViewOp(plan.op)) throw std::invalid_argument("View operations do not emit kernels.");

  GeneratedKernel kernel;
  kernel.functionName = std::string("tensor_") + opName(plan.op);
  kernel.threadgroupCount = plan.threadgroupCount;
  kernel.threadsPerThreadgroup = plan.threadsPerThreadgroup;
  auto workItems = plan.outputType.elementCount();
  if (plan.op == OpType::RoPE) workItems /= 2;

  std::ostringstream source;
  source.imbue(std::locale::classic());
  source << "#include <metal_stdlib>\nusing namespace metal;\n\nkernel void "
         << kernel.functionName << "(\n";
  for (std::size_t i = 0; i < plan.inputTypes.size(); ++i) {
    source << "  device const " << mslType(plan.inputTypes[i].dtype) << " *input" << i
           << " [[buffer(" << i << ")]],\n";
  }
  source << "  device " << mslType(plan.outputType.dtype) << " *output [[buffer("
         << plan.inputTypes.size() << ")]],\n";

  if (plan.op == OpType::LayerNorm) {
    const auto &attributes = std::get<LayerNormAttributes>(plan.attributes);
    std::size_t normalized = 1;
    for (auto axis : attributes.axes) normalized *= plan.outputType.shape[axis];
    source << "  uint tid [[thread_index_in_threadgroup]],\n"
           << "  uint3 group [[threadgroup_position_in_grid]]) {\n"
           << "  threadgroup float partialSum[" << plan.threadsPerThreadgroup << "];\n"
           << "  threadgroup float partialSquares[" << plan.threadsPerThreadgroup << "];\n"
           << "  const uint base = group.x * " << normalized << "u;\n"
           << "  float sum = 0.0f; float squares = 0.0f;\n"
           << "  for (uint i = tid; i < " << normalized << "u; i += "
           << plan.threadsPerThreadgroup << "u) { const float x = float(input0[base + i]); "
           << "sum += x; squares += x * x; }\n"
           << "  partialSum[tid] = sum; partialSquares[tid] = squares;\n"
           << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
           << "  for (uint stride = " << plan.threadsPerThreadgroup
           << "u / 2u; stride > 0u; stride >>= 1u) {\n"
           << "    if (tid < stride) { partialSum[tid] += partialSum[tid + stride]; "
           << "partialSquares[tid] += partialSquares[tid + stride]; }\n"
           << "    threadgroup_barrier(mem_flags::mem_threadgroup);\n  }\n"
           << "  const float mean = partialSum[0] / " << normalized << ".0f;\n"
           << "  const float variance = partialSquares[0] / " << normalized
           << ".0f - mean * mean;\n"
           << "  const float inverse = rsqrt(max(variance, 0.0f) + "
           << std::scientific << std::setprecision(std::numeric_limits<float>::max_digits10)
           << attributes.epsilon << "f);\n"
           << "  for (uint i = tid; i < " << normalized << "u; i += "
           << plan.threadsPerThreadgroup << "u) output[base + i] = "
           << mslType(plan.outputType.dtype)
           << "((float(input0[base + i]) - mean) * inverse * float(input1[i]) + float(input2[i]));\n"
           << "}\n";
    kernel.source = source.str();
    return kernel;
  }

  source << "  uint gid [[thread_position_in_grid]]) {\n"
         << "  if (gid >= " << workItems << "u) return;\n";
  const auto outputType = std::string(mslType(plan.outputType.dtype));
  switch (plan.op) {
  case OpType::Add:
  case OpType::Mul:
    source << "  output[gid] = " << outputType << "(float(input0["
           << logicalIndex(plan.inputTypes[0], plan.outputType.shape, "gid") << "]) "
           << (plan.op == OpType::Add ? "+" : "*") << " float(input1["
           << logicalIndex(plan.inputTypes[1], plan.outputType.shape, "gid") << "]));\n";
    break;
  case OpType::ReLU:
    source << "  output[gid] = " << outputType << "(max(float(input0["
           << logicalIndex(plan.inputTypes[0], plan.outputType.shape, "gid")
           << "]), 0.0f));\n";
    break;
  case OpType::SiLU:
    source << "  const float x = float(input0["
           << logicalIndex(plan.inputTypes[0], plan.outputType.shape, "gid") << "]);\n"
           << "  const float y = x >= 0.0f ? x / (1.0f + exp(-x)) : "
           << "x * exp(x) / (1.0f + exp(x));\n"
           << "  output[gid] = " << outputType << "(y);\n";
    break;
  case OpType::Contiguous:
    source << "  output[gid] = input0["
           << logicalIndex(plan.inputTypes[0], plan.outputType.shape, "gid") << "];\n";
    break;
  case OpType::MatMul: {
    const auto &a = plan.inputTypes[0];
    const auto &b = plan.inputTypes[1];
    const auto aStrides = a.strides();
    const auto bStrides = b.strides();
    const auto m = a.shape[a.shape.size() - 2];
    const auto k = a.shape.back();
    const auto n = b.shape.back();
    source << "  const uint batch = gid / " << m * n << "u;\n"
           << "  const uint row = (gid / " << n << "u) % " << m << "u;\n"
           << "  const uint column = gid % " << n << "u;\n"
           << "  const uint offsetA = " << batchBase(a, plan.outputType, "batch") << ";\n"
           << "  const uint offsetB = " << batchBase(b, plan.outputType, "batch") << ";\n"
           << "  float sum = 0.0f;\n"
           << "  for (uint inner = 0; inner < " << k << "u; ++inner) sum += "
           << "float(input0[offsetA + row * " << aStrides[a.shape.size() - 2]
           << "u + inner * " << aStrides.back() << "u]) * "
           << "float(input1[offsetB + inner * " << bStrides[b.shape.size() - 2]
           << "u + column * " << bStrides.back() << "u]);\n"
           << "  output[gid] = " << outputType << "(sum);\n";
    break;
  }
  case OpType::Softmax: {
    const auto axis = static_cast<std::size_t>(std::get<SoftmaxAttributes>(plan.attributes).axis);
    const auto width = plan.inputTypes[0].shape[axis];
    source << "  float maximum = -INFINITY;\n"
           << "  for (uint i = 0; i < " << width << "u; ++i) maximum = max(maximum, float(input0["
           << axisIndex(plan.inputTypes[0], plan.outputType, axis, "gid", "i") << "]));\n"
           << "  float denominator = 0.0f;\n"
           << "  for (uint i = 0; i < " << width << "u; ++i) denominator += exp(float(input0["
           << axisIndex(plan.inputTypes[0], plan.outputType, axis, "gid", "i") << "]) - maximum);\n"
           << "  const uint coordinate = (gid / " << plan.outputType.strides()[axis]
           << "u) % " << width << "u;\n"
           << "  output[gid] = " << outputType << "(exp(float(input0["
           << axisIndex(plan.inputTypes[0], plan.outputType, axis, "gid", "coordinate")
           << "]) - maximum) / denominator);\n";
    break;
  }
  case OpType::RoPE: {
    auto pairShape = plan.outputType.shape;
    pairShape.back() /= 2;
    source << "  const float c = float(input1["
           << logicalIndex(plan.inputTypes[1], pairShape, "gid") << "]);\n"
           << "  const float s = float(input2["
           << logicalIndex(plan.inputTypes[2], pairShape, "gid") << "]);\n"
           << "  const float even = float(input0["
           << logicalIndex(plan.inputTypes[0], plan.outputType.shape, "2u * gid") << "]);\n"
           << "  const float odd = float(input0["
           << logicalIndex(plan.inputTypes[0], plan.outputType.shape, "2u * gid + 1u") << "]);\n"
           << "  output[2u * gid] = " << outputType << "(even * c - odd * s);\n"
           << "  output[2u * gid + 1u] = " << outputType << "(even * s + odd * c);\n";
    break;
  }
  case OpType::ReduceSum:
  case OpType::ReduceMean: {
    const auto &attributes = std::get<ReductionAttributes>(plan.attributes);
    std::size_t reductionCount = 1;
    for (auto axis : attributes.axes) reductionCount *= plan.inputTypes[0].shape[axis];
    source << "  float sum = 0.0f;\n"
           << "  for (uint reduce = 0; reduce < " << reductionCount
           << "u; ++reduce) sum += float(input0["
           << reductionIndex(plan.inputTypes[0], plan.outputType, attributes.axes,
                             attributes.keepDimensions) << "]);\n"
           << "  output[gid] = " << outputType << "(sum";
    if (plan.op == OpType::ReduceMean) source << " / " << reductionCount << ".0f";
    source << ");\n";
    break;
  }
  case OpType::MaskedFill: {
    const auto value = std::get<MaskedFillAttributes>(plan.attributes).value;
    source << "  const bool masked = input1["
           << logicalIndex(plan.inputTypes[1], plan.outputType.shape, "gid") << "] != 0;\n"
           << "  output[gid] = masked ? " << outputType << "(" << std::scientific
           << std::setprecision(std::numeric_limits<float>::max_digits10) << value
           << "f) : input0[" << logicalIndex(plan.inputTypes[0], plan.outputType.shape, "gid")
           << "];\n";
    break;
  }
  case OpType::Slice:
    source << "  output[gid] = input0["
           << sliceIndex(plan.inputTypes[0], plan.outputType,
                         std::get<SliceAttributes>(plan.attributes)) << "];\n";
    break;
  case OpType::Embedding: {
    const auto width = plan.inputTypes[0].shape[1];
    source << "  const uint indexPosition = gid / " << width << "u;\n"
           << "  const uint column = gid % " << width << "u;\n"
           << "  const int row = input1["
           << logicalIndex(plan.inputTypes[1], plan.inputTypes[1].shape, "indexPosition") << "];\n"
           << "  output[gid] = input0[uint(row) * " << width << "u + column];\n";
    break;
  }
  case OpType::Gather: {
    const auto axis = static_cast<std::size_t>(std::get<GatherAttributes>(plan.attributes).axis);
    const auto &data = plan.inputTypes[0];
    const auto &indices = plan.inputTypes[1];
    const auto outputStrides = plan.outputType.strides();
    const auto dataStrides = data.strides();
    const auto indexStrides = indices.strides();
    std::ostringstream indexExpression;
    indexExpression << "0u";
    for (std::size_t i = 0; i < indices.shape.size(); ++i) {
      const auto outAxis = axis + i;
      indexExpression << " + ((gid / " << outputStrides[outAxis] << "u) % "
                      << plan.outputType.shape[outAxis] << "u) * " << indexStrides[i] << "u";
    }
    std::ostringstream dataExpression;
    dataExpression << "0u";
    for (std::size_t i = 0; i < data.shape.size(); ++i) {
      if (i == axis) dataExpression << " + uint(selected) * " << dataStrides[i] << "u";
      else {
        const auto outAxis = i < axis ? i : i + indices.shape.size() - 1;
        dataExpression << " + ((gid / " << outputStrides[outAxis] << "u) % "
                       << plan.outputType.shape[outAxis] << "u) * " << dataStrides[i] << "u";
      }
    }
    source << "  const int selected = input1[(" << indexExpression.str() << ")];\n"
           << "  output[gid] = input0[(" << dataExpression.str() << ")];\n";
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
