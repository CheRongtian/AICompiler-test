#include "backend/metal/MetalEmitter.hpp"

#include "analyzer/PatternAnalyzer.hpp"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace tensor::metal {
namespace {

const char *typeName(DType type) {
  if (type == DType::Float16) return "half";
  if (type == DType::Float32) return "float";
  throw std::invalid_argument("Fusion supports floating tensors only.");
}

std::size_t inputSlot(const analyzer::Region &region, ValueId value) {
  const auto found = std::find(region.inputs.begin(), region.inputs.end(), value);
  if (found == region.inputs.end()) throw std::invalid_argument("Fusion input is absent from region interface.");
  return static_cast<std::size_t>(found - region.inputs.begin());
}

std::string logicalIndex(const TensorType &input, const TensorType &output,
                         const std::string &linear) {
  const auto inputStrides = input.strides();
  const auto outputStrides = output.strides();
  const auto leading = output.shape.size() - input.shape.size();
  std::ostringstream expression;
  expression << "0u";
  for (std::size_t i = 0; i < input.shape.size(); ++i) {
    if (input.shape[i] != 1) {
      expression << " + ((" << '(' << linear << ") / " << outputStrides[leading + i]
                 << "u) % " << input.shape[i] << "u) * " << inputStrides[i] << "u";
    }
  }
  return "(" + expression.str() + ")";
}

std::string batchBase(const TensorType &input, const TensorType &output) {
  const auto inputStrides = input.strides();
  std::vector<std::size_t> batchShape(output.shape.begin(), output.shape.end() - 2);
  const auto batchStrides = TensorType{batchShape}.strides();
  const auto inputRank = input.shape.size() - 2;
  const auto leading = batchShape.size() - inputRank;
  std::ostringstream expression;
  expression << "0u";
  for (std::size_t i = 0; i < inputRank; ++i) {
    if (input.shape[i] != 1) {
      expression << " + ((batch / " << batchStrides[leading + i] << "u) % "
                 << input.shape[i] << "u) * " << inputStrides[i] << "u";
    }
  }
  return "(" + expression.str() + ")";
}

void emitInterface(std::ostringstream &source, const analyzer::Region &region,
                   const analyzer::AnalyzedGraph &graph, const char *function,
                   bool reduction) {
  source << "#include <metal_stdlib>\nusing namespace metal;\n\nkernel void " << function << "(\n";
  for (std::size_t i = 0; i < region.inputs.size(); ++i) {
    source << "  device const " << typeName(graph.types[region.inputs[i]].dtype)
           << " *input" << i << " [[buffer(" << i << ")]],\n";
  }
  source << "  device " << typeName(graph.types[region.outputs.front()].dtype)
         << " *output [[buffer(" << region.inputs.size() << ")]],\n";
  if (reduction) {
    source << "  uint tid [[thread_index_in_threadgroup]],\n"
           << "  uint3 group [[threadgroup_position_in_grid]]) {\n";
  } else {
    source << "  uint gid [[thread_position_in_grid]]) {\n";
  }
}

} // namespace

GeneratedKernel emitFusion(const analyzer::Region &region,
                           const analyzer::AnalyzedGraph &graph,
                           std::size_t threads) {
  if (region.fusion == analyzer::FusionPattern::None || region.outputs.size() != 1 ||
      (threads != 64 && threads != 128 && threads != 256)) {
    throw std::invalid_argument("Invalid fused region or thread count.");
  }
  GeneratedKernel kernel;
  kernel.threadsPerThreadgroup = threads;
  const auto &outputType = graph.types[region.outputs.front()];
  const auto elementType = typeName(outputType.dtype);
  std::ostringstream source;
  source.imbue(std::locale::classic());

  if (region.fusion == analyzer::FusionPattern::SiLUMul) {
    kernel.functionName = "fused_silu_mul";
    kernel.threadgroupCount = (outputType.elementCount() + threads - 1) / threads;
    const auto &silu = graph.graph.nodes[region.nodes[0]];
    const auto &mul = graph.graph.nodes[region.nodes[1]];
    const auto xSlot = inputSlot(region, silu.inputs[0]);
    const auto other = mul.inputs[0] == silu.outputs[0] ? mul.inputs[1] : mul.inputs[0];
    const auto otherSlot = inputSlot(region, other);
    emitInterface(source, region, graph, kernel.functionName.c_str(), false);
    source << "  if (gid >= " << outputType.elementCount() << "u) return;\n"
           << "  const float x = float(input" << xSlot << "["
           << logicalIndex(graph.types[silu.inputs[0]], outputType, "gid") << "]);\n"
           << "  const float e = x >= 0.0f ? exp(-x) : exp(x);\n"
           << "  const float activated = float(" << elementType
           << "(x >= 0.0f ? x / (1.0f + e) : x * e / (1.0f + e)));\n"
           << "  output[gid] = " << elementType << "(activated * float(input" << otherSlot << "["
           << logicalIndex(graph.types[other], outputType, "gid") << "]));\n}\n";
  } else if (region.fusion == analyzer::FusionPattern::LinearReLU) {
    kernel.functionName = "fused_linear_relu";
    kernel.threadgroupCount = (outputType.elementCount() + threads - 1) / threads;
    const auto &matmul = graph.graph.nodes[region.nodes[0]];
    const auto &add = graph.graph.nodes[region.nodes[1]];
    const auto aSlot = inputSlot(region, matmul.inputs[0]);
    const auto bSlot = inputSlot(region, matmul.inputs[1]);
    const auto bias = add.inputs[0] == matmul.outputs[0] ? add.inputs[1] : add.inputs[0];
    const auto biasSlot = inputSlot(region, bias);
    const auto &a = graph.types[matmul.inputs[0]];
    const auto &b = graph.types[matmul.inputs[1]];
    const auto aStrides = a.strides();
    const auto bStrides = b.strides();
    const auto m = a.shape[a.shape.size() - 2];
    const auto k = a.shape.back();
    const auto n = b.shape.back();
    emitInterface(source, region, graph, kernel.functionName.c_str(), false);
    source << "  if (gid >= " << outputType.elementCount() << "u) return;\n"
           << "  const uint batch = gid / " << m * n << "u;\n"
           << "  const uint row = (gid / " << n << "u) % " << m << "u;\n"
           << "  const uint column = gid % " << n << "u;\n"
           << "  const uint baseA = " << batchBase(a, outputType) << ";\n"
           << "  const uint baseB = " << batchBase(b, outputType) << ";\n"
           << "  float sum = 0.0f;\n"
           << "  for (uint inner = 0; inner < " << k << "u; ++inner) sum += float(input"
           << aSlot << "[baseA + row * " << aStrides[a.shape.size() - 2] << "u + inner * "
           << aStrides.back() << "u]) * float(input" << bSlot << "[baseB + inner * "
           << bStrides[b.shape.size() - 2] << "u + column * " << bStrides.back() << "u]);\n"
           << "  const float linear = float(" << elementType << "(sum));\n"
           << "  const float biased = float(" << elementType << "(linear + float(input"
           << biasSlot << "[" << logicalIndex(graph.types[bias], outputType, "gid") << "])));\n"
           << "  output[gid] = " << elementType << "(max(biased, 0.0f));\n}\n";
  } else {
    const auto &add = graph.graph.nodes[region.nodes[0]];
    const auto &norm = graph.graph.nodes[region.nodes[1]];
    const auto addType = graph.types[add.outputs[0]];
    const auto lhsSlot = inputSlot(region, add.inputs[0]);
    const auto rhsSlot = inputSlot(region, add.inputs[1]);
    std::size_t normalized = addType.shape.back();
    float epsilon = 1e-5f;
    bool layerNorm = region.fusion == analyzer::FusionPattern::AddLayerNorm;
    if (layerNorm) {
      const auto &attributes = std::get<LayerNormAttributes>(norm.attributes);
      normalized = 1;
      for (auto axis : attributes.axes) normalized *= addType.shape[axis];
      epsilon = attributes.epsilon;
      kernel.functionName = "fused_add_layer_norm";
    } else {
      epsilon = std::get<RMSNormAttributes>(norm.attributes).epsilon;
      kernel.functionName = "fused_add_rms_norm";
    }
    kernel.threadgroupCount = addType.elementCount() / normalized;
    const auto weightSlot = inputSlot(region, norm.inputs[1]);
    const auto biasSlot = layerNorm ? inputSlot(region, norm.inputs[2]) : 0;
    emitInterface(source, region, graph, kernel.functionName.c_str(), true);
    source << "  threadgroup float partialA[" << threads << "];\n";
    if (layerNorm) source << "  threadgroup float partialB[" << threads << "];\n";
    source << "  const uint base = group.x * " << normalized << "u;\n"
           << "  float sum = 0.0f; float squares = 0.0f;\n"
           << "  for (uint i = tid; i < " << normalized << "u; i += " << threads << "u) {\n"
           << "    const uint logical = base + i;\n"
           << "    const float value = float(" << elementType << "(float(input" << lhsSlot << "["
           << logicalIndex(graph.types[add.inputs[0]], addType, "logical") << "]) + float(input"
           << rhsSlot << "[" << logicalIndex(graph.types[add.inputs[1]], addType, "logical") << "])));\n"
           << "    sum += value; squares += value * value;\n  }\n"
           << "  partialA[tid] = squares;\n";
    if (layerNorm) source << "  partialB[tid] = sum;\n";
    source << "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
           << "  for (uint stride = " << threads << "u / 2u; stride > 0u; stride >>= 1u) {\n"
           << "    if (tid < stride) { partialA[tid] += partialA[tid + stride];";
    if (layerNorm) source << " partialB[tid] += partialB[tid + stride];";
    source << " }\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n  }\n";
    if (layerNorm) {
      source << "  const float mean = partialB[0] / " << normalized << ".0f;\n"
             << "  const float variance = max(partialA[0] / " << normalized
             << ".0f - mean * mean, 0.0f);\n"
             << "  const float inverse = rsqrt(variance + ";
    } else {
      source << "  const float mean = 0.0f;\n"
             << "  const float inverse = rsqrt(partialA[0] / " << normalized << ".0f + ";
    }
    source << std::scientific << std::setprecision(std::numeric_limits<float>::max_digits10)
           << epsilon << "f);\n"
           << "  for (uint i = tid; i < " << normalized << "u; i += " << threads << "u) {\n"
           << "    const uint logical = base + i;\n"
           << "    const float value = float(" << elementType << "(float(input" << lhsSlot << "["
           << logicalIndex(graph.types[add.inputs[0]], addType, "logical") << "]) + float(input"
           << rhsSlot << "[" << logicalIndex(graph.types[add.inputs[1]], addType, "logical") << "])));\n"
           << "    float normalizedValue = (value - mean) * inverse * float(input" << weightSlot << "[i]);\n";
    if (layerNorm) source << "    normalizedValue += float(input" << biasSlot << "[i]);\n";
    source << "    output[logical] = " << elementType << "(normalizedValue);\n  }\n}\n";
  }
  kernel.source = source.str();
  return kernel;
}

} // namespace tensor::metal
