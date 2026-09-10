#include "llm/KernelContract.hpp"

#include "backend/metal/DecodeGEMVMetalEmitter.hpp"
#include "backend/metal/DecoderLLMMetalEmitter.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace tensor::llm {
namespace {
constexpr std::size_t kMaximumGeneratedGEMVWeightElements = 8u * 1024u * 1024u;

planner::DecoderLLMPlan decoderPlan() {
  planner::DecoderLLMPlan plan;
  plan.attention = {1, 4, 16, 32, 8, DType::Float32, 128};
  plan.layerCount = 2;
  plan.intermediateSize = 128;
  plan.vocabularySize = 128;
  return plan;
}
std::vector<float> values(std::size_t count, std::size_t seed, float scale = 1.0f) {
  std::vector<float> result(count);
  for (std::size_t i = 0; i < count; ++i)
    result[i] = float(int((i * 17 + seed * 31) % 257) - 128) / 128 * scale;
  return result;
}
double dot(const std::vector<float> &x, const std::vector<float> &w,
           std::size_t feature, std::size_t width) {
  double sum = 0;
  for (std::size_t i = 0; i < width; ++i)
    sum += double(x[i]) * w[feature * width + i];
  return sum;
}
std::pair<std::size_t, std::size_t> gemvShape(const std::string &pattern) {
  if (pattern == "decoder_gemv_64_64") return {64,64};
  if (pattern == "decoder_gemv_64_128") return {64,128};
  if (pattern == "decoder_gemv_128_64") return {128,64};
  const std::string prefix="decoder_gemv_";
  if (pattern.rfind(prefix,0)==0) {
    const auto split=pattern.find('_',prefix.size());
    if(split==std::string::npos) throw std::invalid_argument("Expected decoder_gemv_K_N.");
    std::size_t usedK=0,usedN=0;
    const auto k=std::stoull(pattern.substr(prefix.size(),split-prefix.size()),&usedK);
    const auto n=std::stoull(pattern.substr(split+1),&usedN);
    if(!k || !n || usedK!=split-prefix.size() || usedN!=pattern.size()-split-1)
      throw std::invalid_argument("Invalid GEMV contract dimensions.");
    return {k,n};
  }
  return {0,0};
}

std::string baselineName(std::size_t threads, std::size_t vectorWidth,
                         bool simdgroup) {
  return "threads=" + std::to_string(threads) +
         ", vector=" + std::to_string(vectorWidth) +
         ", simdgroup=" + (simdgroup ? "true" : "false");
}

metal::GeneratedKernel emitDecoderLinearBaseline(
    std::size_t inputSize, std::size_t outputSize, std::size_t threads,
    std::size_t vectorWidth, bool simdgroup, metal::ElementType storageType,
    const std::string &functionName) {
  if (vectorWidth == 0) {
    return metal::emitLinearBaseline(1, inputSize, outputSize, threads,
                                     functionName, storageType, 1,
                                     storageType);
  }
  return metal::emitDecodeGEMV(
      1, inputSize, outputSize, {threads, vectorWidth, 0, simdgroup},
      functionName, storageType, storageType);
}

KernelBaseline makeGEMVBaseline(
    std::size_t inputSize, std::size_t outputSize, std::size_t threads,
    std::size_t vectorWidth, bool simdgroup,
    metal::ElementType storageType) {
  KernelBaseline baseline;
  baseline.name = baselineName(threads, vectorWidth, simdgroup);
  baseline.steps = {{emitDecoderLinearBaseline(
                         inputSize, outputSize, threads, vectorWidth,
                         simdgroup, storageType, "template_decoder_gemv"),
                     {0, 1}, {2}}};
  return baseline;
}

KernelBaseline makeGatedMLPBaseline(
    std::size_t threads, std::size_t vectorWidth, bool simdgroup,
    metal::ElementType storageType) {
  KernelBaseline baseline;
  baseline.name = baselineName(threads, vectorWidth, simdgroup);
  baseline.intermediateCounts = {128, 128};
  baseline.steps = {
      {emitDecoderLinearBaseline(64, 128, threads, vectorWidth, simdgroup,
                                 storageType, "template_gate"),
       {0, 1}, {4}},
      {emitDecoderLinearBaseline(64, 128, threads, vectorWidth, simdgroup,
                                 storageType, "template_up"),
       {0, 2}, {5}},
      {metal::emitDecoderSiLUMul(128, threads, "template_gated",
                                 storageType),
       {4, 5}, {3}}};
  return baseline;
}

KernelBaseline makeFusedBaseline(const std::string &pattern,
                                 std::size_t threads,
                                 metal::ElementType storageType) {
  KernelBaseline baseline;
  baseline.name = "fused threads=" + std::to_string(threads);
  if (pattern == "decoder_residual_rmsnorm") {
    baseline.steps = {{metal::emitDecoderResidualRMSNorm(
                           1, 64, 1e-5f, threads,
                           "template_fused_residual_norm", storageType),
                       {0, 1, 2}, {3, 4}}};
    return baseline;
  }
  if (pattern == "decoder_gated_mlp") {
    baseline.steps = {{metal::emitDecoderGatedMLP(
                           1, 64, 128, threads, "template_fused_gated",
                           storageType),
                       {0, 1, 2}, {3}}};
    return baseline;
  }
  throw std::invalid_argument("Missing fused decoder baseline: " + pattern);
}
}

KernelContract makeDecoderKernelContract(const std::string &pattern) {
  KernelContract c;
  c.pattern = pattern;
  c.functionName = "generated_" + pattern;
  c.absoluteTolerance = 2e-4;
  c.relativeTolerance = 2e-3;
  c.workgroupSizes = {32,64,128};
  c.groupPerWorkItem = true;
  c.applicability = "Decoder-only fp32, batch=1, sequence_length=1; no prefill replacement";
  const auto [k,n] = gemvShape(pattern);
  if (k) {
    if (n > kMaximumGeneratedGEMVWeightElements / k) {
      throw std::invalid_argument(
          "Generated GEMV contract fixture exceeds 8M weight elements; "
          "the local template planner remains available for this model shape.");
    }
    c.inputNames = {"input", "weight"};
    c.outputNames = {"output"};
    c.workItem = "one output feature per threadgroup; reduce over K";
    c.semantics = "Bias-free output[j]=sum_i input[i]*weight[j*K+i], "
        "row-major weight [N,K], M=1, K=" + std::to_string(k) +
        ", N=" + std::to_string(n) + ". Group j owns output[j].";
  } else if (pattern == "decoder_rope") {
    c.groupPerWorkItem = false;
    c.inputNames = {"query", "key", "cosine", "sine", "valid_length"};
    c.outputNames = {"rotated_query", "rotated_key"};
    c.workItem = "one interleaved pair in both Q and K";
    c.applicability += "; hidden=64, heads=4, head_dim=16, cache_capacity=32";
    c.semantics = "Q/K are contiguous [1,1,64]. Outputs are head-major [1,4,1,16]. "
        "cosine/sine are position tables [32,8]. valid_length is a read-only int buffer [1], "
        "including the token being appended (1..32). For p<32, use "
        "c=cosine[(valid_length[0]-1)*8+p%8], s=sine[(valid_length[0]-1)*8+p%8]. "
        "For both inputs, out[2*p]=in[2*p]*c-in[2*p+1]*s; "
        "out[2*p+1]=in[2*p]*s+in[2*p+1]*c. Do not modify cache or length.";
  } else if (pattern == "decoder_residual_rmsnorm") {
    c.inputNames = {"residual", "update", "weight"};
    c.outputNames = {"sum", "normalized"};
    c.workItem = "one width-64 row per threadgroup";
    c.applicability += "; hidden=64, rms_epsilon=1e-5";
    c.semantics = "For i<64, sum[i]=fp32(residual[i]+update[i]); "
        "normalized[i]=sum[i]*rsqrt(mean_j(sum[j]*sum[j])+1e-5)*weight[i]. "
        "Both outputs [1,64] are required. Use group-uniform reduction barriers.";
  } else if (pattern == "decoder_gated_mlp") {
    c.inputNames = {"input", "gate_weight", "up_weight"};
    c.outputNames = {"gated"};
    c.workItem = "one intermediate feature per threadgroup; reduce over hidden=64";
    c.applicability += "; hidden=64, intermediate=128";
    c.semantics = "Input [1,64], gate_weight/up_weight row-major [128,64], no biases. "
        "For j<128: gate=sum_i input[i]*gate_weight[j*64+i], "
        "up=sum_i input[i]*up_weight[j*64+i], gated[j]=(gate/(1+exp(-gate)))*up. "
        "This replaces Gate/Up projections and SiLU*Mul; Down projection stays separate.";
  } else {
    throw std::invalid_argument("Unsupported generated-kernel pattern: " + pattern);
  }
  c.inputTypes.assign(c.inputNames.size(), metal::ElementType::Float32);
  if (pattern == "decoder_rope") c.inputTypes.back() = metal::ElementType::Int32;
  for (std::size_t seed = 1; seed <= 3; ++seed) {
    KernelCase data;
    if (k) {
      data.shape = {1,k,n};
      data.inputs = {values(k, seed), values(k*n, seed+9, 0.125f)};
      data.references = {std::vector<double>(n)};
      data.workItems = n;
      for (std::size_t j = 0; j < n; ++j)
        data.references[0][j] = dot(data.inputs[0], data.inputs[1], j, k);
    } else if (pattern == "decoder_rope") {
      const std::size_t length = seed == 1 ? 9 : seed == 2 ? 16 : 32;
      data.shape = {1,4,1,16};
      data.inputs = {values(64,seed), values(64,seed+1),
                     std::vector<float>(256), std::vector<float>(256),
                     {float(length)}};
      data.references.assign(2, std::vector<double>(64));
      data.workItems = 32;
      for (std::size_t position = 0; position < 32; ++position)
        for (std::size_t pair = 0; pair < 8; ++pair) {
          const auto angle = position * std::pow(10000.0, -double(pair)/8);
          data.inputs[2][position*8+pair] = float(std::cos(angle));
          data.inputs[3][position*8+pair] = float(std::sin(angle));
        }
      for (std::size_t output = 0; output < 2; ++output)
        for (std::size_t p = 0; p < 32; ++p) {
          const auto index = (length-1)*8+p%8;
          const double a = data.inputs[output][2*p], b = data.inputs[output][2*p+1];
          data.references[output][2*p] = a*data.inputs[2][index]-b*data.inputs[3][index];
          data.references[output][2*p+1] = a*data.inputs[3][index]+b*data.inputs[2][index];
        }
    } else if (pattern == "decoder_residual_rmsnorm") {
      data.shape = {1,64};
      data.inputs = {values(64,seed), values(64,seed+7,0.5f), values(64,seed+4,0.25f)};
      for (auto &w : data.inputs[2]) w += 1;
      data.references.assign(2,std::vector<double>(64));
      data.workItems = 1;
      double squares = 0;
      for (std::size_t i=0;i<64;++i) {
        const float sum = data.inputs[0][i]+data.inputs[1][i];
        data.references[0][i] = sum;
        squares += double(sum)*sum;
      }
      for (std::size_t i=0;i<64;++i)
        data.references[1][i] = data.references[0][i] /
            std::sqrt(squares/64+1e-5)*data.inputs[2][i];
    } else {
      data.shape = {1,64,128};
      data.inputs = {values(64,seed), values(128*64,seed+3,0.125f),
                     values(128*64,seed+5,0.125f)};
      data.references = {std::vector<double>(128)};
      data.workItems = 128;
      for (std::size_t j=0;j<128;++j) {
        const double gate = dot(data.inputs[0],data.inputs[1],j,64);
        data.references[0][j] = gate/(1+std::exp(-gate))*
            dot(data.inputs[0],data.inputs[2],j,64);
      }
    }
    data.constants = {static_cast<std::uint32_t>(data.workItems)};
    c.cases.push_back(std::move(data));
  }
  return c;
}

std::vector<KernelBaseline> makeDecoderBaselines(
    const KernelContract &contract, const KernelCase &) {
  auto normalizedContract = contract;
  if (normalizedContract.storageType != metal::ElementType::Float32) {
    normalizedContract.pattern.resize(normalizedContract.pattern.size() - 5);
  }

  std::vector<KernelBaseline> result;
  const auto shape = gemvShape(normalizedContract.pattern);
  const auto inputSize = shape.first;
  const auto outputSize = shape.second;
  const bool isGEMV = inputSize != 0;
  const bool isGatedMLP =
      normalizedContract.pattern == "decoder_gated_mlp";

  if (isGEMV || isGatedMLP) {
    for (const auto threads : normalizedContract.workgroupSizes) {
      for (const std::size_t vectorWidth : {0u, 1u, 4u}) {
        if (isGEMV && vectorWidth != 0 && inputSize % vectorWidth != 0) {
          continue;
        }
        result.push_back(
            isGEMV
                ? makeGEMVBaseline(inputSize, outputSize, threads,
                                   vectorWidth, false,
                                   normalizedContract.storageType)
                : makeGatedMLPBaseline(threads, vectorWidth, false,
                                       normalizedContract.storageType));
      }
      if (threads == 32) {
        for (const std::size_t vectorWidth : {1u, 4u}) {
          if (isGEMV && inputSize % vectorWidth != 0) {
            continue;
          }
          result.push_back(
              isGEMV
                  ? makeGEMVBaseline(inputSize, outputSize, threads,
                                     vectorWidth, true,
                                     normalizedContract.storageType)
                  : makeGatedMLPBaseline(threads, vectorWidth, true,
                                         normalizedContract.storageType));
        }
      }
    }
  } else {
    for (const auto threads : normalizedContract.workgroupSizes) {
      KernelBaseline baseline;
      baseline.name = "threads=" + std::to_string(threads);
      if (normalizedContract.pattern == "decoder_rope") {
        auto plan = decoderPlan();
        plan.attention.threadsPerThreadgroup = threads;
        baseline.steps = {
            {metal::emitDecoderRoPE(plan, 1, "template_decoder_q_rope",
                                    normalizedContract.storageType),
             {0, 2, 3, 4}, {5}},
            {metal::emitDecoderRoPE(plan, 1, "template_decoder_k_rope",
                                    normalizedContract.storageType),
             {1, 2, 3, 4}, {6}}};
      } else if (normalizedContract.pattern ==
                 "decoder_residual_rmsnorm") {
        baseline.steps = {
            {metal::emitDecoderAdd(64, threads, "template_residual",
                                   normalizedContract.storageType),
             {0, 1}, {3}},
            {metal::emitDecoderRMSNorm(
                 1, 64, 1e-5f, threads, "template_residual_norm",
                 normalizedContract.storageType),
             {3, 2}, {4}}};
      } else {
        throw std::invalid_argument("Missing decoder baseline: " +
                                    normalizedContract.pattern);
      }
      result.push_back(std::move(baseline));
    }
  }

  if (normalizedContract.pattern == "decoder_residual_rmsnorm" ||
      normalizedContract.pattern == "decoder_gated_mlp") {
    for (const auto threads : normalizedContract.workgroupSizes) {
      result.push_back(makeFusedBaseline(normalizedContract.pattern, threads,
                                         normalizedContract.storageType));
    }
  }
  return result;
}
} // namespace tensor::llm
