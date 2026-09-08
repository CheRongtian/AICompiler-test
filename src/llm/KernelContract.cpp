#include "llm/KernelContract.hpp"

#include "analyzer/PatternAnalyzer.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace tensor::llm {
namespace {
std::string quote(const std::string &value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << c;
    else if (c < 32) out << "\\u" << std::hex << std::setw(4)
                         << std::setfill('0') << unsigned(c) << std::dec;
    else out << c;
  }
  out << '"';
  return out.str();
}
}

KernelContract makeKernelContract(const std::string &pattern) {
  KernelContract contract;
  contract.pattern = pattern;
  if (pattern == "silu_mul") {
    contract.functionName = "generated_silu_mul";
    contract.semantics = "output[i] = (input0[i] / (1 + exp(-input0[i]))) * input1[i]";
    contract.inputNames = {"input0", "input1"};
    contract.outputNames = {"output"};
  } else if (pattern == "rope") {
    contract.functionName = "generated_rope";
    contract.semantics =
        "For each flattened interleaved pair p, rotate both query and key: "
        "out[2*p]=in[2*p]*cosine[p]-in[2*p+1]*sine[p]; "
        "out[2*p+1]=in[2*p]*sine[p]+in[2*p+1]*cosine[p]. "
        "cosine/sine are pre-expanded per pair in head-major [B,H,S,D] order.";
    contract.inputNames = {"query", "key", "cosine", "sine"};
    contract.outputNames = {"rotated_query", "rotated_key"};
  } else {
    return makeDecoderKernelContract(pattern);
  }
  for (std::size_t caseIndex = 0; caseIndex < 2; ++caseIndex) {
    KernelCase data;
    std::size_t count;
    if (pattern == "silu_mul") {
      data.shape = caseIndex == 0 ? std::vector<std::size_t>{1,4096}
                                  : std::vector<std::size_t>{3,4097};
      count = data.shape[0] * data.shape[1];
    } else {
      data.shape = {1,4,caseIndex == 0 ? 1u : 3u,16};
      count = 4 * data.shape[2] * 16;
    }
    data.inputs.resize(contract.inputNames.size());
    data.inputs[0].resize(count);
    data.inputs[1].resize(count);
    for (std::size_t i = 0; i < count; ++i) {
      data.inputs[0][i] = float(int((i * 17 + (caseIndex + 1) * 29) % 257) - 128) / 64;
      data.inputs[1][i] = float(int((i * 31 + (caseIndex + 1) * 11) % 193) - 96) / 80;
    }
    data.references.resize(contract.outputNames.size(), std::vector<double>(count));
    if (pattern == "silu_mul") {
      data.workItems = count;
      for (std::size_t i = 0; i < count; ++i) {
        const double x = data.inputs[0][i];
        data.references[0][i] = x / (1 + std::exp(-x)) * data.inputs[1][i];
      }
    } else {
      data.workItems = count / 2;
      data.inputs[2].resize(data.workItems);
      data.inputs[3].resize(data.workItems);
      for (std::size_t p = 0; p < data.workItems; ++p) {
        const auto position = caseIndex * 8 + (p / 8) % data.shape[2];
        const double angle = position * std::pow(10000.0, -double(p % 8) / 8);
        data.inputs[2][p] = float(std::cos(angle));
        data.inputs[3][p] = float(std::sin(angle));
        for (std::size_t output = 0; output < 2; ++output) {
          const double even = data.inputs[output][2*p];
          const double odd = data.inputs[output][2*p+1];
          data.references[output][2*p] = even * data.inputs[2][p] - odd * data.inputs[3][p];
          data.references[output][2*p+1] = even * data.inputs[3][p] + odd * data.inputs[2][p];
        }
      }
    }
    data.constants = {static_cast<std::uint32_t>(data.workItems)};
    contract.cases.push_back(std::move(data));
  }
  contract.inputTypes.assign(contract.inputNames.size(), metal::ElementType::Float32);
  contract.workItem = pattern == "rope" ? "interleaved pair" : "element";
  return contract;
}

metal::DispatchSize contractDispatch(const KernelContract &contract,
                                     std::size_t workItems, std::size_t threads) {
  return {contract.groupPerWorkItem ? workItems : (workItems + threads - 1) / threads, threads};
}

std::vector<KernelBaseline> makeContractBaselines(
    const KernelContract &contract, const KernelCase &data) {
  if (contract.pattern != "silu_mul" && contract.pattern != "rope")
    return makeDecoderBaselines(contract, data);
  std::vector<KernelBaseline> result;
  for (auto threads : contract.workgroupSizes) {
    BaselineStep step;
    step.kernel = emitContractBaseline(contract, data, threads);
    for (std::size_t i = 0; i < data.inputs.size(); ++i) step.inputs.push_back(i);
    for (std::size_t i = 0; i < data.references.size(); ++i)
      step.outputs.push_back(data.inputs.size() + i);
    result.push_back({"threads=" + std::to_string(threads), {}, {std::move(step)}});
  }
  return result;
}

metal::GeneratedKernel emitContractBaseline(
    const KernelContract &contract, const KernelCase &data, std::size_t threads) {
  if (contract.pattern == "silu_mul") {
    TensorGraph graph;
    const TensorType type{data.shape, DType::Float32};
    const auto x = graph.addInput("input", type);
    const auto y = graph.addInput("multiplier", type);
    const auto activated = graph.addNode(OpType::SiLU, {x});
    graph.outputs = {graph.addNode(OpType::Mul, {activated, y})};
    auto regions = analyzer::formRegions(analyzer::analyze(graph));
    for (const auto &region : regions.regions)
      if (region.fusion == analyzer::FusionPattern::SiLUMul)
        return metal::emitFusion(region, regions.analyzed, threads);
    throw std::runtime_error("SiLU + Mul baseline region was not formed.");
  }
  if (contract.pattern != "rope") throw std::invalid_argument("Missing pattern baseline.");
  metal::GeneratedKernel kernel;
  kernel.functionName = "template_rope";
  kernel.threadsPerThreadgroup = threads;
  kernel.threadgroupCount = (data.workItems + threads - 1) / threads;
  std::ostringstream source;
  source << "#include <metal_stdlib>\nusing namespace metal;\n"
         << "kernel void template_rope(device const float *q [[buffer(0)]],"
         << "device const float *k [[buffer(1)]],"
         << "device const float *c [[buffer(2)]],"
         << "device const float *s [[buffer(3)]],"
         << "device float *oq [[buffer(4)]],device float *ok [[buffer(5)]],"
         << "uint p [[thread_position_in_grid]]) {\n"
         << "if (p >= " << data.workItems << "u) return;\n"
         << "float a=q[2*p], b=q[2*p+1], x=k[2*p], y=k[2*p+1];\n"
         << "oq[2*p]=a*c[p]-b*s[p]; oq[2*p+1]=a*s[p]+b*c[p];\n"
         << "ok[2*p]=x*c[p]-y*s[p]; ok[2*p+1]=x*s[p]+y*c[p];\n}\n";
  kernel.source = source.str();
  return kernel;
}

std::string serializeKernelContract(const metal::MetalRuntime &runtime,
                                    const KernelContract &contract) {
  std::ostringstream out;
  const auto hw = runtime.hardwareInfo();
  out << "{\"version\":2,\"pattern\":" << quote(contract.pattern)
      << ",\"applicability\":" << quote(contract.applicability)
      << ",\"target\":{\"backend\":\"metal\",\"device\":" << quote(runtime.deviceName())
      << ",\"max_threads_per_threadgroup\":" << hw.maxThreadsPerThreadgroup
      << ",\"max_threadgroup_memory_bytes\":" << hw.maxThreadgroupMemoryLength
      << "},\"semantics\":" << quote(contract.semantics)
      << ",\"interface\":{\"function_name\":" << quote(contract.functionName)
      << ",\"buffers\":[";
  std::size_t binding = 0;
  for (const auto &name : contract.inputNames) {
    if (binding) out << ',';
    out << "{\"index\":" << binding++ << ",\"role\":" << quote(name)
        << ",\"type\":" << quote(contract.inputTypes[binding - 1] == metal::ElementType::Int32
                                      ? "device const int*" : "device const float*")
        << ",\"access\":\"read\"}";
  }
  for (const auto &name : contract.outputNames)
    out << ",{\"index\":" << binding++ << ",\"role\":" << quote(name)
        << ",\"type\":\"device float*\",\"access\":\"write\"}";
  out << ",{\"index\":" << binding
      << ",\"role\":\"work_item_count\",\"type\":\"constant uint&\",\"access\":\"read\"}]}"
      << ",\"dispatch\":{\"policy\":"
      << quote(contract.groupPerWorkItem ? "work_item_count threadgroups"
                                         : "ceil(work_item_count/workgroup_size) threadgroups")
      << ",\"allowed_thread_parameters\":"
      << quote(contract.groupPerWorkItem
          ? "uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]"
          : "uint gid [[thread_position_in_grid]]")
      << ",\"work_item\":" << quote(contract.workItem) << "}"
      << ",\"cases\":[";
  for (std::size_t i = 0; i < contract.cases.size(); ++i) {
    if (i) out << ',';
    const auto &data = contract.cases[i];
    out << "{\"shape\":[";
    for (std::size_t j = 0; j < data.shape.size(); ++j) {
      if (j) out << ',';
      out << data.shape[j];
    }
    out << "],\"dtype\":\"float32\",\"layout\":\"contiguous\",\"input_element_counts\":[";
    for (std::size_t j = 0; j < data.inputs.size(); ++j) {
      if (j) out << ',';
      out << data.inputs[j].size();
    }
    out << "],\"output_element_counts\":[";
    for (std::size_t j = 0; j < data.references.size(); ++j) {
      if (j) out << ',';
      out << data.references[j].size();
    }
    out << "],\"constants\":[" << data.constants[0]
        << "],\"work_item_count\":" << data.workItems << '}';
  }
  out << "],\"legal_workgroup_sizes\":[";
  for (std::size_t i = 0; i < contract.workgroupSizes.size(); ++i) {
    if (i) out << ',';
    out << contract.workgroupSizes[i];
  }
  out << "],\"requirements\":[\"Preserve every binding, type, output and function name\","
      << "\"Guard the grid or group work-item index against work_item_count; keep group barriers uniform\","
      << "\"Support all cases with one source; constants are supplied by the host\","
      << "\"Only use valid Metal Shading Language; no out-of-bounds access\"],"
      << "\"admission\":{\"absolute_tolerance\":" << contract.absoluteTolerance
      << ",\"relative_tolerance\":" << contract.relativeTolerance
      << ",\"minimum_speedup\":" << contract.minimumSpeedup
      << ",\"confirmation_rounds\":2,\"performance_cases\":\"all\"},"
      << "\"response_schema\":{\"version\":1,\"function_name\":" << quote(contract.functionName)
      << ",\"workgroup_size\":64,\"msl_source\":\"complete source\"}}\n";
  return out.str();
}

std::string writeKernelContract(const metal::MetalRuntime &runtime,
                                const KernelContract &contract,
                                const std::string &path) {
  std::ofstream out(path);
  if (!out) return "Unable to open kernel contract: " + path;
  out << serializeKernelContract(runtime, contract);
  return out ? std::string{} : "Unable to write kernel contract: " + path;
}

} // namespace tensor::llm
