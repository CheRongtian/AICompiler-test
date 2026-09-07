#include "llm/KernelContract.hpp"

#include <fstream>

namespace tensor::llm {

std::string writeSiLUMulKernelContract(const metal::MetalRuntime &runtime,
                                       const std::string &path) {
  const auto hardware = runtime.hardwareInfo();
  std::ofstream output(path);
  if (!output) return "Unable to open generated-kernel contract output '" + path + "'.";

  output
      << "{\n"
      << "  \"version\":1,\n"
      << "  \"target\":{\"backend\":\"metal\",\"device\":\""
      << runtime.deviceName() << "\",\"max_threads_per_threadgroup\":"
      << hardware.maxThreadsPerThreadgroup
      << ",\"max_threadgroup_memory_bytes\":"
      << hardware.maxThreadgroupMemoryLength << "},\n"
      << "  \"pattern\":\"SiLU + Mul\",\n"
      << "  \"semantics\":\"output[i] = silu(input0[i]) * input1[i]\",\n"
      << "  \"cases\":[\n"
      << "    {\"shape\":[1,4096],\"dtype\":\"float32\",\"layout\":\"contiguous\",\"element_count\":4096},\n"
      << "    {\"shape\":[3,4097],\"dtype\":\"float32\",\"layout\":\"contiguous\",\"element_count\":12291}\n"
      << "  ],\n"
      << "  \"interface\":{\n"
      << "    \"function_name\":\"generated_silu_mul\",\n"
      << "    \"buffers\":[\n"
      << "      {\"index\":0,\"type\":\"device const float*\",\"role\":\"input0\"},\n"
      << "      {\"index\":1,\"type\":\"device const float*\",\"role\":\"input1\"},\n"
      << "      {\"index\":2,\"type\":\"device float*\",\"role\":\"output\"},\n"
      << "      {\"index\":3,\"type\":\"constant uint&\",\"role\":\"element_count\"}\n"
      << "    ],\n"
      << "    \"grid_index\":\"uint gid [[thread_position_in_grid]]\"\n"
      << "  },\n"
      << "  \"legal_workgroup_sizes\":[64,128,256],\n"
      << "  \"requirements\":[\n"
      << "    \"Use the exact function name and buffer bindings\",\n"
      << "    \"Return immediately when gid is greater than or equal to element_count\",\n"
      << "    \"Support both cases with one kernel source\",\n"
      << "    \"Do not read or write outside the declared buffers\",\n"
      << "    \"Use fp32 arithmetic and Metal Shading Language\"\n"
      << "  ],\n"
      << "  \"admission\":{\"absolute_tolerance\":1e-5,\"relative_tolerance\":1e-4,\"minimum_speedup\":1.05,\"confirmation_rounds\":2},\n"
      << "  \"response_schema\":{\"version\":1,\"function_name\":\"generated_silu_mul\",\"workgroup_size\":256,\"msl_source\":\"complete source\"}\n"
      << "}\n";
  if (!output) return "Failed while writing generated-kernel contract '" + path + "'.";
  return {};
}

} // namespace tensor::llm
