#pragma once

#include "backend/metal/MetalRuntime.hpp"
#include "backend/metal/MetalEmitter.hpp"

#include <string>

namespace tensor::llm {

struct KernelCase {
  std::vector<std::size_t> shape;
  std::vector<std::vector<float>> inputs;
  std::vector<std::vector<double>> references;
  std::vector<std::uint32_t> constants;
  std::size_t workItems = 0;
};

// Owned by the compiler; remote responses can only supply source and workgroup size.
struct KernelContract {
  std::string pattern;
  std::string functionName;
  std::string semantics;
  std::vector<std::string> inputNames;
  std::vector<std::string> outputNames;
  std::vector<metal::ElementType> inputTypes;
  metal::ElementType storageType = metal::ElementType::Float32;
  // Group dispatch allows a cooperative reduction per work item.
  bool groupPerWorkItem = false;
  std::string workItem = "element";
  std::string applicability = "Standalone workload";
  std::vector<KernelCase> cases;
  std::vector<std::size_t> workgroupSizes{64, 128, 256};
  double absoluteTolerance = 1e-5;
  double relativeTolerance = 1e-4;
  double minimumSpeedup = 1.05;
};

struct BaselineStep {
  metal::GeneratedKernel kernel;
  // Buffer indices: inputs, outputs, then fp32 intermediates.
  std::vector<std::size_t> inputs;
  std::vector<std::size_t> outputs;
};
struct KernelBaseline {
  std::string name;
  std::vector<std::size_t> intermediateCounts;
  std::vector<BaselineStep> steps;
};
[[nodiscard]] metal::DispatchSize contractDispatch(
    const KernelContract &contract, std::size_t workItems, std::size_t threads);
[[nodiscard]] std::vector<KernelBaseline> makeContractBaselines(
    const KernelContract &contract, const KernelCase &data);
[[nodiscard]] KernelContract makeDecoderKernelContract(const std::string &pattern);
[[nodiscard]] std::vector<KernelBaseline> makeDecoderBaselines(
    const KernelContract &contract, const KernelCase &data);
[[nodiscard]] std::string serializeKernelContract(
    const metal::MetalRuntime &runtime, const KernelContract &contract);
[[nodiscard]] KernelContract makeKernelContract(const std::string &pattern);
[[nodiscard]] std::string typedKernelPattern(const std::string &pattern,
                                            metal::ElementType storageType);
[[nodiscard]] metal::GeneratedKernel emitContractBaseline(
    const KernelContract &contract, const KernelCase &data, std::size_t threads);
[[nodiscard]] std::string writeKernelContract(
    const metal::MetalRuntime &runtime, const KernelContract &contract,
    const std::string &path);

} // namespace tensor::llm
