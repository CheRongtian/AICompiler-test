#include "generated_kernel_main.hpp"

#include "llm/KernelContract.hpp"
#include "runtime/GeneratedKernelAdmission.hpp"

#include <ostream>

bool emitGeneratedKernelContract(const tensor::metal::MetalRuntime &runtime,
                                 const std::string &path,
                                 std::ostream &log) {
  if (!runtime.isAvailable()) {
    log << "Generated-kernel contract: FAIL\n"
        << "Metal error: " << runtime.initializationError() << '\n';
    return false;
  }
  const auto error = tensor::llm::writeSiLUMulKernelContract(runtime, path);
  if (!error.empty()) {
    log << "Generated-kernel contract: FAIL\n"
        << "Contract error: " << error << '\n';
    return false;
  }
  log << "Generated-kernel contract: PASS\n"
      << "Pattern: SiLU + Mul\n"
      << "Cases: [1, 4096] and [3, 4097], fp32\n"
      << "Contract file: " << path << '\n';
  return true;
}

bool runGeneratedKernelAdmission(tensor::metal::MetalRuntime &runtime,
                                 const std::string &responsePath,
                                 const std::string &feedbackPath,
                                 std::ostream &log) {
  return tensor::runtime::admitGeneratedSiLUMulKernel(
      runtime, responsePath, feedbackPath, log);
}
