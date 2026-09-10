#include "generated_kernel_main.hpp"

#include "llm/KernelContract.hpp"
#include "runtime/GeneratedKernelAdmission.hpp"

#include <ostream>

bool emitGeneratedKernelContract(const tensor::metal::MetalRuntime &runtime,
                                 const std::string &path,
                                 std::ostream &log, const std::string &pattern) {
  if (!runtime.isAvailable()) {
    log << "Generated-kernel contract: FAIL\n"
        << "Metal error: " << runtime.initializationError() << '\n';
    return false;
  }
  const auto contract = tensor::llm::makeKernelContract(pattern);
  if(contract.storageType==tensor::metal::ElementType::BFloat16 &&
     !runtime.hardwareInfo().supportsBFloat16) {
    log << "Generated-kernel contract: FAIL\nNative bf16 unavailable; request an fp16 pattern.\n";
    return false;
  }
  const auto error = tensor::llm::writeKernelContract(runtime, contract, path);
  if (!error.empty()) {
    log << "Generated-kernel contract: FAIL\n"
        << "Contract error: " << error << '\n';
    return false;
  }
  log << "Generated-kernel contract: PASS\n"
      << "Pattern: " << contract.pattern << '\n'
      << "Cases: " << contract.cases.size()
      << ", storage=" << (contract.storageType==tensor::metal::ElementType::Float16 ? "fp16" :
                           contract.storageType==tensor::metal::ElementType::BFloat16 ? "bf16" : "fp32")
      << ", outputs=" << contract.outputNames.size() << '\n'
      << "Contract file: " << path << '\n';
  return true;
}

bool runGeneratedKernelAdmission(tensor::metal::MetalRuntime &runtime,
                                 const std::string &responsePath,
                                 const std::string &feedbackPath,
                                 std::ostream &log, const std::string &pattern,
                                 const std::string &artifactPath) {
  return tensor::runtime::admitGeneratedKernel(
      runtime, pattern, responsePath, feedbackPath, log, artifactPath);
}
