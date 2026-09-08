#pragma once

#include "llm/GeneratedKernelProtocol.hpp"
#include "llm/KernelContract.hpp"

#include <iosfwd>
#include <unordered_map>

namespace tensor::runtime {

// Written only after local admission, with the complete current contract.
[[nodiscard]] std::string writeAdmittedKernel(
    const metal::MetalRuntime &runtime, const llm::KernelContract &contract,
    const llm::GeneratedKernelResponse &response, const std::string &path);

struct KernelUsage {
  std::string node;
  std::string pattern;
  std::string implementation;
  std::string functions;
  std::size_t dispatchesPerCall = 0;
  std::size_t completedCalls = 0;
};

class KernelRegistry {
public:
  void load(const metal::MetalRuntime &runtime, const std::string &directory,
            std::ostream &log);
  [[nodiscard]] const llm::GeneratedKernelResponse *find(const std::string &pattern) const;
  std::size_t track(KernelUsage usage);
  void completed(const std::vector<std::size_t> &uses);
  void resetUsage() noexcept;
  void report(std::ostream &log) const;

private:
  std::unordered_map<std::string, llm::GeneratedKernelResponse> entries_;
  std::vector<KernelUsage> usage_;
};
} // namespace tensor::runtime
