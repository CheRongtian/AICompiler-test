#pragma once

#include <cstddef>
#include <memory>
#include <string>

namespace tensor::metal {

struct ComputePipelineResult {
  bool libraryCompilePassed = false;
  bool kernelLookupPassed = false;
  bool pipelineCreationPassed = false;
  std::size_t threadExecutionWidth = 0;
  std::size_t maxTotalThreadsPerThreadgroup = 0;
  std::string errorMessage;
};

class MetalRuntime {
public:
  MetalRuntime();
  ~MetalRuntime();

  MetalRuntime(MetalRuntime &&) noexcept;
  MetalRuntime &operator=(MetalRuntime &&) noexcept;

  MetalRuntime(const MetalRuntime &) = delete;
  MetalRuntime &operator=(const MetalRuntime &) = delete;

  [[nodiscard]] bool isAvailable() const noexcept;
  [[nodiscard]] std::string deviceName() const;
  [[nodiscard]] std::string initializationError() const;

  [[nodiscard]] ComputePipelineResult
  createComputePipeline(const std::string &source,
                        const std::string &functionName) const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace tensor::metal
