#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace tensor::metal {

struct ComputePipelineResult {
  bool libraryCompilePassed = false;
  bool kernelLookupPassed = false;
  bool pipelineCreationPassed = false;
  std::size_t threadExecutionWidth = 0;
  std::size_t maxTotalThreadsPerThreadgroup = 0;
  std::string errorMessage;
};

// Borrowed CPU input; storage must remain valid throughout synchronous run().
struct FloatBufferView {
  const float *data = nullptr;
  std::size_t elementCount = 0;
};

struct DispatchSize {
  std::size_t threadgroupCount = 0;
  std::size_t threadsPerThreadgroup = 0;
};

struct ExecutionResult {
  bool executionPassed = false;
  std::vector<float> output;
  // Command-buffer GPU duration; empty when Metal provides no usable timestamps.
  std::optional<double> gpuExecutionTimeUs;
  // CPU elapsed time from commit through waitUntilCompleted, excluding readback.
  double cpuSubmitToCompletionTimeUs = 0.0;
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
                        const std::string &functionName);

  // Inputs bind at indices [0, inputs.size()), then one fp32 output buffer.
  // Optional uint32 constants bind at inputs.size() + 1 using setBytes.
  // The caller supplies bindings and dispatch matching the compiled kernel.
  // A new compilation attempt clears the previous pipeline, including on failure.
  [[nodiscard]] ExecutionResult
  run(const std::vector<FloatBufferView> &inputs,
      std::size_t outputElementCount, const DispatchSize &dispatch,
      const std::vector<std::uint32_t> &constants = {}) const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace tensor::metal
