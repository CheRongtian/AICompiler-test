#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace tensor::metal {

enum class ElementType { Float16, Float32, Int32 };

struct HardwareInfo {
  std::size_t maxThreadsPerThreadgroup = 0;
  std::size_t maxThreadgroupMemoryLength = 0;
  std::size_t maxBufferLength = 0;
};

struct PipelineBinding {
  std::size_t index = 0;
  bool isBuffer = false;
  bool isFloat32 = false;
  bool isFloat16 = false;
  bool isInt32 = false;
  bool readOnly = false;
  bool writable = false;
};

struct ComputePipelineResult {
  bool libraryCompilePassed = false;
  bool kernelLookupPassed = false;
  bool pipelineCreationPassed = false;
  std::size_t threadExecutionWidth = 0;
  std::size_t maxTotalThreadsPerThreadgroup = 0;
  std::size_t staticThreadgroupMemoryLength = 0;
  bool reflectionAvailable = false;
  // Active argument bindings, translated from Metal reflection to C++ values.
  std::vector<PipelineBinding> bindings;
  std::string errorMessage;
};

[[nodiscard]] std::string checkFloatBufferInterface(const ComputePipelineResult &pipeline,
                                                    std::size_t inputCount);
[[nodiscard]] std::string
checkBufferInterface(const ComputePipelineResult &pipeline,
                     const std::vector<ElementType> &inputs, ElementType output);

// Shared GPU storage, opaque to C++. read() requires completed GPU execution.
class MetalBuffer {
public:
  ~MetalBuffer();
  MetalBuffer(const MetalBuffer &) = delete;
  MetalBuffer &operator=(const MetalBuffer &) = delete;
  [[nodiscard]] std::size_t elementCount() const;
  [[nodiscard]] ElementType elementType() const;
  [[nodiscard]] std::vector<float> read() const;

private:
  friend class MetalRuntime;
  class Impl;
  explicit MetalBuffer(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

using BufferHandle = std::shared_ptr<MetalBuffer>;
struct BufferResult {
  BufferHandle buffer;
  std::string errorMessage;
};

// Borrowed CPU input; storage must remain valid throughout synchronous run().
struct FloatBufferView {
  const float *data = nullptr;
  std::size_t elementCount = 0;
  ElementType elementType = ElementType::Float32;
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

// Owns a pipeline and its buffers independently of later runtime compilations.
// Each submission creates a new command buffer; pipeline/buffers are reused.
class PreparedExecution {
public:
  ~PreparedExecution();
  PreparedExecution(const PreparedExecution &) = delete;
  PreparedExecution &operator=(const PreparedExecution &) = delete;

  // Reset output to NaNs, execute once and read output for numerical validation.
  [[nodiscard]] ExecutionResult run() const;
  // Execute once without CPU readback, for graph nodes, warmup and timing samples.
  [[nodiscard]] ExecutionResult execute() const;

private:
  friend class MetalRuntime;
  class Impl;
  explicit PreparedExecution(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

struct PreparationResult {
  std::unique_ptr<PreparedExecution> execution;
  std::string errorMessage;
};

// A dependency-ordered sequence encoded into one Metal command buffer.
class PreparedSequence {
public:
  ~PreparedSequence();
  PreparedSequence(const PreparedSequence &) = delete;
  PreparedSequence &operator=(const PreparedSequence &) = delete;
  [[nodiscard]] ExecutionResult execute() const;

private:
  friend class MetalRuntime;
  class Impl;
  explicit PreparedSequence(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

struct SequencePreparationResult {
  std::unique_ptr<PreparedSequence> execution;
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
  [[nodiscard]] HardwareInfo hardwareInfo() const;

  [[nodiscard]] ComputePipelineResult
  createComputePipeline(const std::string &source,
                        const std::string &functionName);

  // Inputs bind at indices [0, inputs.size()), followed by one typed output buffer.
  // Optional uint32 constants bind at inputs.size() + 1 using setBytes.
  // The caller supplies bindings and dispatch matching the compiled kernel.
  // A new compilation attempt clears the previous pipeline, including on failure.
  [[nodiscard]] ExecutionResult
  run(const std::vector<FloatBufferView> &inputs,
      std::size_t outputElementCount, const DispatchSize &dispatch,
      const std::vector<std::uint32_t> &constants = {},
      ElementType outputType = ElementType::Float32) const;

  [[nodiscard]] PreparationResult
  prepare(const std::vector<FloatBufferView> &inputs,
          std::size_t outputElementCount, const DispatchSize &dispatch,
          const std::vector<std::uint32_t> &constants = {},
          ElementType outputType = ElementType::Float32) const;

  [[nodiscard]] BufferResult createBuffer(std::size_t elementCount,
                                          const float *initialData = nullptr,
                                          ElementType type = ElementType::Float32) const;
  [[nodiscard]] PreparationResult
  prepareBuffers(const std::vector<BufferHandle> &inputs, const BufferHandle &output,
                 const DispatchSize &dispatch,
                 const std::vector<std::uint32_t> &constants = {}) const;
  [[nodiscard]] SequencePreparationResult
  prepareSequence(const std::vector<const PreparedExecution *> &steps) const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace tensor::metal
