#pragma once

#include "backend/metal/MetalRuntime.hpp"
#include "tensor/DecoderLLM.hpp"

#include <iosfwd>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace tensor::runtime {

struct DecoderLLMCompilation;

struct DecoderLLMRunResult {
  bool passed = false;
  std::vector<float> logits;
  std::vector<float> nextTokenIds;
  std::optional<double> gpuExecutionTimeUs;
  double cpuSubmitToCompletionTimeUs = 0.0;
  std::string errorMessage;
};

class CompiledDecoderLLM {
public:
  ~CompiledDecoderLLM();
  CompiledDecoderLLM(const CompiledDecoderLLM &) = delete;
  CompiledDecoderLLM &operator=(const CompiledDecoderLLM &) = delete;

  [[nodiscard]] std::string reset();
  [[nodiscard]] DecoderLLMRunResult prefill(const std::vector<float> &tokenIds);
  [[nodiscard]] DecoderLLMRunResult decode(const std::vector<float> &tokenIds);

  [[nodiscard]] std::size_t currentLength() const noexcept;
  [[nodiscard]] std::vector<float> readKeyPrefix(std::size_t layer) const;
  [[nodiscard]] std::vector<float> readValuePrefix(std::size_t layer) const;
  [[nodiscard]] bool cacheStorageReused() const noexcept;
  void resetKernelUsage() noexcept;
  void reportKernelUsage(std::ostream &log) const;

private:
  class Impl;
  explicit CompiledDecoderLLM(std::unique_ptr<Impl> impl);
  friend DecoderLLMCompilation compileDecoderLLM(
      metal::MetalRuntime &, const DecoderLLMWorkload &, std::ostream &,
      const std::string &);
  std::unique_ptr<Impl> impl_;
};

struct DecoderLLMCompilation {
  std::unique_ptr<CompiledDecoderLLM> executable;
  std::string errorMessage;
};

[[nodiscard]] DecoderLLMCompilation compileDecoderLLM(
    metal::MetalRuntime &runtime, const DecoderLLMWorkload &workload,
    std::ostream &log, const std::string &kernelLibrary = {});

} // namespace tensor::runtime
