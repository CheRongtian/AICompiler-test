#pragma once

#include "backend/metal/MetalRuntime.hpp"
#include "tensor/TransformerDecode.hpp"

#include <iosfwd>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace tensor::runtime {

struct TransformerDecodeCompilation;

struct TransformerDecodeRunResult {
  bool passed = false;
  std::vector<float> logits;
  std::vector<float> nextTokenIds;
  std::optional<double> gpuExecutionTimeUs;
  double cpuSubmitToCompletionTimeUs = 0.0;
  std::string errorMessage;
};

class CompiledTransformerDecoder {
public:
  ~CompiledTransformerDecoder();
  CompiledTransformerDecoder(const CompiledTransformerDecoder &) = delete;
  CompiledTransformerDecoder &operator=(const CompiledTransformerDecoder &) = delete;

  [[nodiscard]] std::string reset();
  [[nodiscard]] TransformerDecodeRunResult
  prefill(const std::vector<float> &tokenIds);
  [[nodiscard]] TransformerDecodeRunResult
  decode(const std::vector<float> &tokenIds);

  [[nodiscard]] std::size_t currentLength() const noexcept;
  [[nodiscard]] std::vector<float> readKeyPrefix(std::size_t layer) const;
  [[nodiscard]] std::vector<float> readValuePrefix(std::size_t layer) const;
  [[nodiscard]] bool cacheStorageReused() const noexcept;

private:
  class Impl;
  explicit CompiledTransformerDecoder(std::unique_ptr<Impl> impl);
  friend TransformerDecodeCompilation compileTransformerDecoder(
      metal::MetalRuntime &, const TransformerDecodeWorkload &, std::ostream &);
  std::unique_ptr<Impl> impl_;
};

struct TransformerDecodeCompilation {
  std::unique_ptr<CompiledTransformerDecoder> executable;
  std::string errorMessage;
};

[[nodiscard]] TransformerDecodeCompilation compileTransformerDecoder(
    metal::MetalRuntime &runtime, const TransformerDecodeWorkload &workload,
    std::ostream &log);

} // namespace tensor::runtime
