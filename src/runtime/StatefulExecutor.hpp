#pragma once

#include "backend/metal/MetalRuntime.hpp"
#include "planner/KVCachePlan.hpp"

#include <iosfwd>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace tensor::runtime {

struct KVCacheCompilation;

struct KVCacheRunResult {
  bool passed = false;
  std::vector<float> output;
  std::optional<double> gpuExecutionTimeUs;
  double cpuSubmitToCompletionTimeUs = 0.0;
  std::string errorMessage;
};

class CompiledKVCacheAttention {
public:
  ~CompiledKVCacheAttention();
  CompiledKVCacheAttention(const CompiledKVCacheAttention &) = delete;
  CompiledKVCacheAttention &operator=(const CompiledKVCacheAttention &) = delete;

  [[nodiscard]] std::string reset();
  [[nodiscard]] KVCacheRunResult prefill(const std::vector<float> &query,
                                         const std::vector<float> &key,
                                         const std::vector<float> &value);
  [[nodiscard]] KVCacheRunResult decode(const std::vector<float> &query,
                                        const std::vector<float> &key,
                                        const std::vector<float> &value);

  [[nodiscard]] std::size_t currentLength() const noexcept;
  [[nodiscard]] std::vector<float> readKeyPrefix() const;
  [[nodiscard]] std::vector<float> readValuePrefix() const;
  [[nodiscard]] bool cacheStorageReused() const noexcept;

private:
  class Impl;
  explicit CompiledKVCacheAttention(std::unique_ptr<Impl> impl);
  friend struct KVCacheCompilation;
  friend KVCacheCompilation compileKVCacheAttention(
      metal::MetalRuntime &, const planner::KVCachePlan &,
      const std::vector<float> &, const std::vector<float> &,
      const std::vector<float> &, const std::vector<float> &, std::ostream &);
  std::unique_ptr<Impl> impl_;
};

struct KVCacheCompilation {
  std::unique_ptr<CompiledKVCacheAttention> executable;
  std::string errorMessage;
};

[[nodiscard]] KVCacheCompilation compileKVCacheAttention(
    metal::MetalRuntime &runtime, const planner::KVCachePlan &plan,
    const std::vector<float> &queryWeight,
    const std::vector<float> &keyWeight,
    const std::vector<float> &valueWeight,
    const std::vector<float> &outputWeight,
    std::ostream &log);

} // namespace tensor::runtime
