#pragma once

#include "backend/metal/MetalRuntime.hpp"
#include "planner/KVCachePlan.hpp"

#include <memory>
#include <string>
#include <vector>

namespace tensor::runtime {

class KVCacheState;
struct KVCacheStateCreation;

[[nodiscard]] KVCacheStateCreation
createKVCacheState(metal::MetalRuntime &runtime,
                   const planner::KVCachePlan &plan);

class KVCacheState {
public:
  KVCacheState(const KVCacheState &) = delete;
  KVCacheState &operator=(const KVCacheState &) = delete;

  [[nodiscard]] std::size_t currentLength() const noexcept;
  [[nodiscard]] std::size_t capacity() const noexcept;
  [[nodiscard]] std::string reset(metal::MetalRuntime &runtime);
  [[nodiscard]] std::string stageLength(metal::MetalRuntime &runtime,
                                        std::size_t length);
  void commitLength(std::size_t length);

  [[nodiscard]] std::vector<float> readKeyPrefix() const;
  [[nodiscard]] std::vector<float> readValuePrefix() const;
  [[nodiscard]] bool storageReused() const noexcept;

  [[nodiscard]] const metal::BufferHandle &keyBuffer() const noexcept;
  [[nodiscard]] const metal::BufferHandle &valueBuffer() const noexcept;
  [[nodiscard]] const metal::BufferHandle &lengthBuffer() const noexcept;

private:
  friend KVCacheStateCreation
  createKVCacheState(metal::MetalRuntime &, const planner::KVCachePlan &);
  KVCacheState(planner::KVCachePlan plan, metal::BufferHandle key,
               metal::BufferHandle value, metal::BufferHandle length);
  [[nodiscard]] std::vector<float>
  readPrefix(const metal::BufferHandle &buffer) const;

  planner::KVCachePlan plan_;
  metal::BufferHandle key_;
  metal::BufferHandle value_;
  metal::BufferHandle length_;
  const metal::MetalBuffer *initialKey_ = nullptr;
  const metal::MetalBuffer *initialValue_ = nullptr;
  std::size_t currentLength_ = 0;
};

struct KVCacheStateCreation {
  std::unique_ptr<KVCacheState> state;
  std::string errorMessage;
};

} // namespace tensor::runtime
