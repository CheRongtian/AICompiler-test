#pragma once

#include "backend/metal/MetalRuntime.hpp"
#include "planner/KVCachePlan.hpp"

#include <memory>
#include <cstdint>
#include <string>
#include <vector>

namespace tensor::runtime {

class KVCacheState;
struct KVCacheStateCreation;

[[nodiscard]] KVCacheStateCreation
createKVCacheState(metal::MetalRuntime &runtime,
                   const planner::KVCachePlan &plan,
                   std::size_t pageSize = 0);

class KVCacheState {
public:
  KVCacheState(const KVCacheState &) = delete;
  KVCacheState &operator=(const KVCacheState &) = delete;

  [[nodiscard]] std::size_t currentLength() const noexcept;
  [[nodiscard]] std::size_t capacity() const noexcept;
  [[nodiscard]] std::string reset(metal::MetalRuntime &runtime);
  [[nodiscard]] std::string stageLength(metal::MetalRuntime &runtime,
                                        std::size_t length);
  [[nodiscard]] std::string rollbackLength(metal::MetalRuntime &runtime,
                                           std::size_t length);
  void commitLength(std::size_t length);

  [[nodiscard]] std::vector<float> readKeyPrefix() const;
  [[nodiscard]] std::vector<float> readValuePrefix() const;
  [[nodiscard]] bool storageReused() const noexcept;
  [[nodiscard]] bool isPaged() const noexcept;
  [[nodiscard]] std::size_t pageSize() const noexcept;
  [[nodiscard]] std::size_t allocatedPageCount() const noexcept;
  [[nodiscard]] std::vector<std::int32_t> pageTable() const;

  [[nodiscard]] const metal::BufferHandle &keyBuffer() const noexcept;
  [[nodiscard]] const metal::BufferHandle &valueBuffer() const noexcept;
  [[nodiscard]] const metal::BufferHandle &lengthBuffer() const noexcept;
  [[nodiscard]] const metal::BufferHandle &blockTableBuffer() const noexcept;

private:
  friend KVCacheStateCreation
  createKVCacheState(metal::MetalRuntime &, const planner::KVCachePlan &,
                     std::size_t);
  KVCacheState(planner::KVCachePlan plan, metal::BufferHandle key,
               metal::BufferHandle value, metal::BufferHandle length,
               metal::BufferHandle blockTable, std::size_t pageSize);
  [[nodiscard]] std::vector<float>
  readPrefix(const metal::BufferHandle &buffer) const;
  [[nodiscard]] std::string writeBlockTable(metal::MetalRuntime &runtime);
  void resizePageAllocation(std::size_t pageCount);

  planner::KVCachePlan plan_;
  metal::BufferHandle key_;
  metal::BufferHandle value_;
  metal::BufferHandle length_;
  metal::BufferHandle blockTable_;
  const metal::MetalBuffer *initialKey_ = nullptr;
  const metal::MetalBuffer *initialValue_ = nullptr;
  std::size_t currentLength_ = 0;
  std::size_t pageSize_ = 0;
  std::vector<std::int32_t> logicalToPhysical_;
  std::vector<std::int32_t> freePhysicalPages_;
};

struct KVCacheStateCreation {
  std::unique_ptr<KVCacheState> state;
  std::string errorMessage;
};

} // namespace tensor::runtime
