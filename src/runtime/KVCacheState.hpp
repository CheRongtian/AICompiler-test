#pragma once

#include "backend/metal/MetalRuntime.hpp"
#include "planner/KVCachePlan.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace tensor::runtime {

class KVPagePool;
struct KVPagePoolCreation;
class KVCacheState;
struct KVCacheStateCreation;

[[nodiscard]] KVPagePoolCreation
createKVPagePool(metal::MetalRuntime &runtime,
                 const planner::KVCachePlan &plan, std::size_t pageSize,
                 std::size_t physicalPageCount);

class KVPagePool {
public:
  KVPagePool(const KVPagePool &) = delete;
  KVPagePool &operator=(const KVPagePool &) = delete;

  [[nodiscard]] std::size_t pageSize() const noexcept;
  [[nodiscard]] std::size_t capacityPages() const noexcept;
  [[nodiscard]] std::size_t usedPages() const noexcept;
  [[nodiscard]] std::size_t availablePages() const noexcept;
  [[nodiscard]] std::size_t peakUsedPages() const noexcept;
  [[nodiscard]] std::size_t allocationCount() const noexcept;
  [[nodiscard]] std::size_t releaseCount() const noexcept;
  [[nodiscard]] const metal::BufferHandle &keyBuffer() const noexcept;
  [[nodiscard]] const metal::BufferHandle &valueBuffer() const noexcept;

private:
  friend KVPagePoolCreation
  createKVPagePool(metal::MetalRuntime &, const planner::KVCachePlan &,
                   std::size_t, std::size_t);
  friend KVCacheStateCreation
  createKVCacheState(metal::MetalRuntime &, const planner::KVCachePlan &,
                     std::shared_ptr<KVPagePool>);
  friend class KVCacheState;
  KVPagePool(planner::KVCachePlan plan, std::size_t pageSize,
             metal::BufferHandle key, metal::BufferHandle value,
             std::size_t physicalPageCount);
  [[nodiscard]] std::int32_t allocatePage();
  void releasePage(std::int32_t page);
  [[nodiscard]] bool compatible(const planner::KVCachePlan &plan) const noexcept;

  planner::KVCachePlan plan_;
  std::size_t pageSize_ = 0;
  metal::BufferHandle key_;
  metal::BufferHandle value_;
  std::vector<std::int32_t> freePages_;
  std::vector<bool> allocated_;
  std::size_t peakUsedPages_ = 0;
  std::size_t allocationCount_ = 0;
  std::size_t releaseCount_ = 0;
};

struct KVPagePoolCreation {
  std::shared_ptr<KVPagePool> pool;
  std::string errorMessage;
};

[[nodiscard]] KVCacheStateCreation
createKVCacheState(metal::MetalRuntime &runtime,
                   const planner::KVCachePlan &plan,
                   std::size_t pageSize = 0);

[[nodiscard]] KVCacheStateCreation
createKVCacheState(metal::MetalRuntime &runtime,
                   const planner::KVCachePlan &plan,
                   std::shared_ptr<KVPagePool> pagePool);

class KVCacheState {
public:
  ~KVCacheState();
  KVCacheState(const KVCacheState &) = delete;
  KVCacheState &operator=(const KVCacheState &) = delete;

  [[nodiscard]] std::size_t currentLength() const noexcept;
  [[nodiscard]] std::size_t capacity() const noexcept;
  [[nodiscard]] std::string reset(metal::MetalRuntime &runtime);
  [[nodiscard]] std::string stageLength(metal::MetalRuntime &runtime,
                                        std::size_t length);
  [[nodiscard]] std::string rollbackLength(metal::MetalRuntime &runtime,
                                           std::size_t length);
  [[nodiscard]] std::string restore(
      metal::MetalRuntime &runtime, std::size_t length,
      const std::vector<float> &keyPrefix,
      const std::vector<float> &valuePrefix);
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
  friend KVCacheStateCreation
  createKVCacheState(metal::MetalRuntime &, const planner::KVCachePlan &,
                     std::shared_ptr<KVPagePool>);
  KVCacheState(planner::KVCachePlan plan, metal::BufferHandle key,
               metal::BufferHandle value, metal::BufferHandle length,
               metal::BufferHandle blockTable,
               std::shared_ptr<KVPagePool> pagePool);
  [[nodiscard]] std::vector<float>
  readPrefix(const metal::BufferHandle &buffer) const;
  [[nodiscard]] std::string writeBlockTable(metal::MetalRuntime &runtime);
  [[nodiscard]] std::string writePagedPrefix(
      metal::MetalRuntime &runtime, const metal::BufferHandle &buffer,
      const std::vector<float> &prefix, std::size_t length);
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
  std::shared_ptr<KVPagePool> pagePool_;
  std::vector<std::int32_t> logicalToPhysical_;
};

struct KVCacheStateCreation {
  std::unique_ptr<KVCacheState> state;
  std::string errorMessage;
};

} // namespace tensor::runtime
