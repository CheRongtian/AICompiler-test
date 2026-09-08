#include "runtime/KVCacheState.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tensor::runtime {

KVCacheState::KVCacheState(planner::KVCachePlan plan,
                           metal::BufferHandle key,
                           metal::BufferHandle value,
                           metal::BufferHandle length,
                           metal::BufferHandle blockTable,
                           std::size_t pageSize)
    : plan_(std::move(plan)), key_(std::move(key)), value_(std::move(value)),
      length_(std::move(length)), blockTable_(std::move(blockTable)),
      initialKey_(key_.get()), initialValue_(value_.get()), pageSize_(pageSize) {
  if (pageSize_ != 0) {
    const auto pageCount = (plan_.capacity + pageSize_ - 1) / pageSize_;
    logicalToPhysical_.assign(pageCount, -1);
    freePhysicalPages_.reserve(pageCount);
    for (std::size_t page = 0; page < pageCount; ++page) {
      freePhysicalPages_.push_back(static_cast<std::int32_t>(page));
    }
  }
}

std::size_t KVCacheState::currentLength() const noexcept { return currentLength_; }
std::size_t KVCacheState::capacity() const noexcept { return plan_.capacity; }

std::string KVCacheState::reset(metal::MetalRuntime &runtime) {
  const float zero = 0.0f;
  auto error = runtime.writeBuffer(length_, &zero, 1);
  if (!error.empty()) return error;
  if (isPaged()) {
    resizePageAllocation(0);
    error = writeBlockTable(runtime);
    if (!error.empty()) return error;
  }
  currentLength_ = 0;
  return error;
}

std::string KVCacheState::stageLength(metal::MetalRuntime &runtime,
                                      std::size_t length) {
  if (length == 0 || length > plan_.capacity) {
    return "KV cache valid length is outside the configured capacity.";
  }
  const auto previousPageTable = logicalToPhysical_;
  const auto previousFreePages = freePhysicalPages_;
  if (isPaged()) {
    const auto requiredPages = (length + pageSize_ - 1) / pageSize_;
    if (requiredPages > logicalToPhysical_.size()) {
      return "Paged KV cache has no block-table capacity for this length.";
    }
    resizePageAllocation(requiredPages);
    auto error = writeBlockTable(runtime);
    if (!error.empty()) {
      logicalToPhysical_ = previousPageTable;
      freePhysicalPages_ = previousFreePages;
      return error;
    }
  }
  const float value = static_cast<float>(length);
  const auto error = runtime.writeBuffer(length_, &value, 1);
  if (!error.empty() && isPaged()) {
    logicalToPhysical_ = previousPageTable;
    freePhysicalPages_ = previousFreePages;
    (void)writeBlockTable(runtime);
  }
  return error;
}

std::string KVCacheState::rollbackLength(metal::MetalRuntime &runtime,
                                         std::size_t length) {
  if (length > plan_.capacity) {
    return "Cannot roll back KV cache beyond its configured capacity.";
  }
  if (isPaged()) {
    const auto requiredPages = length == 0 ? 0 : (length + pageSize_ - 1) / pageSize_;
    resizePageAllocation(requiredPages);
    const auto tableError = writeBlockTable(runtime);
    if (!tableError.empty()) return tableError;
  }
  const float value = static_cast<float>(length);
  return runtime.writeBuffer(length_, &value, 1);
}

void KVCacheState::commitLength(std::size_t length) {
  if (length == 0 || length > plan_.capacity) {
    throw std::invalid_argument("Cannot commit an invalid KV cache length.");
  }
  currentLength_ = length;
}

std::vector<float>
KVCacheState::readPrefix(const metal::BufferHandle &buffer) const {
  const auto storage = buffer->read();
  const auto prefixCount = plan_.batch * plan_.heads * currentLength_ *
                           plan_.headDimension;
  std::vector<float> result;
  result.reserve(prefixCount);
  for (std::size_t batch = 0; batch < plan_.batch; ++batch) {
    for (std::size_t head = 0; head < plan_.heads; ++head) {
      for (std::size_t position = 0; position < currentLength_; ++position) {
        std::size_t base = 0;
        if (isPaged()) {
          const auto logicalPage = position / pageSize_;
          const auto pageOffset = position % pageSize_;
          const auto physicalPage = static_cast<std::size_t>(
              logicalToPhysical_.at(logicalPage));
          base = (((physicalPage * plan_.batch + batch) * plan_.heads + head) *
                      pageSize_ +
                  pageOffset) *
                 plan_.headDimension;
        } else {
          base = ((batch * plan_.heads + head) * plan_.capacity + position) *
                 plan_.headDimension;
        }
        result.insert(result.end(), storage.begin() + base,
                      storage.begin() + base + plan_.headDimension);
      }
    }
  }
  return result;
}

std::vector<float> KVCacheState::readKeyPrefix() const {
  return readPrefix(key_);
}

std::vector<float> KVCacheState::readValuePrefix() const {
  return readPrefix(value_);
}

bool KVCacheState::storageReused() const noexcept {
  return key_.get() == initialKey_ && value_.get() == initialValue_;
}

bool KVCacheState::isPaged() const noexcept { return pageSize_ != 0; }
std::size_t KVCacheState::pageSize() const noexcept { return pageSize_; }

std::size_t KVCacheState::allocatedPageCount() const noexcept {
  return logicalToPhysical_.size() - freePhysicalPages_.size();
}

std::vector<std::int32_t> KVCacheState::pageTable() const {
  return {logicalToPhysical_.begin(),
          logicalToPhysical_.begin() +
              static_cast<std::ptrdiff_t>(allocatedPageCount())};
}

const metal::BufferHandle &KVCacheState::keyBuffer() const noexcept { return key_; }
const metal::BufferHandle &KVCacheState::valueBuffer() const noexcept { return value_; }
const metal::BufferHandle &KVCacheState::lengthBuffer() const noexcept { return length_; }
const metal::BufferHandle &KVCacheState::blockTableBuffer() const noexcept {
  return blockTable_;
}

std::string KVCacheState::writeBlockTable(metal::MetalRuntime &runtime) {
  if (!isPaged()) return {};
  std::vector<float> values;
  values.reserve(logicalToPhysical_.size());
  for (const auto page : logicalToPhysical_) {
    values.push_back(static_cast<float>(page));
  }
  return runtime.writeBuffer(blockTable_, values.data(), values.size());
}

void KVCacheState::resizePageAllocation(std::size_t pageCount) {
  const auto allocated = allocatedPageCount();
  if (pageCount > allocated) {
    if (pageCount - allocated > freePhysicalPages_.size()) {
      throw std::runtime_error("Paged KV cache physical page pool is exhausted.");
    }
    for (std::size_t logical = allocated; logical < pageCount; ++logical) {
      logicalToPhysical_[logical] = freePhysicalPages_.back();
      freePhysicalPages_.pop_back();
    }
    return;
  }
  for (std::size_t logical = allocated; logical > pageCount; --logical) {
    const auto index = logical - 1;
    freePhysicalPages_.push_back(logicalToPhysical_[index]);
    logicalToPhysical_[index] = -1;
  }
}

KVCacheStateCreation createKVCacheState(metal::MetalRuntime &runtime,
                                        const planner::KVCachePlan &plan,
                                        std::size_t pageSize) {
  KVCacheStateCreation result;
  try {
    plan.validate();
    if (pageSize > plan.capacity) {
      throw std::invalid_argument("Paged KV page size exceeds cache capacity.");
    }
    if (pageSize != 0 && plan.batch != 1) {
      throw std::invalid_argument("Paged KV cache currently supports one request.");
    }
    const auto pageCount = pageSize == 0 ? 0 :
        (plan.capacity + pageSize - 1) / pageSize;
    if (pageCount > static_cast<std::size_t>(
                        std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument("Paged KV block table exceeds int32 capacity.");
    }
    const auto storageCount = pageSize == 0
                                  ? plan.cacheElementCount()
                                  : pageCount * pageSize * plan.batch *
                                        plan.heads * plan.headDimension;
    auto key = runtime.createBuffer(storageCount);
    if (!key.buffer) throw std::runtime_error(key.errorMessage);
    auto value = runtime.createBuffer(storageCount);
    if (!value.buffer) throw std::runtime_error(value.errorMessage);
    const float zero = 0.0f;
    auto length = runtime.createBuffer(1, &zero, metal::ElementType::Int32);
    if (!length.buffer) throw std::runtime_error(length.errorMessage);
    metal::BufferHandle blockTable;
    if (pageSize != 0) {
      std::vector<float> initialTable(pageCount, -1.0f);
      auto table = runtime.createBuffer(pageCount, initialTable.data(),
                                        metal::ElementType::Int32);
      if (!table.buffer) throw std::runtime_error(table.errorMessage);
      blockTable = std::move(table.buffer);
    }
    result.state = std::unique_ptr<KVCacheState>(new KVCacheState(
        plan, std::move(key.buffer), std::move(value.buffer),
        std::move(length.buffer), std::move(blockTable), pageSize));
  } catch (const std::exception &error) {
    result.errorMessage = error.what();
  }
  return result;
}

} // namespace tensor::runtime
