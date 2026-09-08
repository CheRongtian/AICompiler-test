#include "runtime/KVCacheState.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tensor::runtime {
namespace {

std::size_t checkedMultiply(std::size_t left, std::size_t right) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error("Paged KV storage size exceeds size_t capacity.");
  }
  return left * right;
}

} // namespace

KVPagePool::KVPagePool(planner::KVCachePlan plan, std::size_t pageSize,
                       metal::BufferHandle key, metal::BufferHandle value,
                       std::size_t physicalPageCount)
    : plan_(std::move(plan)), pageSize_(pageSize), key_(std::move(key)),
      value_(std::move(value)), allocated_(physicalPageCount, false) {
  freePages_.reserve(physicalPageCount);
  for (std::size_t page = 0; page < physicalPageCount; ++page) {
    freePages_.push_back(static_cast<std::int32_t>(page));
  }
}

std::size_t KVPagePool::pageSize() const noexcept { return pageSize_; }
std::size_t KVPagePool::capacityPages() const noexcept {
  return allocated_.size();
}
std::size_t KVPagePool::usedPages() const noexcept {
  return allocated_.size() - freePages_.size();
}
std::size_t KVPagePool::availablePages() const noexcept {
  return freePages_.size();
}
std::size_t KVPagePool::peakUsedPages() const noexcept {
  return peakUsedPages_;
}
std::size_t KVPagePool::allocationCount() const noexcept {
  return allocationCount_;
}
std::size_t KVPagePool::releaseCount() const noexcept {
  return releaseCount_;
}
const metal::BufferHandle &KVPagePool::keyBuffer() const noexcept {
  return key_;
}
const metal::BufferHandle &KVPagePool::valueBuffer() const noexcept {
  return value_;
}

std::int32_t KVPagePool::allocatePage() {
  if (freePages_.empty()) return -1;
  const auto page = freePages_.back();
  freePages_.pop_back();
  allocated_[static_cast<std::size_t>(page)] = true;
  ++allocationCount_;
  peakUsedPages_ = std::max(peakUsedPages_, usedPages());
  return page;
}

void KVPagePool::releasePage(std::int32_t page) {
  if (page < 0 || static_cast<std::size_t>(page) >= allocated_.size() ||
      !allocated_[static_cast<std::size_t>(page)]) {
    throw std::invalid_argument("Paged KV attempted to release an invalid page.");
  }
  allocated_[static_cast<std::size_t>(page)] = false;
  freePages_.push_back(page);
  ++releaseCount_;
}

bool KVPagePool::compatible(const planner::KVCachePlan &plan) const noexcept {
  return plan.batch == plan_.batch && plan.heads == plan_.heads &&
         plan.headDimension == plan_.headDimension && plan.dtype == plan_.dtype;
}

KVPagePoolCreation createKVPagePool(metal::MetalRuntime &runtime,
                                    const planner::KVCachePlan &plan,
                                    std::size_t pageSize,
                                    std::size_t physicalPageCount) {
  KVPagePoolCreation result;
  try {
    plan.validate();
    if (plan.batch != 1) {
      throw std::invalid_argument("Shared paged KV pool supports one tensor batch per request.");
    }
    if (pageSize == 0 || pageSize > plan.capacity || physicalPageCount == 0 ||
        physicalPageCount > static_cast<std::size_t>(
                                std::numeric_limits<std::int32_t>::max())) {
      throw std::invalid_argument("Shared paged KV pool dimensions are invalid.");
    }
    auto storageCount = checkedMultiply(physicalPageCount, pageSize);
    storageCount = checkedMultiply(storageCount, plan.batch);
    storageCount = checkedMultiply(storageCount, plan.heads);
    storageCount = checkedMultiply(storageCount, plan.headDimension);
    auto key = runtime.createBuffer(storageCount);
    if (!key.buffer) throw std::runtime_error(key.errorMessage);
    auto value = runtime.createBuffer(storageCount);
    if (!value.buffer) throw std::runtime_error(value.errorMessage);
    result.pool = std::shared_ptr<KVPagePool>(new KVPagePool(
        plan, pageSize, std::move(key.buffer), std::move(value.buffer),
        physicalPageCount));
  } catch (const std::exception &error) {
    result.errorMessage = error.what();
  }
  return result;
}

KVCacheState::KVCacheState(planner::KVCachePlan plan,
                           metal::BufferHandle key,
                           metal::BufferHandle value,
                           metal::BufferHandle length,
                           metal::BufferHandle blockTable,
                           std::shared_ptr<KVPagePool> pagePool)
    : plan_(std::move(plan)), key_(std::move(key)), value_(std::move(value)),
      length_(std::move(length)), blockTable_(std::move(blockTable)),
      initialKey_(key_.get()), initialValue_(value_.get()),
      pageSize_(pagePool ? pagePool->pageSize() : 0),
      pagePool_(std::move(pagePool)) {
  if (pageSize_ != 0) {
    const auto pageCount = (plan_.capacity + pageSize_ - 1) / pageSize_;
    logicalToPhysical_.assign(pageCount, -1);
  }
}

KVCacheState::~KVCacheState() {
  if (!pagePool_) return;
  for (auto &page : logicalToPhysical_) {
    if (page < 0) continue;
    try {
      pagePool_->releasePage(page);
    } catch (...) {
    }
    page = -1;
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
  const auto previousPageCount = allocatedPageCount();
  if (isPaged()) {
    const auto requiredPages = (length + pageSize_ - 1) / pageSize_;
    if (requiredPages > logicalToPhysical_.size()) {
      return "Paged KV cache has no block-table capacity for this length.";
    }
    try {
      resizePageAllocation(requiredPages);
    } catch (const std::exception &error) {
      return error.what();
    }
    auto error = writeBlockTable(runtime);
    if (!error.empty()) {
      resizePageAllocation(previousPageCount);
      return error;
    }
  }
  const float value = static_cast<float>(length);
  const auto error = runtime.writeBuffer(length_, &value, 1);
  if (!error.empty() && isPaged()) {
    resizePageAllocation(previousPageCount);
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

std::string KVCacheState::restore(
    metal::MetalRuntime &runtime, std::size_t length,
    const std::vector<float> &keyPrefix,
    const std::vector<float> &valuePrefix) {
  if (!isPaged()) return "KV cache restore requires paged storage.";
  if (currentLength_ != 0) return "KV cache restore requires an empty request state.";
  const auto expected = plan_.batch * plan_.heads * length * plan_.headDimension;
  if (length == 0 || length > plan_.capacity || keyPrefix.size() != expected ||
      valuePrefix.size() != expected) {
    return "KV cache restore snapshot shape is invalid.";
  }
  auto error = stageLength(runtime, length);
  if (error.empty()) {
    error = writePagedPrefix(runtime, key_, keyPrefix, length);
  }
  if (error.empty()) {
    error = writePagedPrefix(runtime, value_, valuePrefix, length);
  }
  if (!error.empty()) {
    (void)rollbackLength(runtime, 0);
    return error;
  }
  commitLength(length);
  return {};
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
  std::size_t count = 0;
  while (count < logicalToPhysical_.size() &&
         logicalToPhysical_[count] >= 0) {
    ++count;
  }
  return count;
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

std::string KVCacheState::writePagedPrefix(
    metal::MetalRuntime &runtime, const metal::BufferHandle &buffer,
    const std::vector<float> &prefix, std::size_t length) {
  for (std::size_t batch = 0; batch < plan_.batch; ++batch) {
    for (std::size_t head = 0; head < plan_.heads; ++head) {
      for (std::size_t position = 0; position < length; ++position) {
        const auto logicalPage = position / pageSize_;
        const auto pageOffset = position % pageSize_;
        const auto physicalPage = static_cast<std::size_t>(
            logicalToPhysical_.at(logicalPage));
        const auto source =
            ((batch * plan_.heads + head) * length + position) *
            plan_.headDimension;
        const auto destination =
            (((physicalPage * plan_.batch + batch) * plan_.heads + head) *
                 pageSize_ +
             pageOffset) *
            plan_.headDimension;
        const auto error = runtime.writeBuffer(
            buffer, prefix.data() + source, plan_.headDimension, destination);
        if (!error.empty()) return error;
      }
    }
  }
  return {};
}

void KVCacheState::resizePageAllocation(std::size_t pageCount) {
  const auto allocated = allocatedPageCount();
  if (pageCount > allocated) {
    std::size_t logical = allocated;
    for (; logical < pageCount; ++logical) {
      const auto physical = pagePool_->allocatePage();
      if (physical < 0) break;
      logicalToPhysical_[logical] = physical;
    }
    if (logical != pageCount) {
      while (logical > allocated) {
        --logical;
        pagePool_->releasePage(logicalToPhysical_[logical]);
        logicalToPhysical_[logical] = -1;
      }
      throw std::runtime_error("Paged KV cache physical page pool is exhausted.");
    }
    return;
  }
  for (std::size_t logical = allocated; logical > pageCount; --logical) {
    const auto index = logical - 1;
    pagePool_->releasePage(logicalToPhysical_[index]);
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
    if (pageSize != 0) {
      const auto pageCount = (plan.capacity + pageSize - 1) / pageSize;
      auto pool = createKVPagePool(runtime, plan, pageSize, pageCount);
      if (!pool.pool) throw std::runtime_error(pool.errorMessage);
      return createKVCacheState(runtime, plan, std::move(pool.pool));
    }
    const auto storageCount = plan.cacheElementCount();
    auto key = runtime.createBuffer(storageCount);
    if (!key.buffer) throw std::runtime_error(key.errorMessage);
    auto value = runtime.createBuffer(storageCount);
    if (!value.buffer) throw std::runtime_error(value.errorMessage);
    const float zero = 0.0f;
    auto length = runtime.createBuffer(1, &zero, metal::ElementType::Int32);
    if (!length.buffer) throw std::runtime_error(length.errorMessage);
    result.state = std::unique_ptr<KVCacheState>(new KVCacheState(
        plan, std::move(key.buffer), std::move(value.buffer),
        std::move(length.buffer), {}, {}));
  } catch (const std::exception &error) {
    result.errorMessage = error.what();
  }
  return result;
}

KVCacheStateCreation createKVCacheState(
    metal::MetalRuntime &runtime, const planner::KVCachePlan &plan,
    std::shared_ptr<KVPagePool> pagePool) {
  KVCacheStateCreation result;
  try {
    plan.validate();
    if (!pagePool || !pagePool->compatible(plan)) {
      throw std::invalid_argument("Paged KV pool is incompatible with the request plan.");
    }
    const auto logicalPageCount =
        (plan.capacity + pagePool->pageSize_ - 1) / pagePool->pageSize_;
    if (logicalPageCount > pagePool->capacityPages()) {
      throw std::invalid_argument(
          "Paged KV pool cannot hold one request at full capacity.");
    }
    const float zero = 0.0f;
    auto length = runtime.createBuffer(1, &zero, metal::ElementType::Int32);
    if (!length.buffer) throw std::runtime_error(length.errorMessage);
    std::vector<float> initialTable(logicalPageCount, -1.0f);
    auto table = runtime.createBuffer(logicalPageCount, initialTable.data(),
                                      metal::ElementType::Int32);
    if (!table.buffer) throw std::runtime_error(table.errorMessage);
    result.state = std::unique_ptr<KVCacheState>(new KVCacheState(
        plan, pagePool->key_, pagePool->value_, std::move(length.buffer),
        std::move(table.buffer), std::move(pagePool)));
  } catch (const std::exception &error) {
    result.errorMessage = error.what();
  }
  return result;
}

} // namespace tensor::runtime
