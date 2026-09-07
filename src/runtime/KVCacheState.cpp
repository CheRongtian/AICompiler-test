#include "runtime/KVCacheState.hpp"

#include <stdexcept>
#include <utility>

namespace tensor::runtime {

KVCacheState::KVCacheState(planner::KVCachePlan plan,
                           metal::BufferHandle key,
                           metal::BufferHandle value,
                           metal::BufferHandle length)
    : plan_(std::move(plan)), key_(std::move(key)), value_(std::move(value)),
      length_(std::move(length)), initialKey_(key_.get()),
      initialValue_(value_.get()) {}

std::size_t KVCacheState::currentLength() const noexcept { return currentLength_; }
std::size_t KVCacheState::capacity() const noexcept { return plan_.capacity; }

std::string KVCacheState::reset(metal::MetalRuntime &runtime) {
  const float zero = 0.0f;
  const auto error = runtime.writeBuffer(length_, &zero, 1);
  if (error.empty()) currentLength_ = 0;
  return error;
}

std::string KVCacheState::stageLength(metal::MetalRuntime &runtime,
                                      std::size_t length) {
  if (length == 0 || length > plan_.capacity) {
    return "KV cache valid length is outside the configured capacity.";
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
      const auto base = (batch * plan_.heads + head) * plan_.capacity *
                        plan_.headDimension;
      result.insert(result.end(), storage.begin() + base,
                    storage.begin() + base + currentLength_ * plan_.headDimension);
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

const metal::BufferHandle &KVCacheState::keyBuffer() const noexcept { return key_; }
const metal::BufferHandle &KVCacheState::valueBuffer() const noexcept { return value_; }
const metal::BufferHandle &KVCacheState::lengthBuffer() const noexcept { return length_; }

KVCacheStateCreation createKVCacheState(metal::MetalRuntime &runtime,
                                        const planner::KVCachePlan &plan) {
  KVCacheStateCreation result;
  try {
    plan.validate();
    auto key = runtime.createBuffer(plan.cacheElementCount());
    if (!key.buffer) throw std::runtime_error(key.errorMessage);
    auto value = runtime.createBuffer(plan.cacheElementCount());
    if (!value.buffer) throw std::runtime_error(value.errorMessage);
    const float zero = 0.0f;
    auto length = runtime.createBuffer(1, &zero, metal::ElementType::Int32);
    if (!length.buffer) throw std::runtime_error(length.errorMessage);
    result.state = std::unique_ptr<KVCacheState>(new KVCacheState(
        plan, std::move(key.buffer), std::move(value.buffer),
        std::move(length.buffer)));
  } catch (const std::exception &error) {
    result.errorMessage = error.what();
  }
  return result;
}

} // namespace tensor::runtime
