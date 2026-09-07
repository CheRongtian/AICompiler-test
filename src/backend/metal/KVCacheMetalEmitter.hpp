#pragma once

#include "backend/metal/MetalEmitter.hpp"
#include "planner/KVCachePlan.hpp"

#include <cstddef>

namespace tensor::metal {

[[nodiscard]] GeneratedKernel
emitKVQueryProjection(const planner::KVCachePlan &plan,
                      std::size_t queryLength);

[[nodiscard]] GeneratedKernel
emitKVCacheProjection(const planner::KVCachePlan &plan,
                      std::size_t queryLength, bool keyCache);

[[nodiscard]] GeneratedKernel
emitKVAttention(const planner::KVCachePlan &plan,
                std::size_t queryLength);

[[nodiscard]] GeneratedKernel
emitKVOutputProjection(const planner::KVCachePlan &plan,
                       std::size_t queryLength);

} // namespace tensor::metal
