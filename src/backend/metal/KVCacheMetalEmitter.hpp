#pragma once

#include "backend/metal/MetalEmitter.hpp"
#include "planner/KVCachePlan.hpp"

#include <cstddef>
#include <string>

namespace tensor::metal {

[[nodiscard]] GeneratedKernel
emitKVQueryProjection(const planner::KVCachePlan &plan,
                      std::size_t queryLength);

[[nodiscard]] GeneratedKernel
emitKVCacheProjection(const planner::KVCachePlan &plan,
                      std::size_t queryLength, bool keyCache);

[[nodiscard]] GeneratedKernel
emitKVAttention(const planner::KVCachePlan &plan,
                std::size_t queryLength, bool causalPrefill = false);

[[nodiscard]] GeneratedKernel
emitPagedKVAttention(const planner::KVCachePlan &plan,
                     std::size_t queryLength, std::size_t pageSize,
                     bool causalPrefill = false);

[[nodiscard]] GeneratedKernel emitServingPagedKVAttention(
    const planner::KVCachePlan &plan, std::size_t maximumBatchSize,
    std::size_t pageSize, const std::string &functionName);

[[nodiscard]] GeneratedKernel
emitKVOutputProjection(const planner::KVCachePlan &plan,
                       std::size_t queryLength);

} // namespace tensor::metal
