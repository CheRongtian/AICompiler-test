#pragma once

#include "backend/metal/MetalEmitter.hpp"
#include "planner/DecoderLLMPlan.hpp"

#include <cstddef>
#include <string>

namespace tensor::metal {

[[nodiscard]] GeneratedKernel emitDecoderEmbedding(
    const planner::DecoderLLMPlan &plan, std::size_t sequenceLength,
    const std::string &functionName);

[[nodiscard]] GeneratedKernel emitDecoderRMSNorm(
    std::size_t rows, std::size_t width, float epsilon,
    std::size_t threads, const std::string &functionName);

[[nodiscard]] GeneratedKernel emitDecoderRoPE(
    const planner::DecoderLLMPlan &plan, std::size_t sequenceLength,
    const std::string &functionName);

[[nodiscard]] GeneratedKernel emitDecoderCacheAppend(
    const planner::DecoderLLMPlan &plan, std::size_t sequenceLength,
    bool inputIsHeadMajor, const std::string &functionName);

[[nodiscard]] GeneratedKernel emitDecoderPagedCacheAppend(
    const planner::DecoderLLMPlan &plan, std::size_t sequenceLength,
    bool inputIsHeadMajor, std::size_t pageSize,
    const std::string &functionName);

[[nodiscard]] GeneratedKernel emitServingRoPE(
    const planner::DecoderLLMPlan &plan, std::size_t maximumBatchSize,
    const std::string &functionName);

[[nodiscard]] GeneratedKernel emitServingPagedCacheAppend(
    const planner::DecoderLLMPlan &plan, std::size_t maximumBatchSize,
    bool inputIsHeadMajor, std::size_t pageSize,
    const std::string &functionName);

[[nodiscard]] GeneratedKernel emitDecoderAdd(
    std::size_t elementCount, std::size_t threads,
    const std::string &functionName);

[[nodiscard]] GeneratedKernel emitDecoderSiLUMul(
    std::size_t elementCount, std::size_t threads,
    const std::string &functionName);

} // namespace tensor::metal
