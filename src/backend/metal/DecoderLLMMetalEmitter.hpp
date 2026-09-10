#pragma once

#include "backend/metal/MetalEmitter.hpp"
#include "backend/metal/MetalRuntime.hpp"
#include "planner/DecoderLLMPlan.hpp"

#include <cstddef>
#include <string>

namespace tensor::metal {

[[nodiscard]] GeneratedKernel emitDecoderEmbedding(
    const planner::DecoderLLMPlan &plan, std::size_t sequenceLength,
    const std::string &functionName,
    ElementType storageType = ElementType::Float32);

[[nodiscard]] GeneratedKernel emitDecoderRMSNorm(
    std::size_t rows, std::size_t width, float epsilon,
    std::size_t threads, const std::string &functionName,
    ElementType storageType = ElementType::Float32);

[[nodiscard]] GeneratedKernel emitDecoderRoPE(
    const planner::DecoderLLMPlan &plan, std::size_t sequenceLength,
    const std::string &functionName,
    ElementType storageType = ElementType::Float32);

[[nodiscard]] GeneratedKernel emitDecoderCacheAppend(
    const planner::DecoderLLMPlan &plan, std::size_t sequenceLength,
    bool inputIsHeadMajor, const std::string &functionName,
    ElementType storageType = ElementType::Float32);

[[nodiscard]] GeneratedKernel emitDecoderPagedCacheAppend(
    const planner::DecoderLLMPlan &plan, std::size_t sequenceLength,
    bool inputIsHeadMajor, std::size_t pageSize,
    const std::string &functionName,
    ElementType storageType = ElementType::Float32);

[[nodiscard]] GeneratedKernel emitServingRoPE(
    const planner::DecoderLLMPlan &plan, std::size_t maximumBatchSize,
    const std::string &functionName,
    ElementType storageType = ElementType::Float32);

[[nodiscard]] GeneratedKernel emitServingPagedCacheAppend(
    const planner::DecoderLLMPlan &plan, std::size_t maximumBatchSize,
    bool inputIsHeadMajor, std::size_t pageSize,
    const std::string &functionName,
    ElementType storageType = ElementType::Float32);

[[nodiscard]] GeneratedKernel emitDecoderAdd(
    std::size_t elementCount, std::size_t threads,
    const std::string &functionName,
    ElementType storageType = ElementType::Float32);

[[nodiscard]] GeneratedKernel emitDecoderSiLUMul(
    std::size_t elementCount, std::size_t threads,
    const std::string &functionName,
    ElementType storageType = ElementType::Float32);

[[nodiscard]] GeneratedKernel emitDecoderResidualRMSNorm(
    std::size_t rows, std::size_t width, float epsilon,
    std::size_t threads, const std::string &functionName,
    ElementType storageType = ElementType::Float32);

[[nodiscard]] GeneratedKernel emitDecoderGatedMLP(
    std::size_t rows, std::size_t hiddenSize, std::size_t intermediateSize,
    std::size_t threads, const std::string &functionName,
    ElementType storageType = ElementType::Float32);

[[nodiscard]] GeneratedKernel emitLMHeadPartialArgmax(
    std::size_t rows, std::size_t hidden, std::size_t vocabulary,
    std::size_t threads, const std::string &name, ElementType storageType);
[[nodiscard]] GeneratedKernel emitLMHeadFinalArgmax(
    std::size_t rows, std::size_t chunks, std::size_t threads,
    const std::string &name);

[[nodiscard]] GeneratedKernel emitDecoderQKVFusion(
    const planner::DecoderLLMPlan &plan, std::size_t sequenceLength,
    bool normalize, const std::string &name, ElementType storageType);
[[nodiscard]] GeneratedKernel emitLinearResidual(
    std::size_t rows, std::size_t inputSize, std::size_t outputSize,
    std::size_t threads, const std::string &name, ElementType storageType);
[[nodiscard]] GeneratedKernel emitPagedAttentionOutputProjection(
    const planner::DecoderLLMPlan &plan, std::size_t queryLength,
    std::size_t pageSize, const std::string &name, ElementType storageType);

} // namespace tensor::metal
