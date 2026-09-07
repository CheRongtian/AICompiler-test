#pragma once

#include "backend/metal/MetalEmitter.hpp"
#include "planner/TransformerDecodePlan.hpp"

#include <cstddef>
#include <string>

namespace tensor::metal {

[[nodiscard]] GeneratedKernel emitTokenEmbedding(
    const planner::TransformerDecodePlan &plan, std::size_t sequenceLength,
    const std::string &functionName);

[[nodiscard]] GeneratedKernel emitBiasedLinear(
    std::size_t rows, std::size_t inputDimension,
    std::size_t outputDimension, bool relu,
    std::size_t threadsPerThreadgroup, const std::string &functionName);

[[nodiscard]] GeneratedKernel emitResidualLayerNorm(
    std::size_t rows, std::size_t width, float epsilon,
    std::size_t threadsPerThreadgroup, const std::string &functionName);

[[nodiscard]] GeneratedKernel emitCrossAttention(
    const planner::TransformerDecodePlan &plan, std::size_t queryLength,
    const std::string &functionName);

[[nodiscard]] GeneratedKernel emitTokenArgmax(
    std::size_t rows, std::size_t vocabularySize,
    std::size_t threadsPerThreadgroup, const std::string &functionName);

} // namespace tensor::metal
