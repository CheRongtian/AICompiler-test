#pragma once

#include "backend/metal/MetalEmitter.hpp"

#include <cstddef>
#include <string>

namespace tensor::metal {

struct DecodeGEMVConfig {
  std::size_t threads = 128;
  std::size_t vectorWidth = 1;
};

[[nodiscard]] GeneratedKernel emitLinearBaseline(
    std::size_t batch, std::size_t inputSize, std::size_t outputSize,
    std::size_t threads, const std::string &functionName);

[[nodiscard]] GeneratedKernel emitDecodeGEMV(
    std::size_t batch, std::size_t inputSize, std::size_t outputSize,
    DecodeGEMVConfig config, const std::string &functionName);

} // namespace tensor::metal
