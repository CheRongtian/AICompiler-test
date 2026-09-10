#pragma once

#include "backend/metal/MetalEmitter.hpp"
#include "backend/metal/MetalRuntime.hpp"

#include <cstddef>
#include <string>

namespace tensor::metal {

struct DecodeGEMVConfig {
  std::size_t threads = 128;
  std::size_t vectorWidth = 1;
  std::size_t tileSize = 0;
  bool simdgroupReduction = false;
};

[[nodiscard]] GeneratedKernel emitTiledGEMM(
    std::size_t rows, std::size_t inputSize, std::size_t outputSize,
    std::size_t tileSize, const std::string &functionName,
    ElementType storageType = ElementType::Float32,
    ElementType outputType = ElementType::Float32);

[[nodiscard]] GeneratedKernel emitLinearBaseline(
    std::size_t batch, std::size_t inputSize, std::size_t outputSize,
    std::size_t threads, const std::string &functionName,
    ElementType storageType = ElementType::Float32,
    std::size_t vectorWidth = 1,
    ElementType outputType = ElementType::Float32);

[[nodiscard]] GeneratedKernel emitDecodeGEMV(
    std::size_t batch, std::size_t inputSize, std::size_t outputSize,
    DecodeGEMVConfig config, const std::string &functionName,
    ElementType storageType = ElementType::Float32,
    ElementType outputType = ElementType::Float32);

} // namespace tensor::metal
