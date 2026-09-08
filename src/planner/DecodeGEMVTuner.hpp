#pragma once

#include "backend/metal/DecodeGEMVMetalEmitter.hpp"
#include "backend/metal/MetalRuntime.hpp"

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace tensor::planner {

struct DecodeGEMVSelection {
  bool success = false;
  bool useBaseline = true;
  metal::DecodeGEMVConfig config;
  double conservativeSpeedup = 1.0;
  std::string errorMessage;
};

[[nodiscard]] DecodeGEMVSelection tuneDecodeGEMV(
    metal::MetalRuntime &runtime, std::size_t batch, std::size_t inputSize,
    std::size_t outputSize, const std::vector<float> &input,
    const std::vector<float> &weight, std::ostream &log);

} // namespace tensor::planner
