#pragma once

#include "backend/metal/MetalRuntime.hpp"

#include <cstddef>
#include <iosfwd>
#include <string>

[[nodiscard]] bool runPagedKVWorkload(
    tensor::metal::MetalRuntime &runtime, const std::string &manifestPath,
    std::size_t pageSize, std::size_t chunkSize, std::ostream &log,
    const std::string &kernelLibrary = {});
