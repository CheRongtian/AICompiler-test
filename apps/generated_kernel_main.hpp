#pragma once

#include "backend/metal/MetalRuntime.hpp"

#include <iosfwd>
#include <string>

bool emitGeneratedKernelContract(const tensor::metal::MetalRuntime &runtime,
                                 const std::string &path,
                                 std::ostream &log,
                                 const std::string &pattern = "silu_mul");
bool runGeneratedKernelAdmission(tensor::metal::MetalRuntime &runtime,
                                 const std::string &responsePath,
                                 const std::string &feedbackPath,
                                 std::ostream &log,
                                 const std::string &pattern = "silu_mul",
                                 const std::string &artifactPath = {});
