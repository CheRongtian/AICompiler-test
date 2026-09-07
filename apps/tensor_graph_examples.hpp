#pragma once

#include "backend/metal/MetalRuntime.hpp"

#include <iosfwd>

bool runTensorGraphExamples(tensor::metal::MetalRuntime &runtime, std::ostream &log);
