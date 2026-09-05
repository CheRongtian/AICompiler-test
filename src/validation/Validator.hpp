#pragma once

#include "tensor/TensorIR.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace tensor::validation {

struct ValidationResult {
  bool passed = false;
  double maxAbsoluteError = 0.0;
  std::size_t mismatchCount = 0;
  std::string errorMessage;
};

[[nodiscard]] ValidationResult compare(const std::vector<float> &actual,
                                       const std::vector<double> &reference,
                                       double absoluteTolerance,
                                       double relativeTolerance);

[[nodiscard]] std::vector<double>
rmsNormReference(const RMSNormOp &op, const std::vector<float> &input,
                 const std::vector<float> &weight);

} // namespace tensor::validation
