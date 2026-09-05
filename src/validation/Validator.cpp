#include "validation/Validator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace tensor::validation {

ValidationResult compare(const std::vector<float> &actual,
                         const std::vector<double> &reference,
                         double absoluteTolerance, double relativeTolerance) {
  ValidationResult result;
  if (reference.empty() || actual.size() != reference.size()) {
    result.errorMessage = "Empty reference or unexpected output length.";
    return result;
  }
  for (std::size_t i = 0; i < reference.size(); ++i) {
    const double expected = reference[i];
    const double value = actual[i];
    const bool finite = std::isfinite(value) && std::isfinite(expected);
    const double error = finite ? std::abs(value - expected)
                                : std::numeric_limits<double>::infinity();
    result.maxAbsoluteError = std::max(result.maxAbsoluteError, error);
    if (!finite || error > absoluteTolerance + relativeTolerance * std::abs(expected)) {
      if (result.mismatchCount == 0) {
        std::ostringstream message;
        message << "First mismatch at index " << i << ": expected=" << expected
                << ", actual=" << value;
        result.errorMessage = message.str();
      }
      ++result.mismatchCount;
    }
  }
  result.passed = result.mismatchCount == 0;
  return result;
}

std::vector<double> rmsNormReference(const RMSNormOp &op,
                                     const std::vector<float> &input,
                                     const std::vector<float> &weight) {
  op.validate();
  if (input.size() != op.input.elementCount() ||
      weight.size() != op.weight.elementCount()) {
    throw std::invalid_argument("RMSNorm host inputs do not match TensorIR shapes.");
  }
  const std::size_t width = op.input.shape.back();
  const std::size_t rows = input.size() / width;
  std::vector<double> reference(input.size());
  for (std::size_t row = 0; row < rows; ++row) {
    double sumSquares = 0.0;
    for (std::size_t column = 0; column < width; ++column) {
      const double x = input[row * width + column];
      sumSquares += x * x;
    }
    const double inverseRms =
        1.0 / std::sqrt(sumSquares / static_cast<double>(width) + op.epsilon);
    for (std::size_t column = 0; column < width; ++column) {
      reference[row * width + column] =
          static_cast<double>(input[row * width + column]) * inverseRms *
          static_cast<double>(weight[column]);
    }
  }
  return reference;
}

} // namespace tensor::validation
