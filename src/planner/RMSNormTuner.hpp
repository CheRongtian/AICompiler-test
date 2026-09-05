#pragma once

#include "backend/metal/MetalRuntime.hpp"
#include "tensor/TensorIR.hpp"
#include "validation/Validator.hpp"

#include <iosfwd>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace tensor::planner {

struct CandidateReport {
  std::string name;
  std::size_t threads = 0;
  std::string outcome;
  std::string reason;
  std::optional<double> medianGpuUs;
  std::optional<double> pairedBaselineUs;
  std::optional<double> speedup;
};

struct TuningResult {
  bool success = false;
  bool usedBaseline = true;
  std::size_t selectedThreads = 256;
  CandidateReport baseline;
  std::vector<CandidateReport> candidates;
  // Selection is executable, with retained pipeline and the supplied input data.
  // Admission is scoped to this concrete op/shape/dtype/epsilon and input data.
  std::unique_ptr<metal::PreparedExecution> selectedExecution;
  metal::ExecutionResult finalExecution;
  validation::ValidationResult finalValidation;
  std::string errorMessage;
};

[[nodiscard]] TuningResult tuneRMSNorm(metal::MetalRuntime &runtime,
                                     const RMSNormOp &op,
                                     const std::vector<float> &input,
                                     const std::vector<float> &weight,
                                     std::ostream &log);

} // namespace tensor::planner
