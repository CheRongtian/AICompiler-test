#include "pytorch_import_main.hpp"

#include "importer/PyTorchImporter.hpp"
#include "runtime/GraphExecutor.hpp"
#include "validation/Validator.hpp"

#include <algorithm>
#include <array>
#include <iomanip>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace {

const char *passFail(bool passed) { return passed ? "PASS" : "FAIL"; }

std::pair<double, double> tolerance(tensor::DType dtype) {
  return dtype == tensor::DType::Float16 ? std::make_pair(3e-3, 3e-3)
                                         : std::make_pair(1e-4, 1e-3);
}

bool sameType(const tensor::TensorType &left, const tensor::TensorType &right) {
  return left.shape == right.shape && left.dtype == right.dtype;
}

} // namespace

bool runImportedPyTorchGraph(tensor::metal::MetalRuntime &runtime,
                             const std::string &manifestPath,
                             std::ostream &log) {
  auto imported = tensor::importer::importPyTorchGraph(manifestPath);
  if (!imported.model) {
    log << "PyTorch import: FAIL\n"
        << "Import error: " << imported.errorMessage << '\n';
    return false;
  }

  std::array<std::size_t, 4> kindCounts{};
  for (const auto &input : imported.model->importedInputs) {
    ++kindCounts[static_cast<std::size_t>(input.kind)];
  }
  log << "PyTorch import: PASS\n"
      << "Model: " << imported.model->modelName << '\n'
      << "Imported graph: values=" << imported.model->graph.values.size()
      << ", nodes=" << imported.model->graph.nodes.size()
      << ", outputs=" << imported.model->graph.outputs.size() << '\n'
      << "Imported inputs: user=" << kindCounts[0]
      << ", parameters=" << kindCounts[1]
      << ", buffers=" << kindCounts[2]
      << ", constants=" << kindCounts[3] << '\n';

  auto compilation = tensor::runtime::compileGraph(
      runtime, imported.model->graph, imported.model->inputs, log);
  if (!compilation.executable) {
    log << "Imported graph compilation: FAIL\n"
        << "Compiler error: " << compilation.errorMessage << '\n';
    return false;
  }
  if (compilation.outputTypes.size() != imported.model->expectedOutputTypes.size() ||
      compilation.referenceOutputs.size() != imported.model->expectedOutputs.size()) {
    log << "Imported graph interface: FAIL\n"
        << "Output count differs from the PyTorch reference.\n";
    return false;
  }

  bool referencePassed = true;
  for (std::size_t index = 0; index < compilation.outputTypes.size(); ++index) {
    if (!sameType(compilation.outputTypes[index],
                  imported.model->expectedOutputTypes[index])) {
      log << "TensorIR reference output " << index
          << ": FAIL (shape or dtype mismatch)\n";
      referencePassed = false;
      continue;
    }
    std::vector<float> actual(compilation.referenceOutputs[index].begin(),
                              compilation.referenceOutputs[index].end());
    const auto limits = tolerance(compilation.outputTypes[index].dtype);
    const auto validation = tensor::validation::compare(
        actual, imported.model->expectedOutputs[index], limits.first, limits.second);
    log << "TensorIR reference output " << index << ": "
        << passFail(validation.passed)
        << ", max absolute error=" << validation.maxAbsoluteError << '\n';
    if (!validation.passed) {
      log << "Reference error: " << validation.errorMessage << '\n';
      referencePassed = false;
    }
  }
  log << "TensorIR reference vs PyTorch: " << passFail(referencePassed) << '\n';
  if (!referencePassed) return false;

  const auto execution = compilation.executable->run();
  if (!execution.passed) {
    log << "Imported Metal execution: FAIL\n"
        << "Metal error: " << execution.errorMessage << '\n';
    return false;
  }
  if (execution.outputs.size() != imported.model->expectedOutputs.size()) {
    log << "Imported Metal execution: FAIL\n"
        << "Metal output count differs from the PyTorch reference.\n";
    return false;
  }

  bool metalPassed = true;
  for (std::size_t index = 0; index < execution.outputs.size(); ++index) {
    const auto limits = tolerance(compilation.outputTypes[index].dtype);
    const auto validation = tensor::validation::compare(
        execution.outputs[index], imported.model->expectedOutputs[index],
        limits.first, limits.second);
    log << "Metal output " << index << " vs PyTorch: "
        << passFail(validation.passed)
        << ", max absolute error=" << validation.maxAbsoluteError << '\n';
    if (!validation.passed) {
      log << "Validation error: " << validation.errorMessage << '\n';
      metalPassed = false;
    }
  }
  if (execution.gpuExecutionTimeUs) {
    log << "Imported graph GPU time (us): " << *execution.gpuExecutionTimeUs << '\n';
  }
  log << "PyTorch graph validation: " << passFail(metalPassed) << '\n';
  return metalPassed;
}
