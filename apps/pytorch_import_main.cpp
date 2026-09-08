#include "pytorch_import_main.hpp"

#include "importer/PyTorchImporter.hpp"
#include "benchmark/Benchmark.hpp"
#include "llm/AdvisorProtocol.hpp"
#include "planner/KernelPlan.hpp"
#include "planner/RegionPlan.hpp"
#include "runtime/GraphExecutor.hpp"
#include "validation/Validator.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <iomanip>
#include <optional>
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

void printTiming(const std::string &label,
                 const std::vector<double> &values, std::ostream &log) {
  const auto stats = tensor::benchmark::summarizeTimings(values);
  if (!stats) {
    log << label << ": unavailable\n";
    return;
  }
  log << label << ": p50=" << stats->medianUs
      << ", p90=" << stats->p90Us << ", min=" << stats->minUs
      << ", max=" << stats->maxUs << ", samples=" << stats->samples
      << '\n';
}

bool benchmarkGraph(const tensor::runtime::CompiledGraph &graph,
                    const PyTorchAdvisorOptions &options,
                    std::ostream &log) {
  if (options.benchmarkMeasuredRuns == 0) return true;
  for (std::size_t run = 0; run < options.benchmarkWarmupRuns; ++run) {
    const auto result = graph.run();
    if (!result.passed) {
      log << "Imported graph benchmark warmup: FAIL\n"
          << "Metal error: " << result.errorMessage << '\n';
      return false;
    }
  }

  std::vector<double> gpuTimes;
  std::vector<double> cpuTimes;
  std::vector<double> endToEndTimes;
  for (std::size_t run = 0; run < options.benchmarkMeasuredRuns; ++run) {
    const auto start = std::chrono::steady_clock::now();
    const auto result = graph.run();
    const auto end = std::chrono::steady_clock::now();
    if (!result.passed) {
      log << "Imported graph benchmark: FAIL\n"
          << "Metal error: " << result.errorMessage << '\n';
      return false;
    }
    if (result.gpuExecutionTimeUs) gpuTimes.push_back(*result.gpuExecutionTimeUs);
    cpuTimes.push_back(result.cpuSubmitToCompletionTimeUs);
    endToEndTimes.push_back(
        std::chrono::duration<double, std::micro>(end - start).count());
  }

  log << "Imported graph benchmark: "
      << (options.benchmarkLabel.empty() ? "unnamed" : options.benchmarkLabel)
      << '\n'
      << "Warmup runs: " << options.benchmarkWarmupRuns
      << ", measured runs: " << options.benchmarkMeasuredRuns << '\n';
  printTiming("  GPU command time (us)", gpuTimes, log);
  printTiming("  CPU submit-to-completion time (us)", cpuTimes, log);
  printTiming("  End-to-end time (us)", endToEndTimes, log);
  log << "Imported graph benchmark validation: PASS\n";
  return true;
}

} // namespace

bool runImportedPyTorchGraph(tensor::metal::MetalRuntime &runtime,
                             const std::string &manifestPath,
                             std::ostream &log,
                             const PyTorchAdvisorOptions &advisor) {
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

  if (!advisor.requestOutputPath.empty()) {
    auto program = tensor::planner::planRegions(tensor::planner::planGraph(
        tensor::analyzer::analyze(imported.model->graph)));
    const auto request = tensor::llm::makeAdvisorRequest(
        std::move(program), runtime.deviceName(), runtime.hardwareInfo());
    const auto error = tensor::llm::writeAdvisorRequest(
        request, advisor.requestOutputPath);
    if (!error.empty()) {
      log << "Advisor request: FAIL\n"
          << "Advisor error: " << error << '\n';
      return false;
    }
    log << "Advisor request: PASS\n"
        << "Advisor regions: " << request.program.regions.size() << '\n'
        << "Advisor legal candidates: " << request.candidates.size() << '\n'
        << "Advisor request file: " << advisor.requestOutputPath << '\n';
    return true;
  }

  std::optional<tensor::llm::AdvisorResponse> response;
  if (!advisor.responsePath.empty()) {
    auto loaded = tensor::llm::loadAdvisorResponse(advisor.responsePath);
    if (loaded.response) {
      response = std::move(loaded.response);
      log << "Advisor response parse: PASS\n";
    } else {
      log << "Advisor response parse: FAIL\n"
          << "Advisor: FALLBACK\n"
          << "Fallback reason: " << loaded.errorMessage << '\n'
          << "Planner mode: deterministic fallback\n";
    }
  }

  auto compilation = tensor::runtime::compileGraph(
      runtime, imported.model->graph, imported.model->inputs, log,
      response ? &*response : nullptr);
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
  if (!metalPassed) return false;
  return benchmarkGraph(*compilation.executable, advisor, log);
}
