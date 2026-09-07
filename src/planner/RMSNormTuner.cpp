#include "planner/RMSNormTuner.hpp"

#include "backend/metal/MetalEmitter.hpp"
#include "benchmark/Benchmark.hpp"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <limits>
#include <ostream>
#include <utility>

namespace tensor::planner {
namespace {

constexpr std::size_t kWarmupIterations = 5;
constexpr std::size_t kSamples = 31;
constexpr double kMinimumSpeedup = 1.05;
constexpr double kAbsoluteTolerance = 1e-5;
constexpr double kRelativeTolerance = 1e-4;

void reject(CandidateReport &report, const std::string &stage,
             const std::string &reason, std::ostream &log) {
  report.outcome = stage + " FAIL";
  report.reason = reason;
  log << report.name << " | " << report.outcome << ": " << reason << '\n';
}

std::string hardwareFilter(const RMSNormOp &op, std::size_t threads,
                           const metal::HardwareInfo &hardware) {
  if (threads == 0 || (threads & (threads - 1)) != 0 ||
      threads > hardware.maxThreadsPerThreadgroup ||
      threads > hardware.maxThreadgroupMemoryLength / sizeof(float)) {
    return "Thread count or reduction scratch memory exceeds device limits.";
  }
  if (op.input.elementCount() > std::numeric_limits<std::uint32_t>::max() ||
      op.input.elementCount() > hardware.maxBufferLength / sizeof(float) ||
      op.weight.elementCount() > hardware.maxBufferLength / sizeof(float)) {
    return "Tensor size exceeds the kernel index or device buffer limit.";
  }
  return {};
}

std::string interfaceCheck(const RMSNormOp &op, const metal::GeneratedKernel &kernel,
                           const metal::ComputePipelineResult &pipeline,
                           const metal::HardwareInfo &hardware,
                           std::size_t expectedThreads) {
  const auto bindingError = metal::checkFloatBufferInterface(pipeline, 2);
  if (!bindingError.empty()) return bindingError;
  if (kernel.threadgroupCount != op.input.elementCount() / op.input.shape.back() ||
      kernel.threadsPerThreadgroup != expectedThreads ||
      expectedThreads > pipeline.maxTotalThreadsPerThreadgroup ||
      pipeline.staticThreadgroupMemoryLength > hardware.maxThreadgroupMemoryLength) {
    return "Dispatch metadata or compiled pipeline resource limits do not match the plan.";
  }
  return {};
}

validation::ValidationResult validateExecution(const metal::ExecutionResult &execution,
                                               const std::vector<double> &reference) {
  if (!execution.executionPassed) {
    validation::ValidationResult result;
    result.errorMessage = execution.errorMessage;
    return result;
  }
  return validation::compare(execution.output, reference,
                             kAbsoluteTolerance, kRelativeTolerance);
}

std::unique_ptr<metal::PreparedExecution>
buildAndValidate(metal::MetalRuntime &runtime, const RMSNormOp &op,
                  const std::vector<float> &input, const std::vector<float> &weight,
                  const std::vector<double> &reference, bool baseline,
                  CandidateReport &report, std::ostream &log) {
  const auto hardware = runtime.hardwareInfo();
  std::string error = hardwareFilter(op, report.threads, hardware);
  if (!error.empty()) {
    reject(report, "Hardware Filter", error, log);
    return {};
  }
  log << report.name << " | Hardware Filter: PASS\n";

  metal::GeneratedKernel kernel;
  try {
    kernel = baseline ? metal::emitRMSNormBaseline(op)
                      : metal::emitRMSNorm(op, report.threads);
  } catch (const std::exception &exception) {
    reject(report, "Emit", exception.what(), log);
    return {};
  }
  log << report.name << " | Emit: PASS\n";
  const auto pipeline = runtime.createComputePipeline(kernel.source, kernel.functionName);
  log << report.name << " | Library compile: "
      << (pipeline.libraryCompilePassed ? "PASS" : "FAIL")
      << ", Kernel lookup: " << (pipeline.kernelLookupPassed ? "PASS" : "FAIL")
      << ", Pipeline: " << (pipeline.pipelineCreationPassed ? "PASS" : "FAIL") << '\n';
  if (!pipeline.pipelineCreationPassed) {
    reject(report, "Compile", pipeline.errorMessage, log);
    return {};
  }
  error = interfaceCheck(op, kernel, pipeline, hardware, report.threads);
  if (!error.empty()) {
    reject(report, "Interface Check", error, log);
    return {};
  }
  log << report.name << " | Interface Check: PASS\n";

  auto prepared = runtime.prepare(
      {{input.data(), input.size()}, {weight.data(), weight.size()}},
      op.output.elementCount(), {kernel.threadgroupCount, kernel.threadsPerThreadgroup});
  if (!prepared.execution) {
    reject(report, "Prepare", prepared.errorMessage, log);
    return {};
  }
  const auto validation = validateExecution(prepared.execution->run(), reference);
  if (!validation.passed) {
    reject(report, "Numerical Validate", validation.errorMessage, log);
    return {};
  }
  log << report.name << " | Numerical Validate: PASS, max absolute error="
      << validation.maxAbsoluteError << '\n';
  report.outcome = "Correctness PASS";
  return std::move(prepared.execution);
}

void printReport(const CandidateReport &report, std::ostream &log) {
  log << "  " << report.name << " | threads=" << report.threads;
  if (report.medianGpuUs) {
    log << " | GPU median(us)=" << *report.medianGpuUs;
  }
  if (report.pairedBaselineUs) {
    log << " | paired baseline(us)=" << *report.pairedBaselineUs;
  }
  if (report.speedup) {
    log << " | speedup=" << *report.speedup << 'x';
  }
  log << " | " << report.outcome;
  if (!report.reason.empty()) {
    log << " (" << report.reason << ')';
  }
  log << '\n';
}

} // namespace

TuningResult tuneRMSNorm(metal::MetalRuntime &runtime, const RMSNormOp &op,
                         const std::vector<float> &input,
                         const std::vector<float> &weight, std::ostream &log) {
  TuningResult result;
  op.validate();
  const auto reference = validation::rmsNormReference(op, input, weight);
  log << "Enumerate: threads={64, 128, 256}\n"
      << "CPU correctness baseline: double-accumulation reference\n"
      << "GPU performance baseline: fixed V0b 256-thread RMSNorm\n"
      << "Warmup=" << kWarmupIterations << ", samples=" << kSamples
      << ", admission requires >= " << kMinimumSpeedup << "x in two paired rounds\n"
      << "Metric: one-dispatch GPU command-buffer duration; compile, allocation, "
         "CPU encoding and readback excluded\n"
      << "Numerical tolerance: atol=" << kAbsoluteTolerance
      << ", rtol=" << kRelativeTolerance << '\n';

  result.baseline.name = "baseline";
  result.baseline.threads = 256;
  auto baseline = buildAndValidate(runtime, op, input, weight, reference,
                                   true, result.baseline, log);
  if (!baseline) {
    result.errorMessage = "Baseline failed; no valid fallback: " + result.baseline.reason;
    return result;
  }
  const std::string warmupError = benchmark::warmup(*baseline, kWarmupIterations);
  if (!warmupError.empty()) {
    reject(result.baseline, "Warmup", warmupError, log);
    result.errorMessage = "Baseline warmup failed; no valid fallback: " + warmupError;
    return result;
  }
  log << "baseline | Warmup: PASS\n";
  const auto baselineTiming = benchmark::measure(*baseline, kSamples);
  if (baselineTiming.passed) {
    result.baseline.medianGpuUs = baselineTiming.stats.medianUs;
    result.baseline.outcome = "Fallback ready";
    log << "baseline | Benchmark: PASS, median(us)="
        << baselineTiming.stats.medianUs << '\n';
  } else {
    result.baseline.outcome = "Performance unavailable";
    result.baseline.reason = baselineTiming.errorMessage;
    log << "baseline | Benchmark unavailable: " << baselineTiming.errorMessage << '\n';
  }

  std::unique_ptr<metal::PreparedExecution> winner;
  std::size_t winnerIndex = 0;
  double bestSpeedup = 1.0;
  for (const std::size_t threads : {64u, 128u, 256u}) {
    CandidateReport report;
    report.name = "candidate_" + std::to_string(threads);
    report.threads = threads;
    result.candidates.push_back(std::move(report));
    auto &candidateReport = result.candidates.back();

    auto candidate = buildAndValidate(runtime, op, input, weight, reference,
                                      false, candidateReport, log);
    if (!candidate) {
      continue;
    }
    const std::string error = benchmark::warmup(*candidate, kWarmupIterations);
    if (!error.empty()) {
      reject(candidateReport, "Warmup", error, log);
      continue;
    }
    log << candidateReport.name << " | Warmup: PASS\n";
    if (!baselineTiming.passed) {
      candidateReport.outcome = "Not admitted";
      candidateReport.reason = "No usable baseline timing; performance comparison skipped.";
      continue;
    }

    const auto first = benchmark::measurePair(*baseline, *candidate, kSamples);
    if (!first.passed) {
      reject(candidateReport, "Benchmark", first.errorMessage, log);
      continue;
    }
    candidateReport.medianGpuUs = first.candidate.medianUs;
    candidateReport.pairedBaselineUs = first.baseline.medianUs;
    candidateReport.speedup = first.speedup;
    log << candidateReport.name << " | Benchmark: PASS, median(us)="
        << first.candidate.medianUs << ", paired baseline(us)="
        << first.baseline.medianUs << ", speedup=" << first.speedup << "x\n";
    if (first.speedup < kMinimumSpeedup) {
      reject(candidateReport, "Compare Baseline", "Speedup is below 1.05x.", log);
      continue;
    }

    const auto confirmation = benchmark::measurePair(*baseline, *candidate, kSamples);
    if (!confirmation.passed) {
      reject(candidateReport, "Confirmation Benchmark", confirmation.errorMessage, log);
      continue;
    }
    log << candidateReport.name << " | Confirmation: median(us)="
        << confirmation.candidate.medianUs << ", paired baseline(us)="
        << confirmation.baseline.medianUs << ", speedup=" << confirmation.speedup << "x\n";
    // Rank by the lower of two paired speedups, normalizing against clock drift.
    candidateReport.speedup = std::min(first.speedup, confirmation.speedup);
    if (*candidateReport.speedup < kMinimumSpeedup) {
      reject(candidateReport, "Compare Baseline", "Confirmation did not reach 1.05x.", log);
      continue;
    }
    candidateReport.outcome = "Admitted";
    log << candidateReport.name << " | Compare Baseline: PASS -> Admit\n";
    if (*candidateReport.speedup > bestSpeedup) {
      bestSpeedup = *candidateReport.speedup;
      winnerIndex = result.candidates.size() - 1;
      winner = std::move(candidate);
    }
  }

  // Execute the actual selection once more with fresh NaN output and readback.
  result.finalExecution = winner ? winner->run() : baseline->run();
  result.finalValidation = validateExecution(result.finalExecution, reference);
  if (!result.finalValidation.passed && winner) {
    reject(result.candidates[winnerIndex], "Final execution",
            result.finalValidation.errorMessage, log);
    log << "Selected candidate failed; executing the retained baseline.\n";
    winner.reset();
    result.finalExecution = baseline->run();
    result.finalValidation = validateExecution(result.finalExecution, reference);
  }
  if (!result.finalValidation.passed) {
    result.errorMessage = "Final baseline execution/validation failed: " +
                          result.finalValidation.errorMessage;
    return result;
  }
  result.usedBaseline = !winner;
  result.selectedThreads = winner ? result.candidates[winnerIndex].threads : 256;
  result.selectedExecution = winner ? std::move(winner) : std::move(baseline);
  result.success = true;

  log << "Admission report (speedup is the lower of both rounds when confirmed):\n";
  printReport(result.baseline, log);
  for (const auto &report : result.candidates) {
    printReport(report, log);
  }
  log << "Selected: " << (result.usedBaseline ? "baseline (fallback)"
                                              : result.candidates[winnerIndex].name)
      << ", threads=" << result.selectedThreads << '\n';
  log << "Final GPU execution: PASS\nFinal numerical validation: PASS"
      << ", max absolute error=" << result.finalValidation.maxAbsoluteError << '\n';
  return result;
}

} // namespace tensor::planner
