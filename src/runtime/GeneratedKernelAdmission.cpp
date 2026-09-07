#include "runtime/GeneratedKernelAdmission.hpp"

#include "analyzer/PatternAnalyzer.hpp"
#include "backend/metal/MetalEmitter.hpp"
#include "benchmark/Benchmark.hpp"
#include "llm/GeneratedKernelProtocol.hpp"
#include "validation/Validator.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tensor::runtime {
namespace {

constexpr std::size_t kWarmup = 5;
constexpr std::size_t kSamples = 31;
constexpr double kMinimumSpeedup = 1.05;
constexpr double kAbsoluteTolerance = 1e-5;
constexpr double kRelativeTolerance = 1e-4;
constexpr std::size_t kGuardElements = 256;
const std::vector<std::size_t> kTemplateThreads{64, 128, 256};

struct CaseData {
  std::size_t rows = 0;
  std::size_t width = 0;
  std::vector<float> input;
  std::vector<float> multiplier;
  std::vector<double> reference;

  [[nodiscard]] std::size_t elementCount() const { return rows * width; }
};

struct TemplateCandidate {
  std::size_t threads = 0;
  double medianUs = 0.0;
  std::vector<std::unique_ptr<metal::PreparedExecution>> executions;
};

struct Feedback {
  std::string status;
  std::string stage;
  std::string message;
  std::optional<double> baselineUs;
  std::optional<double> candidateUs;
  std::optional<double> speedup;
};

std::string escapeJson(const std::string &text) {
  std::ostringstream output;
  const char hex[] = "0123456789abcdef";
  for (const unsigned char character : text) {
    switch (character) {
    case '\"': output << "\\\""; break;
    case '\\': output << "\\\\"; break;
    case '\b': output << "\\b"; break;
    case '\f': output << "\\f"; break;
    case '\n': output << "\\n"; break;
    case '\r': output << "\\r"; break;
    case '\t': output << "\\t"; break;
    default:
      if (character < 0x20) {
        output << "\\u00" << hex[character >> 4] << hex[character & 0xf];
      } else {
        output << static_cast<char>(character);
      }
    }
  }
  return output.str();
}

std::string writeFeedback(const Feedback &feedback, const std::string &path) {
  std::ofstream output(path);
  if (!output) return "Unable to open generated-kernel feedback output '" + path + "'.";
  output << "{\n  \"version\":1,\n  \"status\":\""
         << escapeJson(feedback.status) << "\",\n  \"stage\":\""
         << escapeJson(feedback.stage) << "\",\n  \"message\":\""
         << escapeJson(feedback.message) << "\",\n  \"metrics\":";
  if (feedback.baselineUs && feedback.candidateUs && feedback.speedup) {
    output << "{\"baseline_median_us\":" << *feedback.baselineUs
           << ",\"candidate_median_us\":" << *feedback.candidateUs
           << ",\"speedup\":" << *feedback.speedup << '}';
  } else {
    output << "null";
  }
  output << "\n}\n";
  if (!output) return "Failed while writing generated-kernel feedback '" + path + "'.";
  return {};
}

CaseData makeCase(std::size_t rows, std::size_t width, std::size_t seed) {
  CaseData result;
  result.rows = rows;
  result.width = width;
  result.input.resize(result.elementCount());
  result.multiplier.resize(result.elementCount());
  result.reference.resize(result.elementCount());
  for (std::size_t index = 0; index < result.elementCount(); ++index) {
    const int centeredInput =
        static_cast<int>((index * 17 + seed * 29) % 257) - 128;
    const int centeredMultiplier =
        static_cast<int>((index * 31 + seed * 11) % 193) - 96;
    result.input[index] = static_cast<float>(centeredInput) / 64.0f;
    result.multiplier[index] =
        static_cast<float>(centeredMultiplier) / 80.0f;
    const double x = result.input[index];
    const double activated = x / (1.0 + std::exp(-x));
    result.reference[index] = activated * result.multiplier[index];
  }
  return result;
}

metal::GeneratedKernel templateKernel(const CaseData &data,
                                      std::size_t threads) {
  TensorGraph graph;
  const TensorType type{{data.rows, data.width}, DType::Float32};
  const auto input = graph.addInput("input", type);
  const auto multiplier = graph.addInput("multiplier", type);
  const auto activated = graph.addNode(OpType::SiLU, {input});
  graph.outputs = {graph.addNode(OpType::Mul, {activated, multiplier})};
  auto regions = analyzer::formRegions(analyzer::analyze(graph));
  const auto found = std::find_if(
      regions.regions.begin(), regions.regions.end(),
      [](const analyzer::Region &region) {
        return region.fusion == analyzer::FusionPattern::SiLUMul;
      });
  if (found == regions.regions.end()) {
    throw std::runtime_error("SiLU + Mul template region was not formed.");
  }
  return metal::emitFusion(*found, regions.analyzed, threads);
}

std::string validateOutput(const metal::ExecutionResult &execution,
                           const CaseData &data) {
  if (!execution.executionPassed) return execution.errorMessage;
  const auto comparison = validation::compare(
      execution.output, data.reference, kAbsoluteTolerance, kRelativeTolerance);
  return comparison.passed ? std::string{} : comparison.errorMessage;
}

std::string validateGeneratedOutput(const metal::ExecutionResult &execution,
                                    const CaseData &data) {
  if (!execution.executionPassed) return execution.errorMessage;
  if (execution.output.size() != data.elementCount() + kGuardElements) {
    return "Generated output buffer has an unexpected length.";
  }
  std::vector<float> logical(execution.output.begin(),
                             execution.output.begin() + data.elementCount());
  const auto comparison = validation::compare(
      logical, data.reference, kAbsoluteTolerance, kRelativeTolerance);
  if (!comparison.passed) return comparison.errorMessage;
  const auto guard = execution.output.begin() + data.elementCount();
  if (!std::all_of(guard, execution.output.end(),
                   [](float value) { return std::isnan(value); })) {
    return "Output guard was modified by a thread outside element_count.";
  }
  return {};
}

std::string generatedInterfaceError(
    const metal::ComputePipelineResult &pipeline) {
  if (!pipeline.reflectionAvailable || pipeline.bindings.size() != 4) {
    return "Reflection must contain two fp32 inputs, one fp32 output, and one uint constant.";
  }
  const metal::PipelineBinding *bindings[4]{};
  for (const auto &binding : pipeline.bindings) {
    if (!binding.isBuffer || binding.index >= 4 || bindings[binding.index]) {
      return "Generated kernel must use unique buffer bindings 0 through 3.";
    }
    bindings[binding.index] = &binding;
  }
  if (!bindings[0] || !bindings[1] || !bindings[2] || !bindings[3]) {
    return "Generated kernel is missing a required buffer binding.";
  }
  if (!bindings[0]->isFloat32 || !bindings[0]->readOnly ||
      !bindings[1]->isFloat32 || !bindings[1]->readOnly) {
    return "Bindings 0 and 1 must be read-only fp32 buffers.";
  }
  if (!bindings[2]->isFloat32 || !bindings[2]->writable) {
    return "Binding 2 must be a writable fp32 buffer.";
  }
  if (!bindings[3]->isUInt32 || !bindings[3]->readOnly) {
    return "Binding 3 must be a read-only uint element count.";
  }
  return {};
}

std::unique_ptr<metal::PreparedExecution>
prepareTemplate(metal::MetalRuntime &runtime, const CaseData &data,
                std::size_t threads, std::string &error) {
  const auto kernel = templateKernel(data, threads);
  const auto pipeline = runtime.createComputePipeline(kernel.source,
                                                      kernel.functionName);
  if (!pipeline.pipelineCreationPassed) {
    error = pipeline.errorMessage;
    return {};
  }
  error = metal::checkFloatBufferInterface(pipeline, 2);
  if (!error.empty()) return {};
  auto prepared = runtime.prepare(
      {{data.input.data(), data.input.size()},
       {data.multiplier.data(), data.multiplier.size()}},
      data.elementCount(),
      {kernel.threadgroupCount, kernel.threadsPerThreadgroup});
  error = prepared.errorMessage;
  return std::move(prepared.execution);
}

std::vector<TemplateCandidate>
prepareTemplateCandidates(metal::MetalRuntime &runtime,
                          const std::vector<CaseData> &cases,
                          std::ostream &log) {
  std::vector<TemplateCandidate> results;
  for (const auto threads : kTemplateThreads) {
    if (threads > runtime.hardwareInfo().maxThreadsPerThreadgroup) continue;
    TemplateCandidate candidate;
    candidate.threads = threads;
    bool valid = true;
    for (const auto &data : cases) {
      std::string error;
      auto execution = prepareTemplate(runtime, data, threads, error);
      if (!execution) {
        log << "Template baseline threads=" << threads
            << " preparation: FAIL: " << error << '\n';
        valid = false;
        break;
      }
      error = validateOutput(execution->run(), data);
      if (!error.empty()) {
        log << "Template baseline threads=" << threads
            << " validation: FAIL: " << error << '\n';
        valid = false;
        break;
      }
      candidate.executions.push_back(std::move(execution));
    }
    if (!valid) continue;
    const auto warmup = benchmark::warmup(*candidate.executions.back(), kWarmup);
    if (!warmup.empty()) {
      log << "Template baseline threads=" << threads
          << " warmup: FAIL: " << warmup << '\n';
      continue;
    }
    const auto timing = benchmark::measure(*candidate.executions.back(), kSamples);
    if (!timing.passed) {
      log << "Template baseline threads=" << threads
          << " benchmark: FAIL: " << timing.errorMessage << '\n';
      continue;
    }
    candidate.medianUs = timing.stats.medianUs;
    log << "Template baseline threads=" << threads
        << " validation: PASS, median(us)=" << candidate.medianUs << '\n';
    results.push_back(std::move(candidate));
  }
  return results;
}

} // namespace

bool admitGeneratedSiLUMulKernel(metal::MetalRuntime &runtime,
                                 const std::string &responsePath,
                                 const std::string &feedbackPath,
                                 std::ostream &log) {
  const std::vector<CaseData> cases{
      makeCase(1, 4096, 1), makeCase(3, 4097, 2)};
  auto baselines = prepareTemplateCandidates(runtime, cases, log);
  if (baselines.empty()) {
    const Feedback feedback{"fatal", "baseline",
                            "No valid template baseline is available."};
    const auto error = writeFeedback(feedback, feedbackPath);
    if (!error.empty()) log << "Feedback error: " << error << '\n';
    log << "Generated kernel admission: FAIL\n";
    return false;
  }
  auto baseline = std::min_element(
      baselines.begin(), baselines.end(),
      [](const auto &left, const auto &right) {
        return left.medianUs < right.medianUs;
      });
  log << "Selected performance baseline: template fused SiLU + Mul, threads="
      << baseline->threads << ", median(us)=" << baseline->medianUs << '\n';

  const auto fallback = [&](const std::string &stage,
                            const std::string &message,
                            std::optional<double> pairedBaselineUs = std::nullopt,
                            std::optional<double> candidateUs = std::nullopt,
                            std::optional<double> speedup = std::nullopt) {
    log << "Generated candidate " << stage << ": FAIL: " << message << '\n'
        << "Generated kernel admission: FALLBACK\n"
        << "Selected: template fused SiLU + Mul\n";
    Feedback feedback{"retry", stage, message};
    if (pairedBaselineUs && candidateUs && speedup) {
      feedback.baselineUs = pairedBaselineUs;
      feedback.candidateUs = candidateUs;
      feedback.speedup = speedup;
    }
    const auto error = writeFeedback(feedback, feedbackPath);
    if (!error.empty()) {
      log << "Feedback error: " << error << '\n';
      return false;
    }
    return true;
  };

  auto loaded = llm::loadGeneratedKernelResponse(responsePath);
  if (!loaded.response) return fallback("response_parse", loaded.errorMessage);
  const auto &generated = *loaded.response;
  if (generated.functionName != "generated_silu_mul") {
    return fallback("contract",
                    "function_name must be generated_silu_mul.");
  }
  if (std::find(kTemplateThreads.begin(), kTemplateThreads.end(),
                generated.workgroupSize) == kTemplateThreads.end() ||
      generated.workgroupSize > runtime.hardwareInfo().maxThreadsPerThreadgroup) {
    return fallback("hardware_filter",
                    "workgroup_size must be a legal value: 64, 128, or 256.");
  }

  const auto pipeline = runtime.createComputePipeline(
      generated.source, generated.functionName);
  if (!pipeline.pipelineCreationPassed) {
    return fallback("compile", pipeline.errorMessage);
  }
  log << "Generated candidate compile: PASS\n";
  const auto interfaceError = generatedInterfaceError(pipeline);
  if (!interfaceError.empty()) return fallback("interface", interfaceError);
  if (generated.workgroupSize > pipeline.maxTotalThreadsPerThreadgroup) {
    return fallback("interface",
                    "workgroup_size exceeds the compiled pipeline limit.");
  }
  log << "Generated candidate interface: PASS\n";

  std::vector<std::unique_ptr<metal::PreparedExecution>> candidates;
  for (std::size_t index = 0; index < cases.size(); ++index) {
    const auto &data = cases[index];
    const auto count = data.elementCount();
    if (count > std::numeric_limits<std::uint32_t>::max()) {
      return fallback("prepare", "element_count exceeds uint32.");
    }
    auto guardedInput = data.input;
    auto guardedMultiplier = data.multiplier;
    guardedInput.resize(count + kGuardElements, 0.75f);
    guardedMultiplier.resize(count + kGuardElements, 1.25f);
    auto prepared = runtime.prepare(
        {{guardedInput.data(), guardedInput.size()},
         {guardedMultiplier.data(), guardedMultiplier.size()}},
        count + kGuardElements,
        {(count + generated.workgroupSize - 1) / generated.workgroupSize,
         generated.workgroupSize},
        {static_cast<std::uint32_t>(count)});
    if (!prepared.execution) return fallback("prepare", prepared.errorMessage);
    const auto validationError =
        validateGeneratedOutput(prepared.execution->run(), data);
    if (!validationError.empty()) {
      return fallback("numerical_validation",
                      "case " + std::to_string(index) + ": " + validationError);
    }
    log << "Generated candidate case " << index
        << " shape=[" << data.rows << ',' << data.width
        << "] numerical validation: PASS\n";
    candidates.push_back(std::move(prepared.execution));
  }

  const auto warmup = benchmark::warmup(*candidates.back(), kWarmup);
  if (!warmup.empty()) return fallback("warmup", warmup);
  log << "Generated candidate warmup: PASS\n";
  const auto first = benchmark::measurePair(
      *baseline->executions.back(), *candidates.back(), kSamples);
  if (!first.passed) return fallback("benchmark", first.errorMessage);
  log << "Generated candidate benchmark: PASS, baseline(us)="
      << first.baseline.medianUs << ", candidate(us)="
      << first.candidate.medianUs << ", speedup=" << first.speedup << "x\n";
  if (first.speedup < kMinimumSpeedup) {
    return fallback("performance", "Speedup is below 1.05x.",
                    first.baseline.medianUs, first.candidate.medianUs,
                    first.speedup);
  }

  const auto confirmation = benchmark::measurePair(
      *baseline->executions.back(), *candidates.back(), kSamples);
  if (!confirmation.passed) {
    return fallback("confirmation_benchmark", confirmation.errorMessage);
  }
  const double conservativeSpeedup =
      std::min(first.speedup, confirmation.speedup);
  log << "Generated candidate confirmation: baseline(us)="
      << confirmation.baseline.medianUs << ", candidate(us)="
      << confirmation.candidate.medianUs << ", speedup="
      << confirmation.speedup << "x\n";
  if (conservativeSpeedup < kMinimumSpeedup) {
    return fallback("performance",
                    "Confirmation speedup is below 1.05x.",
                    confirmation.baseline.medianUs,
                    confirmation.candidate.medianUs, conservativeSpeedup);
  }

  const auto finalError =
      validateGeneratedOutput(candidates.back()->run(), cases.back());
  if (!finalError.empty()) {
    return fallback("final_validation", finalError);
  }
  Feedback feedback{"admitted", "admission",
                    "Compile, interface, correctness, and performance passed.",
                    confirmation.baseline.medianUs,
                    confirmation.candidate.medianUs, conservativeSpeedup};
  const auto feedbackError = writeFeedback(feedback, feedbackPath);
  if (!feedbackError.empty()) {
    log << "Feedback error: " << feedbackError << '\n';
    return false;
  }
  log << "Generated kernel admission: PASS\n"
      << "Selected: LLM-generated SiLU + Mul, threads="
      << generated.workgroupSize << ", conservative speedup="
      << conservativeSpeedup << "x\n";
  return true;
}

} // namespace tensor::runtime
