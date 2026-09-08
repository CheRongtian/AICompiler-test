#include "decoder_benchmark_main.hpp"

#include "benchmark/Benchmark.hpp"
#include "importer/DecoderLLMImporter.hpp"
#include "runtime/DecoderLLMExecutor.hpp"
#include "validation/Validator.hpp"

#include <chrono>
#include <cmath>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

namespace {

constexpr double kAbsoluteTolerance = 1e-3;
constexpr double kRelativeTolerance = 5e-3;

struct TimingSeries {
  std::vector<double> prefillGpu;
  std::vector<double> prefillCpu;
  std::vector<double> prefillEndToEnd;
  std::vector<double> decodeGpu;
  std::vector<double> decodeCpu;
  std::vector<double> decodeEndToEnd;
  std::vector<double> totalGeneration;
};

bool compare(const std::string &label, const std::vector<float> &actual,
             const std::vector<double> &expected, double absoluteTolerance,
             double relativeTolerance, std::string &error) {
  const auto result = tensor::validation::compare(
      actual, expected, absoluteTolerance, relativeTolerance);
  if (result.passed) return true;
  error = label + ": " + result.errorMessage;
  return false;
}

void collect(const tensor::runtime::DecoderLLMRunResult &result,
             double endToEndUs, bool prefill, TimingSeries *timings) {
  if (!timings) return;
  auto &gpu = prefill ? timings->prefillGpu : timings->decodeGpu;
  auto &cpu = prefill ? timings->prefillCpu : timings->decodeCpu;
  auto &endToEnd =
      prefill ? timings->prefillEndToEnd : timings->decodeEndToEnd;
  if (result.gpuExecutionTimeUs) gpu.push_back(*result.gpuExecutionTimeUs);
  cpu.push_back(result.cpuSubmitToCompletionTimeUs);
  endToEnd.push_back(endToEndUs);
}

bool runGeneration(tensor::runtime::CompiledDecoderLLM &decoder,
                   const tensor::DecoderLLMWorkload &workload,
                   TimingSeries *timings, bool validate,
                   std::string &error) {
  error = decoder.reset();
  if (!error.empty()) return false;

  const auto totalStart = std::chrono::steady_clock::now();
  const auto prefillStart = std::chrono::steady_clock::now();
  const auto prefill = decoder.prefill(workload.prefill.tokenIds);
  const auto prefillEnd = std::chrono::steady_clock::now();
  if (!prefill.passed) {
    error = "prefill execution: " + prefill.errorMessage;
    return false;
  }
  collect(prefill,
          std::chrono::duration<double, std::micro>(prefillEnd - prefillStart)
              .count(),
          true, timings);
  if (validate) {
    if (!compare("prefill logits", prefill.logits, workload.prefill.logits,
                 kAbsoluteTolerance, kRelativeTolerance, error) ||
        !compare("prefill token", prefill.nextTokenIds,
                 workload.prefill.nextTokenIds, 0.0, 0.0, error)) {
      return false;
    }
  }

  auto nextTokens = prefill.nextTokenIds;
  for (std::size_t step = 0; step < workload.decodeSteps.size(); ++step) {
    const auto &reference = workload.decodeSteps[step];
    if (validate) {
      const std::vector<double> expectedInput(reference.tokenIds.begin(),
                                              reference.tokenIds.end());
      if (!compare("decode input token " + std::to_string(step), nextTokens,
                   expectedInput, 0.0, 0.0, error)) {
        return false;
      }
    }
    const auto decodeStart = std::chrono::steady_clock::now();
    const auto result = decoder.decode(nextTokens);
    const auto decodeEnd = std::chrono::steady_clock::now();
    if (!result.passed) {
      error = "decode step " + std::to_string(step) + ": " +
              result.errorMessage;
      return false;
    }
    collect(result,
            std::chrono::duration<double, std::micro>(decodeEnd - decodeStart)
                .count(),
            false, timings);
    if (validate) {
      if (!compare("decode logits " + std::to_string(step), result.logits,
                   reference.logits, kAbsoluteTolerance, kRelativeTolerance,
                   error) ||
          !compare("decode token " + std::to_string(step),
                   result.nextTokenIds, reference.nextTokenIds, 0.0, 0.0,
                   error)) {
        return false;
      }
    }
    nextTokens = result.nextTokenIds;
  }

  const auto totalEnd = std::chrono::steady_clock::now();
  if (timings) {
    timings->totalGeneration.push_back(
        std::chrono::duration<double, std::micro>(totalEnd - totalStart)
            .count());
  }

  if (!validate) return true;
  const auto &finalReference = workload.decodeSteps.empty()
                                   ? workload.prefill
                                   : workload.decodeSteps.back();
  for (std::size_t layer = 0; layer < finalReference.keyCaches.size(); ++layer) {
    if (!compare("final key cache " + std::to_string(layer),
                 decoder.readKeyPrefix(layer), finalReference.keyCaches[layer],
                 kAbsoluteTolerance, kRelativeTolerance, error) ||
        !compare("final value cache " + std::to_string(layer),
                 decoder.readValuePrefix(layer),
                 finalReference.valueCaches[layer], kAbsoluteTolerance,
                 kRelativeTolerance, error)) {
      return false;
    }
  }
  const auto expectedLength = workload.plan.attention.prefillLength +
                              workload.decodeSteps.size();
  if (decoder.currentLength() != expectedLength) {
    error = "final cache length mismatch";
    return false;
  }
  if (!decoder.cacheStorageReused()) {
    error = "KV cache storage was replaced during benchmark execution";
    return false;
  }
  return true;
}

void printDistribution(const std::string &label,
                       const std::vector<double> &values,
                       std::ostream &log) {
  const auto stats = tensor::benchmark::summarizeTimings(values);
  if (!stats) {
    log << "  " << label << ": unavailable\n";
    return;
  }
  log << "  " << label << ": p50=" << stats->medianUs
      << ", p90=" << stats->p90Us << ", min=" << stats->minUs
      << ", max=" << stats->maxUs << ", samples=" << stats->samples
      << '\n';
}

std::optional<double> p50(const std::vector<double> &values) {
  const auto stats = tensor::benchmark::summarizeTimings(values);
  if (!stats) return std::nullopt;
  return stats->medianUs;
}

void report(const std::string &name, const TimingSeries &timings,
            std::size_t decodeSteps, std::size_t kvBytes,
            std::ostream &log) {
  log << name << ":\n";
  printDistribution("Prefill GPU command time (us)", timings.prefillGpu, log);
  printDistribution("Prefill CPU submit-to-completion time (us)",
                    timings.prefillCpu, log);
  printDistribution("Prefill end-to-end time (us)",
                    timings.prefillEndToEnd, log);
  printDistribution("Decode GPU command time (us/token)", timings.decodeGpu,
                    log);
  printDistribution("Decode CPU submit-to-completion time (us/token)",
                    timings.decodeCpu, log);
  printDistribution("Decode end-to-end time (us/token)",
                    timings.decodeEndToEnd, log);
  printDistribution("Total generation time (us)", timings.totalGeneration,
                    log);
  const auto decodeMedian = p50(timings.decodeEndToEnd);
  if (decodeMedian && *decodeMedian > 0.0) {
    log << "  Decode throughput (tokens/s): " << 1.0e6 / *decodeMedian
        << '\n';
  } else {
    log << "  Decode throughput (tokens/s): unavailable\n";
  }
  log << "  Decode steps per measured run: " << decodeSteps << '\n'
      << "  Fixed KV storage (bytes): " << kvBytes << '\n';
}

bool prepareForMeasurement(tensor::runtime::CompiledDecoderLLM &decoder,
                           const tensor::DecoderLLMWorkload &workload,
                           const std::string &label, std::size_t warmupRuns,
                           std::ostream &log) {
  std::string error;
  if (!runGeneration(decoder, workload, nullptr, true, error)) {
    log << label << " correctness: FAIL\n"
        << "Benchmark error: " << error << '\n';
    return false;
  }
  log << label << " correctness: PASS\n";
  for (std::size_t run = 0; run < warmupRuns; ++run) {
    if (!runGeneration(decoder, workload, nullptr, false, error)) {
      log << label << " warmup: FAIL\n"
          << "Benchmark error: " << error << '\n';
      return false;
    }
  }
  log << label << " warmup: PASS, runs=" << warmupRuns << '\n';
  decoder.resetKernelUsage();
  return true;
}

} // namespace

bool runDecoderLLMBenchmark(tensor::metal::MetalRuntime &runtime,
                            const std::string &manifestPath,
                            std::ostream &log,
                            const DecoderBenchmarkOptions &options) {
  if (options.warmupRuns < 2 || options.measuredRuns < 10) {
    log << "Decoder benchmark: FAIL\n"
        << "Benchmark requires at least 2 warmup runs and 10 measured runs.\n";
    return false;
  }
  auto imported = tensor::importer::importDecoderLLMWorkload(manifestPath);
  if (!imported.workload) {
    log << "Decoder benchmark import: FAIL\n"
        << "Import error: " << imported.errorMessage << '\n';
    return false;
  }
  const auto &workload = *imported.workload;
  log << "Decoder benchmark import: PASS\n"
      << "Benchmark protocol: paired alternating order, warmup="
      << options.warmupRuns << ", measured_runs=" << options.measuredRuns
      << '\n';

  log << "Compiling template-only decoder:\n";
  auto templateCompilation = tensor::runtime::compileDecoderLLM(
      runtime, workload, log, {});
  if (!templateCompilation.executable) {
    log << "Template-only compilation: FAIL\n"
        << "Compiler error: " << templateCompilation.errorMessage << '\n';
    return false;
  }

  log << "Compiling generated-enabled decoder:\n";
  auto generatedCompilation = tensor::runtime::compileDecoderLLM(
      runtime, workload, log, options.kernelLibrary);
  if (!generatedCompilation.executable) {
    log << "Generated-enabled compilation: FAIL\n"
        << "Compiler error: " << generatedCompilation.errorMessage << '\n';
    return false;
  }

  auto &templateDecoder = *templateCompilation.executable;
  auto &generatedDecoder = *generatedCompilation.executable;
  if (!prepareForMeasurement(templateDecoder, workload, "Template-only",
                             options.warmupRuns, log) ||
      !prepareForMeasurement(generatedDecoder, workload, "Generated-enabled",
                             options.warmupRuns, log)) {
    return false;
  }

  TimingSeries templateTimings;
  TimingSeries generatedTimings;
  std::string error;
  for (std::size_t sample = 0; sample < options.measuredRuns; ++sample) {
    for (std::size_t position = 0; position < 2; ++position) {
      const bool runTemplate = (sample + position) % 2 == 0;
      auto &decoder = runTemplate ? templateDecoder : generatedDecoder;
      auto &timings = runTemplate ? templateTimings : generatedTimings;
      if (!runGeneration(decoder, workload, &timings, false, error)) {
        log << "Decoder benchmark execution: FAIL\n"
            << "Benchmark error: " << error << '\n';
        return false;
      }
    }
  }

  const auto &plan = workload.plan;
  const auto kvBytes = plan.layerCount * 2 * plan.attention.batch *
                       plan.attention.heads * plan.attention.capacity *
                       plan.attention.headDimension * sizeof(float);
  log << "Decoder benchmark results (times in microseconds):\n";
  report("Template-only", templateTimings, workload.decodeSteps.size(),
         kvBytes, log);
  report("Generated-enabled", generatedTimings, workload.decodeSteps.size(),
         kvBytes, log);

  const auto templateGpu = p50(templateTimings.decodeGpu);
  const auto generatedGpu = p50(generatedTimings.decodeGpu);
  const auto templateEndToEnd = p50(templateTimings.decodeEndToEnd);
  const auto generatedEndToEnd = p50(generatedTimings.decodeEndToEnd);
  if (templateGpu && generatedGpu && *generatedGpu > 0.0) {
    log << "Generated on/off decode GPU speedup: "
        << *templateGpu / *generatedGpu << "x\n";
  } else {
    log << "Generated on/off decode GPU speedup: unavailable\n";
  }
  if (templateEndToEnd && generatedEndToEnd && *generatedEndToEnd > 0.0) {
    log << "Generated on/off decode end-to-end speedup: "
        << *templateEndToEnd / *generatedEndToEnd << "x\n";
  } else {
    log << "Generated on/off decode end-to-end speedup: unavailable\n";
  }

  log << "Template-only runtime audit:\n";
  templateDecoder.reportKernelUsage(log);
  log << "Generated-enabled runtime audit:\n";
  generatedDecoder.reportKernelUsage(log);
  log << "Decoder benchmark: PASS\n";
  return true;
}
