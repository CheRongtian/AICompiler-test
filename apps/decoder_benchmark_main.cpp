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
#include <utility>
#include <vector>

namespace {

constexpr double kAbsoluteTolerance = 1e-3;
constexpr double kRelativeTolerance = 5e-3;

std::pair<double, double> tolerances(tensor::DType dtype) {
  if (dtype == tensor::DType::Float16 || dtype == tensor::DType::BFloat16) {
    return {2e-2, 2e-2};
  }
  return {kAbsoluteTolerance, kRelativeTolerance};
}

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
                   std::string &error, bool tokenOnly = false) {
  const auto [absoluteTolerance, relativeTolerance] =
      tolerances(decoder.storageDtype());
  error = decoder.reset();
  if (!error.empty()) return false;

  const auto totalStart = std::chrono::steady_clock::now();
  const auto prefillStart = std::chrono::steady_clock::now();
  const auto prefill = decoder.prefill(workload.prefill.tokenIds,!tokenOnly);
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
    if ((!tokenOnly && !compare("prefill logits", prefill.logits, workload.prefill.logits,
                 absoluteTolerance, relativeTolerance, error)) ||
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
    const auto result = decoder.decode(nextTokens,!tokenOnly);
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
      if ((!tokenOnly && !compare("decode logits " + std::to_string(step), result.logits,
                   reference.logits, absoluteTolerance, relativeTolerance,
                   error)) ||
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
                 absoluteTolerance, relativeTolerance, error) ||
        !compare("final value cache " + std::to_string(layer),
                 decoder.readValuePrefix(layer),
                 finalReference.valueCaches[layer], absoluteTolerance,
                 relativeTolerance, error)) {
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
            std::size_t decodeSteps,
            const tensor::runtime::CompiledDecoderLLM &decoder,
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
  printDistribution("TTFT (us)", timings.prefillEndToEnd, log);
  printDistribution("TPOT (us/token)", timings.decodeEndToEnd, log);
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
      << "  Storage dtype: " << tensor::dtypeName(decoder.storageDtype())
      << '\n'
      << "  Precision fallback: "
      << (decoder.precisionFallbackUsed() ? "true" : "false") << '\n'
      << "  Model storage (bytes): " << decoder.modelStorageBytes() << '\n'
      << "  KV storage (bytes): " << decoder.kvStorageBytes() << '\n'
      << "  Allocated activation storage (bytes): "
      << decoder.activationStorageBytes() << '\n'
      << "  Planned peak model + KV + activation (bytes): "
      << decoder.modelStorageBytes() + decoder.kvStorageBytes() +
             decoder.activationStorageBytes()
      << '\n';
}

bool prepareForMeasurement(tensor::runtime::CompiledDecoderLLM &decoder,
                           const tensor::DecoderLLMWorkload &workload,
                           const std::string &label, std::size_t warmupRuns,
                           std::ostream &log, bool tokenOnly = false) {
  std::string error;
  if (!runGeneration(decoder, workload, nullptr, true, error)) {
    log << label << " correctness: FAIL\n"
        << "Benchmark error: " << error << '\n';
    return false;
  }
  log << label << " correctness: PASS\n";
  if(tokenOnly && !runGeneration(decoder,workload,nullptr,true,error,true)) {
    log << label << " token-only correctness: FAIL: " << error << '\n';
    return false;
  }
  for (std::size_t run = 0; run < warmupRuns; ++run) {
    if (!runGeneration(decoder, workload, nullptr, false, error,tokenOnly)) {
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
      << ", readback=" << (options.tokenOnly?"tokens":"logits+tokens")
      << ", comparison=" << (options.compareFusions?"fusion_off_vs_on":"generated_off_vs_on")
      << '\n';

  const std::string baselineLabel = options.compareFusions ? "Fusion-off" : "Template-only";
  const std::string comparisonLabel = options.compareFusions ? "Fusion-on" : "Generated-enabled";
  log << "Compiling " << baselineLabel << " decoder:\n";
  tensor::runtime::DecoderLLMCompileOptions templateOptions;
  templateOptions.storageDtype = workload.plan.attention.dtype;
  templateOptions.requestedStorageDtype = workload.requestedStorageDtype;
  templateOptions.allowPrecisionFallback = true;
  templateOptions.enableFusion = !options.compareFusions;
  auto templateCompilation = tensor::runtime::compileDecoderLLM(
      runtime, workload, log, templateOptions);
  if (!templateCompilation.executable) {
    log << baselineLabel << " compilation: FAIL\n"
        << "Compiler error: " << templateCompilation.errorMessage << '\n';
    return false;
  }

  log << "Compiling " << comparisonLabel << " decoder:\n";
  tensor::runtime::DecoderLLMCompileOptions generatedOptions;
  if (!options.compareFusions) generatedOptions.kernelLibrary = options.kernelLibrary;
  generatedOptions.storageDtype = workload.plan.attention.dtype;
  generatedOptions.requestedStorageDtype = workload.requestedStorageDtype;
  generatedOptions.allowPrecisionFallback = true;
  generatedOptions.enableFusion = true;
  generatedOptions.sharedModelResources=templateCompilation.executable->sharedModelResources();
  auto generatedCompilation = tensor::runtime::compileDecoderLLM(
      runtime, workload, log, generatedOptions);
  if (!generatedCompilation.executable) {
    log << comparisonLabel << " compilation: FAIL\n"
        << "Compiler error: " << generatedCompilation.errorMessage << '\n';
    return false;
  }

  auto &templateDecoder = *templateCompilation.executable;
  auto &generatedDecoder = *generatedCompilation.executable;
  if (!prepareForMeasurement(templateDecoder, workload, baselineLabel,
                             options.warmupRuns, log,options.tokenOnly) ||
      !prepareForMeasurement(generatedDecoder, workload, comparisonLabel,
                             options.warmupRuns, log,options.tokenOnly)) {
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
      if (!runGeneration(decoder, workload, &timings, false, error,options.tokenOnly)) {
        log << "Decoder benchmark execution: FAIL\n"
            << "Benchmark error: " << error << '\n';
        return false;
      }
    }
  }

  log << "Decoder benchmark results (times in microseconds):\n";
  report(baselineLabel, templateTimings, workload.decodeSteps.size(),
         templateDecoder, log);
  report(comparisonLabel, generatedTimings, workload.decodeSteps.size(),
         generatedDecoder, log);

  const auto templateGpu = p50(templateTimings.decodeGpu);
  const auto generatedGpu = p50(generatedTimings.decodeGpu);
  const auto templateEndToEnd = p50(templateTimings.decodeEndToEnd);
  const auto generatedEndToEnd = p50(generatedTimings.decodeEndToEnd);
  if (templateGpu && generatedGpu && *generatedGpu > 0.0) {
    log << (options.compareFusions ? "Fusion on/off" : "Generated on/off")
        << " decode GPU speedup: "
        << *templateGpu / *generatedGpu << "x\n";
  } else {
    log << (options.compareFusions ? "Fusion on/off" : "Generated on/off")
        << " decode GPU speedup: unavailable\n";
  }
  if (templateEndToEnd && generatedEndToEnd && *generatedEndToEnd > 0.0) {
    log << (options.compareFusions ? "Fusion on/off" : "Generated on/off")
        << " decode end-to-end speedup: "
        << *templateEndToEnd / *generatedEndToEnd << "x\n";
  } else {
    log << (options.compareFusions ? "Fusion on/off" : "Generated on/off")
        << " decode end-to-end speedup: unavailable\n";
  }

  log << baselineLabel << " runtime audit:\n";
  templateDecoder.reportKernelUsage(log);
  log << comparisonLabel << " runtime audit:\n";
  generatedDecoder.reportKernelUsage(log);
  log << "Decoder benchmark: PASS\n";
  return true;
}
