#include "serving_main.hpp"

#include "importer/DecoderLLMImporter.hpp"
#include "runtime/DecoderLLMExecutor.hpp"
#include "runtime/ServingRuntime.hpp"
#include "validation/Validator.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
constexpr double kAbsoluteTolerance = 1e-3;
constexpr double kRelativeTolerance = 5e-3;

const char *passFail(bool passed) { return passed ? "PASS" : "FAIL"; }

double elapsedUs(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double, std::micro>(end - start).count();
}

std::vector<tensor::runtime::ServingRequestSpec> makeRequests(
    const tensor::DecoderLLMWorkload &workload) {
  const std::vector<std::size_t> promptLengths{5, 8, 7};
  const std::vector<std::size_t> decodeLengths{7, 5, 6};
  std::vector<tensor::runtime::ServingRequestSpec> requests;
  for (std::size_t request = 0; request < promptLengths.size(); ++request) {
    tensor::runtime::ServingRequestSpec spec;
    spec.id = "request_" + std::to_string(request);
    spec.decodeTokenCount = decodeLengths[request];
    spec.arrivalStep = request;
    for (std::size_t token = 0; token < promptLengths[request]; ++token) {
      const auto seed = workload.prefill.tokenIds[
          (token + request) % workload.prefill.tokenIds.size()];
      const auto shifted = std::fmod(
          seed + static_cast<float>(request * 11 + token),
          static_cast<float>(workload.plan.vocabularySize));
      spec.promptTokenIds.push_back(shifted);
    }
    requests.push_back(std::move(spec));
  }
  return requests;
}

struct SequentialReference {
  bool passed = false;
  std::vector<tensor::runtime::ServingRequestOutput> requests;
  double totalTimeUs = 0.0;
  double tokensPerSecond = 0.0;
  std::size_t decodeCommandSubmissions = 0;
  std::string errorMessage;
};

SequentialReference runSequentialReference(
    tensor::metal::MetalRuntime &runtime,
    const tensor::DecoderLLMWorkload &baseWorkload,
    const std::vector<tensor::runtime::ServingRequestSpec> &specs,
    const tensor::runtime::ServingConfig &config) {
  SequentialReference reference;
  try {
    struct CompiledRequest {
      tensor::runtime::ServingRequestSpec spec;
      std::unique_ptr<tensor::runtime::CompiledDecoderLLM> decoder;
    };
    std::vector<CompiledRequest> compiledRequests;
    for (const auto &spec : specs) {
      auto workload = baseWorkload;
      workload.plan.attention.prefillLength = spec.promptTokenIds.size();
      workload.decodeCount = spec.decodeTokenCount;
      std::ostringstream compilationLog;
      tensor::runtime::DecoderLLMCompileOptions options;
      options.kvPageSize = config.pageSize;
      options.prefillChunkSize =
          std::min(config.prefillChunkSize, spec.promptTokenIds.size() - 1);
      auto compilation = tensor::runtime::compileDecoderLLM(
          runtime, workload, compilationLog, options);
      if (!compilation.executable) {
        throw std::runtime_error("Sequential request " + spec.id +
                                 " compilation failed: " +
                                 compilation.errorMessage);
      }
      compiledRequests.push_back(
          {spec, std::move(compilation.executable)});
    }

    std::size_t generated = 0;
    const auto start = Clock::now();
    for (auto &compiled : compiledRequests) {
      const auto &spec = compiled.spec;
      tensor::runtime::ServingRequestOutput output;
      output.id = spec.id;
      const auto prefill =
          compiled.decoder->prefillChunked(spec.promptTokenIds);
      if (!prefill.passed) {
        throw std::runtime_error("Sequential request " + spec.id +
                                 " prefill failed: " + prefill.errorMessage);
      }
      output.prefillLogits = prefill.logits;
      output.generatedTokenIds.insert(output.generatedTokenIds.end(),
                                      prefill.nextTokenIds.begin(),
                                      prefill.nextTokenIds.end());
      auto nextTokens = prefill.nextTokenIds;
      for (std::size_t step = 0; step < spec.decodeTokenCount; ++step) {
        const auto decoded = compiled.decoder->decode(nextTokens);
        if (!decoded.passed) {
          throw std::runtime_error("Sequential request " + spec.id +
                                   " decode failed: " + decoded.errorMessage);
        }
        output.decodeLogits.push_back(decoded.logits);
        output.generatedTokenIds.insert(output.generatedTokenIds.end(),
                                        decoded.nextTokenIds.begin(),
                                        decoded.nextTokenIds.end());
        nextTokens = decoded.nextTokenIds;
        ++generated;
        ++reference.decodeCommandSubmissions;
      }
      reference.requests.push_back(std::move(output));
    }
    reference.totalTimeUs = elapsedUs(start, Clock::now());
    if (reference.totalTimeUs > 0.0) {
      reference.tokensPerSecond =
          static_cast<double>(generated) * 1.0e6 / reference.totalTimeUs;
    }
    reference.passed = true;
  } catch (const std::exception &error) {
    reference.errorMessage = error.what();
  }
  return reference;
}

bool validateVector(const std::string &label,
                    const std::vector<float> &actual,
                    const std::vector<float> &expected, std::ostream &log,
                    double absoluteTolerance = kAbsoluteTolerance,
                    double relativeTolerance = kRelativeTolerance) {
  const std::vector<double> expectedDouble(expected.begin(), expected.end());
  const auto validation = tensor::validation::compare(
      actual, expectedDouble, absoluteTolerance, relativeTolerance);
  log << label << ": " << passFail(validation.passed)
      << ", max absolute error=" << validation.maxAbsoluteError << '\n';
  if (!validation.passed) {
    log << "Validation error: " << validation.errorMessage << '\n';
  }
  return validation.passed;
}

bool validateRequests(
    const std::vector<tensor::runtime::ServingRequestOutput> &actual,
    const std::vector<tensor::runtime::ServingRequestOutput> &expected,
    std::ostream &log) {
  if (actual.size() != expected.size()) {
    log << "Multi-request result count: FAIL\n";
    return false;
  }
  bool passed = true;
  for (std::size_t request = 0; request < actual.size(); ++request) {
    bool requestPassed = actual[request].id == expected[request].id;
    requestPassed &= validateVector(
        "  " + actual[request].id + " prefill logits",
        actual[request].prefillLogits, expected[request].prefillLogits, log);
    if (actual[request].decodeLogits.size() !=
        expected[request].decodeLogits.size()) {
      requestPassed = false;
      log << "  " << actual[request].id << " decode result count: FAIL\n";
    } else {
      for (std::size_t step = 0;
           step < actual[request].decodeLogits.size(); ++step) {
        requestPassed &= validateVector(
            "  " + actual[request].id + " decode " +
                std::to_string(step) + " logits",
            actual[request].decodeLogits[step],
            expected[request].decodeLogits[step], log);
      }
    }
    requestPassed &= validateVector(
        "  " + actual[request].id + " generated tokens",
        actual[request].generatedTokenIds,
        expected[request].generatedTokenIds, log, 0.0, 0.0);
    log << "Serving request " << actual[request].id << " validation: "
        << passFail(requestPassed) << '\n';
    passed &= requestPassed;
  }
  return passed;
}

void printStatistics(const std::string &label,
                     const tensor::benchmark::Statistics &stats,
                     std::ostream &log) {
  log << label << ": samples=" << stats.samples
      << ", p50(us)=" << stats.medianUs << ", p90(us)=" << stats.p90Us
      << '\n';
}

} // namespace

bool runServingWorkload(tensor::metal::MetalRuntime &runtime,
                        const std::string &manifestPath, std::ostream &log,
                        const std::string &kernelLibrary) {
  auto imported = tensor::importer::importDecoderLLMWorkload(manifestPath);
  if (!imported.workload) {
    log << "Serving import: FAIL\nImport error: " << imported.errorMessage
        << '\n';
    return false;
  }
  const auto requests = makeRequests(*imported.workload);
  log << "Serving workload: requests=" << requests.size()
      << ", prompt_lengths=[5,8,7], decode_lengths=[7,5,6]\n";

  tensor::runtime::ServingConfig config;
  config.pageSize = 4;
  config.physicalPagesPerLayer = 8;
  config.prefillChunkSize = 3;
  config.kernelLibrary = kernelLibrary;

  const auto sequential =
      runSequentialReference(runtime, *imported.workload, requests, config);
  if (!sequential.passed) {
    log << "Sequential reference: FAIL\nReference error: "
        << sequential.errorMessage << '\n';
    return false;
  }
  log << "Sequential reference: PASS, total(us)=" << sequential.totalTimeUs
      << ", tokens/s=" << sequential.tokensPerSecond
      << ", decode_submissions=" << sequential.decodeCommandSubmissions
      << '\n';
  const auto serving = tensor::runtime::runContinuousBatch(
      runtime, *imported.workload, requests, config, log);
  if (!serving.passed) {
    log << "Continuous batching execution: FAIL\nServing error: "
        << serving.errorMessage << '\n';
    return false;
  }

  bool passed = validateRequests(serving.requests, sequential.requests, log);
  const auto &metrics = serving.metrics;
  const bool schedulerPassed = metrics.maximumActiveBatchSize >= 2;
  const bool preemptionPassed = metrics.preemptionCount > 0 &&
                                metrics.resumeCount == metrics.preemptionCount;
  const bool poolPassed = metrics.peakPagesPerLayer <=
                              config.physicalPagesPerLayer &&
                          metrics.pageAllocationCount ==
                              metrics.pageReleaseCount;
  const bool batchingPassed = metrics.decodedTokens > 0 &&
                              metrics.decodeCommandSubmissions <
                                  metrics.decodedTokens;
  log << "Continuous batching scheduler: " << passFail(schedulerPassed)
      << ", average_active_batch=" << metrics.averageActiveBatchSize
      << ", max_active_batch=" << metrics.maximumActiveBatchSize << '\n';
  log << "Preemption/resume: " << passFail(preemptionPassed)
      << ", preemptions=" << metrics.preemptionCount
      << ", resumes=" << metrics.resumeCount << '\n';
  log << "Shared page-pool lifecycle: " << passFail(poolPassed)
      << ", peak_pages_per_layer=" << metrics.peakPagesPerLayer
      << ", allocations=" << metrics.pageAllocationCount
      << ", releases=" << metrics.pageReleaseCount << '\n';
  log << "Batched decode submissions: " << passFail(batchingPassed)
      << ", decoded_tokens=" << metrics.decodedTokens
      << ", Metal_submissions=" << metrics.decodeCommandSubmissions
      << '\n';
  printStatistics("Chunked prefill GPU per request", metrics.prefillGpu, log);
  printStatistics("Chunked prefill end-to-end per request",
                  metrics.prefillEndToEnd, log);
  printStatistics("Batched decode GPU per submission", metrics.decodeGpu, log);
  printStatistics("Batched decode end-to-end per submission",
                  metrics.decodeEndToEnd, log);
  log << "Continuous serving total(us): " << metrics.totalTimeUs
      << ", tokens/s=" << metrics.tokensPerSecond
      << ", throughput_vs_sequential="
      << (sequential.tokensPerSecond > 0.0
              ? metrics.tokensPerSecond / sequential.tokensPerSecond
              : 0.0)
      << "x\n"
      << "Scheduler overhead(us): " << metrics.schedulerOverheadUs
      << ", KV management overhead(us): "
      << metrics.kvManagementOverheadUs
      << ", preemption overhead(us): " << metrics.preemptionOverheadUs
      << '\n';
  passed &= schedulerPassed && preemptionPassed && poolPassed && batchingPassed;
  log << "Continuous batching + preemption validation: " << passFail(passed)
      << '\n';
  return passed;
}
