#include "decoder_llm_main.hpp"

#include "importer/DecoderLLMImporter.hpp"
#include "runtime/DecoderLLMExecutor.hpp"
#include "validation/Validator.hpp"

#include <algorithm>
#include <optional>
#include <ostream>
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

const char *passFail(bool passed) { return passed ? "PASS" : "FAIL"; }

bool validate(const std::string &label, const std::vector<float> &actual,
              const std::vector<double> &reference, std::ostream &log,
              double absoluteTolerance = kAbsoluteTolerance,
              double relativeTolerance = kRelativeTolerance) {
  const auto result = tensor::validation::compare(
      actual, reference, absoluteTolerance, relativeTolerance);
  log << label << ": " << passFail(result.passed)
      << ", max absolute error=" << result.maxAbsoluteError << '\n';
  if (!result.passed) log << "Validation error: " << result.errorMessage << '\n';
  return result.passed;
}

std::optional<double> median(std::vector<double> values) {
  if (values.empty()) return std::nullopt;
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

bool validateState(const tensor::runtime::CompiledDecoderLLM &decoder,
                   const tensor::DecoderLLMReference &reference,
                   std::size_t expectedLength, std::ostream &log,
                   double absoluteTolerance, double relativeTolerance) {
  bool passed = true;
  for (std::size_t layer = 0; layer < reference.keyCaches.size(); ++layer) {
    passed &= validate("  layer " + std::to_string(layer) + " key cache",
                       decoder.readKeyPrefix(layer), reference.keyCaches[layer],
                       log, absoluteTolerance, relativeTolerance);
    passed &= validate("  layer " + std::to_string(layer) + " value cache",
                       decoder.readValuePrefix(layer),
                       reference.valueCaches[layer], log, absoluteTolerance,
                       relativeTolerance);
  }
  const bool lengthPassed = decoder.currentLength() == expectedLength;
  log << "  cache length: " << passFail(lengthPassed)
      << ", current_length=" << decoder.currentLength() << '\n';
  return passed && lengthPassed;
}

} // namespace

bool runDecoderLLMWorkload(tensor::metal::MetalRuntime &runtime,
                           const std::string &manifestPath,
                           std::ostream &log, const std::string &kernelLibrary) {
  auto imported = tensor::importer::importDecoderLLMWorkload(manifestPath);
  if (!imported.workload) {
    log << "Decoder-only import: FAIL\n"
        << "Import error: " << imported.errorMessage << '\n';
    return false;
  }
  const auto &workload = *imported.workload;
  const auto &plan = workload.plan;
  log << "Decoder-only import: PASS\n"
      << "Model: " << workload.modelName << '\n'
      << "Decoder config: batch=" << plan.attention.batch
      << ", layers=" << plan.layerCount
      << ", hidden=" << plan.hiddenSize()
      << ", heads=" << plan.attention.heads
      << ", head_dim=" << plan.attention.headDimension
      << ", intermediate=" << plan.intermediateSize
      << ", vocab=" << plan.vocabularySize
      << ", prefill=" << plan.attention.prefillLength
      << ", decode_steps=" << workload.decodeSteps.size()
      << ", capacity=" << plan.attention.capacity << '\n';

  tensor::runtime::DecoderLLMCompileOptions options;
  options.kernelLibrary = kernelLibrary;
  options.storageDtype = plan.attention.dtype;
  options.requestedStorageDtype = workload.requestedStorageDtype;
  options.allowPrecisionFallback = true;
  auto compilation =
      tensor::runtime::compileDecoderLLM(runtime, workload, log, options);
  if (!compilation.executable) {
    log << "Decoder-only compilation: FAIL\n"
        << "Compiler error: " << compilation.errorMessage << '\n';
    return false;
  }
  auto &decoder = *compilation.executable;
  const auto [absoluteTolerance, relativeTolerance] =
      tolerances(decoder.storageDtype());
  log << "Decoder storage: " << tensor::dtypeName(decoder.storageDtype())
      << ", precision_fallback="
      << (decoder.precisionFallbackUsed() ? "true" : "false")
      << ", model_bytes=" << decoder.modelStorageBytes()
      << ", kv_bytes=" << decoder.kvStorageBytes()
      << ", activation_bytes=" << decoder.activationStorageBytes() << '\n';
  bool passed = true;
  std::vector<double> decodeTimes;

  const auto prefill = decoder.prefill(workload.prefill.tokenIds);
  if (!prefill.passed) {
    log << "Decoder prefill execution: FAIL\n"
        << "Metal error: " << prefill.errorMessage << '\n';
    return false;
  }
  log << "Decoder prefill execution: PASS\n";
  passed &= validate("  logits", prefill.logits, workload.prefill.logits, log,
                     absoluteTolerance, relativeTolerance);
  passed &= validate("  next token", prefill.nextTokenIds,
                     workload.prefill.nextTokenIds, log, 0.0, 0.0);
  passed &= validateState(decoder, workload.prefill,
                          plan.attention.prefillLength, log, absoluteTolerance,
                          relativeTolerance);
  if (prefill.gpuExecutionTimeUs) {
    log << "Decoder prefill GPU time (us): " << *prefill.gpuExecutionTimeUs
        << '\n';
  }

  auto nextTokens = prefill.nextTokenIds;
  for (std::size_t step = 0; step < workload.decodeSteps.size(); ++step) {
    const auto &reference = workload.decodeSteps[step];
    const std::vector<double> expectedInput(reference.tokenIds.begin(),
                                            reference.tokenIds.end());
    const bool chainPassed = validate(
        "Decoder step " + std::to_string(step) + " input token", nextTokens,
        expectedInput, log, 0.0, 0.0);
    const auto result = decoder.decode(nextTokens);
    if (!result.passed) {
      log << "Decoder step " << step << ": FAIL\n"
          << "Metal error: " << result.errorMessage << '\n';
      return false;
    }
    bool stepPassed = chainPassed;
    stepPassed &= validate("  logits", result.logits, reference.logits, log,
                           absoluteTolerance, relativeTolerance);
    stepPassed &= validate("  next token", result.nextTokenIds,
                           reference.nextTokenIds, log, 0.0, 0.0);
    stepPassed &= validateState(
        decoder, reference, plan.attention.prefillLength + step + 1, log,
        absoluteTolerance, relativeTolerance);
    log << "Decoder step " << step << ": " << passFail(stepPassed) << '\n';
    passed &= stepPassed;
    nextTokens = result.nextTokenIds;
    if (result.gpuExecutionTimeUs) {
      decodeTimes.push_back(*result.gpuExecutionTimeUs);
    }
  }

  const bool reused = decoder.cacheStorageReused();
  log << "Decoder per-layer cache storage reuse: " << passFail(reused) << '\n';
  passed &= reused;

  const auto decodeMedian = median(std::move(decodeTimes));
  if (decodeMedian) {
    log << "Decoder median single-token GPU time (us): " << *decodeMedian
        << '\n';
  } else {
    log << "Decoder median single-token GPU time (us): unavailable\n";
  }
  decoder.reportKernelUsage(log);
  log << "Decoder-only validation: " << passFail(passed) << '\n';
  return passed;
}
