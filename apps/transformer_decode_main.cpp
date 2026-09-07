#include "transformer_decode_main.hpp"

#include "importer/TransformerDecodeImporter.hpp"
#include "runtime/TransformerDecodeExecutor.hpp"
#include "validation/Validator.hpp"

#include <algorithm>
#include <optional>
#include <ostream>
#include <vector>

namespace {

constexpr double kAbsoluteTolerance = 5e-4;
constexpr double kRelativeTolerance = 5e-3;

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

bool validateState(
    const tensor::runtime::CompiledTransformerDecoder &decoder,
    const tensor::TransformerDecodeReference &reference,
    std::size_t expectedLength, std::ostream &log) {
  bool passed = true;
  for (std::size_t layer = 0; layer < reference.keyCaches.size(); ++layer) {
    passed &= validate("  layer " + std::to_string(layer) + " key cache",
                       decoder.readKeyPrefix(layer), reference.keyCaches[layer], log);
    passed &= validate("  layer " + std::to_string(layer) + " value cache",
                       decoder.readValuePrefix(layer), reference.valueCaches[layer], log);
  }
  const bool lengthPassed = decoder.currentLength() == expectedLength;
  log << "  cache length: " << passFail(lengthPassed)
      << ", current_length=" << decoder.currentLength() << '\n';
  return passed && lengthPassed;
}

} // namespace

bool runTransformerDecodeWorkload(tensor::metal::MetalRuntime &runtime,
                                  const std::string &manifestPath,
                                  std::ostream &log) {
  auto imported =
      tensor::importer::importTransformerDecodeWorkload(manifestPath);
  if (!imported.workload) {
    log << "Transformer decode import: FAIL\n"
        << "Import error: " << imported.errorMessage << '\n';
    return false;
  }
  const auto &workload = *imported.workload;
  const auto &plan = workload.plan;
  log << "Transformer decode import: PASS\n"
      << "Model: " << workload.modelName << '\n'
      << "Decoder config: batch=" << plan.selfAttention.batch
      << ", layers=" << plan.layerCount
      << ", heads=" << plan.selfAttention.heads
      << ", head_dim=" << plan.selfAttention.headDimension
      << ", d_ff=" << plan.feedForwardDimension
      << ", vocab=" << plan.vocabularySize
      << ", source_length=" << plan.sourceLength
      << ", prefill=" << plan.selfAttention.prefillLength
      << ", decode_steps=" << workload.decodeSteps.size()
      << ", capacity=" << plan.selfAttention.capacity << '\n';

  auto compilation =
      tensor::runtime::compileTransformerDecoder(runtime, workload, log);
  if (!compilation.executable) {
    log << "Transformer decode compilation: FAIL\n"
        << "Compiler error: " << compilation.errorMessage << '\n';
    return false;
  }
  auto &decoder = *compilation.executable;
  bool passed = true;
  std::vector<double> decodeTimes;

  const auto prefill = decoder.prefill(workload.prefill.tokenIds);
  if (!prefill.passed) {
    log << "Transformer prefill execution: FAIL\n"
        << "Metal error: " << prefill.errorMessage << '\n';
    return false;
  }
  log << "Transformer prefill execution: PASS\n";
  passed &= validate("  logits", prefill.logits, workload.prefill.logits, log);
  passed &= validate("  next token", prefill.nextTokenIds,
                     workload.prefill.nextTokenIds, log, 0.0, 0.0);
  passed &= validateState(decoder, workload.prefill,
                          plan.selfAttention.prefillLength, log);
  if (prefill.gpuExecutionTimeUs) {
    log << "Transformer prefill GPU time (us): "
        << *prefill.gpuExecutionTimeUs << '\n';
  }

  auto nextTokens = prefill.nextTokenIds;
  for (std::size_t step = 0; step < workload.decodeSteps.size(); ++step) {
    const auto &reference = workload.decodeSteps[step];
    const std::vector<double> expectedInput(reference.tokenIds.begin(),
                                            reference.tokenIds.end());
    const bool chainPassed = validate(
        "Transformer decode step " + std::to_string(step) + " input token",
        nextTokens, expectedInput, log, 0.0, 0.0);
    const auto result = decoder.decode(nextTokens);
    if (!result.passed) {
      log << "Transformer decode step " << step << ": FAIL\n"
          << "Metal error: " << result.errorMessage << '\n';
      return false;
    }
    bool stepPassed = chainPassed;
    stepPassed &= validate("  logits", result.logits, reference.logits, log);
    stepPassed &= validate("  next token", result.nextTokenIds,
                           reference.nextTokenIds, log, 0.0, 0.0);
    stepPassed &= validateState(
        decoder, reference, plan.selfAttention.prefillLength + step + 1, log);
    log << "Transformer decode step " << step << ": "
        << passFail(stepPassed) << '\n';
    passed &= stepPassed;
    nextTokens = result.nextTokenIds;
    if (result.gpuExecutionTimeUs) {
      decodeTimes.push_back(*result.gpuExecutionTimeUs);
    }
  }

  const bool reused = decoder.cacheStorageReused();
  log << "Transformer per-layer cache storage reuse: " << passFail(reused) << '\n';
  passed &= reused;

  if (decoder.currentLength() == plan.selfAttention.capacity &&
      !workload.decodeSteps.empty()) {
    const auto length = decoder.currentLength();
    std::vector<std::vector<float>> keys;
    std::vector<std::vector<float>> values;
    for (std::size_t layer = 0; layer < plan.layerCount; ++layer) {
      keys.push_back(decoder.readKeyPrefix(layer));
      values.push_back(decoder.readValuePrefix(layer));
    }
    const auto overflow = decoder.decode(nextTokens);
    bool unchanged = decoder.currentLength() == length;
    for (std::size_t layer = 0; layer < plan.layerCount; ++layer) {
      unchanged &= decoder.readKeyPrefix(layer) == keys[layer];
      unchanged &= decoder.readValuePrefix(layer) == values[layer];
    }
    const bool rejected =
        !overflow.passed &&
        overflow.errorMessage == "Transformer KV cache capacity has been reached.";
    log << "Transformer overflow decode rejected: " << passFail(rejected) << '\n'
        << "Transformer overflow state unchanged: " << passFail(unchanged) << '\n';
    passed &= rejected && unchanged;
  }

  const auto decodeMedian = median(std::move(decodeTimes));
  if (decodeMedian) {
    log << "Transformer decode median GPU time (us): " << *decodeMedian << '\n';
  } else {
    log << "Transformer decode median GPU time (us): unavailable\n";
  }
  log << "Transformer decode validation: " << passFail(passed) << '\n';
  return passed;
}
