#include "kv_cache_main.hpp"

#include "importer/KVCacheImporter.hpp"
#include "runtime/StatefulExecutor.hpp"
#include "validation/Validator.hpp"

#include <algorithm>
#include <optional>
#include <ostream>
#include <utility>
#include <vector>

namespace {

constexpr double kAbsoluteTolerance = 2e-4;
constexpr double kRelativeTolerance = 2e-3;

const char *passFail(bool passed) { return passed ? "PASS" : "FAIL"; }

bool validate(const char *label, const std::vector<float> &actual,
              const std::vector<double> &reference, std::ostream &log) {
  const auto result = tensor::validation::compare(
      actual, reference, kAbsoluteTolerance, kRelativeTolerance);
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

} // namespace

bool runKVCacheWorkload(tensor::metal::MetalRuntime &runtime,
                        const std::string &manifestPath,
                        std::ostream &log) {
  auto imported = tensor::importer::importKVCacheWorkload(manifestPath);
  if (!imported.workload) {
    log << "KV cache import: FAIL\n"
        << "Import error: " << imported.errorMessage << '\n';
    return false;
  }
  const auto &workload = *imported.workload;
  const auto &plan = workload.plan;
  log << "KV cache import: PASS\n"
      << "Model: " << workload.modelName << '\n'
      << "KV config: batch=" << plan.batch
      << ", heads=" << plan.heads
      << ", head_dim=" << plan.headDimension
      << ", prefill=" << plan.prefillLength
      << ", decode_steps=" << workload.decodeSteps.size()
      << ", capacity=" << plan.capacity << '\n';

  auto compilation = tensor::runtime::compileKVCacheAttention(
      runtime, plan, workload.queryWeight, workload.keyWeight,
      workload.valueWeight, workload.outputWeight, log);
  if (!compilation.executable) {
    log << "KV cache compilation: FAIL\n"
        << "Compiler error: " << compilation.errorMessage << '\n';
    return false;
  }

  bool passed = true;
  std::vector<double> decodeGpuTimes;
  const auto prefill = compilation.executable->prefill(
      workload.prefill.query, workload.prefill.key, workload.prefill.value);
  if (!prefill.passed) {
    log << "KV prefill execution: FAIL\n"
        << "Metal error: " << prefill.errorMessage << '\n';
    return false;
  }
  log << "KV prefill execution: PASS\n";
  passed &= validate("KV prefill output", prefill.output,
                     workload.prefill.output, log);
  passed &= validate("KV prefill key cache",
                     compilation.executable->readKeyPrefix(),
                     workload.prefill.keyCache, log);
  passed &= validate("KV prefill value cache",
                     compilation.executable->readValuePrefix(),
                     workload.prefill.valueCache, log);
  const bool prefillLength =
      compilation.executable->currentLength() == plan.prefillLength;
  log << "KV prefill length: " << passFail(prefillLength)
      << ", current_length=" << compilation.executable->currentLength() << '\n';
  passed &= prefillLength;
  if (prefill.gpuExecutionTimeUs) {
    log << "KV prefill GPU time (us): " << *prefill.gpuExecutionTimeUs << '\n';
  }

  for (std::size_t index = 0; index < workload.decodeSteps.size(); ++index) {
    const auto &reference = workload.decodeSteps[index];
    const auto result = compilation.executable->decode(
        reference.query, reference.key, reference.value);
    if (!result.passed) {
      log << "KV decode step " << index << ": FAIL\n"
          << "Metal error: " << result.errorMessage << '\n';
      return false;
    }
    bool stepPassed = true;
    stepPassed &= validate("  output", result.output, reference.output, log);
    stepPassed &= validate("  key cache",
                           compilation.executable->readKeyPrefix(),
                           reference.keyCache, log);
    stepPassed &= validate("  value cache",
                           compilation.executable->readValuePrefix(),
                           reference.valueCache, log);
    const auto expectedLength = plan.prefillLength + index + 1;
    const bool lengthPassed =
        compilation.executable->currentLength() == expectedLength;
    log << "  length: " << passFail(lengthPassed)
        << ", current_length=" << compilation.executable->currentLength() << '\n'
        << "KV decode step " << index << ": "
        << passFail(stepPassed && lengthPassed) << '\n';
    passed &= stepPassed && lengthPassed;
    if (result.gpuExecutionTimeUs) decodeGpuTimes.push_back(*result.gpuExecutionTimeUs);
  }

  const bool reachedCapacity =
      compilation.executable->currentLength() == plan.capacity;
  log << "KV cache reached capacity: " << passFail(reachedCapacity)
      << ", current_length=" << compilation.executable->currentLength() << '\n';
  passed &= reachedCapacity;

  if (reachedCapacity && !workload.decodeSteps.empty()) {
    const auto lengthBeforeOverflow = compilation.executable->currentLength();
    const auto keyBeforeOverflow = compilation.executable->readKeyPrefix();
    const auto valueBeforeOverflow = compilation.executable->readValuePrefix();
    const auto &input = workload.decodeSteps.back();
    const auto overflow = compilation.executable->decode(
        input.query, input.key, input.value);

    const bool overflowRejected =
        !overflow.passed &&
        overflow.errorMessage == "KV cache capacity has been reached.";
    const bool lengthUnchanged =
        compilation.executable->currentLength() == lengthBeforeOverflow;
    const bool cacheUnchanged =
        compilation.executable->readKeyPrefix() == keyBeforeOverflow &&
        compilation.executable->readValuePrefix() == valueBeforeOverflow;

    log << "KV overflow decode rejected: " << passFail(overflowRejected) << '\n'
        << "KV overflow length unchanged: " << passFail(lengthUnchanged) << '\n'
        << "KV overflow cache unchanged: " << passFail(cacheUnchanged) << '\n';
    if (!overflowRejected && !overflow.errorMessage.empty()) {
      log << "Overflow error: " << overflow.errorMessage << '\n';
    }
    passed &= overflowRejected && lengthUnchanged && cacheUnchanged;
  } else {
    log << "KV overflow decode rejected: FAIL\n"
        << "KV overflow length unchanged: FAIL\n"
        << "KV overflow cache unchanged: FAIL\n";
    passed = false;
  }

  const bool reused = compilation.executable->cacheStorageReused();
  log << "KV cache storage reuse: " << passFail(reused) << '\n';
  passed &= reused;
  const auto decodeMedian = median(std::move(decodeGpuTimes));
  if (decodeMedian) log << "KV decode median GPU time (us): " << *decodeMedian << '\n';
  else log << "KV decode median GPU time (us): unavailable\n";
  log << "KV cache stateful validation: " << passFail(passed) << '\n';
  return passed;
}
