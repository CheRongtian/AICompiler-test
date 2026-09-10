#include "paged_kv_main.hpp"

#include "importer/DecoderLLMImporter.hpp"
#include "runtime/DecoderLLMExecutor.hpp"
#include "validation/Validator.hpp"

#include <algorithm>
#include <optional>
#include <ostream>
#include <set>
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

bool validateState(const tensor::runtime::CompiledDecoderLLM &decoder,
                   const tensor::DecoderLLMReference &reference,
                   std::size_t expectedLength, std::ostream &log,
                   double absoluteTolerance, double relativeTolerance) {
  bool passed = decoder.currentLength() == expectedLength;
  log << "  paged cache length: " << passFail(passed)
      << ", current_length=" << decoder.currentLength() << '\n';
  for (std::size_t layer = 0; layer < reference.keyCaches.size(); ++layer) {
    passed &= validate("  layer " + std::to_string(layer) + " key cache",
                       decoder.readKeyPrefix(layer), reference.keyCaches[layer],
                       log, absoluteTolerance, relativeTolerance);
    passed &= validate("  layer " + std::to_string(layer) + " value cache",
                       decoder.readValuePrefix(layer),
                       reference.valueCaches[layer], log, absoluteTolerance,
                       relativeTolerance);
  }
  return passed;
}

std::optional<double> median(std::vector<double> values) {
  if (values.empty()) return std::nullopt;
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

bool validBlockTable(const std::vector<std::int32_t> &table,
                     std::size_t expectedPages) {
  if (table.size() != expectedPages) return false;
  std::set<std::int32_t> physicalPages;
  for (const auto page : table) {
    if (page < 0 || !physicalPages.insert(page).second) return false;
  }
  return true;
}

void printBlockTable(const std::vector<std::int32_t> &table,
                     std::ostream &log) {
  log << "Block table: [";
  for (std::size_t index = 0; index < table.size(); ++index) {
    if (index != 0) log << ',';
    log << table[index];
  }
  log << "]\n";
}

} // namespace

bool runPagedKVWorkload(tensor::metal::MetalRuntime &runtime,
                        const std::string &manifestPath,
                        std::size_t pageSize, std::size_t chunkSize,
                        std::ostream &log,
                        const std::string &kernelLibrary) {
  auto imported = tensor::importer::importDecoderLLMWorkload(manifestPath);
  if (!imported.workload) {
    log << "Paged KV import: FAIL\nImport error: " << imported.errorMessage
        << '\n';
    return false;
  }
  const auto &workload = *imported.workload;
  const auto &plan = workload.plan;
  tensor::runtime::DecoderLLMCompileOptions options;
  options.kernelLibrary = kernelLibrary;
  options.kvPageSize = pageSize;
  options.prefillChunkSize = chunkSize;
  options.storageDtype = plan.attention.dtype;
  options.requestedStorageDtype = workload.requestedStorageDtype;
  options.allowPrecisionFallback = true;
  auto compilation =
      tensor::runtime::compileDecoderLLM(runtime, workload, log, options);
  if (!compilation.executable) {
    log << "Paged KV compilation: FAIL\nCompiler error: "
        << compilation.errorMessage << '\n';
    return false;
  }
  auto &decoder = *compilation.executable;
  const auto [absoluteTolerance, relativeTolerance] =
      tolerances(decoder.storageDtype());
  const auto chunkCount =
      (plan.attention.prefillLength + chunkSize - 1) / chunkSize;
  log << "Paged KV configuration: page_size=" << pageSize
      << ", capacity=" << plan.attention.capacity
      << ", chunk_size=" << chunkSize << ", chunks=" << chunkCount
      << ", storage=" << tensor::dtypeName(decoder.storageDtype())
      << ", precision_fallback="
      << (decoder.precisionFallbackUsed() ? "true" : "false")
      << ", model_bytes=" << decoder.modelStorageBytes()
      << ", kv_bytes=" << decoder.kvStorageBytes()
      << ", activation_bytes=" << decoder.activationStorageBytes() << '\n';

  bool passed = decoder.usesPagedKVCache() && decoder.kvPageSize() == pageSize &&
                decoder.prefillChunkSize() == chunkSize;
  log << "Paged KV runtime selection: " << passFail(passed) << '\n';

  const auto prefill = decoder.prefillChunked(workload.prefill.tokenIds);
  if (!prefill.passed) {
    log << "Chunked prefill execution: FAIL\nMetal error: "
        << prefill.errorMessage << '\n';
    return false;
  }
  log << "Chunked prefill execution: PASS\n";
  passed &= validate("  full prefill logits", prefill.logits,
                     workload.prefill.logits, log, absoluteTolerance,
                     relativeTolerance);
  passed &= validate("  next token", prefill.nextTokenIds,
                     workload.prefill.nextTokenIds, log, 0.0, 0.0);
  passed &= validateState(decoder, workload.prefill,
                          plan.attention.prefillLength, log,
                          absoluteTolerance, relativeTolerance);
  const auto prefillPages =
      (plan.attention.prefillLength + pageSize - 1) / pageSize;
  const auto table = decoder.kvBlockTable(0);
  bool tablePassed = decoder.allocatedKVPageCount() == prefillPages;
  for (std::size_t layer = 0; layer < plan.layerCount; ++layer) {
    tablePassed &= validBlockTable(decoder.kvBlockTable(layer), prefillPages);
  }
  log << "Prefill block table: " << passFail(tablePassed)
      << ", allocated_pages=" << decoder.allocatedKVPageCount() << '\n';
  printBlockTable(table, log);
  passed &= tablePassed;
  if (prefill.gpuExecutionTimeUs) {
    log << "Chunked prefill GPU time (us): " << *prefill.gpuExecutionTimeUs
        << '\n';
  }

  auto nextTokens = prefill.nextTokenIds;
  std::vector<double> decodeTimes;
  for (std::size_t step = 0; step < workload.decodeSteps.size(); ++step) {
    const auto &reference = workload.decodeSteps[step];
    const auto result = decoder.decode(nextTokens);
    if (!result.passed) {
      log << "Paged decode step " << step << ": FAIL\nMetal error: "
          << result.errorMessage << '\n';
      return false;
    }
    bool stepPassed = validate("  logits", result.logits, reference.logits, log,
                               absoluteTolerance, relativeTolerance);
    stepPassed &= validate("  next token", result.nextTokenIds,
                           reference.nextTokenIds, log, 0.0, 0.0);
    stepPassed &= validateState(
        decoder, reference, plan.attention.prefillLength + step + 1, log,
        absoluteTolerance, relativeTolerance);
    log << "Paged decode step " << step << ": " << passFail(stepPassed)
        << '\n';
    passed &= stepPassed;
    nextTokens = result.nextTokenIds;
    if (result.gpuExecutionTimeUs) {
      decodeTimes.push_back(*result.gpuExecutionTimeUs);
    }
  }

  while (decoder.currentLength() < plan.attention.capacity) {
    const auto fill = decoder.decode(nextTokens);
    if (!fill.passed) {
      log << "Paged KV capacity fill: FAIL\nMetal error: "
          << fill.errorMessage << '\n';
      return false;
    }
    nextTokens = fill.nextTokenIds;
  }
  std::vector<std::vector<float>> keysAtCapacity;
  std::vector<std::vector<float>> valuesAtCapacity;
  for (std::size_t layer = 0; layer < plan.layerCount; ++layer) {
    keysAtCapacity.push_back(decoder.readKeyPrefix(layer));
    valuesAtCapacity.push_back(decoder.readValuePrefix(layer));
  }
  const auto overflow = decoder.decode(nextTokens);
  bool capacityPagesPassed =
      decoder.allocatedKVPageCount() ==
      (plan.attention.capacity + pageSize - 1) / pageSize;
  bool overflowPassed =
      !overflow.passed && decoder.currentLength() == plan.attention.capacity;
  const auto capacityPages =
      (plan.attention.capacity + pageSize - 1) / pageSize;
  for (std::size_t layer = 0; layer < plan.layerCount; ++layer) {
    capacityPagesPassed &=
        validBlockTable(decoder.kvBlockTable(layer), capacityPages);
    overflowPassed &= decoder.readKeyPrefix(layer) == keysAtCapacity[layer] &&
                      decoder.readValuePrefix(layer) == valuesAtCapacity[layer];
  }
  log << "Paged KV full-capacity state: "
      << passFail(capacityPagesPassed)
      << ", current_length=" << decoder.currentLength()
      << ", allocated_pages=" << decoder.allocatedKVPageCount() << '\n';
  log << "Paged KV overflow rejection: " << passFail(overflowPassed) << '\n';
  if (!overflow.errorMessage.empty()) {
    log << "Overflow error: " << overflow.errorMessage << '\n';
  }
  passed &= capacityPagesPassed && overflowPassed;

  const bool reused = decoder.cacheStorageReused();
  log << "Paged KV storage reuse: " << passFail(reused) << '\n';
  passed &= reused;
  const auto resetError = decoder.reset();
  bool resetPassed = resetError.empty() && decoder.currentLength() == 0 &&
                     decoder.allocatedKVPageCount() == 0;
  for (std::size_t layer = 0; layer < plan.layerCount; ++layer) {
    resetPassed &= decoder.kvBlockTable(layer).empty();
  }
  log << "Paged KV reset/release: " << passFail(resetPassed) << '\n';
  if (!resetError.empty()) log << "Reset error: " << resetError << '\n';
  passed &= resetPassed;

  const auto decodeMedian = median(std::move(decodeTimes));
  if (decodeMedian) {
    log << "Paged KV median single-token GPU time (us): " << *decodeMedian
        << '\n';
  }
  decoder.reportKernelUsage(log);
  log << "Paged KV + chunked prefill validation: " << passFail(passed) << '\n';
  return passed;
}
