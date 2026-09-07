#include "importer/KVCacheImporter.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace tensor::importer {
namespace {

void expect(std::istream &input, const char *expected) {
  std::string token;
  if (!(input >> token) || token != expected) {
    throw std::runtime_error(std::string("Expected '") + expected +
                             "' in KV cache manifest.");
  }
}

std::size_t readSize(std::istream &input, const char *description) {
  std::size_t value = 0;
  if (!(input >> value)) {
    throw std::runtime_error(std::string("Invalid ") + description +
                             " in KV cache manifest.");
  }
  return value;
}

std::string readQuoted(std::istream &input, const char *description) {
  std::string value;
  if (!(input >> std::quoted(value))) {
    throw std::runtime_error(std::string("Invalid ") + description +
                             " in KV cache manifest.");
  }
  return value;
}

std::vector<float> readPayload(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("Unable to open KV cache payload: " + path.string());
  const auto end = input.tellg();
  if (end < 0) throw std::runtime_error("Unable to determine KV cache payload size.");
  const auto bytes = static_cast<std::streamoff>(end);
  if (bytes % static_cast<std::streamoff>(sizeof(float)) != 0) {
    throw std::runtime_error("KV cache payload is not float32-aligned.");
  }
  std::vector<float> values(static_cast<std::size_t>(
      bytes / static_cast<std::streamoff>(sizeof(float))));
  input.seekg(0);
  if (!values.empty() &&
      !input.read(reinterpret_cast<char *>(values.data()),
                  static_cast<std::streamsize>(bytes))) {
    throw std::runtime_error("Unable to read the complete KV cache payload.");
  }
  return values;
}

std::vector<float> slice(const std::vector<float> &payload,
                         std::size_t offset, std::size_t count) {
  if (offset > payload.size() || count > payload.size() - offset) {
    throw std::runtime_error("KV cache tensor range is outside the payload.");
  }
  return {payload.begin() + static_cast<std::ptrdiff_t>(offset),
          payload.begin() + static_cast<std::ptrdiff_t>(offset + count)};
}

std::vector<double> asDouble(std::vector<float> values) {
  return {values.begin(), values.end()};
}

struct TensorRange { std::size_t offset = 0; std::size_t count = 0; };

std::vector<float> requiredTensor(
    const std::unordered_map<std::string, TensorRange> &ranges,
    const std::vector<float> &payload, const char *name,
    std::size_t expectedCount) {
  const auto found = ranges.find(name);
  if (found == ranges.end()) {
    throw std::runtime_error(std::string("Missing KV cache tensor '") + name + "'.");
  }
  if (found->second.count != expectedCount) {
    throw std::runtime_error(std::string("Unexpected element count for KV cache tensor '") +
                             name + "'.");
  }
  return slice(payload, found->second.offset, found->second.count);
}

KVCacheReferenceStep readStep(std::istream &input,
                              const std::vector<float> &payload,
                              std::size_t expectedIndex,
                              std::size_t inputCount,
                              std::size_t cacheCount) {
  expect(input, "STEP");
  const auto index = readSize(input, "decode step index");
  if (index != expectedIndex) throw std::runtime_error("KV cache decode steps are out of order.");
  auto readRange = [&]() {
    const auto offset = readSize(input, "step tensor offset");
    const auto count = readSize(input, "step tensor length");
    return slice(payload, offset, count);
  };
  KVCacheReferenceStep step;
  step.query = readRange();
  step.key = readRange();
  step.value = readRange();
  step.output = asDouble(readRange());
  step.keyCache = asDouble(readRange());
  step.valueCache = asDouble(readRange());
  if (step.query.size() != inputCount || step.key.size() != inputCount ||
      step.value.size() != inputCount || step.output.size() != inputCount ||
      step.keyCache.size() != cacheCount || step.valueCache.size() != cacheCount) {
    throw std::runtime_error("KV cache decode step tensor shape is inconsistent with CONFIG.");
  }
  return step;
}

} // namespace

KVCacheImportResult importKVCacheWorkload(const std::string &manifestPath) {
  KVCacheImportResult result;
  try {
    std::ifstream input(manifestPath);
    if (!input) throw std::runtime_error("Unable to open KV cache manifest: " + manifestPath);
    std::string magic;
    std::size_t version = 0;
    if (!(input >> magic >> version) || magic != "TMC_KV_CACHE" || version != 1) {
      throw std::runtime_error("Invalid or unsupported KV cache manifest header.");
    }

    auto workload = std::make_unique<ImportedKVCacheWorkload>();
    expect(input, "MODEL");
    workload->modelName = readQuoted(input, "model name");
    expect(input, "PAYLOAD");
    const auto payloadName = readQuoted(input, "payload path");
    const auto manifest = std::filesystem::absolute(std::filesystem::path(manifestPath));
    const auto payload = readPayload(manifest.parent_path() / payloadName);

    expect(input, "CONFIG");
    workload->plan.batch = readSize(input, "batch");
    workload->plan.heads = readSize(input, "head count");
    workload->plan.headDimension = readSize(input, "head dimension");
    workload->plan.capacity = readSize(input, "cache capacity");
    workload->plan.prefillLength = readSize(input, "prefill length");
    const auto decodeCount = readSize(input, "decode step count");
    workload->plan.dtype = DType::Float32;
    workload->plan.threadsPerThreadgroup = 128;
    workload->plan.validate();
    if (decodeCount > workload->plan.capacity - workload->plan.prefillLength) {
      throw std::runtime_error("Decode steps exceed the configured cache capacity.");
    }

    expect(input, "TENSORS");
    const auto tensorCount = readSize(input, "named tensor count");
    std::unordered_map<std::string, TensorRange> ranges;
    for (std::size_t index = 0; index < tensorCount; ++index) {
      expect(input, "TENSOR");
      const auto name = readQuoted(input, "tensor name");
      TensorRange range{readSize(input, "tensor offset"),
                        readSize(input, "tensor length")};
      if (!ranges.emplace(name, range).second) {
        throw std::runtime_error("Duplicate named tensor in KV cache manifest.");
      }
    }

    const auto dimension = workload->plan.modelDimension();
    const auto weightCount = dimension * dimension;
    const auto prefillCount = workload->plan.inputElementCount(workload->plan.prefillLength);
    workload->queryWeight = requiredTensor(ranges, payload, "query_weight", weightCount);
    workload->keyWeight = requiredTensor(ranges, payload, "key_weight", weightCount);
    workload->valueWeight = requiredTensor(ranges, payload, "value_weight", weightCount);
    workload->outputWeight = requiredTensor(ranges, payload, "output_weight", weightCount);
    workload->prefill.query = requiredTensor(ranges, payload, "prefill_query", prefillCount);
    workload->prefill.key = requiredTensor(ranges, payload, "prefill_key", prefillCount);
    workload->prefill.value = requiredTensor(ranges, payload, "prefill_value", prefillCount);
    workload->prefill.output = asDouble(
        requiredTensor(ranges, payload, "prefill_output", prefillCount));
    workload->prefill.keyCache = asDouble(
        requiredTensor(ranges, payload, "prefill_key_cache", prefillCount));
    workload->prefill.valueCache = asDouble(
        requiredTensor(ranges, payload, "prefill_value_cache", prefillCount));

    expect(input, "STEPS");
    const auto declaredSteps = readSize(input, "declared decode step count");
    if (declaredSteps != decodeCount) {
      throw std::runtime_error("KV cache CONFIG and STEPS counts disagree.");
    }
    const auto decodeInputCount = workload->plan.inputElementCount(1);
    for (std::size_t index = 0; index < decodeCount; ++index) {
      const auto length = workload->plan.prefillLength + index + 1;
      const auto cacheCount = workload->plan.batch * workload->plan.heads *
                              length * workload->plan.headDimension;
      workload->decodeSteps.push_back(
          readStep(input, payload, index, decodeInputCount, cacheCount));
    }
    expect(input, "END");
    std::string trailing;
    if (input >> trailing) throw std::runtime_error("Unexpected data after KV cache END marker.");
    result.workload = std::move(workload);
  } catch (const std::exception &error) {
    result.errorMessage = error.what();
  }
  return result;
}

} // namespace tensor::importer
