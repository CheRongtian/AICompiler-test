#include "importer/DecoderLLMImporter.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace tensor::importer {
namespace {

struct TensorRange {
  std::size_t offset = 0;
  std::size_t count = 0;
};

void expect(std::istream &input, const char *expected) {
  std::string token;
  if (!(input >> token) || token != expected) {
    throw std::runtime_error(std::string("Expected '") + expected +
                             "' in decoder-only manifest.");
  }
}

std::size_t readSize(std::istream &input, const char *description) {
  std::size_t value = 0;
  if (!(input >> value)) {
    throw std::runtime_error(std::string("Invalid ") + description +
                             " in decoder-only manifest.");
  }
  return value;
}

float readFloat(std::istream &input, const char *description) {
  float value = 0.0f;
  if (!(input >> value)) {
    throw std::runtime_error(std::string("Invalid ") + description +
                             " in decoder-only manifest.");
  }
  return value;
}

std::string readQuoted(std::istream &input, const char *description) {
  std::string value;
  if (!(input >> std::quoted(value))) {
    throw std::runtime_error(std::string("Invalid ") + description +
                             " in decoder-only manifest.");
  }
  return value;
}

std::vector<float> readPayload(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("Unable to open decoder-only payload: " + path.string());
  }
  const auto end = input.tellg();
  if (end < 0) throw std::runtime_error("Unable to determine decoder-only payload size.");
  const auto bytes = static_cast<std::streamoff>(end);
  if (bytes % static_cast<std::streamoff>(sizeof(float)) != 0) {
    throw std::runtime_error("Decoder-only payload is not float32-aligned.");
  }
  std::vector<float> values(static_cast<std::size_t>(
      bytes / static_cast<std::streamoff>(sizeof(float))));
  input.seekg(0);
  if (!values.empty() &&
      !input.read(reinterpret_cast<char *>(values.data()),
                  static_cast<std::streamsize>(bytes))) {
    throw std::runtime_error("Unable to read the complete decoder-only payload.");
  }
  return values;
}

std::vector<float> required(
    const std::unordered_map<std::string, TensorRange> &ranges,
    const std::vector<float> &payload, const std::string &name,
    std::size_t expectedCount) {
  const auto found = ranges.find(name);
  if (found == ranges.end()) {
    throw std::runtime_error("Missing decoder-only tensor '" + name + "'.");
  }
  const auto range = found->second;
  if (range.count != expectedCount || range.offset > payload.size() ||
      range.count > payload.size() - range.offset) {
    throw std::runtime_error("Invalid decoder-only tensor range for '" + name + "'.");
  }
  return {payload.begin() + static_cast<std::ptrdiff_t>(range.offset),
          payload.begin() + static_cast<std::ptrdiff_t>(range.offset + range.count)};
}

std::vector<double> asDouble(std::vector<float> values) {
  return {values.begin(), values.end()};
}

DecoderLLMReference readReference(
    const std::unordered_map<std::string, TensorRange> &ranges,
    const std::vector<float> &payload, const std::string &prefix,
    const planner::DecoderLLMPlan &plan, std::size_t sequenceLength,
    std::size_t cacheLength) {
  DecoderLLMReference reference;
  reference.tokenIds = required(
      ranges, payload, prefix + ".tokens", plan.attention.batch * sequenceLength);
  reference.logits = asDouble(required(
      ranges, payload, prefix + ".logits", plan.logitsElementCount(sequenceLength)));
  reference.nextTokenIds = asDouble(required(
      ranges, payload, prefix + ".next_tokens", plan.attention.batch));
  const auto cacheCount = plan.attention.batch * plan.attention.heads *
                          cacheLength * plan.attention.headDimension;
  for (std::size_t layer = 0; layer < plan.layerCount; ++layer) {
    const auto layerPrefix = prefix + ".layer" + std::to_string(layer);
    reference.keyCaches.push_back(asDouble(required(
        ranges, payload, layerPrefix + ".key_cache", cacheCount)));
    reference.valueCaches.push_back(asDouble(required(
        ranges, payload, layerPrefix + ".value_cache", cacheCount)));
  }
  return reference;
}

DecoderLLMLayerParameters readLayer(
    const std::unordered_map<std::string, TensorRange> &ranges,
    const std::vector<float> &payload, std::size_t index,
    const planner::DecoderLLMPlan &plan) {
  const auto prefix = "layer" + std::to_string(index);
  const auto hidden = plan.hiddenSize();
  const auto hiddenMatrix = hidden * hidden;
  DecoderLLMLayerParameters layer;
  layer.inputNormWeight = required(
      ranges, payload, prefix + ".input_norm.weight", hidden);
  layer.queryWeight = required(ranges, payload, prefix + ".q.weight", hiddenMatrix);
  layer.keyWeight = required(ranges, payload, prefix + ".k.weight", hiddenMatrix);
  layer.valueWeight = required(ranges, payload, prefix + ".v.weight", hiddenMatrix);
  layer.outputWeight = required(ranges, payload, prefix + ".o.weight", hiddenMatrix);
  layer.postAttentionNormWeight = required(
      ranges, payload, prefix + ".post_norm.weight", hidden);
  layer.gateWeight = required(
      ranges, payload, prefix + ".gate.weight", plan.intermediateSize * hidden);
  layer.upWeight = required(
      ranges, payload, prefix + ".up.weight", plan.intermediateSize * hidden);
  layer.downWeight = required(
      ranges, payload, prefix + ".down.weight", hidden * plan.intermediateSize);
  return layer;
}

} // namespace

DecoderLLMImportResult importDecoderLLMWorkload(const std::string &manifestPath) {
  DecoderLLMImportResult result;
  try {
    std::ifstream input(manifestPath);
    if (!input) {
      throw std::runtime_error("Unable to open decoder-only manifest: " + manifestPath);
    }
    std::string magic;
    std::size_t version = 0;
    if (!(input >> magic >> version) || magic != "TMC_DECODER_LLM" || version != 1) {
      throw std::runtime_error("Invalid or unsupported decoder-only manifest header.");
    }

    auto workload = std::make_unique<DecoderLLMWorkload>();
    expect(input, "MODEL");
    workload->modelName = readQuoted(input, "model name");
    expect(input, "PAYLOAD");
    const auto payloadName = readQuoted(input, "payload path");
    expect(input, "CONFIG");
    auto &plan = workload->plan;
    plan.attention.batch = readSize(input, "batch size");
    const auto hiddenSize = readSize(input, "hidden size");
    plan.attention.heads = readSize(input, "head count");
    if (plan.attention.heads == 0 || hiddenSize % plan.attention.heads != 0) {
      throw std::runtime_error("Decoder hidden size must be divisible by head count.");
    }
    plan.attention.headDimension = hiddenSize / plan.attention.heads;
    plan.intermediateSize = readSize(input, "intermediate size");
    plan.vocabularySize = readSize(input, "vocabulary size");
    plan.layerCount = readSize(input, "layer count");
    plan.attention.capacity = readSize(input, "cache capacity");
    plan.attention.prefillLength = readSize(input, "prefill length");
    workload->decodeCount = readSize(input, "decode step count");
    plan.rmsNormEpsilon = readFloat(input, "RMSNorm epsilon");
    plan.attention.dtype = DType::Float32;
    plan.attention.threadsPerThreadgroup = 128;
    plan.validate();
    if (workload->decodeCount >
        plan.attention.capacity - plan.attention.prefillLength) {
      throw std::runtime_error("Decoder decode steps exceed cache capacity.");
    }

    expect(input, "TENSORS");
    const auto tensorCount = readSize(input, "named tensor count");
    std::unordered_map<std::string, TensorRange> ranges;
    for (std::size_t index = 0; index < tensorCount; ++index) {
      expect(input, "TENSOR");
      const auto name = readQuoted(input, "tensor name");
      const TensorRange range{readSize(input, "tensor offset"),
                              readSize(input, "tensor length")};
      if (!ranges.emplace(name, range).second) {
        throw std::runtime_error("Duplicate decoder-only tensor '" + name + "'.");
      }
    }
    expect(input, "END");
    std::string trailing;
    if (input >> trailing) {
      throw std::runtime_error("Unexpected data after decoder-only END marker.");
    }

    const auto manifest = std::filesystem::absolute(std::filesystem::path(manifestPath));
    const auto payload = readPayload(manifest.parent_path() / payloadName);
    const auto hidden = plan.hiddenSize();
    workload->embeddingWeight = required(
        ranges, payload, "embedding.weight", plan.vocabularySize * hidden);
    workload->finalNormWeight = required(
        ranges, payload, "final_norm.weight", hidden);
    workload->languageModelHeadWeight = required(
        ranges, payload, "lm_head.weight", plan.vocabularySize * hidden);
    const auto ropeCount = plan.attention.capacity * plan.attention.headDimension / 2;
    workload->ropeCosine = required(ranges, payload, "rope.cosine", ropeCount);
    workload->ropeSine = required(ranges, payload, "rope.sine", ropeCount);
    for (std::size_t layer = 0; layer < plan.layerCount; ++layer) {
      workload->layers.push_back(readLayer(ranges, payload, layer, plan));
    }
    workload->prefill = readReference(
        ranges, payload, "prefill", plan, plan.attention.prefillLength,
        plan.attention.prefillLength);
    for (std::size_t step = 0; step < workload->decodeCount; ++step) {
      workload->decodeSteps.push_back(readReference(
          ranges, payload, "decode" + std::to_string(step), plan, 1,
          plan.attention.prefillLength + step + 1));
    }
    result.workload = std::move(workload);
  } catch (const std::exception &error) {
    result.errorMessage = error.what();
  }
  return result;
}

} // namespace tensor::importer
