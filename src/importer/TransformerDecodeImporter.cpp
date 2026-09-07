#include "importer/TransformerDecodeImporter.hpp"

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
                             "' in Transformer decode manifest.");
  }
}

std::size_t readSize(std::istream &input, const char *description) {
  std::size_t value = 0;
  if (!(input >> value)) {
    throw std::runtime_error(std::string("Invalid ") + description +
                             " in Transformer decode manifest.");
  }
  return value;
}

std::string readQuoted(std::istream &input, const char *description) {
  std::string value;
  if (!(input >> std::quoted(value))) {
    throw std::runtime_error(std::string("Invalid ") + description +
                             " in Transformer decode manifest.");
  }
  return value;
}

std::vector<float> readPayload(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("Unable to open Transformer decode payload: " +
                             path.string());
  }
  const auto end = input.tellg();
  if (end < 0) {
    throw std::runtime_error("Unable to determine Transformer decode payload size.");
  }
  const auto bytes = static_cast<std::streamoff>(end);
  if (bytes % static_cast<std::streamoff>(sizeof(float)) != 0) {
    throw std::runtime_error("Transformer decode payload is not float32-aligned.");
  }
  std::vector<float> values(static_cast<std::size_t>(
      bytes / static_cast<std::streamoff>(sizeof(float))));
  input.seekg(0);
  if (!values.empty() &&
      !input.read(reinterpret_cast<char *>(values.data()),
                  static_cast<std::streamsize>(bytes))) {
    throw std::runtime_error("Unable to read the complete Transformer decode payload.");
  }
  return values;
}

std::vector<float> required(
    const std::unordered_map<std::string, TensorRange> &ranges,
    const std::vector<float> &payload, const std::string &name,
    std::size_t expectedCount) {
  const auto found = ranges.find(name);
  if (found == ranges.end()) {
    throw std::runtime_error("Missing Transformer decode tensor '" + name + "'.");
  }
  const auto range = found->second;
  if (range.count != expectedCount || range.offset > payload.size() ||
      range.count > payload.size() - range.offset) {
    throw std::runtime_error("Invalid Transformer decode tensor range for '" + name + "'.");
  }
  return {payload.begin() + static_cast<std::ptrdiff_t>(range.offset),
          payload.begin() + static_cast<std::ptrdiff_t>(range.offset + range.count)};
}

std::vector<double> asDouble(std::vector<float> values) {
  return {values.begin(), values.end()};
}

DecoderLayerParameters readLayer(
    const std::unordered_map<std::string, TensorRange> &ranges,
    const std::vector<float> &payload, std::size_t index,
    std::size_t dimension, std::size_t feedForwardDimension) {
  const auto prefix = "layer" + std::to_string(index);
  const auto matrix = dimension * dimension;
  DecoderLayerParameters layer;
  layer.selfQueryWeight = required(ranges, payload, prefix + ".self.q.weight", matrix);
  layer.selfKeyWeight = required(ranges, payload, prefix + ".self.k.weight", matrix);
  layer.selfValueWeight = required(ranges, payload, prefix + ".self.v.weight", matrix);
  layer.selfOutputWeight = required(ranges, payload, prefix + ".self.o.weight", matrix);
  layer.selfNormWeight = required(ranges, payload, prefix + ".self.norm.weight", dimension);
  layer.selfNormBias = required(ranges, payload, prefix + ".self.norm.bias", dimension);

  layer.crossQueryWeight = required(ranges, payload, prefix + ".cross.q.weight", matrix);
  layer.crossQueryBias = required(ranges, payload, prefix + ".cross.q.bias", dimension);
  layer.crossKeyWeight = required(ranges, payload, prefix + ".cross.k.weight", matrix);
  layer.crossKeyBias = required(ranges, payload, prefix + ".cross.k.bias", dimension);
  layer.crossValueWeight = required(ranges, payload, prefix + ".cross.v.weight", matrix);
  layer.crossValueBias = required(ranges, payload, prefix + ".cross.v.bias", dimension);
  layer.crossOutputWeight = required(ranges, payload, prefix + ".cross.o.weight", matrix);
  layer.crossOutputBias = required(ranges, payload, prefix + ".cross.o.bias", dimension);
  layer.crossNormWeight = required(ranges, payload, prefix + ".cross.norm.weight", dimension);
  layer.crossNormBias = required(ranges, payload, prefix + ".cross.norm.bias", dimension);

  const auto inputWeightCount = feedForwardDimension * dimension;
  const auto outputWeightCount = dimension * feedForwardDimension;
  layer.feedForwardInputWeight =
      required(ranges, payload, prefix + ".ffn.in.weight", inputWeightCount);
  layer.feedForwardInputBias =
      required(ranges, payload, prefix + ".ffn.in.bias", feedForwardDimension);
  layer.feedForwardOutputWeight =
      required(ranges, payload, prefix + ".ffn.out.weight", outputWeightCount);
  layer.feedForwardOutputBias =
      required(ranges, payload, prefix + ".ffn.out.bias", dimension);
  layer.feedForwardNormWeight =
      required(ranges, payload, prefix + ".ffn.norm.weight", dimension);
  layer.feedForwardNormBias =
      required(ranges, payload, prefix + ".ffn.norm.bias", dimension);
  return layer;
}

TransformerDecodeReference readReference(
    const std::unordered_map<std::string, TensorRange> &ranges,
    const std::vector<float> &payload, const std::string &prefix,
    const planner::TransformerDecodePlan &plan, std::size_t sequenceLength,
    std::size_t cacheLength) {
  TransformerDecodeReference reference;
  const auto tokenCount = plan.selfAttention.batch * sequenceLength;
  reference.tokenIds = required(ranges, payload, prefix + ".tokens", tokenCount);
  reference.logits = asDouble(required(
      ranges, payload, prefix + ".logits", plan.logitsElementCount(sequenceLength)));
  reference.nextTokenIds = asDouble(required(
      ranges, payload, prefix + ".next_tokens", plan.selfAttention.batch));
  const auto cacheCount = plan.selfAttention.batch * plan.selfAttention.heads *
                          cacheLength * plan.selfAttention.headDimension;
  for (std::size_t layer = 0; layer < plan.layerCount; ++layer) {
    const auto layerPrefix = prefix + ".layer" + std::to_string(layer);
    reference.keyCaches.push_back(asDouble(required(
        ranges, payload, layerPrefix + ".key_cache", cacheCount)));
    reference.valueCaches.push_back(asDouble(required(
        ranges, payload, layerPrefix + ".value_cache", cacheCount)));
  }
  return reference;
}

} // namespace

TransformerDecodeImportResult
importTransformerDecodeWorkload(const std::string &manifestPath) {
  TransformerDecodeImportResult result;
  try {
    std::ifstream input(manifestPath);
    if (!input) {
      throw std::runtime_error("Unable to open Transformer decode manifest: " +
                               manifestPath);
    }
    std::string magic;
    std::size_t version = 0;
    if (!(input >> magic >> version) || magic != "TMC_TRANSFORMER_DECODE" ||
        version != 1) {
      throw std::runtime_error("Invalid or unsupported Transformer decode manifest header.");
    }

    auto workload = std::make_unique<TransformerDecodeWorkload>();
    expect(input, "MODEL");
    workload->modelName = readQuoted(input, "model name");
    expect(input, "PAYLOAD");
    const auto payloadName = readQuoted(input, "payload path");

    expect(input, "CONFIG");
    auto &plan = workload->plan;
    plan.selfAttention.batch = readSize(input, "batch size");
    plan.selfAttention.heads = readSize(input, "head count");
    plan.selfAttention.headDimension = readSize(input, "head dimension");
    plan.feedForwardDimension = readSize(input, "feed-forward dimension");
    plan.vocabularySize = readSize(input, "vocabulary size");
    plan.sourceLength = readSize(input, "source length");
    plan.selfAttention.capacity = readSize(input, "cache capacity");
    plan.selfAttention.prefillLength = readSize(input, "prefill length");
    const auto decodeCount = readSize(input, "decode step count");
    plan.layerCount = readSize(input, "decoder layer count");
    plan.selfAttention.dtype = DType::Float32;
    plan.selfAttention.threadsPerThreadgroup = 128;
    plan.validate();
    if (decodeCount > plan.selfAttention.capacity - plan.selfAttention.prefillLength) {
      throw std::runtime_error("Transformer decode steps exceed cache capacity.");
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
        throw std::runtime_error("Duplicate Transformer decode tensor '" + name + "'.");
      }
    }
    expect(input, "END");
    std::string trailing;
    if (input >> trailing) {
      throw std::runtime_error("Unexpected data after Transformer decode END marker.");
    }

    const auto manifest = std::filesystem::absolute(std::filesystem::path(manifestPath));
    const auto payload = readPayload(manifest.parent_path() / payloadName);
    const auto dimension = plan.modelDimension();
    workload->encoderMemory = required(
        ranges, payload, "encoder_memory",
        plan.selfAttention.batch * plan.sourceLength * dimension);
    workload->embeddingWeight = required(
        ranges, payload, "embedding.weight", plan.vocabularySize * dimension);
    workload->positionalEncoding = required(
        ranges, payload, "position.table", plan.selfAttention.capacity * dimension);
    workload->languageModelHeadWeight = required(
        ranges, payload, "lm_head.weight", plan.vocabularySize * dimension);
    workload->languageModelHeadBias = required(
        ranges, payload, "lm_head.bias", plan.vocabularySize);
    for (std::size_t layer = 0; layer < plan.layerCount; ++layer) {
      workload->layers.push_back(readLayer(
          ranges, payload, layer, dimension, plan.feedForwardDimension));
    }
    workload->prefill = readReference(
        ranges, payload, "prefill", plan, plan.selfAttention.prefillLength,
        plan.selfAttention.prefillLength);
    for (std::size_t step = 0; step < decodeCount; ++step) {
      workload->decodeSteps.push_back(readReference(
          ranges, payload, "decode" + std::to_string(step), plan, 1,
          plan.selfAttention.prefillLength + step + 1));
    }
    result.workload = std::move(workload);
  } catch (const std::exception &error) {
    result.errorMessage = error.what();
  }
  return result;
}

} // namespace tensor::importer
