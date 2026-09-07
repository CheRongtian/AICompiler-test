#include "importer/PyTorchImporter.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tensor::importer {
namespace {

constexpr const char *kMagic = "TMC_PYTORCH_GRAPH";
constexpr std::size_t kFormatVersion = 1;

void expect(std::istream &input, const char *expected) {
  std::string token;
  if (!(input >> token) || token != expected) {
    throw std::runtime_error(std::string("Expected '") + expected + "' in graph manifest.");
  }
}

std::size_t readSize(std::istream &input, const char *description) {
  std::size_t value = 0;
  if (!(input >> value)) {
    throw std::runtime_error(std::string("Invalid ") + description + " in graph manifest.");
  }
  return value;
}

std::int64_t readSigned(std::istream &input, const char *description) {
  std::int64_t value = 0;
  if (!(input >> value)) {
    throw std::runtime_error(std::string("Invalid ") + description + " in graph manifest.");
  }
  return value;
}

float readFloat(std::istream &input, const char *description) {
  float value = 0.0f;
  if (!(input >> value)) {
    throw std::runtime_error(std::string("Invalid ") + description + " in graph manifest.");
  }
  return value;
}

std::string readQuoted(std::istream &input, const char *description) {
  std::string value;
  if (!(input >> std::quoted(value))) {
    throw std::runtime_error(std::string("Invalid ") + description + " in graph manifest.");
  }
  return value;
}

std::vector<std::size_t> readShape(std::istream &input) {
  const auto rank = readSize(input, "tensor rank");
  std::vector<std::size_t> shape(rank);
  for (auto &dimension : shape) dimension = readSize(input, "tensor dimension");
  return shape;
}

DType parseDType(const std::string &token) {
  if (token == "F16") return DType::Float16;
  if (token == "F32") return DType::Float32;
  if (token == "I32") return DType::Int32;
  throw std::runtime_error("Unsupported dtype '" + token + "' in graph manifest.");
}

PyTorchInputKind parseInputKind(const std::string &token) {
  if (token == "USER") return PyTorchInputKind::UserInput;
  if (token == "PARAMETER") return PyTorchInputKind::Parameter;
  if (token == "BUFFER") return PyTorchInputKind::Buffer;
  if (token == "CONSTANT") return PyTorchInputKind::Constant;
  throw std::runtime_error("Unsupported input kind '" + token + "' in graph manifest.");
}

OpType parseOp(const std::string &token) {
  if (token == "Add") return OpType::Add;
  if (token == "Mul") return OpType::Mul;
  if (token == "MatMul") return OpType::MatMul;
  if (token == "RMSNorm") return OpType::RMSNorm;
  if (token == "LayerNorm") return OpType::LayerNorm;
  if (token == "ReLU") return OpType::ReLU;
  if (token == "SiLU") return OpType::SiLU;
  if (token == "Softmax") return OpType::Softmax;
  if (token == "RoPE") return OpType::RoPE;
  if (token == "ReduceSum") return OpType::ReduceSum;
  if (token == "ReduceMean") return OpType::ReduceMean;
  if (token == "Reshape") return OpType::Reshape;
  if (token == "View") return OpType::View;
  if (token == "Transpose") return OpType::Transpose;
  if (token == "Contiguous") return OpType::Contiguous;
  if (token == "MaskedFill") return OpType::MaskedFill;
  if (token == "Slice") return OpType::Slice;
  if (token == "Embedding") return OpType::Embedding;
  if (token == "Gather") return OpType::Gather;
  throw std::runtime_error("Unsupported TensorIR op '" + token + "' in graph manifest.");
}

std::vector<std::int64_t> readAxes(std::istream &input) {
  const auto count = readSize(input, "axis count");
  std::vector<std::int64_t> axes(count);
  for (auto &axis : axes) axis = readSigned(input, "axis");
  return axes;
}

OpAttributes parseAttributes(std::istream &input, OpType op) {
  std::string tag;
  if (!(input >> tag)) throw std::runtime_error("Missing node attributes in graph manifest.");
  if (tag == "NONE") return {};
  if (tag == "RMSNORM") {
    RMSNormAttributes attributes;
    attributes.epsilon = readFloat(input, "RMSNorm epsilon");
    attributes.axis = readSigned(input, "RMSNorm axis");
    return attributes;
  }
  if (tag == "LAYERNORM") {
    LayerNormAttributes attributes;
    attributes.epsilon = readFloat(input, "LayerNorm epsilon");
    attributes.axes = readAxes(input);
    return attributes;
  }
  if (tag == "SOFTMAX") {
    return SoftmaxAttributes{readSigned(input, "Softmax axis")};
  }
  if (tag == "REDUCTION") {
    const auto keepDimensions = readSize(input, "keep-dimensions value");
    if (keepDimensions > 1) throw std::runtime_error("Reduction keep-dimensions must be 0 or 1.");
    return ReductionAttributes{readAxes(input), keepDimensions == 1};
  }
  if (tag == "RESHAPE") return ReshapeAttributes{readShape(input)};
  if (tag == "TRANSPOSE") {
    return TransposeAttributes{readSigned(input, "first transpose axis"),
                               readSigned(input, "second transpose axis")};
  }
  if (tag == "MASKEDFILL") {
    return MaskedFillAttributes{readFloat(input, "MaskedFill value")};
  }
  if (tag == "SLICE") {
    SliceAttributes attributes;
    const auto rank = readSize(input, "Slice rank");
    attributes.starts.resize(rank);
    attributes.sizes.resize(rank);
    attributes.steps.resize(rank);
    for (auto &value : attributes.starts) value = readSize(input, "Slice start");
    for (auto &value : attributes.sizes) value = readSize(input, "Slice size");
    for (auto &value : attributes.steps) value = readSize(input, "Slice step");
    return attributes;
  }
  if (tag == "GATHER") {
    return GatherAttributes{readSigned(input, "Gather axis")};
  }
  throw std::runtime_error("Unsupported attribute tag '" + tag + "' for " +
                           opName(op) + ".");
}

std::vector<float> readPayload(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("Unable to open tensor payload: " + path.string());
  const auto end = input.tellg();
  if (end < 0) throw std::runtime_error("Unable to determine tensor payload size.");
  const auto byteCount = static_cast<std::streamoff>(end);
  if (byteCount % static_cast<std::streamoff>(sizeof(float)) != 0) {
    throw std::runtime_error("Tensor payload size is not a multiple of float32 storage.");
  }
  const auto elementCount = static_cast<std::size_t>(
      byteCount / static_cast<std::streamoff>(sizeof(float)));
  std::vector<float> values(elementCount);
  input.seekg(0);
  if (elementCount != 0 &&
      !input.read(reinterpret_cast<char *>(values.data()),
                  static_cast<std::streamsize>(byteCount))) {
    throw std::runtime_error("Unable to read the complete tensor payload.");
  }
  return values;
}

std::vector<float> payloadSlice(const std::vector<float> &payload,
                                std::size_t offset, std::size_t count) {
  if (offset > payload.size() || count > payload.size() - offset) {
    throw std::runtime_error("Tensor payload range is outside the data file.");
  }
  return {payload.begin() + static_cast<std::ptrdiff_t>(offset),
          payload.begin() + static_cast<std::ptrdiff_t>(offset + count)};
}

void defineValue(std::vector<bool> &defined, ValueId id, std::size_t valueCount) {
  if (id >= valueCount) throw std::runtime_error("Value id is outside the declared range.");
  if (defined[id]) throw std::runtime_error("Value id is defined more than once.");
  defined[id] = true;
}

} // namespace

const char *inputKindName(PyTorchInputKind kind) {
  switch (kind) {
  case PyTorchInputKind::UserInput: return "user input";
  case PyTorchInputKind::Parameter: return "parameter";
  case PyTorchInputKind::Buffer: return "buffer";
  case PyTorchInputKind::Constant: return "constant";
  }
  return "unknown";
}

PyTorchImportResult importPyTorchGraph(const std::string &manifestPath) {
  PyTorchImportResult result;
  try {
    std::ifstream input(manifestPath);
    if (!input) throw std::runtime_error("Unable to open PyTorch graph manifest: " + manifestPath);

    std::string magic;
    std::size_t version = 0;
    if (!(input >> magic >> version) || magic != kMagic) {
      throw std::runtime_error("Invalid PyTorch graph manifest header.");
    }
    if (version != kFormatVersion) {
      throw std::runtime_error("Unsupported PyTorch graph manifest version " +
                               std::to_string(version) + ".");
    }

    expect(input, "MODEL");
    auto model = std::make_unique<ImportedPyTorchModel>();
    model->modelName = readQuoted(input, "model name");

    expect(input, "PAYLOAD");
    const auto payloadName = readQuoted(input, "payload path");
    const auto manifest = std::filesystem::absolute(std::filesystem::path(manifestPath));
    const auto payloadPath = manifest.parent_path() / payloadName;
    const auto payload = readPayload(payloadPath);

    expect(input, "VALUES");
    const auto valueCount = readSize(input, "value count");
    if (valueCount == 0) throw std::runtime_error("Imported graph must contain values.");
    model->graph.values.resize(valueCount);
    std::vector<bool> defined(valueCount, false);

    expect(input, "INPUTS");
    const auto inputCount = readSize(input, "input count");
    for (std::size_t index = 0; index < inputCount; ++index) {
      expect(input, "INPUT");
      const auto id = readSize(input, "input value id");
      defineValue(defined, id, valueCount);
      const auto name = readQuoted(input, "input name");
      std::string kindToken;
      std::string dtypeToken;
      if (!(input >> kindToken >> dtypeToken)) {
        throw std::runtime_error("Invalid imported input declaration.");
      }
      const auto kind = parseInputKind(kindToken);
      TensorType type{readShape(input), parseDType(dtypeToken)};
      const auto offset = readSize(input, "input payload offset");
      const auto count = readSize(input, "input payload length");
      if (count != type.elementCount()) {
        throw std::runtime_error("Input '" + name + "' payload length does not match its shape.");
      }
      model->graph.values[id] = {name, type};
      model->graph.inputs.push_back(id);
      model->inputs.emplace(id, payloadSlice(payload, offset, count));
      model->importedInputs.push_back({id, kind});
    }

    expect(input, "NODES");
    const auto nodeCount = readSize(input, "node count");
    model->graph.nodes.reserve(nodeCount);
    for (std::size_t index = 0; index < nodeCount; ++index) {
      expect(input, "NODE");
      const auto output = readSize(input, "node output id");
      defineValue(defined, output, valueCount);
      const auto name = readQuoted(input, "node name");
      std::string opToken;
      if (!(input >> opToken)) throw std::runtime_error("Missing TensorIR op in node declaration.");
      const auto op = parseOp(opToken);
      const auto operandCount = readSize(input, "node operand count");
      std::vector<ValueId> operands(operandCount);
      for (auto &operand : operands) {
        operand = readSize(input, "node operand id");
        if (operand >= valueCount) throw std::runtime_error("Node operand id is outside the declared range.");
      }
      auto attributes = parseAttributes(input, op);
      model->graph.values[output] = {name, std::nullopt};
      model->graph.nodes.push_back({op, std::move(operands), {output}, std::move(attributes)});
    }

    for (ValueId id = 0; id < valueCount; ++id) {
      if (!defined[id]) throw std::runtime_error("Graph manifest contains an undefined value id.");
    }

    expect(input, "OUTPUTS");
    const auto outputCount = readSize(input, "output count");
    if (outputCount == 0) throw std::runtime_error("Imported graph must declare an output.");
    for (std::size_t index = 0; index < outputCount; ++index) {
      expect(input, "OUTPUT");
      const auto id = readSize(input, "output value id");
      if (id >= valueCount) throw std::runtime_error("Output value id is outside the declared range.");
      std::string dtypeToken;
      if (!(input >> dtypeToken)) throw std::runtime_error("Missing output dtype.");
      TensorType type{readShape(input), parseDType(dtypeToken)};
      const auto offset = readSize(input, "reference payload offset");
      const auto count = readSize(input, "reference payload length");
      if (count != type.elementCount()) {
        throw std::runtime_error("Reference payload length does not match the output shape.");
      }
      model->graph.outputs.push_back(id);
      model->expectedOutputTypes.push_back(type);
      const auto values = payloadSlice(payload, offset, count);
      model->expectedOutputs.emplace_back(values.begin(), values.end());
    }

    expect(input, "END");
    std::string trailing;
    if (input >> trailing) throw std::runtime_error("Unexpected data after graph manifest END marker.");

    result.model = std::move(model);
  } catch (const std::exception &error) {
    result.errorMessage = error.what();
  }
  return result;
}

} // namespace tensor::importer
