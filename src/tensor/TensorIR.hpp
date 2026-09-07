#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace tensor {

enum class DType { Float32 };
enum class Layout { Contiguous };

// Static, dense, row-major tensors. An empty shape denotes a scalar.
struct TensorType {
  std::vector<std::size_t> shape;
  DType dtype = DType::Float32;
  Layout layout = Layout::Contiguous;

  [[nodiscard]] std::size_t elementCount() const;
  [[nodiscard]] std::vector<std::size_t> strides() const;
};

using ValueId = std::size_t;
// FP32 host bindings for graph inputs, including weights.
using GraphInputs = std::unordered_map<ValueId, std::vector<float>>;
enum class OpType { Add, Mul, MatMul, RMSNorm, SiLU, RoPE };
[[nodiscard]] const char *opName(OpType op);

struct RMSNormAttributes {
  float epsilon = 1e-5f;
  std::int64_t axis = -1; // V2 supports only the last axis.
};

using OpAttributes = std::variant<std::monostate, RMSNormAttributes>;

struct Value {
  std::string name;
  std::optional<TensorType> type; // Node results are inferred by the analyzer.
};

struct Node {
  OpType op;
  std::vector<ValueId> inputs;
  ValueId output;
  OpAttributes attributes;
};

// Inputs include model parameters. Each node defines one fresh result value.
// RoPE(x, cos, sin): adjacent pairs, full even last dimension D;
// cos/sin broadcast to x.shape with its last dimension replaced by D/2.
struct TensorGraph {
  std::vector<Value> values;
  std::vector<ValueId> inputs;
  std::vector<Node> nodes;
  std::vector<ValueId> outputs;

  ValueId addInput(std::string name, TensorType type);
  ValueId addNode(OpType op, std::vector<ValueId> operands,
                  OpAttributes attributes = {}, std::string name = {});
};

// Normalize along the last dimension; weight has shape [input.shape.back()].
// y = x * rsqrt(mean(x * x, last axis) + epsilon) * weight
struct RMSNormOp {
  TensorType input;
  TensorType weight;
  TensorType output;
  float epsilon = 1e-5f;

  // Throws std::invalid_argument or std::overflow_error for unsupported IR.
  void validate() const;
};

} // namespace tensor
