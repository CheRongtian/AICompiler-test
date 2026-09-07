#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace tensor {

enum class DType { Float16, Float32, Int32 };
enum class Layout { Contiguous, Strided };

// Static tensors. Strides are expressed in elements; an empty shape is a scalar.
struct TensorType {
  std::vector<std::size_t> shape;
  DType dtype = DType::Float32;
  Layout layout = Layout::Contiguous;
  std::vector<std::size_t> explicitStrides;

  [[nodiscard]] std::size_t elementCount() const;
  [[nodiscard]] std::size_t storageElementCount() const;
  [[nodiscard]] std::vector<std::size_t> strides() const;
  [[nodiscard]] bool isContiguous() const;
};

using ValueId = std::size_t;
// Host values are supplied as logical floats. The runtime converts fp16/int32 inputs.
using GraphInputs = std::unordered_map<ValueId, std::vector<float>>;

enum class OpType {
  Add, Mul, MatMul, RMSNorm, LayerNorm, ReLU, SiLU, Softmax, RoPE,
  ReduceSum, ReduceMean, Reshape, View, Transpose, Contiguous,
  MaskedFill, Slice, Embedding, Gather
};

[[nodiscard]] const char *opName(OpType op);
[[nodiscard]] bool isViewOp(OpType op);

struct RMSNormAttributes { float epsilon = 1e-5f; std::int64_t axis = -1; };
struct LayerNormAttributes {
  float epsilon = 1e-5f;
  std::vector<std::int64_t> axes{-1};
};
struct SoftmaxAttributes { std::int64_t axis = -1; };
struct ReductionAttributes {
  std::vector<std::int64_t> axes;
  bool keepDimensions = false;
};
struct ReshapeAttributes { std::vector<std::size_t> shape; };
struct TransposeAttributes { std::int64_t first = 0; std::int64_t second = 1; };
struct MaskedFillAttributes { float value = 0.0f; };
struct SliceAttributes {
  std::vector<std::size_t> starts;
  std::vector<std::size_t> sizes;
  std::vector<std::size_t> steps;
};
struct GatherAttributes { std::int64_t axis = 0; };

using OpAttributes =
    std::variant<std::monostate, RMSNormAttributes, LayerNormAttributes,
                 SoftmaxAttributes, ReductionAttributes, ReshapeAttributes,
                 TransposeAttributes, MaskedFillAttributes, SliceAttributes,
                 GatherAttributes>;

struct Value { std::string name; std::optional<TensorType> type; };

struct Node {
  OpType op;
  std::vector<ValueId> inputs;
  std::vector<ValueId> outputs;
  OpAttributes attributes;
};

// Functional SSA graph. Views may alias their first input; materializing ops define
// fresh storage. RoPE uses adjacent interleaved pairs.
struct TensorGraph {
  std::vector<Value> values;
  std::vector<ValueId> inputs;
  std::vector<Node> nodes;
  std::vector<ValueId> outputs;

  ValueId addInput(std::string name, TensorType type);
  ValueId addNode(OpType op, std::vector<ValueId> operands,
                  OpAttributes attributes = {}, std::string name = {});
  std::vector<ValueId> addNodeResults(OpType op, std::vector<ValueId> operands,
                                      std::vector<std::string> names,
                                      OpAttributes attributes = {});
};

// Normalize along the last dimension with fp32 accumulation.
struct RMSNormOp {
  TensorType input;
  TensorType weight;
  TensorType output;
  float epsilon = 1e-5f;

  void validate() const;
};

} // namespace tensor
