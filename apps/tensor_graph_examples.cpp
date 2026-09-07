#include "tensor_graph_examples.hpp"

#include "runtime/GraphExecutor.hpp"
#include "validation/Validator.hpp"

#include <cmath>
#include <ostream>
#include <string>
#include <vector>

namespace {

using namespace tensor;

std::vector<float> data(const TensorType &type, std::size_t seed = 0, float scale = 1.0f) {
  std::vector<float> result(type.elementCount());
  for (std::size_t i = 0; i < result.size(); ++i) {
    const int centered = static_cast<int>((i * 17 + seed * 13) % 257) - 128;
    result[i] = static_cast<float>(centered) / 64.0f * scale;
  }
  return result;
}

bool runCase(metal::MetalRuntime &runtime, const std::string &name,
              const TensorGraph &graph, const GraphInputs &inputs,
              const std::vector<std::vector<std::size_t>> &expectedShapes,
              std::ostream &log) {
  log << "\nGraph: " << name << '\n';
  auto compiled = tensor::runtime::compileGraph(runtime, graph, inputs, log);
  if (!compiled.executable) {
    log << "Graph compilation: FAIL: " << compiled.errorMessage << '\n';
    return false;
  }
  if (compiled.outputTypes.size() != expectedShapes.size()) {
    log << "Output shape validation: FAIL (output count)\n";
    return false;
  }
  for (std::size_t i = 0; i < expectedShapes.size(); ++i) {
    if (compiled.outputTypes[i].shape != expectedShapes[i]) {
      log << "Output shape validation: FAIL\n";
      return false;
    }
  }
  const auto executed = compiled.executable->run();
  if (!executed.passed) {
    log << "Graph GPU execution: FAIL: " << executed.errorMessage << '\n';
    return false;
  }
  log << "Graph GPU execution: PASS\n";
  if (executed.gpuExecutionTimeUs) {
    log << "Sum of node GPU command-buffer times (us): " << *executed.gpuExecutionTimeUs << '\n';
  } else {
    log << "Sum of node GPU command-buffer times (us): Unavailable\n";
  }
  bool passed = true;
  for (std::size_t i = 0; i < executed.outputs.size(); ++i) {
    const auto comparison = tensor::validation::compare(executed.outputs[i], compiled.referenceOutputs[i],
                                                         1e-5, 1e-4);
    log << "Output " << graph.values[graph.outputs[i]].name << " shape=[";
    for (std::size_t axis = 0; axis < expectedShapes[i].size(); ++axis) {
      if (axis != 0) log << ", ";
      log << expectedShapes[i][axis];
    }
    log << "] | numerical validation: " << (comparison.passed ? "PASS" : "FAIL")
        << ", max absolute error=" << comparison.maxAbsoluteError << '\n';
    if (!comparison.passed) {
      log << comparison.errorMessage << '\n';
      passed = false;
    }
  }
  return passed;
}

bool runRMSGraph(metal::MetalRuntime &runtime, std::size_t rows, std::size_t width,
                  bool residual, std::ostream &log) {
  TensorGraph graph;
  GraphInputs inputs;
  const TensorType inputType{{rows, width}};
  const TensorType weightType{{width}};
  const auto x = graph.addInput("x", inputType);
  const auto weight = graph.addInput("weight", weightType);
  inputs[x] = data(inputType);
  // Preserve V0b's zero, epsilon-dominated, and ordinary multi-row inputs.
  if (rows > 1) {
    for (std::size_t column = 0; column < width; ++column) {
      inputs[x][column] = 0.0f;
      inputs[x][width + column] *= 1e-4f;
    }
  }
  inputs[weight].resize(width);
  for (std::size_t i = 0; i < width; ++i) {
    inputs[weight][i] = 0.5f + static_cast<float>(i % 31) / 32.0f;
  }
  auto normInput = x;
  if (residual) {
    const auto bias = graph.addInput("residual", weightType);
    inputs[bias] = data(weightType, 3, 0.1f);
    normInput = graph.addNode(OpType::Add, {x, bias});
  }
  const auto output = graph.addNode(OpType::RMSNorm, {normInput, weight}, RMSNormAttributes{});
  graph.outputs = {output};
  return runCase(runtime, residual ? "Add -> RMSNorm" : "RMSNorm (V1 autotuning)",
                  graph, inputs, {{rows, width}}, log);
}

} // namespace

bool runTensorGraphExamples(tensor::metal::MetalRuntime &runtime, std::ostream &log) {
  using namespace tensor;
  bool passed = true;
  {
    TensorGraph graph;
    const TensorType aType{{2, 1, 5}}, bType{{1, 3, 1}}, scalarType{};
    const auto a = graph.addInput("a", aType);
    const auto b = graph.addInput("b", bType);
    const auto scale = graph.addInput("scale", scalarType);
    const auto sum = graph.addNode(OpType::Add, {a, b});
    const auto product = graph.addNode(OpType::Mul, {sum, scale});
    graph.outputs = {sum, product};
    const bool ok = runCase(runtime, "Add -> Mul (broadcast, scalar, two outputs)", graph,
        {{a, data(aType)}, {b, data(bType, 2)}, {scale, {0.5f}}},
        {{2, 3, 5}, {2, 3, 5}}, log);
    passed = ok && passed;
  }
  for (bool batched : {false, true}) {
    TensorGraph graph;
    const TensorType aType{batched ? std::vector<std::size_t>{2, 1, 3, 7}
                                    : std::vector<std::size_t>{3, 7}};
    const TensorType bType{batched ? std::vector<std::size_t>{1, 4, 7, 5}
                                    : std::vector<std::size_t>{7, 5}};
    const auto a = graph.addInput("a", aType);
    const auto b = graph.addInput("b", bType);
    graph.outputs = {graph.addNode(OpType::MatMul, {a, b})};
    const auto shape = batched ? std::vector<std::size_t>{2, 4, 3, 5}
                               : std::vector<std::size_t>{3, 5};
    const bool ok = runCase(runtime, batched ? "Batched MatMul" : "MatMul", graph,
        {{a, data(aType)}, {b, data(bType, 1)}}, {shape}, log);
    passed = ok && passed;
  }
  {
    TensorGraph graph;
    const TensorType xType{{3, 17}}, gateType{{17}};
    const auto x = graph.addInput("x", xType);
    const auto gate = graph.addInput("gate", gateType);
    const auto activated = graph.addNode(OpType::SiLU, {x});
    graph.outputs = {graph.addNode(OpType::Mul, {activated, gate})};
    const bool ok = runCase(runtime, "SiLU -> Mul", graph,
        {{x, data(xType)}, {gate, data(gateType, 2)}}, {{3, 17}}, log);
    passed = ok && passed;
  }
  {
    TensorGraph graph;
    const TensorType xType{{2, 3, 8}}, tableType{{3, 4}};
    const auto x = graph.addInput("x", xType);
    const auto cos = graph.addInput("cos", tableType);
    const auto sin = graph.addInput("sin", tableType);
    graph.outputs = {graph.addNode(OpType::RoPE, {x, cos, sin})};
    GraphInputs inputs{{x, data(xType)}};
    inputs[cos].resize(tableType.elementCount());
    inputs[sin].resize(tableType.elementCount());
    for (std::size_t position = 0; position < 3; ++position) {
      for (std::size_t pair = 0; pair < 4; ++pair) {
        const double angle = static_cast<double>(position) / std::pow(10000.0, static_cast<double>(pair) / 4.0);
        inputs[cos][position * 4 + pair] = static_cast<float>(std::cos(angle));
        inputs[sin][position * 4 + pair] = static_cast<float>(std::sin(angle));
      }
    }
    const bool ok = runCase(runtime, "RoPE (interleaved pairs)", graph, inputs, {{2, 3, 8}}, log);
    passed = ok && passed;
  }
  const bool normAligned = runRMSGraph(runtime, 1, 4096, false, log);
  const bool normTail = runRMSGraph(runtime, 3, 4097, false, log);
  const bool addNorm = runRMSGraph(runtime, 3, 4097, true, log);
  passed = normAligned && normTail && addNorm && passed;
  log << "\nV2 graph examples: " << (passed ? "PASS" : "FAIL") << '\n';
  return passed;
}
