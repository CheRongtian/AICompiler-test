#include "tensor_graph_examples.hpp"

#include "analyzer/PatternAnalyzer.hpp"
#include "runtime/GraphExecutor.hpp"
#include "validation/Validator.hpp"

#include <algorithm>
#include <cmath>
#include <ostream>
#include <string>
#include <utility>
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
    const bool half = compiled.outputTypes[i].dtype == DType::Float16;
    const auto comparison = tensor::validation::compare(
        executed.outputs[i], compiled.referenceOutputs[i],
        half ? 2e-3 : 1e-5, half ? 2e-3 : 1e-4);
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
                 bool residual, std::ostream &log,
                 DType dtype = DType::Float32) {
  TensorGraph graph;
  GraphInputs inputs;
  const TensorType inputType{{rows, width}, dtype};
  const TensorType weightType{{width}, dtype};
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
  const std::string name = residual ? "Add -> RMSNorm"
      : dtype == DType::Float16 ? "fp16 RMSNorm with fp32 accumulation"
                                : "RMSNorm autotuning";
  return runCase(runtime, name,
                  graph, inputs, {{rows, width}}, log);
}

bool runLayerNormFusion(metal::MetalRuntime &runtime, std::ostream &log) {
  TensorGraph graph;
  const TensorType xType{{3, 17}}, parameterType{{17}};
  const auto x = graph.addInput("x", xType);
  const auto residual = graph.addInput("residual", parameterType);
  const auto weight = graph.addInput("weight", parameterType);
  const auto bias = graph.addInput("bias", parameterType);
  const auto added = graph.addNode(OpType::Add, {x, residual});
  graph.outputs = {graph.addNode(OpType::LayerNorm, {added, weight, bias},
                                 LayerNormAttributes{})};
  auto weights = data(parameterType, 3, 0.2f);
  for (auto &value : weights) value += 1.0f;
  return runCase(runtime, "Residual Add -> LayerNorm", graph,
      {{x, data(xType)}, {residual, data(parameterType, 1, 0.1f)},
       {weight, std::move(weights)}, {bias, data(parameterType, 2, 0.05f)}},
      {{3, 17}}, log);
}

bool runLinearReLUFusion(metal::MetalRuntime &runtime, std::ostream &log) {
  TensorGraph graph;
  const TensorType inputType{{3, 7}}, weightType{{7, 5}}, biasType{{5}};
  const auto input = graph.addInput("input", inputType);
  const auto weight = graph.addInput("weight", weightType);
  const auto bias = graph.addInput("bias", biasType);
  const auto linear = graph.addNode(OpType::MatMul, {input, weight});
  const auto biased = graph.addNode(OpType::Add, {linear, bias});
  graph.outputs = {graph.addNode(OpType::ReLU, {biased})};
  return runCase(runtime, "Linear -> ReLU", graph,
      {{input, data(inputType)}, {weight, data(weightType, 1, 0.25f)},
       {bias, data(biasType, 2, 0.1f)}}, {{3, 5}}, log);
}

bool runViewAndReduction(metal::MetalRuntime &runtime, std::ostream &log) {
  TensorGraph graph;
  const TensorType inputType{{2, 3, 4}};
  const auto input = graph.addInput("input", inputType);
  const auto transposed = graph.addNode(OpType::Transpose, {input},
                                        TransposeAttributes{1, 2});
  const auto contiguous = graph.addNode(OpType::Contiguous, {transposed});
  const auto probabilities = graph.addNode(OpType::Softmax, {contiguous},
                                            SoftmaxAttributes{-1});
  graph.outputs = {graph.addNode(OpType::ReduceMean, {probabilities},
                                 ReductionAttributes{{0, 2}, false})};
  return runCase(runtime, "Transpose -> Contiguous -> Softmax -> ReduceMean",
                 graph, {{input, data(inputType)}}, {{4}}, log);
}

bool checkFusionBoundary(std::ostream &log) {
  TensorGraph graph;
  const TensorType xType{{2, 17}}, parameterType{{17}};
  const auto x = graph.addInput("x", xType);
  const auto residual = graph.addInput("residual", parameterType);
  const auto weight = graph.addInput("weight", parameterType);
  const auto added = graph.addNode(OpType::Add, {x, residual});
  const auto normalized = graph.addNode(OpType::RMSNorm, {added, weight},
                                         RMSNormAttributes{});
  graph.outputs = {added, normalized};
  const auto regions = tensor::analyzer::formRegions(tensor::analyzer::analyze(graph));
  const bool passed = std::none_of(regions.regions.begin(), regions.regions.end(),
      [](const tensor::analyzer::Region &region) {
        return region.fusion == tensor::analyzer::FusionPattern::AddRMSNorm;
      });
  log << "Fusion boundary for externally visible intermediate: "
      << (passed ? "PASS" : "FAIL") << '\n';
  return passed;
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
  const bool halfNorm = runRMSGraph(runtime, 1, 257, false, log, DType::Float16);
  const bool addLayerNorm = runLayerNormFusion(runtime, log);
  const bool linearRelu = runLinearReLUFusion(runtime, log);
  const bool views = runViewAndReduction(runtime, log);
  const bool boundary = checkFusionBoundary(log);
  passed = normAligned && normTail && addNorm && halfNorm && addLayerNorm &&
           linearRelu && views && boundary && passed;
  log << "\nTensor graph and fusion examples: " << (passed ? "PASS" : "FAIL") << '\n';
  return passed;
}
