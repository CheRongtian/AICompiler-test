#include "runtime/GraphExecutor.hpp"

#include "backend/metal/MetalEmitter.hpp"
#include "benchmark/Benchmark.hpp"
#include "planner/RMSNormTuner.hpp"
#include "validation/GraphReference.hpp"
#include "validation/Validator.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <utility>

namespace tensor::runtime {
namespace {

constexpr std::size_t kWarmup = 5;
constexpr std::size_t kSamples = 31;
constexpr double kMinimumSpeedup = 1.05;

metal::ElementType elementType(DType dtype) {
  if (dtype == DType::Float16) return metal::ElementType::Float16;
  if (dtype == DType::Int32) return metal::ElementType::Int32;
  return metal::ElementType::Float32;
}

std::pair<double, double> tolerance(DType dtype) {
  return dtype == DType::Float16 ? std::make_pair(2e-3, 2e-3)
                                 : std::make_pair(1e-5, 1e-4);
}

std::vector<float> logicalRead(const metal::BufferHandle &buffer,
                               const TensorType &type) {
  const auto physical = buffer->read();
  if (type.isContiguous()) return {physical.begin(), physical.begin() + type.elementCount()};
  const auto strides = type.strides();
  const auto logicalStrides = TensorType{type.shape}.strides();
  std::vector<float> result(type.elementCount());
  for (std::size_t linear = 0; linear < result.size(); ++linear) {
    std::size_t remaining = linear;
    std::size_t offset = 0;
    for (std::size_t axis = 0; axis < type.shape.size(); ++axis) {
      const auto coordinate = remaining / logicalStrides[axis];
      remaining %= logicalStrides[axis];
      offset += coordinate * strides[axis];
    }
    result[linear] = physical[offset];
  }
  return result;
}

std::vector<double> expected(const validation::GraphReference &reference, ValueId value) {
  return {reference.values[value].begin(), reference.values[value].end()};
}

std::string checkPipeline(const metal::ComputePipelineResult &pipeline,
                          const std::vector<TensorType> &inputs,
                          const TensorType &output) {
  if (!pipeline.pipelineCreationPassed) return pipeline.errorMessage;
  std::vector<metal::ElementType> inputTypes;
  for (const auto &type : inputs) inputTypes.push_back(elementType(type.dtype));
  return metal::checkBufferInterface(pipeline, inputTypes, elementType(output.dtype));
}

std::unique_ptr<metal::PreparedExecution>
prepareKernel(metal::MetalRuntime &runtime, const planner::KernelPlan &plan,
              const std::vector<metal::BufferHandle> &buffers, std::string &error) {
  const auto generated = metal::emitKernel(plan);
  const auto pipeline = runtime.createComputePipeline(generated.source, generated.functionName);
  error = checkPipeline(pipeline, plan.inputTypes, plan.outputType);
  if (!error.empty()) return {};
  std::vector<metal::BufferHandle> arguments;
  for (auto input : plan.inputs) arguments.push_back(buffers[input]);
  auto prepared = runtime.prepareBuffers(arguments, buffers[plan.output],
      {generated.threadgroupCount, generated.threadsPerThreadgroup});
  error = prepared.errorMessage;
  return std::move(prepared.execution);
}

std::unique_ptr<metal::PreparedExecution>
prepareFusion(metal::MetalRuntime &runtime, const analyzer::Region &region,
              const analyzer::AnalyzedGraph &graph, std::size_t threads,
              const std::vector<metal::BufferHandle> &buffers, std::string &error) {
  const auto generated = metal::emitFusion(region, graph, threads);
  std::vector<TensorType> inputTypes;
  std::vector<metal::BufferHandle> arguments;
  for (auto input : region.inputs) {
    inputTypes.push_back(graph.types[input]);
    arguments.push_back(buffers[input]);
  }
  const auto &outputType = graph.types[region.outputs.front()];
  const auto pipeline = runtime.createComputePipeline(generated.source, generated.functionName);
  error = checkPipeline(pipeline, inputTypes, outputType);
  if (!error.empty()) return {};
  if (threads > pipeline.maxTotalThreadsPerThreadgroup) {
    error = "Fused candidate exceeds the compiled pipeline thread limit.";
    return {};
  }
  auto prepared = runtime.prepareBuffers(arguments, buffers[region.outputs.front()],
      {generated.threadgroupCount, generated.threadsPerThreadgroup});
  error = prepared.errorMessage;
  return std::move(prepared.execution);
}

std::unique_ptr<metal::PreparedSequence>
makeSequence(metal::MetalRuntime &runtime,
             const std::vector<std::unique_ptr<metal::PreparedExecution>> &steps,
             std::string &error) {
  std::vector<const metal::PreparedExecution *> pointers;
  for (const auto &step : steps) pointers.push_back(step.get());
  auto prepared = runtime.prepareSequence(pointers);
  error = prepared.errorMessage;
  return std::move(prepared.execution);
}

std::vector<metal::BufferHandle>
allocateGraphBuffers(metal::MetalRuntime &runtime, const planner::GraphPlan &plan,
                     const GraphInputs *inputs, const validation::GraphReference *reference) {
  std::vector<metal::BufferHandle> buffers(plan.buffers.size());
  for (const auto &buffer : plan.buffers) {
    if (buffer.aliasOf) continue;
    const float *initial = nullptr;
    if (inputs && buffer.role == planner::BufferRole::Input) {
      initial = inputs->at(buffer.value).data();
    } else if (reference && std::find(plan.analyzed.graph.inputs.begin(),
                                     plan.analyzed.graph.inputs.end(), buffer.value) !=
                                plan.analyzed.graph.inputs.end()) {
      initial = reference->values[buffer.value].data();
    }
    auto allocated = runtime.createBuffer(buffer.storageElementCount, initial,
                                          elementType(buffer.type.dtype));
    if (!allocated.buffer) throw std::runtime_error(allocated.errorMessage);
    buffers[buffer.value] = std::move(allocated.buffer);
  }
  for (const auto &buffer : plan.buffers) {
    if (buffer.aliasOf) {
      if (!buffers[*buffer.aliasOf]) throw std::runtime_error("View alias base is unavailable.");
      buffers[buffer.value] = buffers[*buffer.aliasOf];
    }
  }
  return buffers;
}

std::vector<metal::BufferHandle>
allocateRegionBuffers(metal::MetalRuntime &runtime, const planner::RegionPlan &region,
                      const planner::GraphPlan &graph,
                      const validation::GraphReference &reference) {
  std::vector<metal::BufferHandle> buffers(graph.buffers.size());
  for (auto input : region.region.inputs) {
    const auto &type = graph.analyzed.types[input];
    auto allocated = runtime.createBuffer(type.storageElementCount(), reference.values[input].data(),
                                          elementType(type.dtype));
    if (!allocated.buffer) throw std::runtime_error(allocated.errorMessage);
    buffers[input] = std::move(allocated.buffer);
  }
  for (const auto &kernel : region.baseline) {
    if (!buffers[kernel.output]) {
      auto allocated = runtime.createBuffer(kernel.outputType.storageElementCount(), nullptr,
                                            elementType(kernel.outputType.dtype));
      if (!allocated.buffer) throw std::runtime_error(allocated.errorMessage);
      buffers[kernel.output] = std::move(allocated.buffer);
    }
  }
  return buffers;
}

std::vector<std::unique_ptr<metal::PreparedExecution>>
prepareBaseline(metal::MetalRuntime &runtime, const planner::RegionPlan &region,
                const std::vector<metal::BufferHandle> &buffers) {
  std::vector<std::unique_ptr<metal::PreparedExecution>> steps;
  for (const auto &kernel : region.baseline) {
    std::string error;
    auto prepared = prepareKernel(runtime, kernel, buffers, error);
    if (!prepared) throw std::runtime_error(error);
    steps.push_back(std::move(prepared));
  }
  return steps;
}

bool validateBuffer(const metal::BufferHandle &buffer, const TensorType &type,
                    const std::vector<double> &reference, std::string &error) {
  const auto limits = tolerance(type.dtype);
  const auto result = validation::compare(logicalRead(buffer, type), reference,
                                          limits.first, limits.second);
  error = result.errorMessage;
  return result.passed;
}

struct FusionSelection { bool fused = false; std::size_t threads = 0; };

struct AdmittedFusionCandidate {
  std::size_t threads = 0;
  double conservativeSpeedup = 0.0;
};

FusionSelection tuneRegion(metal::MetalRuntime &runtime, planner::RegionPlan &region,
                           const planner::GraphPlan &graph,
                           const validation::GraphReference &reference,
                           std::ostream &log) {
  FusionSelection selection;
  log << "Pattern detected: " << analyzer::fusionName(region.region.fusion) << '\n';
  auto baselineBuffers = allocateRegionBuffers(runtime, region, graph, reference);
  auto baselineSteps = prepareBaseline(runtime, region, baselineBuffers);
  std::string error;
  auto baseline = makeSequence(runtime, baselineSteps, error);
  if (!baseline) throw std::runtime_error("Unfused region preparation failed: " + error);
  const auto baselineRun = baseline->execute();
  if (!baselineRun.executionPassed) throw std::runtime_error("Unfused region execution failed: " + baselineRun.errorMessage);
  const auto output = region.region.outputs.front();
  if (!validateBuffer(baselineBuffers[output], graph.analyzed.types[output],
                      expected(reference, output), error)) {
    throw std::runtime_error("Unfused region validation failed: " + error);
  }
  log << "Unfused validation: PASS\n";
  const auto warmupError = benchmark::warmup(*baseline, kWarmup);
  if (!warmupError.empty()) throw std::runtime_error("Unfused region warmup failed: " + warmupError);

  std::vector<AdmittedFusionCandidate> admittedCandidates;
  for (auto threads : region.candidateThreads) {
    log << "Fused candidate threads=" << threads << " | ";
    const auto hardware = runtime.hardwareInfo();
    const std::size_t scratchArrays =
        region.region.fusion == analyzer::FusionPattern::AddLayerNorm ? 2
      : region.region.fusion == analyzer::FusionPattern::AddRMSNorm ? 1 : 0;
    if (threads > hardware.maxThreadsPerThreadgroup ||
        threads * scratchArrays * sizeof(float) > hardware.maxThreadgroupMemoryLength) {
      log << "Hardware Filter: FAIL\n";
      continue;
    }
    auto candidateBuffers = baselineBuffers;
    auto allocated = runtime.createBuffer(graph.analyzed.types[output].storageElementCount(), nullptr,
                                          elementType(graph.analyzed.types[output].dtype));
    if (!allocated.buffer) {
      log << "Buffer allocation: FAIL: " << allocated.errorMessage << '\n';
      continue;
    }
    candidateBuffers[output] = std::move(allocated.buffer);
    std::string candidateError;
    auto step = prepareFusion(runtime, region.region, graph.analyzed, threads,
                              candidateBuffers, candidateError);
    if (!step) {
      log << "Compile/interface: FAIL: " << candidateError << '\n';
      continue;
    }
    std::vector<std::unique_ptr<metal::PreparedExecution>> candidateSteps;
    candidateSteps.push_back(std::move(step));
    auto candidate = makeSequence(runtime, candidateSteps, candidateError);
    if (!candidate) {
      log << "Prepare: FAIL: " << candidateError << '\n';
      continue;
    }
    const auto run = candidate->execute();
    if (!run.executionPassed ||
        !validateBuffer(candidateBuffers[output], graph.analyzed.types[output],
                        expected(reference, output), candidateError)) {
      log << "Numerical validation: FAIL: "
          << (run.executionPassed ? candidateError : run.errorMessage) << '\n';
      continue;
    }
    log << "Numerical validation: PASS";
    const auto candidateWarmup = benchmark::warmup(*candidate, kWarmup);
    if (!candidateWarmup.empty()) {
      log << ", Warmup: FAIL: " << candidateWarmup << '\n';
      continue;
    }
    const auto first = benchmark::measurePair(*baseline, *candidate, kSamples);
    if (!first.passed) {
      log << ", Benchmark: FAIL: " << first.errorMessage << '\n';
      continue;
    }
    log << ", unfused(us)=" << first.baseline.medianUs
        << ", fused(us)=" << first.candidate.medianUs
        << ", speedup=" << first.speedup << 'x';
    if (first.speedup < kMinimumSpeedup) {
      log << " -> rejected: below admission threshold\n";
      continue;
    }
    const auto confirmation = benchmark::measurePair(*baseline, *candidate, kSamples);
    if (!confirmation.passed) {
      log << ", confirmation: FAIL\n";
      continue;
    }
    const auto admittedSpeedup = std::min(first.speedup, confirmation.speedup);
    log << ", confirmation=" << confirmation.speedup << 'x';
    if (admittedSpeedup < kMinimumSpeedup) {
      log << " -> rejected: confirmation below admission threshold\n";
      continue;
    }

    admittedCandidates.push_back({threads, admittedSpeedup});
    log << ", conservative=" << admittedSpeedup << "x -> admitted\n";
  }

  if (!admittedCandidates.empty()) {
    const auto winner = std::max_element(
        admittedCandidates.begin(), admittedCandidates.end(),
        [](const auto &left, const auto &right) {
          return left.conservativeSpeedup < right.conservativeSpeedup;
        });
    selection = {true, winner->threads};
    log << "Selected: fused, threads=" << selection.threads
        << ", admitted=" << admittedCandidates.size()
        << ", conservative speedup=" << winner->conservativeSpeedup << "x\n";
  } else {
    log << "Selected: unfused fallback, admitted=0\n";
  }
  return selection;
}

} // namespace

CompiledGraph::CompiledGraph(planner::ProgramPlan plan,
                             std::vector<metal::BufferHandle> buffers,
                             std::vector<ExecutionUnit> units)
    : plan_(std::move(plan)), buffers_(std::move(buffers)), units_(std::move(units)) {}

GraphExecutionResult CompiledGraph::run() const {
  GraphExecutionResult result;
  double gpuSum = 0.0;
  bool timestamps = true;
  for (const auto &unit : units_) {
    const auto execution = unit.execution->execute();
    if (!execution.executionPassed) {
      result.errorMessage = unit.name + ": " + execution.errorMessage;
      return result;
    }
    if (execution.gpuExecutionTimeUs) gpuSum += *execution.gpuExecutionTimeUs;
    else timestamps = false;
  }
  for (auto output : plan_.graph.analyzed.graph.outputs) {
    result.outputs.push_back(logicalRead(buffers_[output], plan_.graph.analyzed.types[output]));
  }
  if (timestamps) result.gpuExecutionTimeUs = gpuSum;
  result.passed = true;
  return result;
}

GraphCompilation compileGraph(metal::MetalRuntime &runtime, const TensorGraph &graph,
                              const GraphInputs &inputs, std::ostream &log) {
  GraphCompilation result;
  try {
    if (!runtime.isAvailable()) throw std::runtime_error(runtime.initializationError());
    auto program = planner::planRegions(planner::planGraph(analyzer::analyze(graph)));
    log << "Graph analysis: PASS, nodes=" << graph.nodes.size()
        << ", regions=" << program.regions.size() << '\n';
    const auto reference = validation::evaluateGraph(program.graph.analyzed, inputs);

    for (auto &region : program.regions) {
      for (auto &kernel : region.baseline) {
        if (kernel.op != OpType::RMSNorm) continue;
        const auto &attributes = std::get<RMSNormAttributes>(kernel.attributes);
        const RMSNormOp op{kernel.inputTypes[0], kernel.inputTypes[1], kernel.outputType,
                           attributes.epsilon};
        auto tuned = planner::tuneRMSNorm(runtime, op,
            reference.values[kernel.inputs[0]], reference.values[kernel.inputs[1]], log);
        if (!tuned.success) throw std::runtime_error(tuned.errorMessage);
        kernel.threadsPerThreadgroup = tuned.selectedThreads;
        kernel.useRMSNormBaseline = tuned.usedBaseline;
      }
    }

    std::vector<FusionSelection> selections(program.regions.size());
    for (std::size_t i = 0; i < program.regions.size(); ++i) {
      if (program.regions[i].region.fusion != analyzer::FusionPattern::None) {
        selections[i] = tuneRegion(runtime, program.regions[i], program.graph, reference, log);
      }
    }

    auto buffers = allocateGraphBuffers(runtime, program.graph, &inputs, nullptr);
    std::vector<CompiledGraph::ExecutionUnit> units;
    for (std::size_t i = 0; i < program.regions.size(); ++i) {
      auto &region = program.regions[i];
      if (region.baseline.empty()) continue; // A pure view aliases storage and submits no work.
      std::vector<std::unique_ptr<metal::PreparedExecution>> steps;
      std::string error;
      if (selections[i].fused) {
        auto fused = prepareFusion(runtime, region.region, program.graph.analyzed,
                                   selections[i].threads, buffers, error);
        if (!fused) throw std::runtime_error(error);
        steps.push_back(std::move(fused));
      } else {
        steps = prepareBaseline(runtime, region, buffers);
      }
      auto sequence = makeSequence(runtime, steps, error);
      if (!sequence) throw std::runtime_error(error);
      units.push_back({analyzer::fusionName(region.region.fusion), std::move(sequence)});
    }
    for (auto output : graph.outputs) result.outputTypes.push_back(program.graph.analyzed.types[output]);
    result.referenceOutputs = reference.outputs;
    result.executable = std::make_unique<CompiledGraph>(std::move(program), std::move(buffers),
                                                        std::move(units));
    log << "Graph compilation: PASS\n";
  } catch (const std::exception &error) {
    result.errorMessage = error.what();
  }
  return result;
}

} // namespace tensor::runtime
