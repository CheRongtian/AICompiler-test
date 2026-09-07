#include "runtime/GraphExecutor.hpp"

#include "backend/metal/MetalEmitter.hpp"
#include "planner/RMSNormTuner.hpp"
#include "validation/GraphReference.hpp"

#include <exception>
#include <ostream>
#include <stdexcept>
#include <utility>

namespace tensor::runtime {

CompiledGraph::CompiledGraph(planner::GraphPlan plan, std::vector<metal::BufferHandle> buffers,
                             std::vector<std::unique_ptr<metal::PreparedExecution>> steps)
    : plan_(std::move(plan)), buffers_(std::move(buffers)), steps_(std::move(steps)) {}

GraphExecutionResult CompiledGraph::run() const {
  GraphExecutionResult result;
  double gpuSum = 0.0;
  bool allTimestampsAvailable = true;
  for (std::size_t i = 0; i < steps_.size(); ++i) {
    // Synchronous submission orders dependencies. No intermediate CPU readback.
    const auto step = steps_[i]->execute();
    if (!step.executionPassed) {
      result.errorMessage = plan_.analyzed.graph.values[plan_.kernels[i].output].name +
                            ": " + step.errorMessage;
      return result;
    }
    if (step.gpuExecutionTimeUs) gpuSum += *step.gpuExecutionTimeUs;
    else allTimestampsAvailable = false;
  }
  for (auto id : plan_.analyzed.graph.outputs) {
    result.outputs.push_back(buffers_[id]->read());
  }
  if (allTimestampsAvailable) result.gpuExecutionTimeUs = gpuSum;
  result.passed = true;
  return result;
}

GraphCompilation compileGraph(metal::MetalRuntime &runtime, const TensorGraph &graph,
                               const GraphInputs &inputs, std::ostream &log) {
  GraphCompilation result;
  try {
    if (!runtime.isAvailable()) throw std::runtime_error(runtime.initializationError());
    auto plan = planner::planGraph(analyzer::analyze(graph));
    log << "Graph analysis: PASS, nodes=" << plan.kernels.size()
        << ", values=" << plan.buffers.size() << '\n';
    const auto hardware = runtime.hardwareInfo();
    for (const auto &buffer : plan.buffers) {
      if (buffer.elementCount > hardware.maxBufferLength / sizeof(float)) {
        throw std::runtime_error("Graph tensor exceeds the Metal buffer limit.");
      }
    }
    auto reference = validation::evaluateGraph(plan.analyzed, inputs);
    std::vector<metal::BufferHandle> buffers(plan.buffers.size());
    for (const auto &buffer : plan.buffers) {
      const float *data = buffer.role == planner::BufferRole::Input
                              ? inputs.at(buffer.value).data() : nullptr;
      auto allocated = runtime.createBuffer(buffer.elementCount, data);
      if (!allocated.buffer) throw std::runtime_error(allocated.errorMessage);
      buffers[buffer.value] = std::move(allocated.buffer);
    }

    std::vector<std::unique_ptr<metal::PreparedExecution>> steps;
    for (auto &kernelPlan : plan.kernels) {
      const auto &name = graph.values[kernelPlan.output].name;
      log << "Node " << name << " | " << opName(kernelPlan.op) << '\n';
      if (kernelPlan.op == OpType::RMSNorm) {
        const auto &attributes = std::get<RMSNormAttributes>(kernelPlan.attributes);
        const RMSNormOp op{kernelPlan.inputTypes[0], kernelPlan.inputTypes[1],
                           kernelPlan.outputType, attributes.epsilon};
        auto tuned = planner::tuneRMSNorm(runtime, op,
            reference.values[kernelPlan.inputs[0]], reference.values[kernelPlan.inputs[1]], log);
        if (!tuned.success) throw std::runtime_error(name + ": " + tuned.errorMessage);
        kernelPlan.threadsPerThreadgroup = tuned.selectedThreads;
        kernelPlan.useRMSNormBaseline = tuned.usedBaseline;
      }
      const auto kernel = metal::emitKernel(kernelPlan);
      const auto pipeline = runtime.createComputePipeline(kernel.source, kernel.functionName);
      if (!pipeline.pipelineCreationPassed) {
        throw std::runtime_error(name + " compile: " + pipeline.errorMessage);
      }
      const auto interfaceError = metal::checkFloatBufferInterface(pipeline, kernelPlan.inputs.size());
      if (!interfaceError.empty()) throw std::runtime_error(name + ": " + interfaceError);
      std::vector<metal::BufferHandle> arguments;
      for (auto id : kernelPlan.inputs) arguments.push_back(buffers[id]);
      auto prepared = runtime.prepareBuffers(arguments, buffers[kernelPlan.output],
          {kernel.threadgroupCount, kernel.threadsPerThreadgroup});
      if (!prepared.execution) throw std::runtime_error(name + ": " + prepared.errorMessage);
      steps.push_back(std::move(prepared.execution));
      log << "  Compile/interface: PASS, threadgroups=" << kernel.threadgroupCount
          << ", threads=" << kernel.threadsPerThreadgroup << '\n';
    }
    for (auto id : graph.outputs) result.outputTypes.push_back(plan.analyzed.types[id]);
    result.referenceOutputs = std::move(reference.outputs);
    result.executable = std::make_unique<CompiledGraph>(std::move(plan), std::move(buffers),
                                                       std::move(steps));
    log << "Graph compilation: PASS\n";
  } catch (const std::exception &error) {
    result.errorMessage = error.what();
  }
  return result;
}

} // namespace tensor::runtime
