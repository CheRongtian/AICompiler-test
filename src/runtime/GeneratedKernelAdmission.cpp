#include "runtime/GeneratedKernelAdmission.hpp"
#include "runtime/KernelRegistry.hpp"

#include "benchmark/Benchmark.hpp"
#include "llm/GeneratedKernelProtocol.hpp"
#include "llm/KernelContract.hpp"
#include "validation/Validator.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace tensor::runtime {
namespace {
constexpr std::size_t kWarmup = 5;
constexpr std::size_t kSamples = 31;
// A RoPE work item writes two elements; cover a full 256-thread padded tail.
constexpr std::size_t kGuardElements = 512;
struct Feedback {
  std::string status;
  std::string stage;
  std::string message;
  std::optional<double> baselineUs;
  std::optional<double> candidateUs;
  std::optional<double> speedup;
};

std::string escapeJson(const std::string &text) {
  std::ostringstream output;
  const char hex[] = "0123456789abcdef";
  for (const unsigned char character : text) {
    switch (character) {
    case '\"': output << "\\\""; break;
    case '\\': output << "\\\\"; break;
    case '\b': output << "\\b"; break;
    case '\f': output << "\\f"; break;
    case '\n': output << "\\n"; break;
    case '\r': output << "\\r"; break;
    case '\t': output << "\\t"; break;
    default:
      if (character < 0x20) {
        output << "\\u00" << hex[character >> 4] << hex[character & 0xf];
      } else {
        output << static_cast<char>(character);
      }
    }
  }
  return output.str();
}

std::string writeFeedback(const Feedback &feedback, const std::string &path) {
  std::ofstream output(path);
  if (!output) return "Unable to open generated-kernel feedback output '" + path + "'.";
  output << "{\n  \"version\":1,\n  \"status\":\""
         << escapeJson(feedback.status) << "\",\n  \"stage\":\""
         << escapeJson(feedback.stage) << "\",\n  \"message\":\""
         << escapeJson(feedback.message) << "\",\n  \"metrics\":";
  if (feedback.baselineUs && feedback.candidateUs && feedback.speedup) {
    output << "{\"baseline_median_us\":" << *feedback.baselineUs
           << ",\"candidate_median_us\":" << *feedback.candidateUs
           << ",\"speedup\":" << *feedback.speedup << '}';
  } else {
    output << "null";
  }
  output << "\n}\n";
  if (!output) return "Failed while writing generated-kernel feedback '" + path + "'.";
  return {};
}


struct CaseExecution {
  std::unique_ptr<metal::PreparedSequence> sequence;
  std::vector<metal::BufferHandle> outputs;
};

std::unique_ptr<CaseExecution> prepare(
    metal::MetalRuntime &runtime, const llm::KernelContract &contract,
    const llm::KernelCase &data, const metal::DispatchSize &dispatch,
    const llm::KernelBaseline *baseline = nullptr) {
  auto result = std::make_unique<CaseExecution>();
  const bool generated = baseline == nullptr;
  std::vector<metal::BufferHandle> buffers;
  for (std::size_t i=0;i<data.inputs.size();++i) {
    auto padded = data.inputs[i];
    if (generated) padded.resize(padded.size()+kGuardElements, 1.0f);
    auto buffer = runtime.createBuffer(padded.size(),padded.data(),contract.inputTypes[i]);
    if (!buffer.buffer) throw std::runtime_error(buffer.errorMessage);
    buffers.push_back(std::move(buffer.buffer));
  }
  for (const auto &reference : data.references) {
    auto buffer = runtime.createBuffer(reference.size()+(generated?kGuardElements:0),nullptr,
                                       contract.storageType);
    if (!buffer.buffer) throw std::runtime_error(buffer.errorMessage);
    buffers.push_back(buffer.buffer);
    result->outputs.push_back(std::move(buffer.buffer));
  }
  std::vector<std::unique_ptr<metal::PreparedExecution>> steps;
  if (baseline) {
    for (auto count : baseline->intermediateCounts) {
      auto buffer = runtime.createBuffer(count,nullptr,contract.storageType);
      if (!buffer.buffer) throw std::runtime_error(buffer.errorMessage);
      buffers.push_back(std::move(buffer.buffer));
    }
    for (const auto &step : baseline->steps) {
      const auto pipeline = runtime.createComputePipeline(step.kernel.source,step.kernel.functionName);
      if (!pipeline.pipelineCreationPassed) throw std::runtime_error(pipeline.errorMessage);
      std::vector<metal::BufferHandle> inputs, outputs;
      std::vector<metal::ElementType> inputTypes, outputTypes;
      for (auto i : step.inputs) {
        inputs.push_back(buffers.at(i));
        inputTypes.push_back(buffers.at(i)->elementType());
      }
      for (auto i : step.outputs) {
        outputs.push_back(buffers.at(i));
        outputTypes.push_back(buffers.at(i)->elementType());
      }
      auto error = metal::checkBufferInterface(pipeline,inputTypes,outputTypes);
      if (!error.empty()) throw std::runtime_error(error);
      auto prepared = runtime.prepareBuffers(inputs,outputs,
          {step.kernel.threadgroupCount,step.kernel.threadsPerThreadgroup});
      if (!prepared.execution) throw std::runtime_error(prepared.errorMessage);
      steps.push_back(std::move(prepared.execution));
    }
  } else {
    std::vector<metal::BufferHandle> inputs(
        buffers.begin(),buffers.begin()+data.inputs.size());
    auto prepared = runtime.prepareBuffers(inputs,result->outputs,dispatch,data.constants);
    if (!prepared.execution) throw std::runtime_error(prepared.errorMessage);
    steps.push_back(std::move(prepared.execution));
  }
  std::vector<const metal::PreparedExecution *> pointers;
  for (const auto &step : steps) pointers.push_back(step.get());
  auto sequence = runtime.prepareSequence(pointers);
  if (!sequence.execution) throw std::runtime_error(sequence.errorMessage);
  result->sequence = std::move(sequence.execution);
  return result;
}

std::string validate(metal::MetalRuntime &runtime, CaseExecution &execution,
                     const llm::KernelCase &data,
                     const llm::KernelContract &contract, bool guarded) {
  for (const auto &output : execution.outputs) {
    std::vector<float> sentinel(output->elementCount(),std::numeric_limits<float>::quiet_NaN());
    const auto error = runtime.writeBuffer(output,sentinel.data(),sentinel.size());
    if (!error.empty()) return error;
  }
  const auto result = execution.sequence->execute();
  if (!result.executionPassed) return result.errorMessage;
  for (std::size_t i=0;i<execution.outputs.size();++i) {
    const auto output = execution.outputs[i]->read();
    const auto count = data.references[i].size();
    if (output.size() != count+(guarded?kGuardElements:0)) return "Output size mismatch.";
    const std::vector<float> logical(output.begin(),output.begin()+count);
    const auto comparison = validation::compare(
        logical,data.references[i],contract.absoluteTolerance,contract.relativeTolerance);
    if (!comparison.passed) return contract.outputNames[i]+": "+comparison.errorMessage;
    if (guarded && !std::all_of(output.begin()+count,output.end(),
                               [](float value){return std::isnan(value);}))
      return "Output guard modified: "+contract.outputNames[i];
  }
  return {};
}

struct Baseline {
  std::unique_ptr<CaseExecution> execution;
  double medianUs = std::numeric_limits<double>::infinity();
  std::string name;
};

Baseline baselineFor(metal::MetalRuntime &runtime,
                     const llm::KernelContract &contract,
                     const llm::KernelCase &data, std::ostream &log) {
  Baseline best;
  for (const auto &plan : llm::makeContractBaselines(contract,data)) {
    try {
      auto execution = prepare(runtime,contract,data,{},&plan);
      auto error = validate(runtime,*execution,data,contract,false);
      if (!error.empty()) throw std::runtime_error(error);
      error = benchmark::warmup(*execution->sequence,kWarmup);
      if (!error.empty()) throw std::runtime_error(error);
      const auto timing = benchmark::measure(*execution->sequence,kSamples);
      if (!timing.passed) throw std::runtime_error(timing.errorMessage);
      log << "Template baseline " << plan.name
          << " validation: PASS, median(us)=" << timing.stats.medianUs << '\n';
      if (timing.stats.medianUs<best.medianUs)
        best={std::move(execution),timing.stats.medianUs,plan.name};
    } catch (const std::exception &error) {
      log << "Template baseline " << plan.name << ": FAIL: " << error.what() << '\n';
    }
  }
  if (!best.execution) throw std::runtime_error("No valid template baseline available.");
  log << "Selected baseline " << best.name << ", median(us)=" << best.medianUs << '\n';
  return best;
}
} // namespace

bool admitGeneratedKernel(metal::MetalRuntime &runtime,
                           const std::string &pattern,
                           const std::string &responsePath,
                           const std::string &feedbackPath, std::ostream &log,
                           const std::string &artifactPath) {
  auto report = [&](const Feedback &feedback) {
    const auto error = writeFeedback(feedback, feedbackPath);
    log << "Generated candidate " << feedback.stage << ": "
        << (feedback.status == "admitted" ? "PASS" : "FAIL")
        << ": " << feedback.message << '\n';
    if (!error.empty()) {
      log << "Feedback error: " << error << '\n';
      return false;
    }
    log << "Generated kernel admission: "
        << (feedback.status == "admitted" ? "PASS" :
            feedback.status == "fatal" ? "FAIL" : "FALLBACK") << '\n'
        << "Selected: " << (feedback.status == "admitted" ? "generated " : "template ")
        << pattern << '\n';
    return feedback.status != "fatal";
  };
  try {
    const auto contract = llm::makeKernelContract(pattern);
    if(contract.storageType==metal::ElementType::BFloat16 &&
       !runtime.hardwareInfo().supportsBFloat16)
      return report({"fatal","hardware_filter","Native bf16 is unavailable; generate an fp16 contract."});
    std::vector<Baseline> baselines;
    for (std::size_t i = 0; i < contract.cases.size(); ++i) {
      log << "Template baseline case " << i << '\n';
      baselines.push_back(baselineFor(runtime, contract, contract.cases[i], log));
    }
    auto loaded = llm::loadGeneratedKernelResponse(responsePath);
    if (!loaded.response)
      return report({"retry", "response_parse", loaded.errorMessage});
    const auto &candidate = *loaded.response;
    if (candidate.functionName != contract.functionName)
      return report({"retry", "contract", "Expected function_name: " + contract.functionName});
    if (std::find(contract.workgroupSizes.begin(), contract.workgroupSizes.end(),
                  candidate.workgroupSize) == contract.workgroupSizes.end() ||
        candidate.workgroupSize > runtime.hardwareInfo().maxThreadsPerThreadgroup)
      return report({"retry", "hardware_filter", "workgroup_size is not legal for this contract/device."});
    const auto pipeline = runtime.createComputePipeline(candidate.source, candidate.functionName);
    if (!pipeline.pipelineCreationPassed)
      return report({"retry", "compile", pipeline.errorMessage});
    log << "Generated candidate compile: PASS\n";
    const auto interfaceError = metal::checkBufferInterface(
        pipeline, contract.inputTypes,
        std::vector<metal::ElementType>(contract.outputNames.size(),contract.storageType), true);
    if (!interfaceError.empty())
      return report({"retry", "interface", interfaceError});
    if (candidate.workgroupSize > pipeline.maxTotalThreadsPerThreadgroup ||
        pipeline.staticThreadgroupMemoryLength > runtime.hardwareInfo().maxThreadgroupMemoryLength)
      return report({"retry", "hardware_filter", "Compiled pipeline exceeds device/dispatch limits."});
    log << "Generated candidate interface: PASS, outputs="
        << contract.outputNames.size() << '\n';

    std::vector<std::unique_ptr<CaseExecution>> executions;
    for (std::size_t i = 0; i < contract.cases.size(); ++i) {
      const auto &data = contract.cases[i];
      std::unique_ptr<CaseExecution> execution;
      try {
        execution = prepare(runtime, contract, data,
            llm::contractDispatch(contract,data.workItems,candidate.workgroupSize));
      } catch (const std::exception &error) {
        return report({"retry", "prepare", error.what()});
      }
      const auto error = validate(runtime,*execution, data, contract, true);
      if (!error.empty())
        return report({"retry", "numerical_validation", "case " + std::to_string(i) + ": " + error});
      log << "Generated candidate case " << i << " all outputs numerical validation: PASS\n";
      executions.push_back(std::move(execution));
    }

    double worstSpeedup = std::numeric_limits<double>::infinity();
    std::optional<Feedback> performanceFailure;
    Feedback accepted{"admitted", "admission", "All cases passed correctness and two performance rounds."};
    for (std::size_t i = 0; i < executions.size(); ++i) {
      const auto error = benchmark::warmup(*executions[i]->sequence, kWarmup);
      if (!error.empty()) return report({"retry", "warmup", error});
      for (std::size_t round = 0; round < 2; ++round) {
        const auto timing = benchmark::measurePair(
            *baselines[i].execution->sequence, *executions[i]->sequence, kSamples);
        if (!timing.passed) return report({"retry", "benchmark", timing.errorMessage});
        log << "Generated candidate case " << i << " round " << round + 1
            << ": baseline(us)=" << timing.baseline.medianUs
            << ", candidate(us)=" << timing.candidate.medianUs
            << ", speedup=" << timing.speedup << "x\n";
        if (timing.speedup < worstSpeedup) {
          worstSpeedup = timing.speedup;
          accepted.baselineUs = timing.baseline.medianUs;
          accepted.candidateUs = timing.candidate.medianUs;
          accepted.speedup = timing.speedup;
        }
        if (timing.speedup < contract.minimumSpeedup &&
            (!performanceFailure || timing.speedup < *performanceFailure->speedup))
          performanceFailure = Feedback{
              "retry", "performance", "case " + std::to_string(i) +
              " round " + std::to_string(round + 1) + " speedup is below " +
              std::to_string(contract.minimumSpeedup) + "x.",
              timing.baseline.medianUs, timing.candidate.medianUs, timing.speedup};
      }
      const auto finalError = validate(runtime,*executions[i], contract.cases[i], contract, true);
      if (!finalError.empty())
        return report({"retry", "final_validation", finalError});
    }
    if (performanceFailure) return report(*performanceFailure);
    if (!artifactPath.empty()) {
      const auto error = writeAdmittedKernel(runtime,contract,candidate,artifactPath);
      if (!error.empty()) return report({"fatal","artifact_write",error});
    }
    return report(accepted);
  } catch (const std::exception &error) {
    return report({"fatal", "baseline_or_contract", error.what()});
  }
}
} // namespace tensor::runtime
