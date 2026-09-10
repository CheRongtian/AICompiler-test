#include "planner/DecodeGEMVTuner.hpp"

#include "benchmark/Benchmark.hpp"
#include "validation/Validator.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <utility>

namespace tensor::planner {
namespace {

constexpr std::size_t kWarmup = 5;
constexpr std::size_t kSamples = 31;
constexpr double kMinimumSpeedup = 1.05;
constexpr std::size_t kMaximumCpuReferenceOperations = 64u * 1024u * 1024u;

metal::ElementType elementType(DType dtype) {
  if (dtype == DType::Float16) return metal::ElementType::Float16;
  if (dtype == DType::Float32) return metal::ElementType::Float32;
  if (dtype == DType::BFloat16) return metal::ElementType::BFloat16;
  throw std::invalid_argument("Decode GEMV tuner supports fp16, bf16, or fp32.");
}

std::uint16_t floatToHalf(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t sign = (bits >> 16) & 0x8000u;
  const std::uint32_t exponent = (bits >> 23) & 0xffu;
  const std::uint32_t mantissa = bits & 0x7fffffu;
  if (exponent == 0xffu) {
    return static_cast<std::uint16_t>(sign | (mantissa ? 0x7e00u : 0x7c00u));
  }
  const int halfExponent = static_cast<int>(exponent) - 112;
  if (halfExponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
  if (halfExponent <= 0) {
    if (halfExponent < -10) return static_cast<std::uint16_t>(sign);
    const std::uint32_t normalized = mantissa | 0x800000u;
    const int shift = 14 - halfExponent;
    return static_cast<std::uint16_t>(
        sign | ((normalized + (1u << (shift - 1))) >> shift));
  }
  const std::uint32_t rounded = mantissa + 0x1000u;
  if (rounded & 0x800000u) {
    if (halfExponent + 1 >= 31) {
      return static_cast<std::uint16_t>(sign | 0x7c00u);
    }
    return static_cast<std::uint16_t>(sign | ((halfExponent + 1) << 10));
  }
  return static_cast<std::uint16_t>(sign | (halfExponent << 10) |
                                    (rounded >> 13));
}

float halfToFloat(std::uint16_t value) {
  const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000u) << 16;
  std::uint32_t exponent = (value >> 10) & 0x1fu;
  std::uint32_t mantissa = value & 0x3ffu;
  std::uint32_t bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {
      int shift = 0;
      while ((mantissa & 0x400u) == 0) {
        mantissa <<= 1;
        ++shift;
      }
      mantissa &= 0x3ffu;
      bits = sign | static_cast<std::uint32_t>(113 - shift) << 23 |
             mantissa << 13;
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000u | mantissa << 13;
  } else {
    bits = sign | (exponent + 112) << 23 | mantissa << 13;
  }
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

float storageValue(float value, DType dtype) {
  if (dtype == DType::BFloat16) {
    std::uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }
  return dtype == DType::Float16 ? halfToFloat(floatToHalf(value)) : value;
}

std::vector<double> reference(std::size_t batch, std::size_t inputSize,
                              std::size_t outputSize,
                              const std::vector<float> &input,
                              const std::vector<float> &weight,
                              DType storageDtype) {
  std::vector<double> output(batch * outputSize);
  for (std::size_t row = 0; row < batch; ++row) {
    for (std::size_t feature = 0; feature < outputSize; ++feature) {
      double sum = 0.0;
      for (std::size_t inner = 0; inner < inputSize; ++inner) {
        sum += static_cast<double>(
                   storageValue(input[row * inputSize + inner], storageDtype)) *
               static_cast<double>(storageValue(
                   weight[feature * inputSize + inner], storageDtype));
      }
      output[row * outputSize + feature] =
          storageValue(static_cast<float>(sum), storageDtype);
    }
  }
  return output;
}

std::unique_ptr<metal::PreparedExecution> prepare(
    metal::MetalRuntime &runtime, const metal::GeneratedKernel &kernel,
    const std::vector<float> &input, const std::vector<float> &weight,
    std::size_t outputCount, const std::vector<double> &expected,
    std::string &error, DType storageDtype) {
  const auto type = elementType(storageDtype);
  const auto pipeline =
      runtime.createComputePipeline(kernel.source, kernel.functionName);
  if (!pipeline.pipelineCreationPassed) {
    error = pipeline.errorMessage;
    return {};
  }
  error = metal::checkBufferInterface(
      pipeline, {type, type}, type);
  if (!error.empty()) return {};
  auto prepared = runtime.prepare(
      {{input.data(), input.size(), type}, {weight.data(), weight.size(), type}},
      outputCount, {kernel.threadgroupCount, kernel.threadsPerThreadgroup}, {},
      type);
  if (!prepared.execution) {
    error = prepared.errorMessage;
    return {};
  }
  const auto execution = prepared.execution->run();
  if (!execution.executionPassed) {
    error = execution.errorMessage;
    return {};
  }
  const auto tolerance = storageDtype == DType::Float32 ? 2e-3 : 2e-2;
  if (!expected.empty()) {
    const auto validation = validation::compare(
        execution.output, expected, tolerance, tolerance);
    if (!validation.passed) {
      error = validation.errorMessage;
      return {};
    }
  }
  return std::move(prepared.execution);
}

} // namespace

DecodeGEMVSelection tuneDecodeGEMV(
    metal::MetalRuntime &runtime, std::size_t batch, std::size_t inputSize,
    std::size_t outputSize, const std::vector<float> &input,
    const std::vector<float> &weight, std::ostream &log,
    DType storageDtype, bool prefill) {
  DecodeGEMVSelection result;
  if (input.size() != batch * inputSize ||
      weight.size() != outputSize * inputSize) {
    result.errorMessage = "Decode GEMV tuning inputs do not match the shape.";
    return result;
  }
  const auto type = elementType(storageDtype);
  const bool useCpuReference =
      inputSize <= kMaximumCpuReferenceOperations / outputSize &&
      batch <= kMaximumCpuReferenceOperations / (inputSize * outputSize);
  const auto sampleCount = useCpuReference ? kSamples : std::size_t{9};
  auto expected = useCpuReference
                      ? reference(batch, inputSize, outputSize, input, weight,
                                  storageDtype)
                      : std::vector<double>{};
  const auto baselineKernel = metal::emitLinearBaseline(
      batch, inputSize, outputSize, 128, "decode_gemv_baseline", type, 1,
      type);
  std::string error;
  auto baseline = prepare(runtime, baselineKernel, input, weight,
                          batch * outputSize, expected, error, storageDtype);
  if (!baseline) {
    result.errorMessage = "Decode GEMV baseline failed: " + error;
    return result;
  }
  if (!useCpuReference) {
    const auto output = baseline->readOutputs().front();
    expected.assign(output.begin(), output.end());
    log << "Decode GEMV large-shape validation: template GPU reference selected; "
           "end-to-end PyTorch validation remains required\n";
  }
  const auto warmup = benchmark::warmup(*baseline, kWarmup);
  if (!warmup.empty()) {
    result.errorMessage = "Decode GEMV baseline warmup failed: " + warmup;
    return result;
  }
  log << "Decode GEMV shape M=" << batch << ", K=" << inputSize
      << ", N=" << outputSize
      << ", dtype=" << dtypeName(storageDtype)
      << ", samples=" << sampleCount
      << " | baseline validation/warmup: PASS\n";

  const std::vector<metal::DecodeGEMVConfig> candidates = prefill
      ? std::vector<metal::DecodeGEMVConfig>{{64,1,8}, {256,1,16}}
      : std::vector<metal::DecodeGEMVConfig>{
            {32,1}, {64,1}, {128,1}, {32,4}, {64,4}, {128,4},
            {32,1,0,true}, {32,4,0,true}};
  double bestSpeedup = 1.0;
  for (const auto candidateConfig : candidates) {
    log << "  candidate threads=" << candidateConfig.threads
        << ", vector_width=" << candidateConfig.vectorWidth
        << ", tile=" << candidateConfig.tileSize
        << ", simdgroup="
        << (candidateConfig.simdgroupReduction ? "true" : "false");
    const auto hardware = runtime.hardwareInfo();
    if (candidateConfig.threads > hardware.maxThreadsPerThreadgroup ||
        candidateConfig.threads * sizeof(float) * (prefill ? 2 : 1) >
            hardware.maxThreadgroupMemoryLength ||
        input.size() > hardware.maxBufferLength /
                           dtypeStorageBytes(storageDtype) ||
        weight.size() > hardware.maxBufferLength /
                            dtypeStorageBytes(storageDtype) ||
        inputSize % candidateConfig.vectorWidth != 0) {
      log << " | Hardware Filter: FAIL\n";
      continue;
    }
    metal::GeneratedKernel kernel;
    try {
      if (prefill) {
        kernel = metal::emitTiledGEMM(batch,inputSize,outputSize,
            candidateConfig.tileSize,"prefill_gemm_"+
            std::to_string(candidateConfig.tileSize),type,type);
      } else kernel = metal::emitDecodeGEMV(
          batch, inputSize, outputSize, candidateConfig,
          "decode_gemv_" + std::to_string(candidateConfig.threads) + "_v" +
              std::to_string(candidateConfig.vectorWidth) +
              (candidateConfig.simdgroupReduction ? "_simd" : ""), type,
              type);
    } catch (const std::exception &exception) {
      log << " | Emit: FAIL: " << exception.what() << '\n';
      continue;
    }
    auto candidate = prepare(runtime, kernel, input, weight,
                             batch * outputSize, expected, error,
                             storageDtype);
    if (!candidate) {
      log << " | Compile/interface/numerical: FAIL: " << error << '\n';
      continue;
    }
    const auto candidateWarmup = benchmark::warmup(*candidate, kWarmup);
    if (!candidateWarmup.empty()) {
      log << " | Warmup: FAIL: " << candidateWarmup << '\n';
      continue;
    }
    const auto first = benchmark::measurePair(*baseline, *candidate, sampleCount);
    if (!first.passed) {
      log << " | Benchmark: FAIL: " << first.errorMessage << '\n';
      continue;
    }
    log << " | validation: PASS, speedup=" << first.speedup << 'x';
    if (first.speedup < kMinimumSpeedup) {
      log << " -> fallback\n";
      continue;
    }
    const auto confirmation =
        benchmark::measurePair(*baseline, *candidate, sampleCount);
    if (!confirmation.passed) {
      log << ", confirmation: FAIL: " << confirmation.errorMessage << '\n';
      continue;
    }
    const auto conservative = std::min(first.speedup, confirmation.speedup);
    log << ", confirmation=" << confirmation.speedup << 'x';
    if (conservative < kMinimumSpeedup) {
      log << " -> fallback\n";
      continue;
    }
    log << " -> admitted\n";
    if (conservative > bestSpeedup) {
      bestSpeedup = conservative;
      result.useBaseline = false;
      result.config = candidateConfig;
      result.conservativeSpeedup = conservative;
    }
  }
  result.success = true;
  if (result.useBaseline) {
    result.config = {128, 1};
    log << "Decode GEMV selected: generic Linear fallback\n";
  } else {
    log << "Decode GEMV selected: threads=" << result.config.threads
        << ", vector_width=" << result.config.vectorWidth
        << ", tile=" << result.config.tileSize
        << ", simdgroup="
        << (result.config.simdgroupReduction ? "true" : "false")
        << ", conservative speedup=" << result.conservativeSpeedup << "x\n";
  }
  return result;
}

} // namespace tensor::planner
