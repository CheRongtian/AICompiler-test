#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "backend/metal/MetalRuntime.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

namespace tensor::metal {
namespace {

std::string toStdString(NSString *value) {
  if (value == nil) {
    return {};
  }

  const char *utf8 = value.UTF8String;
  return utf8 == nullptr ? std::string{} : std::string{utf8};
}

std::string errorMessage(NSError *error, const std::string &fallback) {
  if (error == nil) {
    return fallback;
  }

  const std::string message = toStdString(error.localizedDescription);
  return message.empty() ? fallback : message;
}

std::size_t elementSize(ElementType type) {
  return type == ElementType::Float16 ? sizeof(std::uint16_t) : sizeof(std::uint32_t);
}

std::uint16_t floatToHalf(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t sign = (bits >> 16) & 0x8000u;
  const std::uint32_t exponent = (bits >> 23) & 0xffu;
  const std::uint32_t mantissa = bits & 0x7fffffu;
  if (exponent == 0xffu) return static_cast<std::uint16_t>(sign | (mantissa ? 0x7e00u : 0x7c00u));
  const int halfExponent = static_cast<int>(exponent) - 127 + 15;
  if (halfExponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
  if (halfExponent <= 0) {
    if (halfExponent < -10) return static_cast<std::uint16_t>(sign);
    std::uint32_t normalized = mantissa | 0x800000u;
    const int shift = 14 - halfExponent;
    std::uint32_t rounded = normalized >> shift;
    if ((normalized >> (shift - 1)) & 1u) ++rounded;
    return static_cast<std::uint16_t>(sign | rounded);
  }
  std::uint32_t roundedMantissa = mantissa + 0x1000u;
  std::uint32_t roundedExponent = static_cast<std::uint32_t>(halfExponent);
  if (roundedMantissa & 0x800000u) {
    roundedMantissa = 0;
    if (++roundedExponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
  }
  return static_cast<std::uint16_t>(sign | (roundedExponent << 10) |
                                    (roundedMantissa >> 13));
}

float halfToFloat(std::uint16_t value) {
  const std::uint32_t sign = static_cast<std::uint32_t>(value & 0x8000u) << 16;
  std::uint32_t exponent = (value >> 10) & 0x1fu;
  std::uint32_t mantissa = value & 0x3ffu;
  std::uint32_t bits;
  if (exponent == 0) {
    if (mantissa == 0) bits = sign;
    else {
      int shift = 0;
      while ((mantissa & 0x400u) == 0) { mantissa <<= 1; ++shift; }
      mantissa &= 0x3ffu;
      bits = sign | static_cast<std::uint32_t>(127 - 14 - shift) << 23 | mantissa << 13;
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000u | mantissa << 13;
  } else {
    bits = sign | (exponent + 127 - 15) << 23 | mantissa << 13;
  }
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

} // namespace

std::string checkFloatBufferInterface(const ComputePipelineResult &pipeline,
                                       std::size_t inputCount) {
  return checkBufferInterface(pipeline,
                              std::vector<ElementType>(inputCount, ElementType::Float32),
                              ElementType::Float32);
}

std::string checkBufferInterface(const ComputePipelineResult &pipeline,
                                 const std::vector<ElementType> &inputs,
                                 ElementType output) {
  if (!pipeline.reflectionAvailable || pipeline.bindings.size() != inputs.size() + 1) {
    return "Reflection must contain exactly the expected input and output buffers.";
  }
  std::vector<bool> seen(inputs.size() + 1, false);
  for (const auto &binding : pipeline.bindings) {
    if (!binding.isBuffer || binding.index >= seen.size() || seen[binding.index]) {
      return "Expected unique buffer bindings at consecutive indices.";
    }
    seen[binding.index] = true;
    const auto expected = binding.index < inputs.size() ? inputs[binding.index] : output;
    const bool typeMatches = expected == ElementType::Float32 ? binding.isFloat32
                           : expected == ElementType::Float16 ? binding.isFloat16
                                                              : binding.isInt32;
    if (!typeMatches) return "Reflected buffer element type disagrees with TensorIR.";
    if ((binding.index < inputs.size() && !binding.readOnly) ||
        (binding.index == inputs.size() && !binding.writable)) {
      return "Expected read-only inputs and a writable output buffer.";
    }
  }
  return {};
}

class MetalBuffer::Impl {
public:
  id<MTLBuffer> buffer = nil;
  std::size_t count = 0;
  ElementType type = ElementType::Float32;
};

MetalBuffer::MetalBuffer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
MetalBuffer::~MetalBuffer() = default;
std::size_t MetalBuffer::elementCount() const { return impl_->count; }
ElementType MetalBuffer::elementType() const { return impl_->type; }
std::vector<float> MetalBuffer::read() const {
  std::vector<float> result(impl_->count);
  if (impl_->type == ElementType::Float32) {
    std::memcpy(result.data(), impl_->buffer.contents, result.size() * sizeof(float));
  } else if (impl_->type == ElementType::Float16) {
    const auto *source = static_cast<const std::uint16_t *>(impl_->buffer.contents);
    for (std::size_t i = 0; i < result.size(); ++i) result[i] = halfToFloat(source[i]);
  } else {
    const auto *source = static_cast<const std::int32_t *>(impl_->buffer.contents);
    for (std::size_t i = 0; i < result.size(); ++i) result[i] = static_cast<float>(source[i]);
  }
  return result;
}

class PreparedExecution::Impl {
public:
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> pipeline = nil;
  std::vector<id<MTLBuffer>> inputs;
  id<MTLBuffer> output = nil;
  std::size_t outputElementCount = 0;
  ElementType outputType = ElementType::Float32;
  DispatchSize dispatch;
  std::vector<std::uint32_t> constants;

  ExecutionResult execute(bool readback) const {
    ExecutionResult result;
    @autoreleasepool {
      if (readback) {
        if (outputType == ElementType::Float32) {
          std::fill_n(static_cast<float *>(output.contents), outputElementCount,
                      std::numeric_limits<float>::quiet_NaN());
        } else if (outputType == ElementType::Float16) {
          std::fill_n(static_cast<std::uint16_t *>(output.contents), outputElementCount, 0x7e00u);
        } else {
          std::fill_n(static_cast<std::int32_t *>(output.contents), outputElementCount, 0);
        }
      }
      id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
      if (commandBuffer == nil) {
        result.errorMessage = "Failed to create a Metal command buffer.";
        return result;
      }
      id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
      if (encoder == nil) {
        result.errorMessage = "Failed to create a Metal compute command encoder.";
        return result;
      }
      [encoder setComputePipelineState:pipeline];
      for (std::size_t index = 0; index < inputs.size(); ++index) {
        [encoder setBuffer:inputs[index] offset:0 atIndex:index];
      }
      [encoder setBuffer:output offset:0 atIndex:inputs.size()];
      if (!constants.empty()) {
        [encoder setBytes:constants.data()
                  length:constants.size() * sizeof(std::uint32_t)
                 atIndex:inputs.size() + 1];
      }
      [encoder dispatchThreadgroups:MTLSizeMake(dispatch.threadgroupCount, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(dispatch.threadsPerThreadgroup, 1, 1)];
      [encoder endEncoding];

      const auto cpuStart = std::chrono::steady_clock::now();
      [commandBuffer commit];
      [commandBuffer waitUntilCompleted];
      const auto cpuEnd = std::chrono::steady_clock::now();
      result.cpuSubmitToCompletionTimeUs =
          std::chrono::duration<double, std::micro>(cpuEnd - cpuStart).count();
      if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
        result.errorMessage = errorMessage(
            commandBuffer.error, "Metal command buffer did not complete successfully.");
        return result;
      }
      const double gpuStart = commandBuffer.GPUStartTime;
      const double gpuEnd = commandBuffer.GPUEndTime;
      if (std::isfinite(gpuStart) && std::isfinite(gpuEnd) &&
          gpuStart > 0.0 && gpuEnd > gpuStart) {
        result.gpuExecutionTimeUs = (gpuEnd - gpuStart) * 1e6;
      }
      if (readback) {
        result.output.resize(outputElementCount);
        if (outputType == ElementType::Float32) {
          std::memcpy(result.output.data(), output.contents,
                      outputElementCount * sizeof(float));
        } else if (outputType == ElementType::Float16) {
          const auto *source = static_cast<const std::uint16_t *>(output.contents);
          for (std::size_t i = 0; i < outputElementCount; ++i) result.output[i] = halfToFloat(source[i]);
        } else {
          const auto *source = static_cast<const std::int32_t *>(output.contents);
          for (std::size_t i = 0; i < outputElementCount; ++i) result.output[i] = static_cast<float>(source[i]);
        }
      }
      result.executionPassed = true;
    }
    return result;
  }
};

PreparedExecution::PreparedExecution(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
PreparedExecution::~PreparedExecution() = default;
ExecutionResult PreparedExecution::run() const { return impl_->execute(true); }
ExecutionResult PreparedExecution::execute() const { return impl_->execute(false); }

class PreparedSequence::Impl {
public:
  struct Step {
    id<MTLComputePipelineState> pipeline = nil;
    std::vector<id<MTLBuffer>> inputs;
    id<MTLBuffer> output = nil;
    DispatchSize dispatch;
    std::vector<std::uint32_t> constants;
  };

  id<MTLCommandQueue> queue = nil;
  std::vector<Step> steps;

  ExecutionResult execute() const {
    ExecutionResult result;
    @autoreleasepool {
      id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
      if (commandBuffer == nil) {
        result.errorMessage = "Failed to create a Metal command buffer for a region.";
        return result;
      }
      for (const auto &step : steps) {
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (encoder == nil) {
          result.errorMessage = "Failed to create a Metal encoder for a region step.";
          return result;
        }
        [encoder setComputePipelineState:step.pipeline];
        for (std::size_t index = 0; index < step.inputs.size(); ++index) {
          [encoder setBuffer:step.inputs[index] offset:0 atIndex:index];
        }
        [encoder setBuffer:step.output offset:0 atIndex:step.inputs.size()];
        if (!step.constants.empty()) {
          [encoder setBytes:step.constants.data()
                    length:step.constants.size() * sizeof(std::uint32_t)
                   atIndex:step.inputs.size() + 1];
        }
        [encoder dispatchThreadgroups:MTLSizeMake(step.dispatch.threadgroupCount, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(step.dispatch.threadsPerThreadgroup, 1, 1)];
        [encoder endEncoding];
      }
      const auto cpuStart = std::chrono::steady_clock::now();
      [commandBuffer commit];
      [commandBuffer waitUntilCompleted];
      const auto cpuEnd = std::chrono::steady_clock::now();
      result.cpuSubmitToCompletionTimeUs =
          std::chrono::duration<double, std::micro>(cpuEnd - cpuStart).count();
      if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
        result.errorMessage = errorMessage(
            commandBuffer.error, "Metal region command buffer did not complete successfully.");
        return result;
      }
      const double gpuStart = commandBuffer.GPUStartTime;
      const double gpuEnd = commandBuffer.GPUEndTime;
      if (std::isfinite(gpuStart) && std::isfinite(gpuEnd) && gpuStart > 0.0 && gpuEnd > gpuStart) {
        result.gpuExecutionTimeUs = (gpuEnd - gpuStart) * 1e6;
      }
      result.executionPassed = true;
    }
    return result;
  }
};

PreparedSequence::PreparedSequence(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
PreparedSequence::~PreparedSequence() = default;
ExecutionResult PreparedSequence::execute() const { return impl_->execute(); }

class MetalRuntime::Impl {
public:
  Impl() {
    @autoreleasepool {
      device_ = MTLCreateSystemDefaultDevice();
      if (device_ == nil) {
        initializationError_ = "No Metal-capable default device was found.";
        return;
      }

      commandQueue_ = [device_ newCommandQueue];
      if (commandQueue_ == nil) {
        initializationError_ = "Failed to create a Metal command queue.";
      }
    }
  }

  [[nodiscard]] bool isAvailable() const noexcept {
    return device_ != nil && commandQueue_ != nil;
  }

  [[nodiscard]] std::string deviceName() const {
    @autoreleasepool {
      return device_ == nil ? std::string{} : toStdString(device_.name);
    }
  }

  [[nodiscard]] const std::string &initializationError() const noexcept {
    return initializationError_;
  }

  HardwareInfo hardwareInfo() const {
    if (device_ == nil) {
      return {};
    }
    return {device_.maxThreadsPerThreadgroup.width,
            device_.maxThreadgroupMemoryLength, device_.maxBufferLength};
  }

  [[nodiscard]] ComputePipelineResult
  createComputePipeline(const std::string &source,
                        const std::string &functionName) {
    ComputePipelineResult result;

    @autoreleasepool {
      pipeline_ = nil;
      if (!isAvailable()) {
        result.errorMessage = initializationError_.empty()
                                  ? "Metal runtime is unavailable."
                                  : initializationError_;
        return result;
      }

      NSString *sourceString =
          [[NSString alloc] initWithBytes:source.data()
                                  length:source.size()
                                encoding:NSUTF8StringEncoding];
      if (sourceString == nil) {
        result.errorMessage = "MSL source is not valid UTF-8.";
        return result;
      }

      NSError *libraryError = nil;
      id<MTLLibrary> library =
          [device_ newLibraryWithSource:sourceString
                                options:nil
                                  error:&libraryError];
      if (library == nil) {
        result.errorMessage = errorMessage(
            libraryError, "Metal failed to compile the MSL source.");
        return result;
      }
      result.libraryCompilePassed = true;

      NSString *functionNameString =
          [[NSString alloc] initWithBytes:functionName.data()
                                  length:functionName.size()
                                encoding:NSUTF8StringEncoding];
      if (functionNameString == nil) {
        result.errorMessage = "Kernel function name is not valid UTF-8.";
        return result;
      }

      id<MTLFunction> function =
          [library newFunctionWithName:functionNameString];
      if (function == nil) {
        result.errorMessage =
            "Metal kernel function '" + functionName + "' was not found.";
        return result;
      }
      result.kernelLookupPassed = true;

      NSError *pipelineError = nil;
      MTLComputePipelineReflection *reflection = nil;
      id<MTLComputePipelineState> pipeline =
          [device_ newComputePipelineStateWithFunction:function
                                               options:MTLPipelineOptionBindingInfo |
                                                       MTLPipelineOptionBufferTypeInfo
                                            reflection:&reflection
                                                 error:&pipelineError];
      if (pipeline == nil) {
        result.errorMessage = errorMessage(
            pipelineError, "Metal failed to create the compute pipeline.");
        return result;
      }

      result.pipelineCreationPassed = true;
      result.threadExecutionWidth = pipeline.threadExecutionWidth;
      result.maxTotalThreadsPerThreadgroup =
          pipeline.maxTotalThreadsPerThreadgroup;
      result.staticThreadgroupMemoryLength = pipeline.staticThreadgroupMemoryLength;
      result.reflectionAvailable = reflection != nil;
      for (id<MTLBinding> binding in reflection.bindings) {
        if (!binding.isArgument || !binding.isUsed) {
          continue;
        }
        PipelineBinding info;
        info.index = binding.index;
        info.isBuffer = binding.type == MTLBindingTypeBuffer;
        info.readOnly = binding.access == MTLBindingAccessReadOnly;
        info.writable = binding.access == MTLBindingAccessReadWrite ||
                        binding.access == MTLBindingAccessWriteOnly;
        if (info.isBuffer) {
          id<MTLBufferBinding> buffer = (id<MTLBufferBinding>)binding;
          info.isFloat32 = buffer.bufferDataType == MTLDataTypeFloat;
          info.isFloat16 = buffer.bufferDataType == MTLDataTypeHalf;
          info.isInt32 = buffer.bufferDataType == MTLDataTypeInt;
        }
        result.bindings.push_back(info);
      }
      pipeline_ = pipeline;
    }

    return result;
  }

  BufferResult createBuffer(std::size_t count, const float *initialData,
                            ElementType type) const {
    BufferResult result;
    @autoreleasepool {
      if (!isAvailable()) {
        result.errorMessage = initializationError_;
        return result;
      }
      if (count == 0 || count > device_.maxBufferLength / elementSize(type)) {
        result.errorMessage = "Buffer size is zero or exceeds the device limit.";
        return result;
      }
      const auto byteCount = count * elementSize(type);
      auto storage = std::make_unique<MetalBuffer::Impl>();
      storage->count = count;
      storage->type = type;
      storage->buffer = [device_ newBufferWithLength:byteCount
                                            options:MTLResourceStorageModeShared];
      if (storage->buffer == nil) {
        result.errorMessage = "Failed to allocate a shared Metal buffer.";
        return result;
      }
      if (initialData && type == ElementType::Float32) {
        std::memcpy(storage->buffer.contents, initialData, byteCount);
      } else if (initialData && type == ElementType::Float16) {
        auto *destination = static_cast<std::uint16_t *>(storage->buffer.contents);
        for (std::size_t i = 0; i < count; ++i) destination[i] = floatToHalf(initialData[i]);
      } else if (initialData) {
        auto *destination = static_cast<std::int32_t *>(storage->buffer.contents);
        for (std::size_t i = 0; i < count; ++i) destination[i] = static_cast<std::int32_t>(initialData[i]);
      } else if (type == ElementType::Float32) {
        std::fill_n(static_cast<float *>(storage->buffer.contents), count,
                    std::numeric_limits<float>::quiet_NaN());
      } else if (type == ElementType::Float16) {
        std::fill_n(static_cast<std::uint16_t *>(storage->buffer.contents), count, 0x7e00u);
      } else {
        std::fill_n(static_cast<std::int32_t *>(storage->buffer.contents), count, 0);
      }
      result.buffer = BufferHandle(new MetalBuffer(std::move(storage)));
    }
    return result;
  }

  PreparationResult prepareBuffers(const std::vector<BufferHandle> &inputs,
                                     const BufferHandle &output,
                                     const DispatchSize &dispatch,
                                     const std::vector<std::uint32_t> &constants) const {
    PreparationResult result;
    @autoreleasepool {
      if (!isAvailable()) {
        result.errorMessage = initializationError_;
        return result;
      }
      if (pipeline_ == nil) {
        result.errorMessage = "Create a compute pipeline before execution.";
        return result;
      }
      const auto bindingCount = inputs.size() + 1 + !constants.empty();
      if (inputs.empty() || bindingCount > 31 || constants.size() > 1024) {
        result.errorMessage = "Invalid buffer binding count or oversized inline constants.";
        return result;
      }
      if (dispatch.threadgroupCount == 0 ||
          dispatch.threadgroupCount > std::numeric_limits<std::uint32_t>::max() ||
          dispatch.threadsPerThreadgroup == 0 ||
          dispatch.threadsPerThreadgroup > pipeline_.maxTotalThreadsPerThreadgroup ||
          dispatch.threadsPerThreadgroup > device_.maxThreadsPerThreadgroup.width ||
          pipeline_.staticThreadgroupMemoryLength > device_.maxThreadgroupMemoryLength) {
        result.errorMessage = "Kernel dispatch exceeds device or pipeline limits.";
        return result;
      }
      if (!output || output->impl_->buffer.device != device_) {
        result.errorMessage = "Output buffer is missing or belongs to another device.";
        return result;
      }
      auto prepared = std::make_unique<PreparedExecution::Impl>();
      prepared->queue = commandQueue_;
      prepared->pipeline = pipeline_;
      prepared->dispatch = dispatch;
      prepared->constants = constants;
      prepared->outputElementCount = output->elementCount();
      prepared->outputType = output->elementType();
      prepared->output = output->impl_->buffer;
      for (const auto &input : inputs) {
        if (!input || input->impl_->buffer.device != device_ || input == output) {
          result.errorMessage = "Input buffer is missing, on another device, or aliases output.";
          return result;
        }
        prepared->inputs.push_back(input->impl_->buffer);
      }
      result.execution = std::unique_ptr<PreparedExecution>(
          new PreparedExecution(std::move(prepared)));
    }
    return result;
  }

  PreparationResult prepare(const std::vector<FloatBufferView> &inputs,
                              std::size_t outputElementCount, const DispatchSize &dispatch,
                              const std::vector<std::uint32_t> &constants,
                              ElementType outputType) const {
    std::vector<BufferHandle> buffers;
    for (const auto &input : inputs) {
      if (!input.data) return {nullptr, "Host input data is missing."};
      auto buffer = createBuffer(input.elementCount, input.data, input.elementType);
      if (!buffer.buffer) return {nullptr, buffer.errorMessage};
      buffers.push_back(std::move(buffer.buffer));
    }
    auto output = createBuffer(outputElementCount, nullptr, outputType);
    if (!output.buffer) return {nullptr, output.errorMessage};
    return prepareBuffers(buffers, output.buffer, dispatch, constants);
  }

private:
  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> commandQueue_ = nil;
  id<MTLComputePipelineState> pipeline_ = nil;
  std::string initializationError_;
};

MetalRuntime::MetalRuntime() : impl_(std::make_unique<Impl>()) {}

MetalRuntime::~MetalRuntime() = default;

MetalRuntime::MetalRuntime(MetalRuntime &&) noexcept = default;

MetalRuntime &MetalRuntime::operator=(MetalRuntime &&) noexcept = default;

bool MetalRuntime::isAvailable() const noexcept {
  return impl_->isAvailable();
}

std::string MetalRuntime::deviceName() const {
  return impl_->deviceName();
}

std::string MetalRuntime::initializationError() const {
  return impl_->initializationError();
}

HardwareInfo MetalRuntime::hardwareInfo() const { return impl_->hardwareInfo(); }

ComputePipelineResult
MetalRuntime::createComputePipeline(const std::string &source,
                                    const std::string &functionName) {
  return impl_->createComputePipeline(source, functionName);
}

ExecutionResult
MetalRuntime::run(const std::vector<FloatBufferView> &inputs,
                  std::size_t outputElementCount, const DispatchSize &dispatch,
                  const std::vector<std::uint32_t> &constants,
                  ElementType outputType) const {
  auto prepared = prepare(inputs, outputElementCount, dispatch, constants, outputType);
  if (!prepared.execution) {
    ExecutionResult result;
    result.errorMessage = prepared.errorMessage;
    return result;
  }
  return prepared.execution->run();
}

PreparationResult
MetalRuntime::prepare(const std::vector<FloatBufferView> &inputs,
                      std::size_t outputElementCount, const DispatchSize &dispatch,
                      const std::vector<std::uint32_t> &constants,
                      ElementType outputType) const {
  return impl_->prepare(inputs, outputElementCount, dispatch, constants, outputType);
}

BufferResult MetalRuntime::createBuffer(std::size_t elementCount,
                                        const float *initialData,
                                        ElementType type) const {
  return impl_->createBuffer(elementCount, initialData, type);
}

PreparationResult MetalRuntime::prepareBuffers(const std::vector<BufferHandle> &inputs,
                                               const BufferHandle &output,
                                               const DispatchSize &dispatch,
                                               const std::vector<std::uint32_t> &constants) const {
  return impl_->prepareBuffers(inputs, output, dispatch, constants);
}

SequencePreparationResult
MetalRuntime::prepareSequence(const std::vector<const PreparedExecution *> &steps) const {
  SequencePreparationResult result;
  if (steps.empty()) {
    result.errorMessage = "A prepared region sequence cannot be empty.";
    return result;
  }
  auto sequence = std::make_unique<PreparedSequence::Impl>();
  for (const auto *execution : steps) {
    if (execution == nullptr || execution->impl_->queue == nil ||
        execution->impl_->pipeline == nil) {
      result.errorMessage = "A region contains an invalid prepared execution.";
      return result;
    }
    if (sequence->queue == nil) sequence->queue = execution->impl_->queue;
    if (sequence->queue != execution->impl_->queue) {
      result.errorMessage = "All region steps must use the same Metal command queue.";
      return result;
    }
    PreparedSequence::Impl::Step step;
    step.pipeline = execution->impl_->pipeline;
    step.inputs = execution->impl_->inputs;
    step.output = execution->impl_->output;
    step.dispatch = execution->impl_->dispatch;
    step.constants = execution->impl_->constants;
    sequence->steps.push_back(std::move(step));
  }
  result.execution = std::unique_ptr<PreparedSequence>(
      new PreparedSequence(std::move(sequence)));
  return result;
}

} // namespace tensor::metal
