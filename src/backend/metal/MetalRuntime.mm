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

} // namespace

std::string checkFloatBufferInterface(const ComputePipelineResult &pipeline,
                                       std::size_t inputCount) {
  if (!pipeline.reflectionAvailable || pipeline.bindings.size() != inputCount + 1) {
    return "Reflection must contain exactly the expected input and output buffers.";
  }
  std::vector<bool> seen(inputCount + 1, false);
  for (const auto &binding : pipeline.bindings) {
    if (!binding.isBuffer || !binding.isFloat32 || binding.index >= seen.size() ||
        seen[binding.index]) {
      return "Expected unique fp32 buffer bindings at consecutive indices.";
    }
    seen[binding.index] = true;
    if ((binding.index < inputCount && !binding.readOnly) ||
        (binding.index == inputCount && !binding.writable)) {
      return "Expected read-only inputs and a writable output buffer.";
    }
  }
  return {};
}

class MetalBuffer::Impl {
public:
  id<MTLBuffer> buffer = nil;
  std::size_t count = 0;
};

MetalBuffer::MetalBuffer(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
MetalBuffer::~MetalBuffer() = default;
std::size_t MetalBuffer::elementCount() const { return impl_->count; }
std::vector<float> MetalBuffer::read() const {
  std::vector<float> result(impl_->count);
  std::memcpy(result.data(), impl_->buffer.contents, result.size() * sizeof(float));
  return result;
}

class PreparedExecution::Impl {
public:
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> pipeline = nil;
  std::vector<id<MTLBuffer>> inputs;
  id<MTLBuffer> output = nil;
  std::size_t outputElementCount = 0;
  DispatchSize dispatch;
  std::vector<std::uint32_t> constants;

  ExecutionResult execute(bool readback) const {
    ExecutionResult result;
    @autoreleasepool {
      if (readback) {
        std::fill_n(static_cast<float *>(output.contents), outputElementCount,
                    std::numeric_limits<float>::quiet_NaN());
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
        std::memcpy(result.output.data(), output.contents,
                    outputElementCount * sizeof(float));
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
        }
        result.bindings.push_back(info);
      }
      pipeline_ = pipeline;
    }

    return result;
  }

  BufferResult createBuffer(std::size_t count, const float *initialData) const {
    BufferResult result;
    @autoreleasepool {
      if (!isAvailable()) {
        result.errorMessage = initializationError_;
        return result;
      }
      if (count == 0 || count > device_.maxBufferLength / sizeof(float)) {
        result.errorMessage = "Buffer size is zero or exceeds the device limit.";
        return result;
      }
      auto storage = std::make_unique<MetalBuffer::Impl>();
      storage->count = count;
      if (initialData) {
        storage->buffer = [device_ newBufferWithBytes:initialData
                                              length:count * sizeof(float)
                                             options:MTLResourceStorageModeShared];
      } else {
        storage->buffer = [device_ newBufferWithLength:count * sizeof(float)
                                              options:MTLResourceStorageModeShared];
      }
      if (storage->buffer == nil) {
        result.errorMessage = "Failed to allocate a shared Metal buffer.";
        return result;
      }
      if (!initialData) {
        std::fill_n(static_cast<float *>(storage->buffer.contents), count,
                    std::numeric_limits<float>::quiet_NaN());
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
                              const std::vector<std::uint32_t> &constants) const {
    std::vector<BufferHandle> buffers;
    for (const auto &input : inputs) {
      if (!input.data) return {nullptr, "Host input data is missing."};
      auto buffer = createBuffer(input.elementCount, input.data);
      if (!buffer.buffer) return {nullptr, buffer.errorMessage};
      buffers.push_back(std::move(buffer.buffer));
    }
    auto output = createBuffer(outputElementCount, nullptr);
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
                  const std::vector<std::uint32_t> &constants) const {
  auto prepared = prepare(inputs, outputElementCount, dispatch, constants);
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
                      const std::vector<std::uint32_t> &constants) const {
  return impl_->prepare(inputs, outputElementCount, dispatch, constants);
}

BufferResult MetalRuntime::createBuffer(std::size_t elementCount,
                                        const float *initialData) const {
  return impl_->createBuffer(elementCount, initialData);
}

PreparationResult MetalRuntime::prepareBuffers(const std::vector<BufferHandle> &inputs,
                                               const BufferHandle &output,
                                               const DispatchSize &dispatch,
                                               const std::vector<std::uint32_t> &constants) const {
  return impl_->prepareBuffers(inputs, output, dispatch, constants);
}

} // namespace tensor::metal
