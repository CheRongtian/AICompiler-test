#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "backend/metal/MetalRuntime.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

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
      id<MTLComputePipelineState> pipeline =
          [device_ newComputePipelineStateWithFunction:function
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
      pipeline_ = pipeline;
    }

    return result;
  }

  [[nodiscard]] ExecutionResult
  run(const std::vector<FloatBufferView> &inputs,
      std::size_t outputElementCount, const DispatchSize &dispatch,
      const std::vector<std::uint32_t> &constants) const {
    ExecutionResult result;

    @autoreleasepool {
      if (!isAvailable()) {
        result.errorMessage = initializationError_;
        return result;
      }
      if (pipeline_ == nil) {
        result.errorMessage = "Create a compute pipeline before execution.";
        return result;
      }
      // Metal provides 31 buffer argument slots, including output and constants.
      const std::size_t bindingCount = inputs.size() + 1 + !constants.empty();
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
      const std::size_t maxElements = device_.maxBufferLength / sizeof(float);
      if (outputElementCount == 0 || outputElementCount > maxElements) {
        result.errorMessage = "Output buffer size is zero or exceeds the device limit.";
        return result;
      }
      for (const auto &input : inputs) {
        if (input.data == nullptr || input.elementCount == 0 ||
            input.elementCount > maxElements) {
          result.errorMessage = "Input buffer is empty or exceeds the device limit.";
          return result;
        }
      }

      std::vector<id<MTLBuffer>> inputBuffers;
      inputBuffers.reserve(inputs.size());
      for (const auto &input : inputs) {
        id<MTLBuffer> buffer =
            [device_ newBufferWithBytes:input.data
                                length:input.elementCount * sizeof(float)
                               options:MTLResourceStorageModeShared];
        if (buffer == nil) {
          result.errorMessage = "Failed to allocate a shared input buffer.";
          return result;
        }
        inputBuffers.push_back(buffer);
      }

      const std::size_t byteCount = outputElementCount * sizeof(float);
      id<MTLBuffer> outputBuffer =
          [device_ newBufferWithLength:byteCount
                              options:MTLResourceStorageModeShared];
      if (outputBuffer == nil) {
        result.errorMessage = "Failed to allocate a shared output buffer.";
        return result;
      }

      // Unwritten output elements must fail the numerical comparison.
      std::fill_n(static_cast<float *>(outputBuffer.contents), outputElementCount,
                  std::numeric_limits<float>::quiet_NaN());

      id<MTLCommandBuffer> commandBuffer = [commandQueue_ commandBuffer];
      if (commandBuffer == nil) {
        result.errorMessage = "Failed to create a Metal command buffer.";
        return result;
      }
      id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
      if (encoder == nil) {
        result.errorMessage = "Failed to create a Metal compute command encoder.";
        return result;
      }

      [encoder setComputePipelineState:pipeline_];
      for (std::size_t index = 0; index < inputBuffers.size(); ++index) {
        [encoder setBuffer:inputBuffers[index] offset:0 atIndex:index];
      }
      [encoder setBuffer:outputBuffer offset:0 atIndex:inputs.size()];
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

      // Metal timestamps are in seconds and are read only after completion.
      const double gpuStart = commandBuffer.GPUStartTime;
      const double gpuEnd = commandBuffer.GPUEndTime;
      if (std::isfinite(gpuStart) && std::isfinite(gpuEnd) &&
          gpuStart > 0.0 && gpuEnd > gpuStart) {
        result.gpuExecutionTimeUs = (gpuEnd - gpuStart) * 1e6;
      }

      // Shared storage is CPU-readable after GPU completion.
      result.output.resize(outputElementCount);
      std::memcpy(result.output.data(), outputBuffer.contents, byteCount);
      result.executionPassed = true;
    }

    return result;
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

ComputePipelineResult
MetalRuntime::createComputePipeline(const std::string &source,
                                    const std::string &functionName) {
  return impl_->createComputePipeline(source, functionName);
}

ExecutionResult
MetalRuntime::run(const std::vector<FloatBufferView> &inputs,
                  std::size_t outputElementCount, const DispatchSize &dispatch,
                  const std::vector<std::uint32_t> &constants) const {
  return impl_->run(inputs, outputElementCount, dispatch, constants);
}

} // namespace tensor::metal
