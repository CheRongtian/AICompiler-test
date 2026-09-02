#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "backend/metal/MetalRuntime.hpp"

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
                        const std::string &functionName) const {
    ComputePipelineResult result;

    @autoreleasepool {
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
    }

    return result;
  }

private:
  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> commandQueue_ = nil;
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
                                    const std::string &functionName) const {
  return impl_->createComputePipeline(source, functionName);
}

} // namespace tensor::metal
