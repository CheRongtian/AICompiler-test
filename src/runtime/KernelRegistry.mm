#include "runtime/KernelRegistry.hpp"

#import <Foundation/Foundation.h>

#include <filesystem>
#include <algorithm>
#include <exception>
#include <ostream>
#include <utility>

namespace tensor::runtime {
namespace {
NSString *ns(const std::string &s) {
  return [[NSString alloc] initWithBytes:s.data() length:s.size()
                               encoding:NSUTF8StringEncoding];
}
std::string str(NSString *s) {
  return s.UTF8String ? std::string(s.UTF8String) : std::string{};
}
}

std::string writeAdmittedKernel(
    const metal::MetalRuntime &runtime, const llm::KernelContract &contract,
    const llm::GeneratedKernelResponse &response, const std::string &path) {
  @autoreleasepool {
    NSDictionary *candidate = @{
      @"version": @(response.version), @"function_name": ns(response.functionName),
      @"workgroup_size": @(response.workgroupSize), @"msl_source": ns(response.source)};
    NSDictionary *record = @{
      @"version": @1, @"status": @"admitted",
      @"contract": ns(llm::serializeKernelContract(runtime, contract)),
      @"candidate": candidate};
    NSError *error = nil;
    NSData *data = [NSJSONSerialization dataWithJSONObject:record
        options:NSJSONWritingPrettyPrinted error:&error];
    if (!data || ![data writeToFile:ns(path) options:NSDataWritingAtomic error:&error])
      return error ? str(error.localizedDescription) : "Unable to write admitted kernel.";
  }
  return {};
}

void KernelRegistry::load(const metal::MetalRuntime &runtime,
                           const std::string &directory, std::ostream &log) {
  entries_.clear();
  usage_.clear();
  if (directory.empty()) {
    log << "Kernel registry: template-only (no library supplied)\n";
    return;
  }
  if (!std::filesystem::is_directory(directory)) {
    log << "Kernel registry: no library directory; template\n";
    return;
  }
  std::vector<std::string> availablePatterns;
  for(const auto &entry:std::filesystem::directory_iterator(directory))
    if(entry.is_directory() && entry.path().filename().string().rfind("decoder_",0)==0)
      availablePatterns.push_back(entry.path().filename().string());
  std::sort(availablePatterns.begin(),availablePatterns.end());
  for (const auto &pattern : availablePatterns) {
    const auto path = std::filesystem::path(directory) / pattern / "admitted.json";
    if (!std::filesystem::exists(path)) {
      log << "Kernel registry " << pattern << ": no admitted artifact; template\n";
      continue;
    }
    @autoreleasepool {
      NSError *error = nil;
      NSData *data = [NSData dataWithContentsOfFile:ns(path.string()) options:0 error:&error];
      id decoded = data ? [NSJSONSerialization JSONObjectWithData:data options:0 error:&error] : nil;
      if (![decoded isKindOfClass:NSDictionary.class]) {
        log << "Kernel registry " << pattern << ": invalid artifact; template\n";
        continue;
      }
      NSDictionary *record = decoded;
      llm::KernelContract contract;
      try { contract=llm::makeKernelContract(pattern); }
      catch(const std::exception &exception) {
        log << "Kernel registry " << pattern << ": " << exception.what() << "; template\n";
        continue;
      }
      if(contract.storageType==metal::ElementType::BFloat16 &&
         !runtime.hardwareInfo().supportsBFloat16) continue;
      if (![record[@"version"] isEqual:@1] ||
          ![record[@"status"] isEqual:@"admitted"] ||
          ![record[@"contract"] isEqual:ns(llm::serializeKernelContract(runtime, contract))] ||
          ![record[@"candidate"] isKindOfClass:NSDictionary.class]) {
        log << "Kernel registry " << pattern
            << ": admission record or current contract/device mismatch; template\n";
        continue;
      }
      NSData *candidate = [NSJSONSerialization dataWithJSONObject:record[@"candidate"]
          options:0 error:&error];
      NSString *json = candidate ? [[NSString alloc] initWithData:candidate
          encoding:NSUTF8StringEncoding] : nil;
      auto parsed = llm::parseGeneratedKernelResponse(str(json));
      if (!parsed.response || parsed.response->functionName != contract.functionName) {
        log << "Kernel registry " << pattern << ": invalid candidate; template\n";
        continue;
      }
      entries_.emplace(pattern, std::move(*parsed.response));
      log << "Kernel registry " << pattern << ": admitted artifact loaded\n";
    }
  }
}

const llm::GeneratedKernelResponse *KernelRegistry::find(const std::string &pattern) const {
  auto found = entries_.find(pattern);
  return found == entries_.end() ? nullptr : &found->second;
}
std::size_t KernelRegistry::track(KernelUsage usage) {
  usage_.push_back(std::move(usage));
  return usage_.size()-1;
}
void KernelRegistry::completed(const std::vector<std::size_t> &uses) {
  for (auto index : uses) ++usage_.at(index).completedCalls;
}
void KernelRegistry::resetUsage() noexcept {
  for (auto &use : usage_) use.completedCalls = 0;
}
void KernelRegistry::report(std::ostream &log) const {
  std::size_t generated = 0, templates = 0;
  log << "Decoder kernel runtime usage audit:\n";
  for (const auto &use : usage_) {
    const auto dispatches = use.completedCalls * use.dispatchesPerCall;
    (use.implementation == "generated" ? generated : templates) += dispatches;
    log << "  " << use.node << " | " << use.pattern << " | "
        << use.implementation << " | " << use.functions
        << " | completed_calls=" << use.completedCalls
        << " | completed_dispatches=" << dispatches << '\n';
  }
  for (const auto &entry : entries_) {
    bool used = false;
    for (const auto &use : usage_)
      used |= use.pattern == entry.first && use.implementation == "generated" &&
              use.completedCalls != 0;
    log << "  Registry entry " << entry.first << ": "
        << (used ? "executed" : "loaded but not executed") << '\n';
  }
  log << "Audited replacement-region dispatches: generated=" << generated
      << ", template=" << templates << '\n';
}
} // namespace tensor::runtime
