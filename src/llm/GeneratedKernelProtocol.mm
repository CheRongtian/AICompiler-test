#include "llm/GeneratedKernelProtocol.hpp"

#import <Foundation/Foundation.h>

#include <cstring>
#include <limits>
#include <utility>

namespace tensor::llm {
namespace {

std::string toString(NSString *value) {
  if (value == nil) return {};
  const char *text = value.UTF8String;
  return text == nullptr ? std::string{} : std::string(text);
}

std::string errorText(NSError *error, const char *fallback) {
  if (error == nil) return fallback;
  const auto description = toString(error.localizedDescription);
  return description.empty() ? fallback : description;
}

bool isInteger(NSNumber *value) {
  if (value == nil || ![value isKindOfClass:NSNumber.class]) return false;
  const char *type = value.objCType;
  return std::strcmp(type, @encode(BOOL)) != 0 &&
         std::strcmp(type, @encode(float)) != 0 &&
         std::strcmp(type, @encode(double)) != 0;
}

} // namespace

GeneratedKernelResponseResult
loadGeneratedKernelResponse(const std::string &path) {
  GeneratedKernelResponseResult result;
  @autoreleasepool {
    NSString *file = [[NSString alloc] initWithBytes:path.data()
                                              length:path.size()
                                            encoding:NSUTF8StringEncoding];
    if (file == nil) {
      result.errorMessage = "Generated-kernel response path is not valid UTF-8.";
      return result;
    }
    NSError *readError = nil;
    NSData *data = [NSData dataWithContentsOfFile:file
                                          options:0
                                            error:&readError];
    if (data == nil) {
      result.errorMessage = errorText(readError,
          "Unable to read the generated-kernel response.");
      return result;
    }
    NSString *json = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
    if (json == nil) return {std::nullopt, "Generated response is not UTF-8."};
    return parseGeneratedKernelResponse(toString(json));
  }
}

GeneratedKernelResponseResult parseGeneratedKernelResponse(const std::string &json) {
  GeneratedKernelResponseResult result;
  @autoreleasepool {
    NSData *data = [NSData dataWithBytes:json.data() length:json.size()];
    NSError *jsonError = nil;
    id decoded = [NSJSONSerialization JSONObjectWithData:data
                                                 options:0
                                                   error:&jsonError];
    if (decoded == nil || ![decoded isKindOfClass:NSDictionary.class]) {
      result.errorMessage = errorText(jsonError,
          "Generated-kernel response must be a JSON object.");
      return result;
    }
    NSDictionary *object = (NSDictionary *)decoded;
    NSSet *expected = [NSSet setWithArray:@[
        @"version", @"function_name", @"workgroup_size", @"msl_source"]];
    if (object.count != expected.count ||
        ![[NSSet setWithArray:object.allKeys] isEqualToSet:expected]) {
      result.errorMessage =
          "Generated-kernel response contains missing or unknown fields.";
      return result;
    }

    NSNumber *version = object[@"version"];
    NSNumber *workgroup = object[@"workgroup_size"];
    NSString *function = object[@"function_name"];
    NSString *source = object[@"msl_source"];
    if (!isInteger(version) || version.longLongValue != 1) {
      result.errorMessage = "Generated-kernel response must use version 1.";
      return result;
    }
    if (!isInteger(workgroup) || workgroup.longLongValue <= 0 ||
        workgroup.unsignedLongLongValue >
            std::numeric_limits<std::size_t>::max()) {
      result.errorMessage = "workgroup_size must be a positive integer.";
      return result;
    }
    if (![function isKindOfClass:NSString.class] ||
        ![source isKindOfClass:NSString.class]) {
      result.errorMessage = "function_name and msl_source must be strings.";
      return result;
    }

    GeneratedKernelResponse response;
    response.functionName = toString(function);
    response.workgroupSize =
        static_cast<std::size_t>(workgroup.unsignedLongLongValue);
    response.source = toString(source);
    if (response.functionName.empty() || response.source.empty()) {
      result.errorMessage = "function_name and msl_source cannot be empty.";
      return result;
    }
    result.response = std::move(response);
  }
  return result;
}

} // namespace tensor::llm
