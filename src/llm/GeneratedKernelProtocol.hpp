#pragma once

#include <cstddef>
#include <optional>
#include <string>

namespace tensor::llm {

struct GeneratedKernelResponse {
  std::size_t version = 1;
  std::string functionName;
  std::size_t workgroupSize = 0;
  std::string source;
};

struct GeneratedKernelResponseResult {
  std::optional<GeneratedKernelResponse> response;
  std::string errorMessage;
};

[[nodiscard]] GeneratedKernelResponseResult
loadGeneratedKernelResponse(const std::string &path);

} // namespace tensor::llm
