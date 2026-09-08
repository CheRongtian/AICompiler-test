#pragma once

#include "tensor/DecoderLLM.hpp"

#include <memory>
#include <string>

namespace tensor::importer {

struct DecoderLLMImportResult {
  std::unique_ptr<DecoderLLMWorkload> workload;
  std::string errorMessage;
};

[[nodiscard]] DecoderLLMImportResult
importDecoderLLMWorkload(const std::string &manifestPath);

} // namespace tensor::importer
