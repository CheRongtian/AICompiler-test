#pragma once

#include "tensor/TransformerDecode.hpp"

#include <memory>
#include <string>

namespace tensor::importer {

struct TransformerDecodeImportResult {
  std::unique_ptr<TransformerDecodeWorkload> workload;
  std::string errorMessage;
};

[[nodiscard]] TransformerDecodeImportResult
importTransformerDecodeWorkload(const std::string &manifestPath);

} // namespace tensor::importer
