#pragma once

#include "planner/KVCachePlan.hpp"

#include <memory>
#include <string>
#include <vector>

namespace tensor::importer {

struct KVCacheReferenceStep {
  std::vector<float> query;
  std::vector<float> key;
  std::vector<float> value;
  std::vector<double> output;
  std::vector<double> keyCache;
  std::vector<double> valueCache;
};

struct ImportedKVCacheWorkload {
  std::string modelName;
  planner::KVCachePlan plan;
  std::vector<float> queryWeight;
  std::vector<float> keyWeight;
  std::vector<float> valueWeight;
  std::vector<float> outputWeight;
  KVCacheReferenceStep prefill;
  std::vector<KVCacheReferenceStep> decodeSteps;
};

struct KVCacheImportResult {
  std::unique_ptr<ImportedKVCacheWorkload> workload;
  std::string errorMessage;
};

[[nodiscard]] KVCacheImportResult
importKVCacheWorkload(const std::string &manifestPath);

} // namespace tensor::importer
