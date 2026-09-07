#pragma once

#include "backend/metal/MetalRuntime.hpp"
#include "planner/RegionPlan.hpp"

#include <cstddef>
#include <iosfwd>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace tensor::llm {

enum class CandidateKind { RMSNorm, Fusion };

struct AdvisorCandidate {
  std::string id;
  std::size_t regionId = 0;
  CandidateKind kind = CandidateKind::Fusion;
  std::size_t threads = 0;
};

struct AdvisorRequest {
  std::size_t version = 1;
  std::string deviceName;
  metal::HardwareInfo hardware;
  std::size_t candidateBudgetPerKind = 2;
  planner::ProgramPlan program;
  std::vector<AdvisorCandidate> candidates;
};

struct AdvisorRegionAdvice {
  std::size_t regionId = 0;
  std::vector<std::string> rankedCandidateIds;
};

struct AdvisorResponse {
  std::size_t version = 1;
  std::vector<AdvisorRegionAdvice> regions;
};

struct AdvisorResponseResult {
  std::optional<AdvisorResponse> response;
  std::string errorMessage;
};

struct AdvisorGuidance {
  bool active = false;
  std::map<std::string, std::vector<std::size_t>> rankedThreads;

  [[nodiscard]] std::vector<std::size_t>
  threadsFor(std::size_t regionId, CandidateKind kind,
             const std::vector<std::size_t> &deterministic) const;
};

[[nodiscard]] const char *candidateKindName(CandidateKind kind);
[[nodiscard]] AdvisorRequest
makeAdvisorRequest(planner::ProgramPlan program, std::string deviceName,
                   const metal::HardwareInfo &hardware);
[[nodiscard]] std::string writeAdvisorRequest(const AdvisorRequest &request,
                                              const std::string &path);
[[nodiscard]] AdvisorResponseResult loadAdvisorResponse(const std::string &path);
[[nodiscard]] AdvisorGuidance
makeAdvisorGuidance(const AdvisorRequest &request,
                    const AdvisorResponse &response, std::ostream &log);

} // namespace tensor::llm
