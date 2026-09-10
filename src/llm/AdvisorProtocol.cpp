#include "llm/AdvisorProtocol.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <map>
#include <ostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <variant>

namespace tensor::llm {
namespace {

constexpr std::size_t kProtocolVersion = 1;
constexpr std::size_t kCandidateBudget = 2;
const std::vector<std::size_t> kThreadCandidates{64, 128, 256};

std::string guidanceKey(std::size_t regionId, CandidateKind kind) {
  return std::to_string(regionId) + ":" + candidateKindName(kind);
}

std::string candidateId(std::size_t regionId, CandidateKind kind,
                        std::size_t threads) {
  return "r" + std::to_string(regionId) + "_" + candidateKindName(kind) +
         "_t" + std::to_string(threads);
}

bool hardwareAllows(const planner::RegionPlan &region, CandidateKind kind,
                    std::size_t threads, const metal::HardwareInfo &hardware) {
  std::size_t scratchArrays = 0;
  if (kind == CandidateKind::RMSNorm ||
      region.region.fusion == analyzer::FusionPattern::AddRMSNorm) {
    scratchArrays = 1;
  }
  if (kind == CandidateKind::Fusion &&
      region.region.fusion == analyzer::FusionPattern::AddLayerNorm) {
    scratchArrays = 2;
  }
  return threads > 0 && threads <= hardware.maxThreadsPerThreadgroup &&
         threads * scratchArrays * sizeof(float) <=
             hardware.maxThreadgroupMemoryLength;
}

bool hasRMSNorm(const planner::RegionPlan &region) {
  return std::any_of(region.baseline.begin(), region.baseline.end(),
                     [](const planner::KernelPlan &kernel) {
                       return kernel.op == OpType::RMSNorm;
                     });
}

const char *layoutName(Layout layout) {
  return layout == Layout::Contiguous ? "contiguous" : "strided";
}

std::string escapeJson(const std::string &text) {
  std::ostringstream output;
  for (const unsigned char character : text) {
    switch (character) {
    case '\"': output << "\\\""; break;
    case '\\': output << "\\\\"; break;
    case '\b': output << "\\b"; break;
    case '\f': output << "\\f"; break;
    case '\n': output << "\\n"; break;
    case '\r': output << "\\r"; break;
    case '\t': output << "\\t"; break;
    default:
      if (character < 0x20) {
        const char hex[] = "0123456789abcdef";
        output << "\\u00" << hex[character >> 4] << hex[character & 0xf];
      } else {
        output << static_cast<char>(character);
      }
    }
  }
  return output.str();
}

void writeSizeArray(std::ostream &output,
                    const std::vector<std::size_t> &values) {
  output << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) output << ',';
    output << values[index];
  }
  output << ']';
}

std::vector<std::int64_t> reductionAxes(const Node &node) {
  if (node.op == OpType::RMSNorm) {
    return {std::get<RMSNormAttributes>(node.attributes).axis};
  }
  if (node.op == OpType::LayerNorm) {
    return std::get<LayerNormAttributes>(node.attributes).axes;
  }
  if (node.op == OpType::Softmax) {
    return {std::get<SoftmaxAttributes>(node.attributes).axis};
  }
  if (node.op == OpType::ReduceSum || node.op == OpType::ReduceMean) {
    return std::get<ReductionAttributes>(node.attributes).axes;
  }
  return {};
}

void writeSignedArray(std::ostream &output,
                      const std::vector<std::int64_t> &values) {
  output << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0) output << ',';
    output << values[index];
  }
  output << ']';
}

void writeTensor(std::ostream &output, ValueId value,
                 const TensorType &type) {
  output << "{\"value_id\":" << value << ",\"shape\":";
  writeSizeArray(output, type.shape);
  output << ",\"dtype\":\"" << tensor::dtypeName(type.dtype)
         << "\",\"layout\":\"" << layoutName(type.layout)
         << "\",\"strides\":";
  writeSizeArray(output, type.strides());
  output << '}';
}

struct JsonValue {
  using Array = std::vector<JsonValue>;
  using Object = std::map<std::string, JsonValue>;
  std::variant<std::nullptr_t, bool, double, std::string, Array, Object> value;
};

class JsonParser {
public:
  explicit JsonParser(std::string text) : text_(std::move(text)) {}

  JsonValue parse() {
    skipWhitespace();
    auto result = parseValue();
    skipWhitespace();
    if (position_ != text_.size()) fail("Unexpected trailing content");
    return result;
  }

private:
  [[noreturn]] void fail(const std::string &message) const {
    throw std::invalid_argument(message + " at byte " +
                                std::to_string(position_) + ".");
  }

  void skipWhitespace() {
    while (position_ < text_.size() &&
           (text_[position_] == ' ' || text_[position_] == '\n' ||
            text_[position_] == '\r' || text_[position_] == '\t')) {
      ++position_;
    }
  }

  char take() {
    if (position_ == text_.size()) fail("Unexpected end of JSON");
    return text_[position_++];
  }

  void expect(char expected) {
    if (take() != expected) fail(std::string("Expected '") + expected + "'");
  }

  JsonValue parseValue() {
    skipWhitespace();
    if (position_ == text_.size()) fail("Expected JSON value");
    switch (text_[position_]) {
    case '{': return JsonValue{parseObject()};
    case '[': return JsonValue{parseArray()};
    case '\"': return JsonValue{parseString()};
    case 't': parseLiteral("true"); return JsonValue{true};
    case 'f': parseLiteral("false"); return JsonValue{false};
    case 'n': parseLiteral("null"); return JsonValue{nullptr};
    default:
      if (text_[position_] == '-' ||
          (text_[position_] >= '0' && text_[position_] <= '9')) {
        return JsonValue{parseNumber()};
      }
      fail("Expected JSON value");
    }
  }

  JsonValue::Object parseObject() {
    JsonValue::Object object;
    expect('{');
    skipWhitespace();
    if (position_ < text_.size() && text_[position_] == '}') {
      ++position_;
      return object;
    }
    while (true) {
      skipWhitespace();
      if (position_ == text_.size() || text_[position_] != '\"') {
        fail("Expected object key");
      }
      auto key = parseString();
      skipWhitespace();
      expect(':');
      skipWhitespace();
      if (!object.emplace(key, parseValue()).second) {
        fail("Duplicate object key '" + key + "'");
      }
      skipWhitespace();
      const char delimiter = take();
      if (delimiter == '}') break;
      if (delimiter != ',') fail("Expected ',' or '}'");
    }
    return object;
  }

  JsonValue::Array parseArray() {
    JsonValue::Array array;
    expect('[');
    skipWhitespace();
    if (position_ < text_.size() && text_[position_] == ']') {
      ++position_;
      return array;
    }
    while (true) {
      array.push_back(parseValue());
      skipWhitespace();
      const char delimiter = take();
      if (delimiter == ']') break;
      if (delimiter != ',') fail("Expected ',' or ']'");
    }
    return array;
  }

  static void appendUtf8(std::string &output, unsigned codePoint) {
    if (codePoint <= 0x7f) {
      output.push_back(static_cast<char>(codePoint));
    } else if (codePoint <= 0x7ff) {
      output.push_back(static_cast<char>(0xc0 | (codePoint >> 6)));
      output.push_back(static_cast<char>(0x80 | (codePoint & 0x3f)));
    } else {
      output.push_back(static_cast<char>(0xe0 | (codePoint >> 12)));
      output.push_back(static_cast<char>(0x80 | ((codePoint >> 6) & 0x3f)));
      output.push_back(static_cast<char>(0x80 | (codePoint & 0x3f)));
    }
  }

  std::string parseString() {
    std::string result;
    expect('\"');
    while (true) {
      const char character = take();
      if (character == '\"') return result;
      if (static_cast<unsigned char>(character) < 0x20) {
        fail("Unescaped control character in string");
      }
      if (character != '\\') {
        result.push_back(character);
        continue;
      }
      const char escape = take();
      switch (escape) {
      case '\"': result.push_back('\"'); break;
      case '\\': result.push_back('\\'); break;
      case '/': result.push_back('/'); break;
      case 'b': result.push_back('\b'); break;
      case 'f': result.push_back('\f'); break;
      case 'n': result.push_back('\n'); break;
      case 'r': result.push_back('\r'); break;
      case 't': result.push_back('\t'); break;
      case 'u': {
        unsigned codePoint = 0;
        for (int index = 0; index < 4; ++index) {
          const char digit = take();
          codePoint <<= 4;
          if (digit >= '0' && digit <= '9') codePoint += digit - '0';
          else if (digit >= 'a' && digit <= 'f') codePoint += digit - 'a' + 10;
          else if (digit >= 'A' && digit <= 'F') codePoint += digit - 'A' + 10;
          else fail("Invalid Unicode escape");
        }
        if (codePoint >= 0xd800 && codePoint <= 0xdfff) {
          fail("Unicode surrogate escapes are unsupported");
        }
        appendUtf8(result, codePoint);
        break;
      }
      default: fail("Invalid string escape");
      }
    }
  }

  double parseNumber() {
    const char *begin = text_.c_str() + position_;
    char *end = nullptr;
    errno = 0;
    const double result = std::strtod(begin, &end);
    if (begin == end || errno == ERANGE || !std::isfinite(result)) {
      fail("Invalid JSON number");
    }
    position_ += static_cast<std::size_t>(end - begin);
    return result;
  }

  void parseLiteral(const char *literal) {
    for (const char *character = literal; *character != '\0'; ++character) {
      if (take() != *character) fail("Invalid JSON literal");
    }
  }

  std::string text_;
  std::size_t position_ = 0;
};

const JsonValue::Object &asObject(const JsonValue &value,
                                  const std::string &context) {
  const auto *result = std::get_if<JsonValue::Object>(&value.value);
  if (!result) throw std::invalid_argument(context + " must be an object.");
  return *result;
}

const JsonValue::Array &asArray(const JsonValue &value,
                                const std::string &context) {
  const auto *result = std::get_if<JsonValue::Array>(&value.value);
  if (!result) throw std::invalid_argument(context + " must be an array.");
  return *result;
}

const std::string &asString(const JsonValue &value,
                            const std::string &context) {
  const auto *result = std::get_if<std::string>(&value.value);
  if (!result) throw std::invalid_argument(context + " must be a string.");
  return *result;
}

std::size_t asSize(const JsonValue &value, const std::string &context) {
  const auto *number = std::get_if<double>(&value.value);
  if (!number || *number < 0.0 || std::floor(*number) != *number ||
      *number > static_cast<double>(std::numeric_limits<std::size_t>::max())) {
    throw std::invalid_argument(context + " must be a non-negative integer.");
  }
  return static_cast<std::size_t>(*number);
}

const JsonValue &required(const JsonValue::Object &object,
                          const std::string &key,
                          const std::string &context) {
  const auto found = object.find(key);
  if (found == object.end()) {
    throw std::invalid_argument(context + " is missing '" + key + "'.");
  }
  return found->second;
}

void requireOnly(const JsonValue::Object &object,
                 const std::set<std::string> &allowed,
                 const std::string &context) {
  for (const auto &field : object) {
    if (allowed.find(field.first) == allowed.end()) {
      throw std::invalid_argument(context + " contains unknown field '" +
                                  field.first + "'.");
    }
  }
}

AdvisorResponse decodeResponse(const JsonValue &root) {
  const auto &object = asObject(root, "Advisor response");
  requireOnly(object, {"version", "regions"}, "Advisor response");
  AdvisorResponse response;
  response.version = asSize(required(object, "version", "Advisor response"),
                            "Advisor response.version");
  if (response.version != kProtocolVersion) {
    throw std::invalid_argument("Unsupported advisor protocol version " +
                                std::to_string(response.version) + ".");
  }
  const auto &regions = asArray(required(object, "regions", "Advisor response"),
                                "Advisor response.regions");
  std::set<std::size_t> seenRegions;
  for (std::size_t index = 0; index < regions.size(); ++index) {
    const std::string context = "Advisor response.regions[" +
                                std::to_string(index) + "]";
    const auto &regionObject = asObject(regions[index], context);
    requireOnly(regionObject, {"region_id", "ranked_candidate_ids"}, context);
    AdvisorRegionAdvice advice;
    advice.regionId = asSize(required(regionObject, "region_id", context),
                             context + ".region_id");
    if (!seenRegions.insert(advice.regionId).second) {
      throw std::invalid_argument("Advisor response repeats region_id " +
                                  std::to_string(advice.regionId) + ".");
    }
    const auto &ids = asArray(required(regionObject, "ranked_candidate_ids", context),
                              context + ".ranked_candidate_ids");
    for (std::size_t idIndex = 0; idIndex < ids.size(); ++idIndex) {
      advice.rankedCandidateIds.push_back(
          asString(ids[idIndex], context + ".ranked_candidate_ids[" +
                                      std::to_string(idIndex) + "]"));
    }
    response.regions.push_back(std::move(advice));
  }
  return response;
}

} // namespace

const char *candidateKindName(CandidateKind kind) {
  switch (kind) {
  case CandidateKind::RMSNorm: return "rmsnorm";
  case CandidateKind::Fusion: return "fusion";
  }
  throw std::invalid_argument("Unknown advisor candidate kind.");
}

std::vector<std::size_t>
AdvisorGuidance::threadsFor(std::size_t regionId, CandidateKind kind,
                            const std::vector<std::size_t> &deterministic) const {
  if (!active) return deterministic;
  const auto found = rankedThreads.find(guidanceKey(regionId, kind));
  return found == rankedThreads.end() ? deterministic : found->second;
}

AdvisorRequest makeAdvisorRequest(planner::ProgramPlan program,
                                  std::string deviceName,
                                  const metal::HardwareInfo &hardware) {
  AdvisorRequest request;
  request.deviceName = std::move(deviceName);
  request.hardware = hardware;
  request.candidateBudgetPerKind = kCandidateBudget;
  request.program = std::move(program);
  for (const auto &region : request.program.regions) {
    const auto add = [&](CandidateKind kind) {
      for (const auto threads : kThreadCandidates) {
        if (hardwareAllows(region, kind, threads, hardware)) {
          request.candidates.push_back(
              {candidateId(region.region.id, kind, threads), region.region.id,
               kind, threads});
        }
      }
    };
    if (hasRMSNorm(region)) add(CandidateKind::RMSNorm);
    if (region.region.fusion != analyzer::FusionPattern::None) {
      add(CandidateKind::Fusion);
    }
  }
  return request;
}

std::string writeAdvisorRequest(const AdvisorRequest &request,
                                const std::string &path) {
  std::ofstream output(path);
  if (!output) return "Unable to open advisor request output '" + path + "'.";
  output << "{\n  \"version\":" << request.version
         << ",\n  \"device\":{\"name\":\"" << escapeJson(request.deviceName)
         << "\",\"max_threads_per_threadgroup\":"
         << request.hardware.maxThreadsPerThreadgroup
         << ",\"max_threadgroup_memory_bytes\":"
         << request.hardware.maxThreadgroupMemoryLength
         << ",\"max_buffer_bytes\":" << request.hardware.maxBufferLength
         << "},\n  \"candidate_budget_per_kind\":"
         << request.candidateBudgetPerKind << ",\n  \"regions\":[\n";
  for (std::size_t regionIndex = 0;
       regionIndex < request.program.regions.size(); ++regionIndex) {
    const auto &region = request.program.regions[regionIndex];
    const auto &graph = request.program.graph.analyzed;
    if (regionIndex != 0) output << ",\n";
    output << "    {\"region_id\":" << region.region.id
           << ",\"pattern\":\""
           << escapeJson(analyzer::fusionName(region.region.fusion))
           << "\",\"operations\":[";
    for (std::size_t nodeOffset = 0;
         nodeOffset < region.region.nodes.size(); ++nodeOffset) {
      if (nodeOffset != 0) output << ',';
      const auto nodeIndex = region.region.nodes[nodeOffset];
      const auto &node = graph.graph.nodes[nodeIndex];
      output << "{\"node_index\":" << nodeIndex << ",\"op\":\""
             << opName(node.op) << "\",\"reduction_axes\":";
      writeSignedArray(output, reductionAxes(node));
      output << '}';
    }
    output << "],\"inputs\":[";
    for (std::size_t index = 0; index < region.region.inputs.size(); ++index) {
      if (index != 0) output << ',';
      const auto value = region.region.inputs[index];
      writeTensor(output, value, graph.types[value]);
    }
    output << "],\"outputs\":[";
    for (std::size_t index = 0; index < region.region.outputs.size(); ++index) {
      if (index != 0) output << ',';
      const auto value = region.region.outputs[index];
      writeTensor(output, value, graph.types[value]);
    }
    output << "],\"candidates\":[";
    bool firstCandidate = true;
    for (const auto &candidate : request.candidates) {
      if (candidate.regionId != region.region.id) continue;
      if (!firstCandidate) output << ',';
      firstCandidate = false;
      output << "{\"id\":\"" << candidate.id << "\",\"kind\":\""
             << candidateKindName(candidate.kind)
             << "\",\"workgroup_size\":" << candidate.threads << '}';
    }
    output << "]}";
  }
  output << "\n  ]\n}\n";
  if (!output) return "Failed while writing advisor request '" + path + "'.";
  return {};
}

AdvisorResponseResult loadAdvisorResponse(const std::string &path) {
  AdvisorResponseResult result;
  try {
    std::ifstream input(path);
    if (!input) throw std::invalid_argument(
        "Unable to open advisor response '" + path + "'.");
    std::ostringstream contents;
    contents << input.rdbuf();
    if (!input.good() && !input.eof()) {
      throw std::invalid_argument(
          "Failed while reading advisor response '" + path + "'.");
    }
    result.response = decodeResponse(JsonParser(contents.str()).parse());
  } catch (const std::exception &error) {
    result.errorMessage = error.what();
  }
  return result;
}

AdvisorGuidance makeAdvisorGuidance(const AdvisorRequest &request,
                                    const AdvisorResponse &response,
                                    std::ostream &log) {
  AdvisorGuidance guidance;
  std::map<std::string, const AdvisorCandidate *> candidates;
  std::set<std::size_t> regionIds;
  for (const auto &region : request.program.regions) {
    regionIds.insert(region.region.id);
  }
  for (const auto &candidate : request.candidates) {
    candidates.emplace(candidate.id, &candidate);
  }

  std::size_t recommendations = 0;
  std::size_t accepted = 0;
  std::size_t rejected = 0;
  std::set<std::string> seenCandidates;
  for (const auto &region : response.regions) {
    recommendations += region.rankedCandidateIds.size();
    if (regionIds.find(region.regionId) == regionIds.end()) {
      rejected += region.rankedCandidateIds.size();
      log << "Advisor hint rejected: unknown region " << region.regionId << '\n';
      continue;
    }
    for (const auto &id : region.rankedCandidateIds) {
      const auto found = candidates.find(id);
      if (found == candidates.end()) {
        ++rejected;
        log << "Advisor hint rejected: unknown candidate '" << id << "'\n";
        continue;
      }
      const auto &candidate = *found->second;
      if (candidate.regionId != region.regionId) {
        ++rejected;
        log << "Advisor hint rejected: candidate '" << id
            << "' belongs to another region\n";
        continue;
      }
      if (!seenCandidates.insert(id).second) {
        ++rejected;
        log << "Advisor hint rejected: duplicate candidate '" << id << "'\n";
        continue;
      }
      auto &threads = guidance.rankedThreads[
          guidanceKey(candidate.regionId, candidate.kind)];
      if (threads.size() >= request.candidateBudgetPerKind) {
        ++rejected;
        log << "Advisor hint skipped by Top-K budget: '" << id << "'\n";
        continue;
      }
      threads.push_back(candidate.threads);
      ++accepted;
    }
  }
  guidance.active = accepted != 0;
  log << "Advisor recommendations: " << recommendations << '\n'
      << "Planner accepted hints: " << accepted << '\n'
      << "Planner rejected hints: " << rejected << '\n'
      << "Planner mode: "
      << (guidance.active ? "LLM-guided Top-K" : "deterministic fallback")
      << '\n';
  return guidance;
}

} // namespace tensor::llm
