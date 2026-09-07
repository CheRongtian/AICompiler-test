#pragma once

#include "tensor/TensorIR.hpp"

#include <memory>
#include <string>
#include <vector>

namespace tensor::importer {

enum class PyTorchInputKind { UserInput, Parameter, Buffer, Constant };

[[nodiscard]] const char *inputKindName(PyTorchInputKind kind);

struct ImportedInput {
  ValueId value = 0;
  PyTorchInputKind kind = PyTorchInputKind::UserInput;
};

struct ImportedPyTorchModel {
  std::string modelName;
  TensorGraph graph;
  GraphInputs inputs;
  std::vector<ImportedInput> importedInputs;
  std::vector<TensorType> expectedOutputTypes;
  std::vector<std::vector<double>> expectedOutputs;
};

struct PyTorchImportResult {
  std::unique_ptr<ImportedPyTorchModel> model;
  std::string errorMessage;
};

// Imports the canonical static graph emitted by tools/export_pytorch.py.
// Tensor payload values are stored as float32 host values; TensorIR dtype controls
// conversion when Metal buffers are created.
[[nodiscard]] PyTorchImportResult importPyTorchGraph(const std::string &manifestPath);

} // namespace tensor::importer
