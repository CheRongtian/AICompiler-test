import argparse
import array
import importlib.util
import json
import math
import operator
from dataclasses import dataclass
from pathlib import Path

import torch

class ExportError(RuntimeError):
    pass

@dataclass
class InputRecord:
    value_id: int
    name: str
    kind: str
    dtype: str
    shape: tuple
    tensor: torch.Tensor

@dataclass
class NodeRecord:
    value_id: int
    name: str
    op: str
    inputs: list
    attributes: str

def _dtype_token(dtype):
    if dtype == torch.float16:
        return "F16"
    if dtype == torch.float32:
        return "F32"
    if dtype in (torch.int32, torch.int64):
        return "I32"
    raise ExportError(f"Unsupported PyTorch dtype: {dtype}")

def _static_shape(value, node_name):
    try:
        return tuple(int(dimension) for dimension in value.shape)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ExportError(
            f"Dynamic shape at node '{node_name}' is outside the static V4 importer"
        ) from error

def _quoted(value):
    return json.dumps(value, ensure_ascii=False)

def _flatten(value):
    if isinstance(value, (tuple, list)):
        flattened = []
        for item in value:
            flattened.extend(_flatten(item))
        return flattened
    return [value]

class GraphBuilder:
    def __init__(self):
        self.inputs = []
        self.nodes = []
        self.shapes = {}
        self.dtypes = {}
        self.next_value = 0
        self.constant_cache = {}

    def add_input(self, name, kind, tensor):
        if not isinstance(tensor, torch.Tensor):
            raise ExportError(f"Imported value '{name}' is not a tensor")
        tensor = tensor.detach().cpu().contiguous()
        dtype = _dtype_token(tensor.dtype)
        if dtype == "I32":
            if tensor.numel() and tensor.dtype == torch.int64:
                minimum = int(tensor.min().item())
                maximum = int(tensor.max().item())
                if minimum < -(2**31) or maximum >= 2**31:
                    raise ExportError(f"Integer tensor '{name}' exceeds int32 range")
            tensor = tensor.to(torch.int32)
        value_id = self.next_value
        self.next_value += 1
        shape = tuple(int(dimension) for dimension in tensor.shape)
        self.inputs.append(InputRecord(value_id, name, kind, dtype, shape, tensor))
        self.shapes[value_id] = shape
        self.dtypes[value_id] = dtype
        return value_id

    def add_constant(self, value, dtype="F32"):
        numeric = int(value) if dtype == "I32" else float(value)
        key = (dtype, numeric)
        if key in self.constant_cache:
            return self.constant_cache[key]
        torch_dtype = {
            "F16": torch.float16,
            "F32": torch.float32,
            "I32": torch.int32,
        }[dtype]
        value_id = self.add_input(
            f"constant_{len(self.constant_cache)}", "CONSTANT", torch.tensor(numeric, dtype=torch_dtype)
        )
        self.constant_cache[key] = value_id
        return value_id

    def emit(self, name, op, inputs, attributes, shape, dtype):
        value_id = self.next_value
        self.next_value += 1
        self.nodes.append(NodeRecord(value_id, name, op, list(inputs), attributes))
        self.shapes[value_id] = tuple(shape)
        self.dtypes[value_id] = dtype
        return value_id

class ExportNormalizer:
    def __init__(self, exported_program, user_inputs):
        self.exported = exported_program
        self.user_inputs = list(user_inputs)
        self.builder = GraphBuilder()
        self.fx_values = {}
        self.scalar_values = {}

    def _node_tensor_meta(self, node):
        value = node.meta.get("val")
        if not isinstance(value, torch.Tensor):
            raise ExportError(
                f"Node '{node.name}' has no tensor metadata; target={node.target}"
            )
        return _static_shape(value, node.name), _dtype_token(value.dtype)

    def _value_id(self, value, context, dtype="F32"):
        if isinstance(value, torch.fx.Node):
            mapped = self.fx_values.get(value)
            if not isinstance(mapped, int):
                raise ExportError(
                    f"Node '{value.name}' is not a tensor value while importing {context}"
                )
            return mapped
        if isinstance(value, (bool, int, float)):
            return self.builder.add_constant(value, dtype)
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            return self.builder.add_input(f"{context}_constant", "CONSTANT", value)
        raise ExportError(f"Unsupported tensor operand in {context}: {value!r}")

    def _scalar(self, value, context):
        if isinstance(value, torch.fx.Node):
            if value in self.scalar_values:
                return self.scalar_values[value]
            metadata = value.meta.get("val")
            try:
                if isinstance(metadata, bool):
                    return metadata
                if isinstance(metadata, int):
                    return int(metadata)
                if isinstance(metadata, float):
                    return float(metadata)
                return int(metadata)
            except (TypeError, ValueError, RuntimeError) as error:
                raise ExportError(
                    f"Unable to resolve static scalar node '{value.name}' in {context}"
                ) from error
        if isinstance(value, (bool, int, float)):
            return value
        raise ExportError(f"Expected a static scalar in {context}, received {value!r}")

    def _placeholder_sources(self):
        signature = self.exported.graph_signature
        parameter_map = dict(signature.inputs_to_parameters)
        buffer_map = dict(signature.inputs_to_buffers)
        user_names = []
        for value in signature.user_inputs:
            user_names.append(value if isinstance(value, str) else value.name)
        if len(user_names) != len(self.user_inputs):
            raise ExportError(
                "PyTorch export user-input signature does not match the supplied sample inputs"
            )
        return parameter_map, buffer_map, dict(zip(user_names, self.user_inputs))

    def _state_tensor(self, target, kind):
        state = self.exported.state_dict
        if target in state:
            return state[target]
        constants = getattr(self.exported, "constants", {})
        if target in constants:
            return constants[target]
        raise ExportError(f"Missing exported {kind.lower()} tensor '{target}'")

    def _import_placeholder(self, node, parameter_map, buffer_map, users):
        name = node.name
        if name in parameter_map:
            value = self.builder.add_input(
                parameter_map[name], "PARAMETER", self._state_tensor(parameter_map[name], "parameter")
            )
        elif name in buffer_map:
            value = self.builder.add_input(
                buffer_map[name], "BUFFER", self._state_tensor(buffer_map[name], "buffer")
            )
        elif name in users:
            value = self.builder.add_input(name, "USER", users[name])
        else:
            raise ExportError(f"Unclassified PyTorch placeholder '{name}'")
        self.fx_values[node] = value

    def _identity(self, node):
        self.fx_values[node] = self._value_id(node.args[0], node.name)

    def _linear(self, node):
        shape, dtype = self._node_tensor_meta(node)
        source = self._value_id(node.args[0], node.name, dtype)
        weight = self._value_id(node.args[1], node.name, dtype)
        weight_shape = self.builder.shapes[weight]
        if len(weight_shape) != 2:
            raise ExportError(f"Linear weight at node '{node.name}' must be rank 2")
        transposed = self.builder.emit(
            f"{node.name}.weight_transpose",
            "Transpose",
            [weight],
            "TRANSPOSE 0 1",
            (weight_shape[1], weight_shape[0]),
            self.builder.dtypes[weight],
        )
        result = self.builder.emit(
            f"{node.name}.matmul", "MatMul", [source, transposed], "NONE", shape, dtype
        )
        bias = node.args[2] if len(node.args) > 2 else None
        if bias is not None:
            bias_id = self._value_id(bias, node.name, dtype)
            result = self.builder.emit(node.name, "Add", [result, bias_id], "NONE", shape, dtype)
        self.fx_values[node] = result

    def _binary(self, node, op):
        shape, dtype = self._node_tensor_meta(node)
        left = self._value_id(node.args[0], node.name, dtype)
        right = self._value_id(node.args[1], node.name, dtype)
        if op == "Add":
            alpha = node.kwargs.get("alpha", node.args[2] if len(node.args) > 2 else 1)
            alpha = self._scalar(alpha, node.name)
            if alpha != 1:
                right_shape = self.builder.shapes[right]
                right = self.builder.emit(
                    f"{node.name}.alpha",
                    "Mul",
                    [right, self.builder.add_constant(alpha, dtype)],
                    "NONE",
                    right_shape,
                    dtype,
                )
        self.fx_values[node] = self.builder.emit(
            node.name, op, [left, right], "NONE", shape, dtype
        )

    def _division(self, node):
        shape, dtype = self._node_tensor_meta(node)
        source = self._value_id(node.args[0], node.name, dtype)
        divisor = self._scalar(node.args[1], node.name)
        if divisor == 0:
            raise ExportError(f"Division by zero at node '{node.name}'")
        reciprocal = self.builder.add_constant(1.0 / float(divisor), dtype)
        self.fx_values[node] = self.builder.emit(
            node.name, "Mul", [source, reciprocal], "NONE", shape, dtype
        )

    def _layer_norm(self, node, native=False):
        source = self._value_id(node.args[0], node.name)
        source_shape = self.builder.shapes[source]
        normalized_shape = tuple(int(value) for value in node.args[1])
        if not normalized_shape or len(normalized_shape) > len(source_shape):
            raise ExportError(f"Invalid LayerNorm shape at node '{node.name}'")
        weight = self._value_id(node.args[2], node.name)
        bias = self._value_id(node.args[3], node.name)
        epsilon = float(node.args[4]) if len(node.args) > 4 else 1e-5
        axes = list(range(len(source_shape) - len(normalized_shape), len(source_shape)))
        attributes = "LAYERNORM {} {} {}".format(
            format(epsilon, ".17g"), len(axes), " ".join(str(axis) for axis in axes)
        )
        dtype = self.builder.dtypes[source]
        output = self.builder.emit(
            node.name, "LayerNorm", [source, weight, bias], attributes, source_shape, dtype
        )
        self.fx_values[node] = (output, None, None) if native else output

    def _slice(self, node):
        shape, dtype = self._node_tensor_meta(node)
        source = self._value_id(node.args[0], node.name, dtype)
        source_shape = self.builder.shapes[source]
        rank = len(source_shape)

        dimension_arg = node.args[1] if len(node.args) > 1 else 0
        dimension = int(self._scalar(dimension_arg, node.name))

        if dimension < 0:
            dimension += rank

        if dimension < 0 or dimension >= rank:
            raise ExportError(
                f"Slice dimension is out of range at node '{node.name}'"
            )

        # aten.slice.Tensor permits None for the default slice boundary.
        start_arg = node.args[2] if len(node.args) > 2 else None
        step_arg = node.args[4] if len(node.args) > 4 else None

        start = (
            0
            if start_arg is None
            else int(self._scalar(start_arg, node.name))
        )

        step = (
            1
            if step_arg is None
            else int(self._scalar(step_arg, node.name))
        )

        dimension_size = source_shape[dimension]

        if start < 0:
            start += dimension_size

        # Python positive-slice semantics clamp the start boundary.
        start = max(0, min(start, dimension_size))

        if step <= 0:
            raise ExportError(
                f"Only positive Slice steps are supported at node '{node.name}'"
            )

        starts = [0] * rank
        steps = [1] * rank

        starts[dimension] = start
        steps[dimension] = step

        # The exported node already contains the statically inferred result shape.
        # TensorIR Slice stores start + result size + step, so parsing end is
        # unnecessary for the current static importer.
        values = [str(rank)]
        values.extend(str(value) for value in starts)
        values.extend(str(value) for value in shape)
        values.extend(str(value) for value in steps)

        self.fx_values[node] = self.builder.emit(
            node.name,
            "Slice",
            [source],
            "SLICE " + " ".join(values),
            shape,
            dtype,
        )

    def _getitem(self, node):
        source = node.args[0]
        if not isinstance(source, torch.fx.Node) or source not in self.fx_values:
            raise ExportError(f"Unsupported getitem source at node '{node.name}'")
        values = self.fx_values[source]
        if not isinstance(values, tuple):
            raise ExportError(f"Node '{node.name}' indexes a non-tuple value")
        index = int(self._scalar(node.args[1], node.name))
        if index < 0:
            index += len(values)
        if index < 0 or index >= len(values) or values[index] is None:
            raise ExportError(f"Unsupported tuple result {index} at node '{node.name}'")
        self.fx_values[node] = values[index]

    def _import_call_function(self, node):
        target = str(node.target)
        metadata = node.meta.get("val")
        if not isinstance(metadata, (torch.Tensor, tuple, list)):
            try:
                self.scalar_values[node] = (
                    bool(metadata) if isinstance(metadata, bool) else
                    float(metadata) if isinstance(metadata, float) else int(metadata)
                )
                return
            except (TypeError, ValueError, RuntimeError):
                pass

        if node.target is operator.getitem:
            self._getitem(node)
        elif target == "aten.linear.default":
            self._linear(node)
        elif target in {"aten.add.Tensor", "aten.add.Scalar"}:
            self._binary(node, "Add")
        elif target in {"aten.mul.Tensor", "aten.mul.Scalar"}:
            self._binary(node, "Mul")
        elif target in {"aten.div.Tensor", "aten.div.Scalar"}:
            self._division(node)
        elif target in {"aten.matmul.default", "aten.mm.default", "aten.bmm.default"}:
            shape, dtype = self._node_tensor_meta(node)
            left = self._value_id(node.args[0], node.name, dtype)
            right = self._value_id(node.args[1], node.name, dtype)
            self.fx_values[node] = self.builder.emit(
                node.name, "MatMul", [left, right], "NONE", shape, dtype
            )
        elif target == "aten.embedding.default":
            shape, dtype = self._node_tensor_meta(node)
            table = self._value_id(node.args[0], node.name, dtype)
            indices = self._value_id(node.args[1], node.name, "I32")
            self.fx_values[node] = self.builder.emit(
                node.name, "Embedding", [table, indices], "NONE", shape, dtype
            )
        elif target in {"aten.view.default", "aten._unsafe_view.default", "aten.reshape.default"}:
            shape, dtype = self._node_tensor_meta(node)
            source = self._value_id(node.args[0], node.name, dtype)
            attributes = "RESHAPE {} {}".format(
                len(shape), " ".join(str(value) for value in shape)
            )
            self.fx_values[node] = self.builder.emit(
                node.name, "View", [source], attributes, shape, dtype
            )
        elif target == "aten.transpose.int":
            shape, dtype = self._node_tensor_meta(node)
            source = self._value_id(node.args[0], node.name, dtype)
            first = int(self._scalar(node.args[1], node.name))
            second = int(self._scalar(node.args[2], node.name))
            self.fx_values[node] = self.builder.emit(
                node.name,
                "Transpose",
                [source],
                f"TRANSPOSE {first} {second}",
                shape,
                dtype,
            )
        elif target == "aten.t.default":
            shape, dtype = self._node_tensor_meta(node)
            source = self._value_id(node.args[0], node.name, dtype)
            self.fx_values[node] = self.builder.emit(
                node.name, "Transpose", [source], "TRANSPOSE 0 1", shape, dtype
            )
        elif target in {"aten.clone.default", "aten.contiguous.default"}:
            shape, dtype = self._node_tensor_meta(node)
            source = self._value_id(node.args[0], node.name, dtype)
            self.fx_values[node] = self.builder.emit(
                node.name, "Contiguous", [source], "NONE", shape, dtype
            )
        elif target in {"aten.relu.default", "aten.relu_.default"}:
            shape, dtype = self._node_tensor_meta(node)
            source = self._value_id(node.args[0], node.name, dtype)
            self.fx_values[node] = self.builder.emit(
                node.name, "ReLU", [source], "NONE", shape, dtype
            )
        elif target in {"aten.softmax.int", "aten._softmax.default"}:
            shape, dtype = self._node_tensor_meta(node)
            source = self._value_id(node.args[0], node.name, dtype)
            axis = int(self._scalar(node.args[1], node.name))
            self.fx_values[node] = self.builder.emit(
                node.name, "Softmax", [source], f"SOFTMAX {axis}", shape, dtype
            )
        elif target == "aten.layer_norm.default":
            self._layer_norm(node)
        elif target == "aten.native_layer_norm.default":
            self._layer_norm(node, native=True)
        elif target == "aten.slice.Tensor":
            self._slice(node)
        elif target in {"aten.dropout.default", "aten.detach.default", "aten.alias.default"}:
            if target == "aten.dropout.default":
                training = bool(self._scalar(node.args[2] if len(node.args) > 2 else False, node.name))
                if training:
                    raise ExportError(f"Training Dropout is unsupported at node '{node.name}'")
            self._identity(node)
        else:
            raise ExportError(
                f"Unsupported PyTorch op at node '{node.name}': target={target}"
            )

    def normalize(self):
        parameter_map, buffer_map, users = self._placeholder_sources()
        output_ids = None
        for node in self.exported.graph_module.graph.nodes:
            if node.op == "placeholder":
                self._import_placeholder(node, parameter_map, buffer_map, users)
            elif node.op == "call_function":
                self._import_call_function(node)
            elif node.op == "output":
                output_ids = []
                for value in _flatten(node.args[0]):
                    if not isinstance(value, torch.fx.Node):
                        raise ExportError("The V4 importer requires tensor graph outputs")
                    mapped = self.fx_values.get(value)
                    if not isinstance(mapped, int):
                        raise ExportError(f"Unsupported graph output from node '{value.name}'")
                    output_ids.append(mapped)
            else:
                raise ExportError(
                    f"Unsupported FX node '{node.name}': kind={node.op}, target={node.target}"
                )
        if not output_ids:
            raise ExportError("PyTorch export produced no tensor outputs")
        return self.builder, output_ids


def _load_transformer_module(project_root):
    source = project_root / "workloads" / "pytorch" / "transformer.py"
    spec = importlib.util.spec_from_file_location("tmc_transformer_workload", source)
    if spec is None or spec.loader is None:
        raise ExportError(f"Unable to load Transformer workload: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_transformer(project_root):
    module = _load_transformer_module(project_root)
    torch.manual_seed(0)
    model = module.Transformer(
        src_vocab=64,
        tgt_vocab=64,
        d_model=64,
        n_heads=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        d_ff=128,
        dropout=0.0,
        max_len=32,
    ).eval()
    src = torch.randint(0, 64, (1, 16), dtype=torch.int64)
    tgt = torch.randint(0, 64, (1, 8), dtype=torch.int64)
    with torch.no_grad():
        reference = model(src, tgt)
        exported = torch.export.export(model, (src, tgt))
    return exported, (src, tgt), reference

def _payload_values(tensor):
    return tensor.detach().cpu().contiguous().to(torch.float32).reshape(-1).tolist()

def _write_archive(output_path, model_name, builder, output_ids, references):
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path = output_path.with_name(output_path.name + ".bin")
    payload = array.array("f")
    input_ranges = {}
    for record in builder.inputs:
        offset = len(payload)
        payload.extend(_payload_values(record.tensor))
        input_ranges[record.value_id] = (offset, record.tensor.numel())

    output_ranges = []
    for reference in references:
        offset = len(payload)
        values = _payload_values(reference)
        payload.extend(values)
        output_ranges.append((offset, len(values)))

    with payload_path.open("wb") as output:
        payload.tofile(output)

    lines = [
        "TMC_PYTORCH_GRAPH 1",
        f"MODEL {_quoted(model_name)}",
        f"PAYLOAD {_quoted(payload_path.name)}",
        f"VALUES {builder.next_value}",
        f"INPUTS {len(builder.inputs)}",
    ]
    for record in builder.inputs:
        offset, count = input_ranges[record.value_id]
        shape = " ".join(str(value) for value in record.shape)
        suffix = f" {shape}" if shape else ""
        lines.append(
            f"INPUT {record.value_id} {_quoted(record.name)} {record.kind} "
            f"{record.dtype} {len(record.shape)}{suffix} {offset} {count}"
        )
    lines.append(f"NODES {len(builder.nodes)}")
    for record in builder.nodes:
        operands = " ".join(str(value) for value in record.inputs)
        suffix = f" {operands}" if operands else ""
        lines.append(
            f"NODE {record.value_id} {_quoted(record.name)} {record.op} "
            f"{len(record.inputs)}{suffix} {record.attributes}"
        )
    lines.append(f"OUTPUTS {len(output_ids)}")
    for value_id, reference, payload_range in zip(output_ids, references, output_ranges):
        shape = tuple(int(value) for value in reference.shape)
        dtype = _dtype_token(reference.dtype)
        dimensions = " ".join(str(value) for value in shape)
        suffix = f" {dimensions}" if dimensions else ""
        offset, count = payload_range
        lines.append(
            f"OUTPUT {value_id} {dtype} {len(shape)}{suffix} {offset} {count}"
        )
    lines.append("END")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload_path

def _build_external_model(project_root, model_path):
    model_path = Path(model_path)

    if not model_path.is_absolute():
        model_path = project_root / model_path

    if not model_path.exists():
        raise ExportError(f"Model file does not exist: {model_path}")

    spec = importlib.util.spec_from_file_location(
        f"tmc_workload_{model_path.stem}",
        model_path,
    )

    if spec is None or spec.loader is None:
        raise ExportError(f"Unable to load model file: {model_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "export_case"):
        raise ExportError(
            f"{model_path} must define export_case() returning (model, inputs)"
        )

    model, inputs = module.export_case()

    if not isinstance(model, torch.nn.Module):
        raise ExportError("export_case() must return a torch.nn.Module")

    if not isinstance(inputs, tuple):
        raise ExportError("export_case() inputs must be a tuple")

    model = model.eval()

    with torch.no_grad():
        reference = model(*inputs)
        exported = torch.export.export(model, inputs)

    return model_path.stem, exported, inputs, reference

def main():
    parser = argparse.ArgumentParser(
        description="Export the static Transformer workload for TensorMetalCompiler"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/transformer.tmc"),
        help="output graph manifest path",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="PyTorch workload file defining export_case()",
    )
    
    arguments = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    if arguments.model is None:
        model_name = "transformer"
        exported, inputs, reference = _build_transformer(project_root)
    else:
        model_name, exported, inputs, reference = _build_external_model(
            project_root,
            arguments.model,
        )
    builder, output_ids = ExportNormalizer(exported, inputs).normalize()

    references = [value for value in _flatten(reference) if isinstance(value, torch.Tensor)]
    if len(references) != len(output_ids):
        raise ExportError("PyTorch reference outputs do not match exported graph outputs")
    payload = _write_archive(
        # arguments.output, "transformer", builder, output_ids, references
        arguments.output, model_name, builder, output_ids, references
    )
    print(f"PyTorch export: PASS")
    print(f"Manifest: {arguments.output.resolve()}")
    print(f"Payload: {payload}")
    print(f"Inputs: {len(builder.inputs)}")
    print(f"Nodes: {len(builder.nodes)}")
    print(f"Outputs: {len(output_ids)}")

if __name__ == "__main__":
    main()
