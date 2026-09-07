#!/usr/bin/env python3

import argparse
import array
import copy
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import torch


class ExportError(RuntimeError):
    pass


@dataclass
class DecodeReference:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    output: torch.Tensor
    key_cache: torch.Tensor
    value_cache: torch.Tensor


def _quoted(value):
    return json.dumps(value, ensure_ascii=False)


def _load_workload(project_root):
    path = project_root / "workloads" / "pytorch" / "KVCache.py"
    spec = importlib.util.spec_from_file_location("tmc_kv_cache_workload", path)
    if spec is None or spec.loader is None:
        raise ExportError(f"Unable to load KV cache workload: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_exported_patterns(prefill, decode):
    prefill_cats = [
        node for node in prefill.graph_module.graph.nodes
        if node.op == "call_function" and str(node.target) == "aten.cat.default"
    ]
    if prefill_cats:
        raise ExportError("Prefill graph unexpectedly contains aten.cat")

    decode_cats = [
        node for node in decode.graph_module.graph.nodes
        if node.op == "call_function" and str(node.target) == "aten.cat.default"
    ]
    if len(decode_cats) != 2:
        targets = sorted({
            str(node.target) for node in decode.graph_module.graph.nodes
            if node.op == "call_function"
        })
        raise ExportError(
            "Decode graph must contain key/value aten.cat nodes; observed targets: "
            + ", ".join(targets)
        )
    for node in decode_cats:
        tensors = node.args[0]
        axis = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        if not isinstance(tensors, (tuple, list)) or len(tensors) != 2 or int(axis) != 2:
            raise ExportError(
                f"Unsupported cache append at node '{node.name}': expected two tensors on axis 2"
            )


def _make_references(project_root):
    workload = _load_workload(project_root)
    torch.manual_seed(0)
    batch = 1
    heads = 4
    head_dimension = 16
    dimension = heads * head_dimension
    prefill_length = 7
    decode_count = 8
    capacity = prefill_length + decode_count

    attention = workload.MultiHeadAttentionKVCache(
        dim=dimension, n_heads=heads, use_kv=True
    ).eval()
    prefill_query = torch.randn(batch, prefill_length, dimension)
    prefill_key = torch.randn(batch, prefill_length, dimension)
    prefill_value = torch.randn(batch, prefill_length, dimension)

    with torch.no_grad():
        prefill_output, (key_cache, value_cache) = attention(
            prefill_query, prefill_key, prefill_value
        )
        prefill_key_cache = key_cache
        prefill_value_cache = value_cache

        prefill_wrapper = workload.KVCachePrefill(copy.deepcopy(attention)).eval()
        decode_wrapper = workload.KVCacheDecode(copy.deepcopy(attention)).eval()
        prefill_export = torch.export.export(
            prefill_wrapper, (prefill_query, prefill_key, prefill_value)
        )

        sample_query = torch.randn(batch, 1, dimension)
        sample_key = torch.randn(batch, 1, dimension)
        sample_value = torch.randn(batch, 1, dimension)
        decode_export = torch.export.export(
            decode_wrapper,
            (sample_query, sample_key, sample_value, key_cache, value_cache),
        )
        _validate_exported_patterns(prefill_export, decode_export)

        steps = []
        for _ in range(decode_count):
            query = torch.randn(batch, 1, dimension)
            key = torch.randn(batch, 1, dimension)
            value = torch.randn(batch, 1, dimension)
            output, (key_cache, value_cache) = attention(
                query, key, value, past_key=key_cache, past_value=value_cache
            )
            steps.append(
                DecodeReference(query, key, value, output, key_cache, value_cache)
            )

    config = {
        "batch": batch,
        "heads": heads,
        "head_dimension": head_dimension,
        "capacity": capacity,
        "prefill_length": prefill_length,
        "decode_count": decode_count,
    }
    tensors = {
        "query_weight": attention.q_proj.weight,
        "key_weight": attention.k_proj.weight,
        "value_weight": attention.v_proj.weight,
        "output_weight": attention.o_proj.weight,
        "prefill_query": prefill_query,
        "prefill_key": prefill_key,
        "prefill_value": prefill_value,
        "prefill_output": prefill_output,
        "prefill_key_cache": prefill_key_cache,
        "prefill_value_cache": prefill_value_cache,
    }
    return config, tensors, steps


def _values(tensor):
    return tensor.detach().cpu().contiguous().to(torch.float32).reshape(-1).tolist()


def _write_archive(output_path, config, tensors, steps):
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path = output_path.with_name(output_path.name + ".bin")
    payload = array.array("f")

    def append(tensor):
        offset = len(payload)
        values = _values(tensor)
        payload.extend(values)
        return offset, len(values)

    tensor_ranges = {name: append(tensor) for name, tensor in tensors.items()}
    step_ranges = []
    for step in steps:
        step_ranges.append([
            append(step.query),
            append(step.key),
            append(step.value),
            append(step.output),
            append(step.key_cache),
            append(step.value_cache),
        ])

    with payload_path.open("wb") as output:
        payload.tofile(output)

    lines = [
        "TMC_KV_CACHE 1",
        f"MODEL {_quoted('MultiHeadAttentionKVCache')}",
        f"PAYLOAD {_quoted(payload_path.name)}",
        "CONFIG {batch} {heads} {head_dimension} {capacity} {prefill_length} {decode_count}".format(
            **config
        ),
        f"TENSORS {len(tensor_ranges)}",
    ]
    for name, (offset, count) in tensor_ranges.items():
        lines.append(f"TENSOR {_quoted(name)} {offset} {count}")
    lines.append(f"STEPS {len(step_ranges)}")
    for index, ranges in enumerate(step_ranges):
        flattened = " ".join(
            f"{offset} {count}" for offset, count in ranges
        )
        lines.append(f"STEP {index} {flattened}")
    lines.append("END")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload_path


def main():
    parser = argparse.ArgumentParser(
        description="Export KVCache.py prefill/decode references for the Metal stateful runtime"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/kv_cache.tmc"),
        help="output KV cache manifest path",
    )
    arguments = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config, tensors, steps = _make_references(project_root)
    payload = _write_archive(arguments.output, config, tensors, steps)
    print("KV cache export: PASS")
    print(f"Manifest: {arguments.output.resolve()}")
    print(f"Payload: {payload}")
    print(
        "Config: batch={batch}, heads={heads}, head_dim={head_dimension}, "
        "prefill={prefill_length}, decode_steps={decode_count}, capacity={capacity}".format(
            **config
        )
    )


if __name__ == "__main__":
    main()
