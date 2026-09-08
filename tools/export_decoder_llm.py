#!/usr/bin/env python3

import argparse
import array
import importlib.util
import json
from pathlib import Path

import torch


class ExportError(RuntimeError):
    pass


def _load_workload(project_root):
    path = project_root / "workloads" / "pytorch" / "decoder_llm.py"
    spec = importlib.util.spec_from_file_location("tmc_decoder_llm", path)
    if spec is None or spec.loader is None:
        raise ExportError(f"Unable to load decoder workload: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _values(tensor):
    return tensor.detach().cpu().contiguous().to(torch.float32).reshape(-1).tolist()


def _quoted(value):
    return json.dumps(value, ensure_ascii=False)


def _collect_reference(tensors, prefix, token_ids, logits, next_tokens, caches):
    tensors[f"{prefix}.tokens"] = token_ids
    tensors[f"{prefix}.logits"] = logits
    tensors[f"{prefix}.next_tokens"] = next_tokens
    for layer_index, (key_cache, value_cache) in enumerate(caches):
        tensors[f"{prefix}.layer{layer_index}.key_cache"] = key_cache
        tensors[f"{prefix}.layer{layer_index}.value_cache"] = value_cache


@torch.no_grad()
def _make_workload(project_root):
    module = _load_workload(project_root)
    torch.manual_seed(11)
    batch = 1
    hidden_size = 64
    heads = 4
    intermediate_size = 128
    vocabulary_size = 128
    layer_count = 2
    prefill_length = 8
    decode_count = 8
    capacity = 32
    epsilon = 1e-5

    model = module.DecoderOnlyLLM(
        vocabulary_size=vocabulary_size,
        hidden_size=hidden_size,
        heads=heads,
        layers=layer_count,
        intermediate_size=intermediate_size,
        capacity=capacity,
        epsilon=epsilon,
    ).eval()
    tensors = {
        "embedding.weight": model.embedding.weight,
        "final_norm.weight": model.final_norm.weight,
        "lm_head.weight": model.lm_head.weight,
        "rope.cosine": model.layers[0].attention.rope_cosine,
        "rope.sine": model.layers[0].attention.rope_sine,
    }
    for layer_index, layer in enumerate(model.layers):
        prefix = f"layer{layer_index}"
        tensors[f"{prefix}.input_norm.weight"] = layer.input_norm.weight
        tensors[f"{prefix}.q.weight"] = layer.attention.q_proj.weight
        tensors[f"{prefix}.k.weight"] = layer.attention.k_proj.weight
        tensors[f"{prefix}.v.weight"] = layer.attention.v_proj.weight
        tensors[f"{prefix}.o.weight"] = layer.attention.o_proj.weight
        tensors[f"{prefix}.post_norm.weight"] = layer.post_attention_norm.weight
        tensors[f"{prefix}.gate.weight"] = layer.mlp.gate_proj.weight
        tensors[f"{prefix}.up.weight"] = layer.mlp.up_proj.weight
        tensors[f"{prefix}.down.weight"] = layer.mlp.down_proj.weight

    token_ids = torch.randint(0, vocabulary_size, (batch, prefill_length))
    logits, caches = model(token_ids)
    next_tokens = logits[:, -1, :].argmax(dim=-1)
    _collect_reference(tensors, "prefill", token_ids, logits, next_tokens, caches)

    current_tokens = next_tokens.unsqueeze(1)
    for step in range(decode_count):
        logits, caches = model(current_tokens, caches)
        next_tokens = logits[:, -1, :].argmax(dim=-1)
        _collect_reference(
            tensors, f"decode{step}", current_tokens, logits, next_tokens, caches
        )
        current_tokens = next_tokens.unsqueeze(1)

    config = (
        batch,
        hidden_size,
        heads,
        intermediate_size,
        vocabulary_size,
        layer_count,
        capacity,
        prefill_length,
        decode_count,
        epsilon,
    )
    return config, tensors


def _write_archive(path, config, tensors):
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_path = path.with_name(path.name + ".bin")
    payload = array.array("f")
    ranges = {}
    for name, tensor in tensors.items():
        offset = len(payload)
        values = _values(tensor)
        payload.extend(values)
        ranges[name] = (offset, len(values))
    with payload_path.open("wb") as output:
        payload.tofile(output)

    integer_config = config[:-1]
    epsilon = config[-1]
    lines = [
        "TMC_DECODER_LLM 1",
        f"MODEL {_quoted('DecoderOnlyLLM')}",
        f"PAYLOAD {_quoted(payload_path.name)}",
        "CONFIG "
        + " ".join(str(value) for value in integer_config)
        + f" {epsilon:.9g}",
        f"TENSORS {len(ranges)}",
    ]
    for name, (offset, count) in ranges.items():
        lines.append(f"TENSOR {_quoted(name)} {offset} {count}")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload_path


def main():
    parser = argparse.ArgumentParser(
        description="Export the decoder-only LLM workload and PyTorch references"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("build/decoder_llm.tmc")
    )
    arguments = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config, tensors = _make_workload(project_root)
    payload = _write_archive(arguments.output, config, tensors)
    print("Decoder-only export: PASS")
    print(f"Manifest: {arguments.output.resolve()}")
    print(f"Payload: {payload}")


if __name__ == "__main__":
    try:
        main()
    except ExportError as error:
        raise SystemExit(f"Decoder-only export: FAIL\nExport error: {error}")
