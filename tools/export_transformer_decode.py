#!/usr/bin/env python3

import argparse
import array
import importlib.util
import json
import sys
from pathlib import Path

import torch


class ExportError(RuntimeError):
    pass


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ExportError(f"Unable to load workload: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.path[0]
    return module


def _values(tensor):
    return tensor.detach().cpu().contiguous().to(torch.float32).reshape(-1).tolist()


def _quoted(value):
    return json.dumps(value, ensure_ascii=False)


def _collect_layer(tensors, prefix, layer):
    self_attention = layer.self_attn
    cross_attention = layer.cross_attn
    feed_forward = layer.ffn

    tensors[f"{prefix}.self.q.weight"] = self_attention.q_proj.weight
    tensors[f"{prefix}.self.k.weight"] = self_attention.k_proj.weight
    tensors[f"{prefix}.self.v.weight"] = self_attention.v_proj.weight
    tensors[f"{prefix}.self.o.weight"] = self_attention.o_proj.weight
    tensors[f"{prefix}.self.norm.weight"] = layer.self_attn_norm.weight
    tensors[f"{prefix}.self.norm.bias"] = layer.self_attn_norm.bias

    tensors[f"{prefix}.cross.q.weight"] = cross_attention.W_q.weight
    tensors[f"{prefix}.cross.q.bias"] = cross_attention.W_q.bias
    tensors[f"{prefix}.cross.k.weight"] = cross_attention.W_k.weight
    tensors[f"{prefix}.cross.k.bias"] = cross_attention.W_k.bias
    tensors[f"{prefix}.cross.v.weight"] = cross_attention.W_v.weight
    tensors[f"{prefix}.cross.v.bias"] = cross_attention.W_v.bias
    tensors[f"{prefix}.cross.o.weight"] = cross_attention.fc.weight
    tensors[f"{prefix}.cross.o.bias"] = cross_attention.fc.bias
    tensors[f"{prefix}.cross.norm.weight"] = cross_attention.norm.weight
    tensors[f"{prefix}.cross.norm.bias"] = cross_attention.norm.bias

    tensors[f"{prefix}.ffn.in.weight"] = feed_forward.fc1.weight
    tensors[f"{prefix}.ffn.in.bias"] = feed_forward.fc1.bias
    tensors[f"{prefix}.ffn.out.weight"] = feed_forward.fc2.weight
    tensors[f"{prefix}.ffn.out.bias"] = feed_forward.fc2.bias
    tensors[f"{prefix}.ffn.norm.weight"] = feed_forward.norm.weight
    tensors[f"{prefix}.ffn.norm.bias"] = feed_forward.norm.bias


def _collect_reference(tensors, prefix, token_ids, logits, next_tokens, caches):
    tensors[f"{prefix}.tokens"] = token_ids
    tensors[f"{prefix}.logits"] = logits
    tensors[f"{prefix}.next_tokens"] = next_tokens
    for layer_index, (key_cache, value_cache) in enumerate(caches):
        tensors[f"{prefix}.layer{layer_index}.key_cache"] = key_cache
        tensors[f"{prefix}.layer{layer_index}.value_cache"] = value_cache


@torch.no_grad()
def _make_workload(project_root):
    workload_root = project_root / "workloads" / "pytorch"
    benchmark = _load_module(
        "tmc_transformer_decode", workload_root / "transformer_kv_benchmark.py"
    )

    torch.manual_seed(7)
    batch = 1
    heads = 2
    head_dimension = 4
    dimension = heads * head_dimension
    feed_forward_dimension = 16
    vocabulary_size = 32
    source_length = 5
    layer_count = 2
    prefill_length = 4
    decode_count = 4
    capacity = prefill_length + decode_count

    model = benchmark.TransformerWithKVCache(
        src_vocab=vocabulary_size,
        tgt_vocab=vocabulary_size,
        d_model=dimension,
        n_heads=heads,
        num_encoder_layers=1,
        num_decoder_layers=layer_count,
        d_ff=feed_forward_dimension,
        dropout=0.0,
        max_len=capacity,
    ).eval()
    memory = torch.randn(batch, source_length, dimension)
    tokens = torch.randint(0, vocabulary_size, (batch, prefill_length))
    causal_mask = torch.triu(
        torch.ones(prefill_length, prefill_length, dtype=torch.bool), diagonal=1
    )

    logits, caches = model.decode_step(
        tokens,
        memory,
        layer_past_key_values=None,
        step=0,
        self_mask=causal_mask,
    )
    next_tokens = logits[:, -1, :].argmax(dim=-1)

    tensors = {
        "encoder_memory": memory,
        "embedding.weight": model.decoder.embedding.weight,
        "position.table": model.decoder.pos_encoding[:, :capacity, :],
        "lm_head.weight": model.decoder.fc_out.weight,
        "lm_head.bias": model.decoder.fc_out.bias,
    }
    for layer_index, layer in enumerate(model.decoder.layers):
        _collect_layer(tensors, f"layer{layer_index}", layer)
    _collect_reference(tensors, "prefill", tokens, logits, next_tokens, caches)

    current_tokens = next_tokens.unsqueeze(1)
    for step_index in range(decode_count):
        logits, caches = model.decode_step(
            current_tokens,
            memory,
            layer_past_key_values=caches,
            step=prefill_length + step_index,
        )
        next_tokens = logits[:, -1, :].argmax(dim=-1)
        _collect_reference(
            tensors,
            f"decode{step_index}",
            current_tokens,
            logits,
            next_tokens,
            caches,
        )
        current_tokens = next_tokens.unsqueeze(1)

    config = (
        batch,
        heads,
        head_dimension,
        feed_forward_dimension,
        vocabulary_size,
        source_length,
        capacity,
        prefill_length,
        decode_count,
        layer_count,
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

    lines = [
        "TMC_TRANSFORMER_DECODE 1",
        f"MODEL {_quoted('TransformerWithKVCache')}",
        f"PAYLOAD {_quoted(payload_path.name)}",
        "CONFIG " + " ".join(str(value) for value in config),
        f"TENSORS {len(ranges)}",
    ]
    for name, (offset, count) in ranges.items():
        lines.append(f"TENSOR {_quoted(name)} {offset} {count}")
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload_path


def main():
    parser = argparse.ArgumentParser(
        description="Export a complete cached Transformer decoder reference"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("build/transformer_decode.tmc")
    )
    arguments = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config, tensors = _make_workload(project_root)
    payload = _write_archive(arguments.output, config, tensors)
    print("Transformer decode export: PASS")
    print(f"Manifest: {arguments.output.resolve()}")
    print(f"Payload: {payload}")


if __name__ == "__main__":
    try:
        main()
    except ExportError as error:
        raise SystemExit(f"Transformer decode export: FAIL\nExport error: {error}")
