#!/usr/bin/env python3
"""Benchmark the decoder-only reference with synchronized PyTorch MPS timing."""

import argparse
import copy
import importlib.util
import math
import statistics
import time
from pathlib import Path

import torch


class BenchmarkError(RuntimeError):
    pass


def _load_workload(project_root):
    path = project_root / "workloads" / "pytorch" / "decoder_llm.py"
    spec = importlib.util.spec_from_file_location("tmc_decoder_mps", path)
    if spec is None or spec.loader is None:
        raise BenchmarkError(f"Unable to load decoder workload: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_case(project_root):
    module = _load_workload(project_root)
    torch.manual_seed(11)
    model = module.DecoderOnlyLLM(
        vocabulary_size=128,
        hidden_size=64,
        heads=4,
        layers=2,
        intermediate_size=128,
        capacity=32,
        epsilon=1e-5,
    ).eval()
    tokens = torch.randint(0, 128, (1, 8))
    return model, tokens, 8


def _timed_mps(call):
    torch.mps.synchronize()
    start = time.perf_counter_ns()
    result = call()
    torch.mps.synchronize()
    return result, (time.perf_counter_ns() - start) / 1_000.0


@torch.no_grad()
def _run_generation(model, prefill_tokens, decode_steps, timed, capture_outputs):
    per_step = []
    decode_times = []
    total_start = time.perf_counter_ns() if timed else 0

    def prefill_call():
        device_logits, caches = model(prefill_tokens)
        device_token = device_logits[:, -1, :].argmax(dim=-1)
        host_logits = device_logits.detach().cpu()
        host_token = device_token.detach().cpu()
        next_device_token = host_token.to(device_logits.device)
        return host_logits, caches, next_device_token, host_token

    if timed:
        (logits, caches, next_token, host_token), prefill_time = _timed_mps(
            prefill_call
        )
    else:
        logits, caches, next_token, host_token = prefill_call()
        prefill_time = None
    if capture_outputs:
        per_step.append((logits, host_token))

    for _ in range(decode_steps):
        current_token = next_token.unsqueeze(1)

        def decode_call():
            device_logits, next_caches = model(current_token, caches)
            device_token = device_logits[:, -1, :].argmax(dim=-1)
            host_logits = device_logits.detach().cpu()
            host_token = device_token.detach().cpu()
            next_device_token = host_token.to(device_logits.device)
            return host_logits, next_caches, next_device_token, host_token

        if timed:
            (logits, caches, next_token, host_token), elapsed = _timed_mps(
                decode_call
            )
            decode_times.append(elapsed)
        else:
            logits, caches, next_token, host_token = decode_call()
        if capture_outputs:
            per_step.append((logits, host_token))

    total_time = (
        (time.perf_counter_ns() - total_start) / 1_000.0 if timed else None
    )
    cpu_caches = (
        [(key.detach().cpu(), value.detach().cpu()) for key, value in caches]
        if capture_outputs
        else []
    )
    return {
        "steps": per_step,
        "caches": cpu_caches,
        "prefill_us": prefill_time,
        "decode_us": decode_times,
        "total_us": total_time,
    }


def _validate(actual, expected):
    if len(actual["steps"]) != len(expected["steps"]):
        raise BenchmarkError("MPS and CPU produced different step counts.")
    max_error = 0.0
    for index, ((actual_logits, actual_token), (expected_logits, expected_token)) in enumerate(
        zip(actual["steps"], expected["steps"])
    ):
        if not torch.equal(actual_token, expected_token):
            raise BenchmarkError(f"MPS token mismatch at generation step {index}.")
        error = (actual_logits - expected_logits).abs().max().item()
        max_error = max(max_error, error)
        if not torch.allclose(actual_logits, expected_logits, atol=1e-3, rtol=5e-3):
            raise BenchmarkError(
                f"MPS logits mismatch at generation step {index}; max error={error}."
            )
    for layer, ((actual_key, actual_value), (expected_key, expected_value)) in enumerate(
        zip(actual["caches"], expected["caches"])
    ):
        for name, actual_cache, expected_cache in (
            ("key", actual_key, expected_key),
            ("value", actual_value, expected_value),
        ):
            error = (actual_cache - expected_cache).abs().max().item()
            max_error = max(max_error, error)
            if not torch.allclose(actual_cache, expected_cache, atol=1e-3, rtol=5e-3):
                raise BenchmarkError(
                    f"MPS layer {layer} {name} cache mismatch; max error={error}."
                )
    return max_error


def _percentile(values, fraction):
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _report(label, values):
    print(
        f"  {label}: p50={statistics.median(values):.3f}, "
        f"p90={_percentile(values, 0.9):.3f}, min={min(values):.3f}, "
        f"max={max(values):.3f}, samples={len(values)}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark the decoder-only PyTorch model on MPS"
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    if args.warmup < 2 or args.samples < 10:
        raise BenchmarkError(
            "Benchmark requires at least 2 warmup runs and 10 measured runs."
        )
    if not torch.backends.mps.is_available():
        print("PyTorch MPS benchmark: SKIP")
        print("Reason: torch.backends.mps.is_available() is false.")
        return

    project_root = Path(__file__).resolve().parents[1]
    cpu_model, cpu_tokens, decode_steps = _build_case(project_root)
    expected = _run_generation(
        cpu_model, cpu_tokens, decode_steps, timed=False, capture_outputs=True
    )
    mps_model = copy.deepcopy(cpu_model).to("mps").eval()
    mps_tokens = cpu_tokens.to("mps")

    actual = _run_generation(
        mps_model, mps_tokens, decode_steps, timed=True, capture_outputs=True
    )
    max_error = _validate(actual, expected)
    print(f"PyTorch MPS correctness: PASS, max absolute error={max_error:.9g}")

    for _ in range(args.warmup):
        _run_generation(
            mps_model, mps_tokens, decode_steps, timed=True, capture_outputs=False
        )

    prefill_times = []
    decode_times = []
    total_times = []
    for _ in range(args.samples):
        result = _run_generation(
            mps_model, mps_tokens, decode_steps, timed=True, capture_outputs=False
        )
        prefill_times.append(result["prefill_us"])
        decode_times.extend(result["decode_us"])
        total_times.append(result["total_us"])

    print("PyTorch MPS benchmark (synchronized end-to-end microseconds):")
    print("Scope: logits/token readback plus autoregressive token feedback")
    print(f"Warmup runs: {args.warmup}, measured runs: {args.samples}")
    _report("Prefill time (us)", prefill_times)
    _report("Decode time (us/token)", decode_times)
    _report("Total generation time (us)", total_times)
    decode_median = statistics.median(decode_times)
    print(f"  Decode throughput (tokens/s): {1_000_000.0 / decode_median:.3f}")
    print("PyTorch MPS benchmark: PASS")


if __name__ == "__main__":
    try:
        main()
    except BenchmarkError as error:
        raise SystemExit(f"PyTorch MPS benchmark: FAIL\nBenchmark error: {error}")
