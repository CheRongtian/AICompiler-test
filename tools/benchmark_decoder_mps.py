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


def _load_exporter(project_root):
    path = project_root / "tools" / "export_decoder_llm.py"
    spec = importlib.util.spec_from_file_location("tmc_decoder_export", path)
    if spec is None or spec.loader is None:
        raise BenchmarkError(f"Unable to load decoder exporter: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_case(project_root, arguments):
    module = _load_workload(project_root)
    exporter = _load_exporter(project_root)
    try:
        state, checkpoint_module = exporter._load_checkpoint(arguments.checkpoint)
        inferred = exporter._infer_config(state, checkpoint_module)
        config = exporter._resolve_config(arguments, inferred)
        exporter._validate_checkpoint_shapes(state, config)
    except exporter.ExportError as error:
        raise BenchmarkError(str(error)) from error

    torch.manual_seed(arguments.seed)
    model = module.DecoderOnlyLLM(
        vocabulary_size=config["vocabulary_size"],
        hidden_size=config["hidden_size"],
        heads=config["heads"],
        layers=config["layers"],
        intermediate_size=config["intermediate_size"],
        capacity=config["capacity"],
        epsilon=config["epsilon"],
    ).eval()
    if state:
        try:
            missing, unexpected = model.load_state_dict(state, strict=False)
        except RuntimeError as error:
            raise BenchmarkError(
                f"Checkpoint parameters are incompatible: {error}"
            ) from error
        missing = [
            name
            for name in missing
            if not name.endswith("attention.rope_cosine")
            and not name.endswith("attention.rope_sine")
        ]
        if missing or unexpected:
            raise BenchmarkError(
                "Checkpoint keys do not match DecoderOnlyLLM; "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )

    tokens = torch.randint(
        0,
        config["vocabulary_size"],
        (config["batch"], config["prefill_length"]),
    )
    try:
        model, arguments.effective_dtype = exporter._choose_precision(
            model, arguments.effective_dtype or arguments.dtype, checkpoint_module,
            tokens, config["decode_count"])
    except exporter.ExportError as error:
        raise BenchmarkError(str(error)) from error
    return model, tokens, config["decode_count"], config


def _timed_mps(call):
    torch.mps.synchronize()
    start = time.perf_counter_ns()
    result = call()
    torch.mps.synchronize()
    return result, (time.perf_counter_ns() - start) / 1_000.0


@torch.no_grad()
def _run_generation(model, prefill_tokens, decode_steps, timed, capture_outputs,
                    token_only=False):
    per_step = []
    decode_times = []
    peak_tensor_bytes = 0
    peak_driver_bytes = 0
    total_start = time.perf_counter_ns() if timed else 0

    def sample_memory():
        nonlocal peak_tensor_bytes, peak_driver_bytes
        if timed:
            peak_tensor_bytes = max(
                peak_tensor_bytes, torch.mps.current_allocated_memory()
            )
            peak_driver_bytes = max(
                peak_driver_bytes, torch.mps.driver_allocated_memory()
            )

    def prefill_call():
        device_logits, caches = model(prefill_tokens)
        device_token = device_logits[:, -1, :].argmax(dim=-1)
        host_logits = None if token_only else device_logits.detach().cpu()
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
    sample_memory()
    if capture_outputs:
        per_step.append((logits, host_token))

    for _ in range(decode_steps):
        current_token = next_token.unsqueeze(1)

        def decode_call():
            device_logits, next_caches = model(current_token, caches)
            device_token = device_logits[:, -1, :].argmax(dim=-1)
            host_logits = None if token_only else device_logits.detach().cpu()
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
        sample_memory()
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
        "peak_tensor_bytes": peak_tensor_bytes,
        "peak_driver_bytes": peak_driver_bytes,
    }


def _validate(actual, expected, absolute_tolerance, relative_tolerance):
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
        if not torch.allclose(
            actual_logits,
            expected_logits,
            atol=absolute_tolerance,
            rtol=relative_tolerance,
        ):
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
            if not torch.allclose(
                actual_cache,
                expected_cache,
                atol=absolute_tolerance,
                rtol=relative_tolerance,
            ):
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
    if not values:
        print(f"  {label}: unavailable (no decode steps)")
        return
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
    parser.add_argument("--token-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--effective-dtype", choices=("fp32", "fp16", "bf16"), default=None)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--hidden-size", dest="hidden_size", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument(
        "--intermediate-size", dest="intermediate_size", type=int, default=None
    )
    parser.add_argument("--vocab-size", dest="vocabulary_size", type=int, default=None)
    parser.add_argument("--capacity", type=int, default=None)
    parser.add_argument(
        "--prefill-length", dest="prefill_length", type=int, default=None
    )
    parser.add_argument("--decode-count", dest="decode_count", type=int, default=None)
    parser.add_argument("--epsilon", type=float, default=None)
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
    cpu_model, cpu_tokens, decode_steps, config = _build_case(
        project_root, args
    )
    comparison_dtype = args.effective_dtype or args.dtype
    absolute_tolerance, relative_tolerance = (
        (1e-3, 5e-3) if comparison_dtype == "fp32" else (2e-2, 2e-2)
    )
    expected = _run_generation(
        cpu_model, cpu_tokens, decode_steps, timed=False, capture_outputs=True
    )
    mps_model = copy.deepcopy(cpu_model).to("mps").eval()
    mps_tokens = cpu_tokens.to("mps")

    actual = _run_generation(
        mps_model, mps_tokens, decode_steps, timed=True, capture_outputs=True
    )
    max_error = _validate(
        actual, expected, absolute_tolerance, relative_tolerance
    )
    print(f"PyTorch MPS correctness: PASS, max absolute error={max_error:.9g}")
    effective_dtype = args.effective_dtype or args.dtype
    print(
        "PyTorch MPS configuration: "
        f"requested_dtype={args.dtype}, effective_dtype={effective_dtype}, "
        f"hidden={config['hidden_size']}, heads={config['heads']}, "
        f"layers={config['layers']}, intermediate={config['intermediate_size']}, "
        f"vocab={config['vocabulary_size']}, prefill={config['prefill_length']}, "
        f"decode={config['decode_count']}"
    )

    for _ in range(args.warmup):
        _run_generation(
            mps_model, mps_tokens, decode_steps, timed=True, capture_outputs=False,
            token_only=args.token_only
        )

    prefill_times = []
    decode_times = []
    total_times = []
    peak_tensor_bytes = 0
    peak_driver_bytes = 0
    for _ in range(args.samples):
        result = _run_generation(
            mps_model, mps_tokens, decode_steps, timed=True, capture_outputs=False,
            token_only=args.token_only
        )
        prefill_times.append(result["prefill_us"])
        decode_times.extend(result["decode_us"])
        total_times.append(result["total_us"])
        peak_tensor_bytes = max(peak_tensor_bytes, result["peak_tensor_bytes"])
        peak_driver_bytes = max(peak_driver_bytes, result["peak_driver_bytes"])

    print("PyTorch MPS benchmark (synchronized end-to-end microseconds):")
    print("Scope: " + ("token readback" if args.token_only else "logits/token readback")
          + " plus autoregressive token feedback")
    print(f"  MPS sampled peak tensor allocation (bytes): {peak_tensor_bytes}")
    print(f"  MPS sampled peak driver allocation (bytes): {peak_driver_bytes}")
    print(f"Warmup runs: {args.warmup}, measured runs: {args.samples}")
    _report("TTFT / prefill time (us)", prefill_times)
    _report("TPOT / decode time (us/token)", decode_times)
    _report("Total generation time (us)", total_times)
    if decode_times:
        decode_median = statistics.median(decode_times)
        print(f"  Decode throughput (tokens/s): {1_000_000.0 / decode_median:.3f}")
    print("PyTorch MPS benchmark: PASS")


if __name__ == "__main__":
    try:
        main()
    except BenchmarkError as error:
        raise SystemExit(f"PyTorch MPS benchmark: FAIL\nBenchmark error: {error}")
