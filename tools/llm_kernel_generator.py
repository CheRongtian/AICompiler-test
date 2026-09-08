#!/usr/bin/env python3
"""Request pattern-specific Metal kernels; keep retry and admission local."""

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path


class GeneratorError(RuntimeError):
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUEST_TIMEOUT_SECONDS = 60.0
MAX_ATTEMPTS = 3
PATTERNS = (
    "silu_mul", "rope",
    "decoder_gemv_64_64", "decoder_gemv_64_128", "decoder_gemv_128_64",
    "decoder_rope", "decoder_residual_rmsnorm", "decoder_gated_mlp",
)


def _load_env(path):
    if not path.exists():
        raise GeneratorError(f"Missing environment file: {path}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise GeneratorError(f"Unable to read environment file: {error}") from error
    for line_number, source_line in enumerate(lines, start=1):
        line = source_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            raise GeneratorError(f"Invalid .env entry on line {line_number}.")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise GeneratorError(f"Invalid .env key on line {line_number}.")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _call_endpoint(endpoint, api_key, session_id, messages):
    body = {
        "messages": messages,
        "sessionId": session_id,
        "source": "api",
        "extra": {},
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": api_key,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request, timeout=REQUEST_TIMEOUT_SECONDS
        ) as response:
            chunks = []
            ended = False
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data:
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError as error:
                    raise GeneratorError(
                        f"Invalid JSON in SSE data event: {error}"
                    ) from error
                content = event.get("content")
                if isinstance(content, str):
                    chunks.append(content)
                if "end" in event:
                    ended = True
                    break
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise GeneratorError(
            f"LLM endpoint returned HTTP {error.code}: {detail}"
        ) from error
    except (urllib.error.URLError, TimeoutError, UnicodeDecodeError) as error:
        raise GeneratorError(f"LLM request failed: {error}") from error

    if not ended:
        raise GeneratorError("SSE stream closed before the end event.")
    if not chunks:
        raise GeneratorError("SSE stream completed without assistant content.")
    return "".join(chunks).strip()


def _read_json(path, label):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GeneratorError(f"Unable to read {label}: {error}") from error


def _write_text(path, value):
    try:
        path.write_text(value + "\n", encoding="utf-8")
    except OSError as error:
        raise GeneratorError(f"Unable to write {path}: {error}") from error


def _run_compiler(arguments):
    completed = subprocess.run(arguments, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise GeneratorError(
            f"TensorMetalCompiler exited with status {completed.returncode}."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate and admit one LLM-authored Metal kernel."
    )
    parser.add_argument(
        "--compiler",
        type=Path,
        default=PROJECT_ROOT / "build" / "TensorMetalCompiler",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=PROJECT_ROOT / "build" / "generated_kernels",
    )
    parser.add_argument(
        "--env-file", type=Path, default=PROJECT_ROOT / ".env"
    )
    parser.add_argument("--pattern", choices=PATTERNS, default="silu_mul")
    args = parser.parse_args()

    _load_env(args.env_file)
    endpoint = os.environ.get("TMC_LLM_KERNEL_GENERATOR_URL")
    api_key = os.environ.get("TMC_LLM_API_KEY")
    if not endpoint:
        raise GeneratorError("Set TMC_LLM_KERNEL_GENERATOR_URL in .env.")
    if not api_key:
        raise GeneratorError("Set TMC_LLM_API_KEY in .env.")
    compiler = args.compiler.resolve()
    if not compiler.is_file():
        raise GeneratorError(f"Compiler executable does not exist: {compiler}")

    library = _read_json(PROJECT_ROOT / "config" / "kernel_principles.json", "principle library")
    if not isinstance(library, dict) or library.get("version") != 2:
        raise GeneratorError("Unsupported principle library version.")
    common = library.get("common")
    patterns = library.get("patterns")
    specific = patterns.get(args.pattern) if isinstance(patterns, dict) else None
    if not isinstance(common, list) or not isinstance(specific, list):
        raise GeneratorError("Missing common or pattern-specific principles.")
    if not all(isinstance(value, str) for value in common + specific):
        raise GeneratorError("Principles must be strings.")
    principles = {"version": library["version"], "common": common, "pattern": specific}

    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    contract_path = work_dir / "contract.json"
    candidate_path = work_dir / "candidate.json"
    feedback_path = work_dir / "feedback.json"
    admitted_path = work_dir / "admitted.json"

    _run_compiler(
        [str(compiler), "--emit-kernel-contract", str(contract_path), "--pattern", args.pattern]
    )
    try:
        contract_text = contract_path.read_text(encoding="utf-8").strip()
        contract = json.loads(contract_text)
    except (OSError, json.JSONDecodeError) as error:
        raise GeneratorError(f"Unable to read kernel contract: {error}") from error

    session_id = str(uuid.uuid4())
    previous_attempts = []
    feedback = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"Generated kernel attempt: {attempt}/{MAX_ATTEMPTS}")
        request = {
            "task": "Return the next GeneratedKernel JSON matching contract.response_schema.",
            "pattern": args.pattern,
            "contract": contract,
            "principles": principles,
            "attempt": attempt,
            "previous_attempts": previous_attempts,
            "compiler_feedback": feedback,
        }
        request_text = json.dumps(request, ensure_ascii=False, separators=(",", ":"))
        _write_text(work_dir / f"request_{attempt}.json", request_text)
        messages = [{"role": "user", "content": request_text}]
        try:
            assistant_content = _call_endpoint(
                endpoint, api_key, session_id, messages
            )
        except GeneratorError as error:
            print(f"LLM request unavailable: {error}")
            print("LLM generated kernel: FALLBACK")
            print("No new candidate was produced; existing admitted artifacts are unchanged.")
            return
        _write_text(candidate_path, assistant_content)

        _run_compiler(
            [
                str(compiler),
                "--admit-generated-kernel",
                str(candidate_path),
                "--feedback-output",
                str(feedback_path),
                "--pattern",
                args.pattern,
                "--artifact-output",
                str(admitted_path),
            ]
        )
        feedback = _read_json(feedback_path, "compiler feedback")
        status = feedback.get("status")
        if status == "admitted":
            if not admitted_path.is_file():
                raise GeneratorError("Compiler reported admission without an artifact.")
            print("LLM generated kernel: ADMITTED")
            print(f"Admitted kernel: {admitted_path}")
            return
        if status == "fatal":
            raise GeneratorError(feedback.get("message", "Fatal compiler failure."))
        if status != "retry":
            raise GeneratorError("Compiler feedback has an unknown status.")

        previous_attempts.append({
            "attempt": attempt,
            "response": assistant_content,
            "compiler_feedback": feedback,
        })

    print("LLM generated kernel: FALLBACK")
    print("Retry budget exhausted; no new kernel was admitted.")
    if admitted_path.exists():
        print("The existing admitted artifact is retained for decoder contract/device checks.")
    else:
        print("No admitted artifact exists; the decoder will use its template.")


if __name__ == "__main__":
    try:
        main()
    except GeneratorError as error:
        print(f"LLM kernel generator: FAIL\nGenerator error: {error}", file=sys.stderr)
        raise SystemExit(1)
