#!/usr/bin/env python3
"""Generate, validate, and retry one Metal SiLU + Mul kernel."""

import argparse
import json
import os
import shutil
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

    work_dir = args.work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    contract_path = work_dir / "contract.json"
    candidate_path = work_dir / "candidate.json"
    feedback_path = work_dir / "feedback.json"
    admitted_path = work_dir / "admitted_silu_mul.json"
    admitted_path.unlink(missing_ok=True)

    _run_compiler(
        [str(compiler), "--emit-kernel-contract", str(contract_path)]
    )
    try:
        contract_text = contract_path.read_text(encoding="utf-8").strip()
        json.loads(contract_text)
    except (OSError, json.JSONDecodeError) as error:
        raise GeneratorError(f"Unable to read kernel contract: {error}") from error

    session_id = str(uuid.uuid4())
    messages = [{"role": "user", "content": contract_text}]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"Generated kernel attempt: {attempt}/{MAX_ATTEMPTS}")
        try:
            assistant_content = _call_endpoint(
                endpoint, api_key, session_id, messages
            )
        except GeneratorError as error:
            print(f"LLM request unavailable: {error}")
            print("LLM generated kernel: FALLBACK")
            print("The existing template kernel remains selected.")
            return
        _write_text(candidate_path, assistant_content)

        _run_compiler(
            [
                str(compiler),
                "--admit-generated-kernel",
                str(candidate_path),
                "--feedback-output",
                str(feedback_path),
            ]
        )
        feedback = _read_json(feedback_path, "compiler feedback")
        status = feedback.get("status")
        if status == "admitted":
            shutil.copyfile(candidate_path, admitted_path)
            print("LLM generated kernel: ADMITTED")
            print(f"Admitted kernel: {admitted_path}")
            return
        if status == "fatal":
            raise GeneratorError(feedback.get("message", "Fatal compiler failure."))
        if status != "retry":
            raise GeneratorError("Compiler feedback has an unknown status.")

        messages.append({"role": "assistant", "content": assistant_content})
        messages.append(
            {
                "role": "user",
                "content": (
                    "The compiler rejected the previous kernel. Generate a corrected "
                    "response using the original contract and compiler feedback below.\n\n"
                    "The original contract remains authoritative and immutable. Preserve "
                    "the exact function name, complete parameter list, parameter order, "
                    "types, address spaces, access qualifiers, buffer indices, grid-index "
                    "attribute, mathematical semantics, and element_count bounds check. "
                    "Do not remove an input, replace constant uint& with a struct, or infer "
                    "element_count from the dispatch size. A retry may change only the "
                    "kernel-body implementation and select a workgroup size from "
                    "legal_workgroup_sizes. Return raw JSON matching response_schema.\n\n"
                    "ORIGINAL CONTRACT:\n"
                    + contract_text
                    + "\n\nCOMPILER FEEDBACK:\n"
                    + json.dumps(feedback, separators=(",", ":"))
                ),
            }
        )

    print("LLM generated kernel: FALLBACK")
    print("Retry budget exhausted; the validated template kernel remains selected.")


if __name__ == "__main__":
    try:
        main()
    except GeneratorError as error:
        print(f"LLM kernel generator: FAIL\nGenerator error: {error}", file=sys.stderr)
        raise SystemExit(1)
