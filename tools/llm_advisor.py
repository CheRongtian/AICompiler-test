#!/usr/bin/env python3
"""Request a ranked Metal candidate list from the Jimo AI SSE endpoint."""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path


class AdvisorError(RuntimeError):
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUEST_TIMEOUT_SECONDS = 60.0


def _load_env(path):
    if not path.exists():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise AdvisorError(f"Unable to read environment file: {error}") from error
    for line_number, source_line in enumerate(lines, start=1):
        line = source_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            raise AdvisorError(f"Invalid .env entry on line {line_number}.")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise AdvisorError(f"Invalid .env key on line {line_number}.")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _load_request(path):
    try:
        with open(path, "r", encoding="utf-8") as source:
            request = json.load(source)
    except (OSError, json.JSONDecodeError) as error:
        raise AdvisorError(f"Unable to read advisor request: {error}") from error
    if request.get("version") != 1 or not isinstance(request.get("regions"), list):
        raise AdvisorError("Advisor request must use protocol version 1 and contain regions.")
    return request


def _prompt(request):
    return json.dumps(request, separators=(",", ":"), ensure_ascii=False)


def _decode_advice(content):
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as error:
        raise AdvisorError(f"LLM content is not valid advisor JSON: {error}") from error


def _call_endpoint(endpoint, api_key, request):
    body = {
        "messages": [{"role": "user", "content": _prompt(request)}],
        "sessionId": str(uuid.uuid4()),
        "source": "api",
        "extra": {},
    }
    http_request = urllib.request.Request(
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
            http_request, timeout=REQUEST_TIMEOUT_SECONDS
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
                    raise AdvisorError(f"Invalid JSON in SSE data event: {error}") from error
                content = event.get("content")
                if isinstance(content, str):
                    chunks.append(content)
                if "end" in event:
                    ended = True
                    break
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise AdvisorError(f"LLM endpoint returned HTTP {error.code}: {detail}") from error
    except (urllib.error.URLError, TimeoutError, UnicodeDecodeError) as error:
        raise AdvisorError(f"LLM request failed: {error}") from error

    if not ended:
        raise AdvisorError("SSE stream closed before the end event.")
    if not chunks:
        raise AdvisorError("SSE stream completed without assistant content.")
    return _decode_advice("".join(chunks))


def _validate_response(response):
    if not isinstance(response, dict) or set(response) != {"version", "regions"}:
        raise AdvisorError("Advisor response must contain only version and regions.")
    if response["version"] != 1 or not isinstance(response["regions"], list):
        raise AdvisorError("Advisor response must use protocol version 1.")
    seen_regions = set()
    for index, region in enumerate(response["regions"]):
        if not isinstance(region, dict) or set(region) != {
            "region_id",
            "ranked_candidate_ids",
        }:
            raise AdvisorError(f"Invalid advisor region at index {index}.")
        region_id = region["region_id"]
        candidate_ids = region["ranked_candidate_ids"]
        if (
            not isinstance(region_id, int)
            or isinstance(region_id, bool)
            or region_id < 0
            or region_id in seen_regions
            or not isinstance(candidate_ids, list)
            or any(not isinstance(candidate_id, str) for candidate_id in candidate_ids)
        ):
            raise AdvisorError(f"Invalid advisor region at index {index}.")
        seen_regions.add(region_id)


def _write_response(path, response):
    try:
        with open(path, "w", encoding="utf-8") as destination:
            json.dump(response, destination, indent=2)
            destination.write("\n")
    except OSError as error:
        raise AdvisorError(f"Unable to write advisor response: {error}") from error


def main():
    default_env = PROJECT_ROOT / ".env"
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--env-file", default=str(default_env))
    pre_args, _ = pre_parser.parse_known_args()
    _load_env(Path(pre_args.env_file))

    parser = argparse.ArgumentParser(
        description="Rank AICompiler candidates using the Jimo AI SSE endpoint."
    )
    parser.add_argument("--env-file", default=str(default_env))
    parser.add_argument("--input", required=True, help="region summary JSON")
    parser.add_argument("--output", required=True, help="advisor response JSON")
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("TMC_LLM_ENDPOINT"),
    )
    args = parser.parse_args()

    api_key = os.environ.get("TMC_LLM_API_KEY")
    if not args.endpoint:
        raise AdvisorError("Set TMC_LLM_ENDPOINT or pass --endpoint.")
    if not api_key:
        raise AdvisorError("Set TMC_LLM_API_KEY before calling the advisor.")
    request = _load_request(args.input)
    response = _call_endpoint(args.endpoint, api_key, request)
    _validate_response(response)
    _write_response(args.output, response)
    print("LLM advisor: PASS")
    print(f"Advisor regions: {len(response['regions'])}")
    print(f"Advisor response file: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except AdvisorError as error:
        print(f"LLM advisor: FAIL\nAdvisor error: {error}", file=sys.stderr)
        raise SystemExit(1)
