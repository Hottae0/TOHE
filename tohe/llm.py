"""Explicit HTTP chat-completions-compatible endpoint. No provider is selected by default."""
import json
import os
import time
import urllib.request
import urllib.error
from .io import digest


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def chat(config, messages):
    endpoint = os.environ.get(config["endpoint_env"])
    if not endpoint or not config.get("model"):
        raise ValueError("Set an explicit model in config and the configured endpoint environment variable")
    key = os.environ.get(config["api_key_env"], "")
    body = {"model": config["model"], "messages": messages,
            "temperature": config["temperature"], "max_tokens": config["max_output_tokens"]}
    if config.get("response_format") is not None:
        body["response_format"] = config["response_format"]
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    request = urllib.request.Request(endpoint, data=json.dumps(body).encode(), headers=headers)
    start = time.perf_counter()
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=config["timeout_seconds"]) as response:
                payload = json.load(response)
            return {"text": payload["choices"][0]["message"]["content"],
                    "usage": payload.get("usage"), "response_model": payload.get("model"),
                    "system_fingerprint": payload.get("system_fingerprint"),
                    "finish_reason": payload["choices"][0].get("finish_reason"),
                    "seconds": time.perf_counter() - start,
                    "request_hash": digest(body), "endpoint_hash": digest(endpoint)}
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504) or attempt == 3:
                # Avoid printing headers, tokens, or provider error bodies.
                raise RuntimeError(f"LLM HTTP error {exc.code}") from None
            time.sleep(2 ** attempt)
    raise RuntimeError("LLM request did not complete")
