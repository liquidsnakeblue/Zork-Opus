"""
LLM client - direct HTTP calls with retry, circuit breaker, and prompt logging.
Replaces the OpenAI SDK to support top_k, min_p, and other non-standard params.
"""

import json
import random
import time
import re
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


REASONING_MODEL_DEFAULT_MAX_TOKENS = 8000

# Models that need special handling (response_format suppression, etc.)
REASONING_MODEL_MARKERS = [
    "deepseek-r1", "deepseek-reasoner", "qwq", "qwen3", "glm-4", "glm4",
    "minimax-m1", "minimax-m2", "o1-", "o3-", "gemini-3",
    "step-3", "step3", "-thinking", "-reasoner", "gpt-oss",
]

# Models where response_format must be suppressed
RESPONSE_FORMAT_BLOCKED = ["step-3", "step3", "gpt-oss"]


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    reasoning_content: Optional[str] = None


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 10, recovery_timeout: float = 300.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = 0.0
        self.is_open = False

    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.failure_threshold:
            self.is_open = True

    def record_success(self):
        self.failures = 0
        self.is_open = False

    def can_proceed(self) -> bool:
        if not self.is_open:
            return True
        if time.time() - self.last_failure_time > self.recovery_timeout:
            self.is_open = False
            return True
        return False


class _CompletionsAPI:
    """Namespace for chat.completions.create() to mirror OpenAI SDK pattern."""

    def __init__(self, client: 'LLMClient'):
        self._client = client

    def create(self, **kwargs) -> LLMResponse:
        return self._client._call(**kwargs)

    def create_streaming(self, on_chunk=None, **kwargs) -> LLMResponse:
        return self._client._call_streaming(on_chunk=on_chunk, **kwargs)


class _ChatAPI:
    def __init__(self, client: 'LLMClient'):
        self.completions = _CompletionsAPI(client)


class LLMClient:
    """Direct HTTP LLM client with retry logic and prompt logging."""

    def __init__(self, config=None, base_url: str = None, api_key: str = None,
                 logger=None, prompt_logger=None):
        self.base_url = (base_url or (config.client_base_url if config else "")).rstrip("/")
        self.api_key = api_key or (config.api_key_for("agent") if config else None)
        self.logger = logger
        self.prompt_logger = prompt_logger
        self.config = config

        retry_cfg = config.retry if config else {}
        self.max_retries = retry_cfg.get("max_retries", 5)
        self.initial_delay = retry_cfg.get("initial_delay", 1.0)
        self.max_delay = retry_cfg.get("max_delay", 60.0)
        self.exp_base = retry_cfg.get("exponential_base", 2.0)
        self.jitter = retry_cfg.get("jitter_factor", 0.1)
        self.timeout = retry_cfg.get("timeout_seconds", 120.0)

        self.circuit_breaker = CircuitBreaker(
            failure_threshold=retry_cfg.get("circuit_breaker_failure_threshold", 10),
            recovery_timeout=retry_cfg.get("circuit_breaker_recovery_timeout", 300.0),
        )

        self.chat = _ChatAPI(self)

    def _is_reasoning_model(self, model: str) -> bool:
        return any(m in model.lower() for m in REASONING_MODEL_MARKERS)

    def _should_suppress_response_format(self, model: str) -> bool:
        return any(m in model.lower() for m in RESPONSE_FORMAT_BLOCKED)

    def _build_request(self, model: str, messages: List[Dict], **kwargs) -> Dict:
        body: Dict[str, Any] = {"model": model, "messages": messages, "stream": False}

        # Standard params
        for key in ["temperature", "top_p", "max_tokens", "stop", "presence_penalty"]:
            val = kwargs.get(key)
            if val is not None:
                body[key] = val

        # Extended params (non-OpenAI standard)
        for key in ["top_k", "min_p", "repetition_penalty"]:
            val = kwargs.get(key)
            if val is not None:
                body[key] = val

        # Thinking/reasoning support (both vLLM and llama.cpp accept chat_template_kwargs)
        enable_thinking = kwargs.get("enable_thinking")
        if enable_thinking is not None:
            body.setdefault("chat_template_kwargs", {})["enable_thinking"] = bool(enable_thinking)

        # Default max_tokens for reasoning models
        if self._is_reasoning_model(model) and "max_tokens" not in body:
            body["max_tokens"] = REASONING_MODEL_DEFAULT_MAX_TOKENS

        return body

    def _call(self, model: str = "", messages: List[Dict] = None,
              name: str = "LLM", **kwargs) -> LLMResponse:
        if not self.circuit_breaker.can_proceed():
            raise RuntimeError("Circuit breaker is open")

        body = self._build_request(model, messages or [], **kwargs)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/chat/completions"
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                start = time.time()
                resp = requests.post(url, json=body, headers=headers, timeout=self.timeout)
                duration_ms = (time.time() - start) * 1000

                if resp.status_code == 429:
                    self._backoff(attempt, "rate_limit")
                    continue
                if resp.status_code >= 500:
                    body_preview = resp.text[:200] if resp.text else "(empty)"
                    self._backoff(attempt, f"server_{resp.status_code}: {body_preview}")
                    continue

                resp.raise_for_status()
                data = resp.json()

                content = ""
                reasoning = None
                usage = None

                if "choices" in data and data["choices"]:
                    msg = data["choices"][0].get("message", {})
                    content = msg.get("content", "") or ""

                    # Extract reasoning from various formats
                    reasoning = msg.get("reasoning_content")
                    if not reasoning and msg.get("reasoning"):
                        reasoning = msg["reasoning"]
                    if not reasoning:
                        details = msg.get("reasoning_details") or []
                        if details:
                            reasoning = "\n".join(d.get("content", "") for d in details if d.get("content"))

                if "usage" in data:
                    usage = data["usage"]

                # Retry on empty response
                if not content.strip() and not reasoning:
                    if attempt < self.max_retries:
                        self._backoff(attempt, "empty_response")
                        continue

                self.circuit_breaker.record_success()

                # Log the call
                if self.prompt_logger:
                    self.prompt_logger.log_call(
                        name=name, model=model, messages=messages or [],
                        response_content=content, reasoning_content=reasoning,
                        usage=usage, temperature=kwargs.get("temperature"),
                        max_tokens=kwargs.get("max_tokens"), duration_ms=duration_ms,
                    )

                return LLMResponse(content=content, model=model, usage=usage, reasoning_content=reasoning)

            except requests.Timeout:
                last_error = TimeoutError(f"Request timed out after {self.timeout}s")
                self._backoff(attempt, "timeout")
            except requests.ConnectionError as e:
                last_error = ConnectionError(str(e))
                self._backoff(attempt, "connection_error")
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    self._backoff(attempt, f"error: {e}")
                    continue
                break

        self.circuit_breaker.record_failure()
        raise last_error or RuntimeError("LLM call failed after all retries")

    def _call_streaming(self, model: str = "", messages: List[Dict] = None,
                        name: str = "LLM", on_chunk=None, **kwargs) -> LLMResponse:
        """Streaming variant - calls on_chunk with accumulated content."""
        if not self.circuit_breaker.can_proceed():
            raise RuntimeError("Circuit breaker is open")

        body = self._build_request(model, messages or [], **kwargs)
        body["stream"] = True
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/chat/completions"
        start = time.time()

        resp = requests.post(url, json=body, headers=headers, timeout=self.timeout, stream=True)
        resp.raise_for_status()

        content_parts = []
        reasoning_parts = []
        usage = None
        result_model = model

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                if "choices" in chunk and chunk["choices"]:
                    delta = chunk["choices"][0].get("delta", {})
                    c = delta.get("content", "")
                    r = delta.get("reasoning_content", "") or delta.get("reasoning", "")
                    if c:
                        content_parts.append(c)
                    if r:
                        reasoning_parts.append(r)

                    if on_chunk:
                        accumulated = "".join(content_parts) if content_parts else "".join(reasoning_parts)
                        if accumulated:
                            on_chunk(accumulated)

                if "usage" in chunk:
                    usage = chunk["usage"]
                if "model" in chunk:
                    result_model = chunk["model"]
            except json.JSONDecodeError:
                continue

        duration_ms = (time.time() - start) * 1000
        content = "".join(content_parts)
        reasoning = "".join(reasoning_parts) or None

        self.circuit_breaker.record_success()

        if self.prompt_logger:
            self.prompt_logger.log_call(
                name=name, model=model, messages=messages or [],
                response_content=content, reasoning_content=reasoning,
                usage=usage, temperature=kwargs.get("temperature"),
                max_tokens=kwargs.get("max_tokens"), duration_ms=duration_ms,
            )

        return LLMResponse(content=content, model=result_model, usage=usage, reasoning_content=reasoning)

    def _backoff(self, attempt: int, reason: str):
        delay = min(self.initial_delay * (self.exp_base ** attempt), self.max_delay)
        delay *= (1 + random.uniform(-self.jitter, self.jitter))
        if self.logger:
            self.logger.warning(f"LLM retry {attempt+1}/{self.max_retries}: {reason}, waiting {delay:.1f}s")
        time.sleep(delay)


# ── JSON extraction utilities ──

def _strip_control_chars(s: str) -> str:
    """Remove control characters that break JSON parsing, preserving \\n \\r \\t."""
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', s)


def extract_json(content: str) -> str:
    """Extract JSON from text that may contain reasoning, markdown fences, or format tokens."""
    # Strip control characters early (LLMs sometimes emit raw \x00-\x1f inside strings)
    content = _strip_control_chars(content)
    # Strip thinking blocks (Qwen 3.5 etc.)
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    # Strip model format tokens (GPT-OSS etc.)
    content = re.sub(r'<\|[^|]+\|>[^<{]*', '', content).strip()

    # Strip markdown fences
    if "```json" in content:
        start = content.find("```json") + 7
        end = content.find("```", start)
        if end != -1:
            return content[start:end].strip()
        else:
            # No closing fence (truncated response) — strip opening and continue
            content = content[start:].strip()

    if content.strip().startswith("```") and content.strip().endswith("```"):
        lines = content.strip().split("\n")
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    elif content.strip().startswith("```"):
        # Opening fence without json tag and no closing fence
        lines = content.strip().split("\n")
        if len(lines) >= 2:
            content = "\n".join(lines[1:]).strip()
            if content.endswith("```"):
                content = content[:-3].strip()

    # Find balanced braces
    start = content.find('{')
    if start == -1:
        return content

    depth = 0
    for i in range(start, len(content)):
        if content[i] == '{': depth += 1
        elif content[i] == '}': depth -= 1
        if depth == 0:
            candidate = content[start:i+1]
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                # Try collapsing double braces
                collapsed = candidate.replace('{{', '{').replace('}}', '}')
                try:
                    json.loads(collapsed)
                    return collapsed
                except json.JSONDecodeError:
                    # Try stripping control characters
                    cleaned = _strip_control_chars(candidate)
                    try:
                        json.loads(cleaned)
                        return cleaned
                    except json.JSONDecodeError:
                        pass
                    # Try repairing unescaped quotes
                    repaired = _repair_json_quotes(cleaned)
                    try:
                        json.loads(repaired)
                        return repaired
                    except json.JSONDecodeError:
                        pass
            break

    return content


def _repair_json_quotes(s: str) -> str:
    """Fix unescaped quotes inside JSON string values."""
    result = []
    i = 0
    in_string = False
    while i < len(s):
        c = s[i]
        if not in_string:
            if c == '"':
                in_string = True
            result.append(c)
        else:
            if c == '\\' and i + 1 < len(s):
                result.append(c)
                result.append(s[i + 1])
                i += 2
                continue
            elif c == '"':
                rest = s[i + 1:].lstrip()
                if not rest or rest[0] in ':,}]':
                    in_string = False
                    result.append(c)
                else:
                    result.append('\\"')
            else:
                result.append(c)
        i += 1
    return ''.join(result)
