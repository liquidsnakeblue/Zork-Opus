"""
Local prompt logger - replaces Langfuse for debugging LLM interactions.
Writes every LLM call (system prompt, user message, response) to timestamped JSONL files.
Browse with any text editor or jq.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List


class PromptLogger:
    """Logs all LLM calls to local JSONL files for debugging."""

    def __init__(self, log_dir: str = "game_files/prompt_logs", max_content: int = 50000, enabled: bool = True):
        self.log_dir = Path(log_dir)
        self.max_content = max_content
        self.enabled = enabled
        self._episode_id = "unknown"
        self._session_file: Optional[Path] = None

        if enabled:
            self.log_dir.mkdir(parents=True, exist_ok=True)

    def set_episode(self, episode_id: str) -> None:
        self._episode_id = episode_id
        if self.enabled:
            episode_dir = self.log_dir / episode_id
            episode_dir.mkdir(parents=True, exist_ok=True)
            self._session_file = episode_dir / "llm_calls.jsonl"

    def log_call(
        self,
        name: str,
        model: str,
        messages: List[Dict[str, str]],
        response_content: str,
        reasoning_content: Optional[str] = None,
        usage: Optional[Dict[str, int]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        duration_ms: Optional[float] = None,
        turn: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a single LLM call with full prompt and response."""
        if not self.enabled or not self._session_file:
            return

        def truncate(s: str) -> str:
            if s and len(s) > self.max_content:
                return s[:self.max_content] + f"\n... [TRUNCATED at {self.max_content} chars, full={len(s)}]"
            return s

        entry = {
            "timestamp": datetime.now().isoformat(),
            "episode_id": self._episode_id,
            "turn": turn,
            "name": name,
            "model": model,
            "messages": [
                {"role": m.get("role", "?"), "content": truncate(m.get("content", ""))}
                for m in messages
            ],
            "response": truncate(response_content or ""),
            "reasoning": truncate(reasoning_content or "") if reasoning_content else None,
            "usage": usage,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "duration_ms": duration_ms,
        }
        if extra:
            entry["extra"] = extra

        try:
            with open(self._session_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass  # Never let logging break the game

    def log_span_start(self, name: str, turn: Optional[int] = None, metadata: Optional[Dict] = None) -> str:
        """Log the start of a logical span (turn, phase, etc). Returns span_id."""
        span_id = f"{name}_{datetime.now().strftime('%H%M%S%f')}"
        if not self.enabled or not self._session_file:
            return span_id

        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "span_start",
            "span_id": span_id,
            "name": name,
            "turn": turn,
            "metadata": metadata,
        }
        try:
            with open(self._session_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass
        return span_id

    def log_span_end(self, span_id: str, output: Optional[Dict] = None) -> None:
        if not self.enabled or not self._session_file:
            return
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "span_end",
            "span_id": span_id,
            "output": output,
        }
        try:
            with open(self._session_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass
