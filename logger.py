"""
Logging setup - human-readable console + JSON file output.
Consolidated from the original's 4 handler classes into 2.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List
from pathlib import Path


class _ConsoleFormatter(logging.Formatter):
    """Human-readable formatter that filters low-value messages."""

    def format(self, record):
        msg = record.getMessage()

        if record.levelname in ("ERROR", "WARNING"):
            return f"{record.levelname}: {msg}"

        if record.levelname == "DEBUG":
            return None

        et = getattr(record, "event_type", None)
        if et == "episode_initialized":
            return f"\n🎮 NEW EPISODE: {getattr(record, 'episode_id', '?')}"
        if et == "turn_completed":
            return (f"Turn {getattr(record, 'turn', '?')}: '{getattr(record, 'action', '?')}' "
                    f"→ Score: {getattr(record, 'score', 0)}, "
                    f"Location: {getattr(record, 'location', '?')}")
        if et in ("episode_completed", "episode_finalized"):
            return (f"🏁 Episode done: {getattr(record, 'turn', '?')} turns, "
                    f"Score: {getattr(record, 'final_score', 0)}")
        if et == "web_search_request":
            return f"  🔍 Search: \"{getattr(record, 'query', '?')}\""
        if et == "web_crawl_request":
            return f"  🌐 Crawl: {getattr(record, 'urls', '?')}"
        if et == "walkthrough_updated":
            return f"  📝 Walkthrough updated ({getattr(record, 'walkthrough_length', 0)} chars)"
        if et == "knowledge_update":
            return f"📚 Knowledge: {msg}"
        if et == "progress":
            stage = getattr(record, "stage", "")
            if stage in ("episode_initialization", "episode_finalization", "inter_episode_synthesis"):
                return f"⚙️  {stage.replace('_', ' ').title()}: {getattr(record, 'details', msg)}"
            return None
        if et in ("agent_raw_response_debug", "reasoning_extraction_debug", "map_consolidation",
                   "agent_llm_response", "zork_response", "search_disabled"):
            return None

        if record.levelname == "INFO" and any(k in msg.lower() for k in
                ["error", "failed", "completed", "initialized"]):
            return msg

        return None


class _JsonFormatter(logging.Formatter):
    def format(self, record):
        d = {"timestamp": datetime.fromtimestamp(record.created).isoformat(),
             "level": record.levelname, "message": record.getMessage()}
        skip = {"name", "msg", "args", "levelname", "levelno", "pathname", "filename",
                "module", "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "exc_info", "exc_text", "stack_info", "getMessage"}
        for k, v in record.__dict__.items():
            if k not in skip and not k.startswith("_"):
                d[k] = v
        return json.dumps(d)


class _FilteringHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            if msg is not None:
                self.stream.write(msg + self.terminator)
                self.flush()
        except Exception:
            self.handleError(record)


class _FilteringFileHandler(logging.FileHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            if msg is not None:
                if self.stream is None:
                    self.stream = self._open()
                self.stream.write(msg + self.terminator)
                self.flush()
        except Exception:
            self.handleError(record)


def setup_logging(episode_log: str, json_log: str, level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger("zorkopus")
    logger.setLevel(level)
    logger.handlers = []

    ch = _FilteringHandler()
    ch.setLevel(level)
    ch.setFormatter(_ConsoleFormatter())
    logger.addHandler(ch)

    fh = _FilteringFileHandler(episode_log, mode="a", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(_ConsoleFormatter())
    logger.addHandler(fh)

    jh = logging.FileHandler(json_log, mode="a", encoding="utf-8")
    jh.setLevel(level)
    jh.setFormatter(_JsonFormatter())
    jh.is_json_handler = True
    logger.addHandler(jh)

    return logger


def setup_episode_logging(episode_id: str, workdir: str = "game_files", level=logging.INFO) -> str:
    episode_dir = Path(workdir) / "episodes" / episode_id
    episode_dir.mkdir(parents=True, exist_ok=True)
    log_file = episode_dir / "episode_log.jsonl"

    logger = logging.getLogger("zorkopus")
    logger.setLevel(level)

    for h in logger.handlers[:]:
        if isinstance(h, logging.FileHandler) and hasattr(h, "is_json_handler"):
            logger.removeHandler(h)
            h.close()

    jh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    jh.setFormatter(_JsonFormatter())
    jh.setLevel(level)
    jh.is_json_handler = True
    logger.addHandler(jh)

    return str(log_file)
