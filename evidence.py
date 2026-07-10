"""
Evidence layer - structured, deterministic learning substrate.

Two pieces:

TrialLog — append-only JSONL of every executed action with before/after state
deltas and a success/failure verdict. Written immediately after the Z-machine
executes, before any LLM post-processing, so a crash can never erase evidence
of an action that already changed the game. This is the raw material for
offline mining of causal procedures.

ProcedureStore — curated, verified multi-step procedures with preconditions,
steps, postconditions, failure modes, and episode provenance. These are
CANONICAL: prompt builders inject them as ground truth that memories and the
generated walkthrough must not contradict. Promotion into this store is
deliberate (human- or miner-driven), never free-form LLM prose.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Response fragments that mark an action as having failed outright.
_FAILURE_MARKERS = (
    "you can't", "you cannot", "won't budge", "won't turn", "doesn't budge",
    "i don't know the word", "don't understand", "you used the word",
    "you don't see", "there is a wall", "too narrow", "nothing happens",
    "not here", "isn't here", "can't go that way", "can't do that",
    "already open", "already closed", "already have",
)


class TrialLog:
    """Append-only per-action outcome journal (deterministic, no LLM)."""

    def __init__(self, config, logger=None):
        self.logger = logger
        self.dir = Path(config.game_workdir) / "evidence"
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        self.path = self.dir / "trials.jsonl"

    @staticmethod
    def judge(response: str, score_delta: int, moved: bool,
              items_gained: List[str]) -> Optional[bool]:
        """Success verdict: True/False when determinable, None when ambiguous."""
        if score_delta > 0 or moved or items_gained:
            return True
        r = (response or "").lower()
        if any(m in r for m in _FAILURE_MARKERS):
            return False
        return None

    def record(self, episode_id: str, turn: int, location_id: int,
               location_name: str, action: str, response: str,
               score_delta: int, moved: bool,
               items_gained: List[str], items_lost: List[str]) -> None:
        entry = {
            "ts": datetime.now().isoformat(),
            "episode": episode_id, "turn": turn,
            "location_id": location_id, "location_name": location_name,
            "action": (action or "").strip().lower(),
            "score_delta": score_delta, "moved": moved,
            "items_gained": items_gained, "items_lost": items_lost,
            "success": self.judge(response, score_delta, moved, items_gained),
            "response": (response or "")[:200],
        }
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
                f.flush()
        except Exception as e:
            if self.logger:
                self.logger.warning(f"TrialLog write failed: {e}")


class ProcedureStore:
    """Verified multi-step procedures — canonical knowledge with provenance."""

    def __init__(self, config, logger=None, path: Optional[str] = None):
        self.logger = logger
        self.path = Path(path) if path else Path(config.game_workdir) / "procedures.json"
        self.procedures: List[Dict[str, Any]] = []
        self.reload()

    def reload(self):
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.procedures = data.get("procedures", [])
            else:
                self.procedures = []
        except Exception as e:
            if self.logger:
                self.logger.warning(f"ProcedureStore: could not load {self.path}: {e}")
            self.procedures = []

    def for_location(self, location_id: int) -> List[Dict[str, Any]]:
        return [p for p in self.procedures
                if location_id in (p.get("location_ids") or [])]

    def format(self, procedures: Optional[List[Dict[str, Any]]] = None) -> str:
        procs = self.procedures if procedures is None else procedures
        procs = [p for p in procs if p.get("status", "verified") == "verified"]
        if not procs:
            return ""
        lines = ["=== VERIFIED PROCEDURES (canonical — confirmed across episodes; "
                 "trust these over memories and walkthrough) ==="]
        for p in procs:
            lines.append(f"▶ {p.get('name', p.get('id', '?'))}")
            if p.get("preconditions"):
                lines.append(f"  Requires: {'; '.join(p['preconditions'])}")
            steps = p.get("steps") or []
            if steps:
                lines.append("  Steps: " + " → ".join(
                    f"{i}. {s}" for i, s in enumerate(steps, 1)))
            if p.get("postconditions"):
                lines.append(f"  Result: {'; '.join(p['postconditions'])}")
            for fm in p.get("failure_modes") or []:
                lines.append(f"  ⚠️ {fm}")
            if p.get("evidence"):
                lines.append(f"  Evidence: {'; '.join(p['evidence'])}")
        return "\n".join(lines)

    def format_for_location(self, location_id: int) -> str:
        return self.format(self.for_location(location_id))
