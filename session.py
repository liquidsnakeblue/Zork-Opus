"""
Session tracker - persistent stats across generations (episodes).
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class SessionStats:
    generation: int = 0
    high_score: int = 0
    total_deaths: int = 0
    total_turns: int = 0
    best_generation: int = 0


class SessionTracker:
    DEFAULT_FILE = "game_files/session_stats.json"

    def __init__(self, stats_file: Optional[str] = None):
        self.path = Path(stats_file or self.DEFAULT_FILE)
        self.stats = self._load()

    def _load(self) -> SessionStats:
        if self.path.exists():
            try:
                with open(self.path) as f:
                    return SessionStats(**json.load(f))
            except Exception:
                pass
        return SessionStats()

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, 'w') as f:
            json.dump(asdict(self.stats), f, indent=2)

    def start_generation(self) -> int:
        self.stats.generation += 1
        self._save()
        return self.stats.generation

    def end_generation(self, score: int, turns: int, died: bool = False) -> bool:
        self.stats.total_turns += turns
        if died: self.stats.total_deaths += 1
        is_new_high = score > self.stats.high_score
        if is_new_high:
            self.stats.high_score = score
            self.stats.best_generation = self.stats.generation
        self._save()
        return is_new_high

    def display(self) -> str:
        s = self.stats
        best = f" (Gen {s.best_generation})" if s.best_generation else ""
        return f"Generation: {s.generation} | High Score: {s.high_score}{best} | Deaths: {s.total_deaths}"

    def header(self) -> str:
        s = self.stats
        gen = s.generation + 1
        best = f" (Gen {s.best_generation})" if s.best_generation else ""
        return (f"\n{'═'*60}\n"
                f"  🎮 GENERATION {gen}  |  🏆 HIGH: {s.high_score}{best}  |  💀 DEATHS: {s.total_deaths}\n"
                f"{'═'*60}\n")

    def reset(self):
        self.stats = SessionStats()
        self._save()
