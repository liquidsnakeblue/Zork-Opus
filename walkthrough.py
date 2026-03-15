"""
Walkthrough manager - generates puzzle-solving guides from memories using the reasoner.
"""

import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from config import Config
from state import GameState
from llm_client import LLMClient


class WalkthroughManager:
    def __init__(self, config: Config, game_state: GameState, logger=None,
                 walkthrough_path: str = None, memories_path: str = None):
        self.config = config
        self.gs = game_state
        self.logger = logger

        workdir = Path(config.game_workdir)
        project_root = workdir.parent if workdir.name == "game_files" else workdir
        self.walkthrough_path = Path(walkthrough_path) if walkthrough_path else project_root / "Zork Walkthrough.md"
        self.memories_path = Path(memories_path) if memories_path else workdir / config.memory_file

        self.client = LLMClient(
            config=config, base_url=config.base_url_for("reasoner"),
            api_key=config.api_key_for("reasoner"), logger=logger,
        )
        self._cache: Optional[str] = None
        self._cache_mtime: Optional[float] = None

    def generate(self) -> Optional[str]:
        """Generate or incrementally update walkthrough from memories."""
        existing = self._load_existing()
        memories = self._load_memories()
        if not memories:
            return None

        if existing:
            prompt = self._incremental_prompt(memories, existing)
        else:
            prompt = self._fresh_prompt(memories)

        try:
            # Backup existing
            if self.walkthrough_path.exists():
                ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
                shutil.copy2(self.walkthrough_path, self.walkthrough_path.parent / f"Zork Walkthrough.{ts}.md")

            rs = self.config.reasoner_sampling
            resp = self.client.chat.completions.create(
                model=self.config.reasoner_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=rs.get("max_tokens", 32000),
                enable_thinking=rs.get("enable_thinking"),
                name="WalkthroughGen",
            )

            content = resp.content
            if not content: return None

            header = (f"# Zork I - Puzzle Solving Guide\n"
                     f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n"
                     f"*Episode: {self.gs.episode_id}*\n\n---\n\n")
            self.walkthrough_path.write_text(header + content, encoding="utf-8")
            self._cache = None
            return content
        except Exception as e:
            if self.logger: self.logger.error(f"Walkthrough generation failed: {e}")
            return None

    def get_content(self) -> str:
        if not self.walkthrough_path.exists(): return ""
        try:
            mtime = self.walkthrough_path.stat().st_mtime
            if self._cache is not None and self._cache_mtime == mtime:
                return self._cache
            self._cache = self.walkthrough_path.read_text(encoding="utf-8")
            self._cache_mtime = mtime
            return self._cache
        except Exception:
            return ""

    def _load_memories(self) -> str:
        try:
            return self.memories_path.read_text(encoding="utf-8") if self.memories_path.exists() else ""
        except Exception:
            return ""

    def _load_existing(self) -> str:
        try:
            return self.walkthrough_path.read_text(encoding="utf-8") if self.walkthrough_path.exists() else ""
        except Exception:
            return ""

    def _fresh_prompt(self, memories: str) -> str:
        return f"""Analyze these Zork I gameplay memories and create a puzzle-solving guide.

=== MEMORIES ===
{memories}

Create sections: Overview, Essential Items, Puzzle Solutions, Safe Order of Play, Combat, Unsolved Mysteries.
Be thorough but organized."""

    def _incremental_prompt(self, memories: str, existing: str) -> str:
        return f"""Update this Zork I guide with new memories.

=== EXISTING GUIDE ===
{existing}

=== CURRENT MEMORIES ===
{memories}

Rules:
1. Preserve confirmed knowledge unless directly contradicted
2. Add new information
3. Remove disproven information
4. Consolidate, don't accumulate
5. The Thief and Guard are the same NPC

Output the complete updated guide."""

    def reset(self):
        self._cache = None
        self._cache_mtime = None
