"""
Knowledge system - consolidated from 6 files into 1.
Handles periodic knowledge base updates and cross-episode synthesis.
"""

import os
import re
import json
from typing import Dict, Optional, List, Any
from pathlib import Path

from config import Config
from llm_client import LLMClient, extract_json


class KnowledgeManager:
    """Manages the knowledgebase.md file via LLM analysis of episode logs."""

    def __init__(self, config: Config, agent=None, map_manager=None,
                 llm_client: Optional[LLMClient] = None, logger=None):
        self.config = config
        self.agent = agent
        self.map_manager = map_manager
        self.logger = logger
        self.last_update_turn = 0
        self.object_events: List[Dict] = []

        self.output_file = str(Path(config.game_workdir) / config.knowledge_file)
        self.client = llm_client or LLMClient(
            config=config, base_url=config.base_url_for("analysis"),
            api_key=config.api_key_for("analysis"),
        )

    def should_update(self, turn: int) -> bool:
        return turn > 0 and (turn - self.last_update_turn) >= self.config.knowledge_update_interval

    def update_from_episode(self, episode_id: str, turn_count: int, is_final: bool = False) -> bool:
        """Run knowledge update from episode log data."""
        # Extract turn data from episode log
        turn_data = self._extract_turns(episode_id, 1, turn_count)
        if not turn_data or not turn_data.get("actions_and_responses"):
            return False

        # Quality check
        actions = turn_data["actions_and_responses"]
        if len(actions) < 3 and not is_final:
            return False

        # Load existing knowledge
        existing = ""
        try:
            if os.path.exists(self.output_file):
                with open(self.output_file, "r", encoding="utf-8") as f:
                    existing = f.read()
        except Exception:
            pass

        # Preserve cross-episode insights
        cross_ep = _extract_section(existing, "CROSS-EPISODE INSIGHTS")
        if cross_ep:
            existing = _remove_section(existing, "CROSS-EPISODE INSIGHTS")

        # Generate new knowledge
        new_knowledge = self._generate(turn_data, existing)
        if not new_knowledge or new_knowledge.startswith("SKIP:"):
            return False

        # Restore cross-episode section
        new_knowledge = _remove_section(new_knowledge, "CROSS-EPISODE INSIGHTS")
        if cross_ep:
            new_knowledge = _update_section(new_knowledge, "CROSS-EPISODE INSIGHTS", cross_ep)

        try:
            with open(self.output_file, "w", encoding="utf-8") as f:
                f.write(new_knowledge)
            self.last_update_turn = turn_count
            if self.agent and hasattr(self.agent, "reload_knowledge"):
                self.agent.reload_knowledge()
            return True
        except Exception as e:
            if self.logger: self.logger.error(f"Failed to write knowledge: {e}")
            return False

    def synthesize_cross_episode(self, episode_data: Dict) -> bool:
        """Update CROSS-EPISODE INSIGHTS section at episode end."""
        existing = ""
        try:
            if os.path.exists(self.output_file):
                with open(self.output_file, "r") as f:
                    existing = f.read()
        except Exception:
            pass

        current_insights = _extract_section(existing, "CROSS-EPISODE INSIGHTS") or ""
        turn_data = self._extract_turns(
            episode_data.get("episode_id", ""),
            1, episode_data.get("turn_count", 0)
        )
        turns_summary = _format_turns(turn_data) if turn_data else "(no turn data)"

        prompt = f"""Analyze this completed Zork episode and update the CROSS-EPISODE INSIGHTS section with UNIVERSAL strategic wisdom that persists across future episodes.

ARCHITECTURE REMINDER:
- **Memory System** handles location-specific procedures (e.g., "At Behind House, enter window")
- **Loop Break System** terminates stuck episodes (20+ turns no score) - NOT a game mechanic
- **Objective System** discovers and tracks goals automatically
- **Map System** manages spatial navigation and connections

CROSS-EPISODE INSIGHTS should contain ONLY universal strategic wisdom that applies regardless of location.

**CURRENT EPISODE SUMMARY:**
- Episode ID: {episode_data.get('episode_id', '?')}
- Total turns: {episode_data.get('turn_count', 0)}
- Final score: {episode_data.get('final_score', 0)}
- Episode ended in death: {episode_data.get('episode_ended_in_death', False)}
- Completed objectives: {episode_data.get('completed_objectives', [])}

KEY TURNS:
{turns_summary[:10000]}

EXISTING CROSS-EPISODE INSIGHTS:
{current_insights or "(No existing insights - this is the first episode)"}

Update the CROSS-EPISODE INSIGHTS with focus on:
1. **Death Patterns**: What killed us and universal avoidance strategies
   - CRITICAL: If episode ended at 20+ turns without score, this is Loop Break timeout (system), NOT game death
2. **Successful Strategies**: Universal approaches worth repeating
3. **Puzzle Meta-Patterns**: Types of puzzle solutions confirmed (not specific instances)
4. **Resource Management**: Inventory/item lessons that apply broadly
5. **Exploration Efficiency**: Navigation and discovery principles

RULES:
- Extract PRINCIPLES, not specific instances
- If it requires "at Location X", it belongs in Memory System, not here
- Preserve confirmed insights from previous episodes unless contradicted
- Do NOT document Loop Break timeouts as game mechanics

Output the updated CROSS-EPISODE INSIGHTS section content only (no section header)."""

        try:
            resp = self.client.chat.completions.create(
                model=self.config.analysis_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.analysis_sampling.get("temperature", 0.3),
                max_tokens=self.config.analysis_sampling.get("max_tokens", 8000),
                enable_thinking=self.config.analysis_sampling.get("enable_thinking"),
                name="CrossEpisodeSynthesis",
            )
            new_insights = resp.content or ""
            if not new_insights.strip():
                return False

            updated = _update_section(existing, "CROSS-EPISODE INSIGHTS", new_insights)
            with open(self.output_file, "w") as f:
                f.write(updated)
            return True
        except Exception as e:
            if self.logger: self.logger.error(f"Cross-episode synthesis failed: {e}")
            return False

    def _generate(self, turn_data: Dict, existing: str) -> str:
        turns_text = _format_turns(turn_data)
        death_text = _format_deaths(turn_data)

        is_first = not existing.strip() or len(existing.strip().split("\n")) < 10

        prompt = f"""Analyze this Zork gameplay data and create/update the strategic knowledge base.

ARCHITECTURE REMINDER:
- **Memory System** handles location-specific procedures (stored at SOURCE locations)
- **Loop Break System** terminates episodes stuck 20+ turns without score (system behavior, NOT game mechanic)
- **Objective System** discovers and tracks goals automatically
- **Map System** manages spatial navigation and connections

THIS knowledge base provides UNIVERSAL strategic wisdom, not location-specific tactics.

GAMEPLAY DATA:
{turns_text[:15000]}
{death_text}

EXISTING KNOWLEDGE BASE:
{'-' * 50}
{existing[:10000] if existing else "No existing knowledge - this is the first update"}
{'-' * 50}

INSTRUCTIONS:
Create a strategic knowledge base with these sections. Focus on UNIVERSAL patterns, not location-specific procedures.

## UNIVERSAL GAME MECHANICS
Document game rules that apply everywhere:
- Parser patterns (e.g., "EXAMINE reveals details", "Containers require OPEN before access")
- Object behaviors (e.g., "Dropped items persist in location")
- Scoring mechanics, action categories
✅ GOOD: "Containers must be opened before contents can be accessed"
❌ BAD: "Open window then enter window at Behind House" (location-specific → Memory System)

## DANGER CATEGORIES
Document TYPES of dangers and recognition patterns (not specific instances):
- Warning signals, death mechanics, safety principles
✅ GOOD: "Dark areas pose dangers - carry light source"
❌ BAD: "Troll at Troll Room attacks on sight" (specific location → Memory System)

## STRATEGIC PRINCIPLES
Universal decision-making strategies:
- Exploration heuristics, resource management, problem-solving patterns
✅ GOOD: "When stuck, try alternative action verbs (SEARCH, MOVE, LOOK UNDER)"
❌ BAD: "Collect egg from tree before exploring forest" (specific tactic → Memory System)

## DEATH & DANGER ANALYSIS
Analyze death events for universal lessons.
**CRITICAL**: If death occurred at 20+ turns without score progress, this is LOOP BREAK TIMEOUT (system behavior), NOT a game mechanic.

## LESSONS LEARNED
Strategic insights: new mechanics discovered, confirmed patterns, updated understanding.
FOCUS: Extract the PRINCIPLE, not the instance.

## CROSS-EPISODE INSIGHTS
**DO NOT GENERATE THIS SECTION** - It is managed separately and updated only at episode end.

CRITICAL REQUIREMENTS:
1. Universal Scope: ONLY include knowledge that applies regardless of location
2. Principle Over Instance: Extract the pattern, not the specific example
3. System Awareness: Ignore Loop Break timeouts as game mechanics
4. No Location Coupling: If it requires "at Location X", it belongs in Memory System"""

        try:
            resp = self.client.chat.completions.create(
                model=self.config.analysis_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.analysis_sampling.get("temperature", 0.3),
                max_tokens=self.config.analysis_sampling.get("max_tokens", 8000),
                enable_thinking=self.config.analysis_sampling.get("enable_thinking"),
                name="KnowledgeGeneration",
            )
            return resp.content or ""
        except Exception as e:
            if self.logger: self.logger.error(f"Knowledge generation failed: {e}")
            return ""

    def _extract_turns(self, episode_id: str, start: int, end: int) -> Optional[Dict]:
        """Extract turn data from episode log."""
        log_path = Path(self.config.game_workdir) / "episodes" / episode_id / "episode_log.jsonl"
        if not log_path.exists():
            # Fallback to main log
            log_path = Path(self.config.json_log_file)
        if not log_path.exists():
            return None

        actions = []
        deaths = []
        try:
            with open(log_path, "r") as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        et = entry.get("event_type", "")
                        turn = entry.get("turn", 0)
                        if start <= turn <= end:
                            if et == "turn_completed":
                                actions.append({
                                    "turn": turn,
                                    "action": entry.get("action", ""),
                                    "response": entry.get("response", ""),
                                    "score": entry.get("score", 0),
                                    "location": entry.get("location", ""),
                                })
                            if "died" in entry.get("message", "").lower() or "death" in str(entry).lower():
                                deaths.append(entry)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass

        return {"actions_and_responses": actions, "death_events": deaths,
                "episode_id": episode_id, "turn_range": f"{start}-{end}"}

    def get_export_data(self) -> Dict:
        try:
            with open(self.output_file, "r") as f:
                content = f.read()
            content = re.sub(r"## CURRENT WORLD MAP\s*\n\s*```mermaid\s*\n.*?\n```", "", content, flags=re.DOTALL)
            return {"content": content.strip(), "last_updated": os.path.getmtime(self.output_file)}
        except FileNotFoundError:
            return {"content": "No knowledge base yet.", "last_updated": None}

    def reset(self):
        self.last_update_turn = 0
        self.object_events.clear()

    def detect_object_events(self, prev_inv, cur_inv, jericho, action, turn):
        acquired = set(cur_inv) - set(prev_inv)
        for item in acquired:
            self.object_events.append({"turn": turn, "type": "acquired", "item": item, "action": action})
        dropped = set(prev_inv) - set(cur_inv)
        for item in dropped:
            self.object_events.append({"turn": turn, "type": "dropped", "item": item, "action": action})


# ── Section utilities ──

def _extract_section(content: str, name: str) -> str:
    pattern = rf"(## {re.escape(name)}\s*\n)(.*?)(?=\n## |\Z)"
    m = re.search(pattern, content, re.DOTALL)
    return m.group(2).strip() if m else ""

def _remove_section(content: str, name: str) -> str:
    pattern = rf"## {re.escape(name)}\s*\n.*?(?=\n## |\Z)"
    return re.sub(pattern, "", content, flags=re.DOTALL).strip()

def _update_section(content: str, name: str, new_content: str) -> str:
    content = _remove_section(content, name)
    if not content.endswith("\n"): content += "\n"
    return f"{content}\n## {name}\n{new_content}\n"


def _format_turns(data: Optional[Dict]) -> str:
    if not data: return ""
    lines = []
    for a in data.get("actions_and_responses", []):
        lines.append(f"Turn {a['turn']}: {a['action']} → score={a.get('score', '?')}, loc={a.get('location', '?')}")
        resp = a.get('response', '')
        if resp:
            lines.append(f"  Response: {resp[:200]}")
    return "\n".join(lines)

def _format_deaths(data: Optional[Dict]) -> str:
    if not data or not data.get("death_events"): return ""
    lines = ["DEATH EVENTS:"]
    for d in data["death_events"]:
        lines.append(f"  Turn {d.get('turn', '?')}: {d.get('message', str(d)[:200])}")
    return "\n".join(lines)
