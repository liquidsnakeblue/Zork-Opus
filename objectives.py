"""
Objective manager - handles discovery, tracking, completion, and refinement of objectives.
Uses the reasoner model for strategic objective generation.
"""

import json
import re
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from collections import deque

from config import Config
from state import GameState, Objective
from llm_client import LLMClient, extract_json


class ObjectiveDefinition(BaseModel):
    category: Literal["exploration", "action"]
    name: str
    text: str
    completion_condition: str
    target_location_id: Optional[int] = None


class ReasonerResponse(BaseModel):
    reasoning: str = Field(default="")
    suggested_approach: str = Field(default="")
    new_objectives: List[ObjectiveDefinition] = Field(default_factory=list)
    abandon_objective_ids: List[str] = Field(default_factory=list)


class CompletionCheck(BaseModel):
    objective_id: str
    completed: bool = False
    reason: str = ""
    progress: Optional[Any] = None
    new_status: Optional[str] = None
    new_progress: Optional[Any] = None
    reasoning: Optional[str] = None


class CompletionResponse(BaseModel):
    reasoning: str = ""
    updates: List[CompletionCheck] = Field(default_factory=list)
    completed_objective_ids: List[str] = Field(default_factory=list)  # Alternate format LLM may use


class ObjectiveManager:
    def __init__(self, config: Config, game_state: GameState,
                 knowledge_manager=None, map_manager=None, memory_manager=None,
                 context_manager=None, walkthrough_manager=None,
                 llm_client: Optional[LLMClient] = None, review_client: Optional[LLMClient] = None,
                 web_search_manager=None, streaming_server=None, logger=None):
        self.config = config
        self.gs = game_state
        self.knowledge = knowledge_manager
        self.map_manager = map_manager
        self.memory = memory_manager
        self.context = context_manager
        self.walkthrough = walkthrough_manager
        self.streaming = streaming_server
        self.web_search = web_search_manager
        self.logger = logger

        # LLM clients
        self.reasoner_client = llm_client or LLMClient(
            config=config, base_url=config.base_url_for("reasoner"),
            api_key=config.api_key_for("reasoner"),
        )
        self.review_client = review_client or self.reasoner_client

    def should_run_reasoner(self) -> bool:
        """Check if it's time for the reasoner to update objectives."""
        turn = self.gs.turn_count
        if turn == 0: return False
        interval = self.config.objective_update_interval
        return (turn - self.gs.objective_update_turn) >= interval

    def run_reasoner(self, game_text: str = "") -> Optional[ReasonerResponse]:
        """Call reasoner to generate/update objectives."""
        if not self.config.enable_deep_reasoning:
            return None

        # Build context for reasoner
        parts = [f"Turn: {self.gs.turn_count}", f"Score: {self.gs.previous_score}",
                 f"Location: {self.gs.current_room_name} (L{self.gs.current_room_id})"]

        # Current objectives
        active = self.gs.active_objectives
        if active:
            parts.append("\nCURRENT OBJECTIVES:")
            for o in active:
                parts.append(f"  [{o.id}] {o.name}: {o.text} (condition: {o.completion_condition})")

        completed = self.gs.completed_objectives_list[-5:]
        if completed:
            parts.append("\nRECENTLY COMPLETED:")
            for c in completed:
                parts.append(f"  ✓ {c['objective']} (turn {c['completed_turn']})")

        # Recent actions
        recent = self.gs.action_history[-10:]
        if recent:
            parts.append("\nRECENT ACTIONS:")
            for e in recent:
                parts.append(f"  T{e.turn}: {e.action} → {e.response[:100]}")

        # Map info
        if self.map_manager:
            metrics = self.map_manager.get_quality_metrics()
            parts.append(f"\nMap: {metrics.get('room_count', 0)} rooms, {metrics.get('connection_count', 0)} connections")

        # Walkthrough
        if self.walkthrough:
            wt = self.walkthrough.get_content()
            if wt:
                parts.append(f"\nWALKTHROUGH GUIDE:\n{wt[:3000]}")

        context = "\n".join(parts)

        prompt = f"""You are a strategic advisor for a text adventure game agent playing Zork. Analyze the game state and output STRUCTURED objectives as JSON. Objectives should be well thought out and reasoned with a focus on high-level goals and milestones that would help the agent beat the game. The game is won when all valuable treasures are in the trophy case.

{context}

=== YOUR TASK ===
Create or update objectives for the agent. You have FULL CONTROL over objectives:
- You can CREATE new objectives
- You can REMOVE objectives that are stale, completed, or no longer relevant

There are TWO types of objectives:

1. ACTION (PREFERRED): Specific tasks with location context
   - Example: "Go to Forest (L42) and climb tree to get jeweled egg"
   - Example: "Return to Living Room (L61) and place egg in trophy case"
   - ALWAYS prefer ACTION objectives when you know what to do next

2. EXPLORATION (USE SPARINGLY): Discovery-focused, only when stuck/blocked
   - Example: "Discover 3 new locations" (progress: "0/3 found")
   - ONLY use when agent is stuck, blocked, or unsure how to proceed

**CRITICAL**: IN NEW OR EARLY GAME SITUATIONS, prefer EXPLORATION to build out the map. Puzzles often involve items from other locations.

=== CRITICAL CONSTRAINTS ===
- Only reference location IDs that exist in the map data above
- Do NOT suggest acquiring items already in inventory: {self.gs.current_inventory}
- Keep objectives SHORT and CLEAN (no long paragraphs)
- Aim for 3-7 total active objectives
- **NEVER create objectives that duplicate existing active ones** (check CURRENT OBJECTIVES above)
- **ONE LOCATION PER OBJECTIVE (MANDATORY)**: Each objective MUST target exactly ONE location. Multi-location objectives break pathfinder navigation. Split multi-location tasks into separate objectives.
  * BAD: "Collect sword from Living Room, rope from Attic, and sack from Kitchen"
  * GOOD: "Collect Sword" at L193, "Collect Rope" at L201, "Collect Sack" at L203
- **NEVER include navigation paths** - The agent has a pathfinder tool. Only specify the DESTINATION, NOT the route.
- **ALWAYS include target_location_id** for ACTION objectives

=== OUTPUT FORMAT ===
For each objective, provide:
- category: "exploration" or "action"
- name: A CONCISE 3-5 word TITLE (e.g., "Collect Brass Lantern", "Open Trap Door")
- text: Short, clean description (1 sentence)
- completion_condition: EXPLICIT, VERIFIABLE condition (e.g., "egg is in inventory", "score increased", "current location is Kitchen")
- target_location_id: For ACTION objectives, the target location number (e.g., 42 for L42)

Also provide:
- suggested_approach: A DETAILED paragraph (3-5 sentences) explaining HOW the agent should approach the next series of turns
- reasoning: Brief summary of the strategic plan

First, THINK DEEPLY about the game state. Then output JSON in ```json fences:
```json
{{
  "reasoning": "Agent needs equipment. Each objective targets ONE location.",
  "suggested_approach": "First collect the sword, then explore underground. Deposit treasures in trophy case.",
  "new_objectives": [
    {{"category": "action", "name": "Collect Sword", "text": "Take the elvish sword from the Living Room", "completion_condition": "sword is in inventory", "target_location_id": 193}}
  ],
  "abandon_objective_ids": ["A001"]
}}
```"""

        try:
            rs = self.config.reasoner_sampling
            resp = self.reasoner_client.chat.completions.create(
                model=self.config.reasoner_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=rs.get("temperature", 0.7),
                max_tokens=rs.get("max_tokens", 16000),
                enable_thinking=rs.get("enable_thinking"),
                name="ObjectiveReasoner",
            )

            content = resp.content or ""
            if not content and resp.reasoning_content:
                content = resp.reasoning_content

            result = ReasonerResponse.model_validate_json(extract_json(content))

            # Apply results
            self._apply_reasoner_result(result)
            self.gs.objective_update_turn = self.gs.turn_count

            # Store for viewer (viewer expects: reasoning, suggested_approach, objectives[], trigger_reason)
            # Build objectives list with the IDs that were just assigned by _apply_reasoner_result
            new_obj_data = []
            for defn in result.new_objectives[:3]:
                # Find the objective we just added (match by name since IDs were generated)
                for obj in reversed(self.gs.objectives):
                    if obj.name == defn.name and obj.created_turn == self.gs.turn_count:
                        new_obj_data.append({
                            "id": obj.id, "name": obj.name, "category": obj.category,
                            "text": obj.text, "completion_condition": obj.completion_condition,
                        })
                        break

            # Determine trigger reason
            if self.gs.turn_count == 0:
                trigger = "Episode start"
            elif result.abandon_objective_ids:
                trigger = f"Periodic review (abandoned {len(result.abandon_objective_ids)})"
            else:
                trigger = f"Periodic review (every {self.config.objective_update_interval} turns)"

            self.gs.reasoner_events.append({
                "turn": self.gs.turn_count,
                "reasoning": result.reasoning,
                "suggested_approach": result.suggested_approach,
                "trigger_reason": trigger,
                "objectives": new_obj_data,
                "abandoned": result.abandon_objective_ids,
            })

            return result

        except Exception as e:
            if self.logger: self.logger.error(f"Reasoner failed: {e}")
            return None

    def _apply_reasoner_result(self, result: ReasonerResponse):
        # Abandon objectives
        for obj_id in result.abandon_objective_ids:
            obj = self.gs.get_objective(obj_id)
            if obj and obj.status in ("pending", "in_progress"):
                obj.status = "abandoned"

        # Add new objectives (skip duplicates)
        existing_names = {o.name.lower().strip() for o in self.gs.objectives
                         if o.status in ("pending", "in_progress")}
        for defn in result.new_objectives[:3]:
            if defn.name.lower().strip() in existing_names:
                continue  # Skip duplicate
            obj_id = self.gs.next_objective_id(defn.category)
            obj = Objective(
                id=obj_id, category=defn.category, name=defn.name,
                text=defn.text, completion_condition=defn.completion_condition,
                created_turn=self.gs.turn_count,
                target_location_id=defn.target_location_id,
            )
            self.gs.objectives.append(obj)
            existing_names.add(defn.name.lower().strip())

        # Update approach
        if result.suggested_approach:
            self.gs.current_approach = result.suggested_approach

    def check_completions(self, action: str, response: str) -> List[str]:
        """Check if any objectives were completed this turn. Returns completed IDs."""
        if not self.config.enable_objective_completion_llm_check:
            return []

        active = self.gs.active_objectives
        if not active:
            return []

        # Build detailed context
        obj_lines = []
        for o in active:
            progress = f" - progress: {o.progress}" if o.progress else ""
            loc = f" - target: L{o.target_location_id}" if o.target_location_id else ""
            obj_lines.append(f"- [{o.id}] ({o.category}) {o.status}: {o.text}{progress}{loc}")
            obj_lines.append(f"    ✓ COMPLETE WHEN: {o.completion_condition}")
        obj_text = "\n".join(obj_lines)

        # Recent action history
        recent = self.gs.action_history[-5:]
        history = "\n".join(f"  T{e.turn}: {e.action} → {e.response[:100]}" for e in recent) if recent else "None"

        # Movement info
        prev_loc = self.gs.prev_room_name or "Unknown"
        moved = prev_loc != self.gs.current_room_name if prev_loc != "Unknown" else False
        move_info = f"MOVED: {prev_loc} → {self.gs.current_room_name}" if moved else f"STAYED: {self.gs.current_room_name}"

        prompt = f"""Review objectives after the agent's latest action.

CURRENT OBJECTIVES (each shows its completion condition):
{obj_text}

## Recent Action History
{history}

## Latest Action and Response
Action: {action}
Response: {response}

## Movement This Turn
{move_info}

## Current Game State
Location: {self.gs.current_room_name} (ID: {self.gs.current_room_id})
Score: {self.gs.previous_score}
Inventory: {', '.join(self.gs.current_inventory) if self.gs.current_inventory else 'empty'}

=== COMPLETION CHECK ===
**KEY RULE: Each objective has a "COMPLETE WHEN" condition. Use THAT condition to determine completion.**

1. **Action Objectives**: The completion condition tells you exactly what to verify.
2. **Exploration Objectives**: Track discovery progress (update progress field).

**DO NOT MARK COMPLETE IF**: condition not satisfied, action failed, or only partial progress.

=== OUTPUT FORMAT ===
Return JSON with "reasoning" FIRST (think before deciding):
```json
{{
  "reasoning": "Analyze each objective's condition against current state",
  "updates": [
    {{"objective_id": "A001", "completed": true, "reason": "egg is now in inventory"}}
  ]
}}
```
If no changes: {{"reasoning": "No conditions met", "updates": []}}"""

        try:
            resp = self.review_client.chat.completions.create(
                model=self.config.agent_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.turn_review_sampling.get("temperature", 0.1),
                max_tokens=self.config.turn_review_sampling.get("max_tokens", 2000),
                enable_thinking=self.config.turn_review_sampling.get("enable_thinking"),
                name="ObjectiveReview",
            )

            content = resp.content or ""
            if not content and resp.reasoning_content:
                content = resp.reasoning_content

            result = CompletionResponse.model_validate_json(extract_json(content))
            completed_ids = []

            # Handle completed_objective_ids (alternate format some LLMs use)
            for oid in result.completed_objective_ids:
                obj = self.gs.get_objective(oid)
                if obj and obj.status in ("pending", "in_progress"):
                    obj.status = "completed"
                    obj.completed_turn = self.gs.turn_count
                    completed_ids.append(oid)

            # Handle updates array
            for update in result.updates:
                obj = self.gs.get_objective(update.objective_id)
                if not obj:
                    continue
                # Check completed flag OR new_status == "completed"
                is_completed = update.completed or (update.new_status == "completed")
                if is_completed and obj.status in ("pending", "in_progress"):
                    obj.status = "completed"
                    obj.completed_turn = self.gs.turn_count
                    if update.objective_id not in completed_ids:
                        completed_ids.append(update.objective_id)
                # Update progress
                if update.progress or update.new_progress:
                    obj.progress = str(update.new_progress or update.progress)

            # Store for viewer (needs {id, name} objects for completed, reasoning text for content)
            completed_objs = []
            for uid in completed_ids:
                obj = self.gs.get_objective(uid)
                completed_objs.append({"id": uid, "name": obj.name if obj else uid})

            review_lines = []
            for update in result.updates:
                status = "completed" if update.completed else "updated"
                review_lines.append(f"[{update.objective_id}] {status}: {update.reason}")

            self.gs.objective_review_results[self.gs.turn_count] = {
                "content": "\n".join(review_lines) if review_lines else "No objective changes.",
                "completed_objectives": completed_objs,
                "updates": [u.model_dump() for u in result.updates],
            }

            return completed_ids

        except Exception as e:
            if self.logger: self.logger.error(f"Completion check failed: {e}")
            return []

    def mark_in_progress(self, obj_id: str) -> bool:
        obj = self.gs.get_objective(obj_id)
        if not obj: return False
        # Reset any other in_progress to pending
        for o in self.gs.objectives:
            if o.status == "in_progress" and o.id != obj_id:
                o.status = "pending"
        obj.status = "in_progress"
        return True

    def reset(self):
        self.gs.objective_update_turn = 0
