"""
Objective manager - handles discovery, tracking, completion, and refinement of objectives.
Uses the reasoner model for strategic objective generation.
"""

import json
import re
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, field_validator
from collections import deque

from config import Config
from state import GameState, Objective
from llm_client import LLMClient, extract_json


# Typed predicate schemas: type name -> (required arg, coercion)
_PREDICATE_TYPES = {
    "inventory_contains": ("item", str),
    "trophy_contains": ("item", str),
    "room_id_equals": ("room_id", int),
    "score_delta_at_least": ("amount", int),
    "new_rooms_since_created": ("count", int),
}


class ObjectiveDefinition(BaseModel):
    category: Literal["exploration", "action"]
    name: str
    text: str
    completion_condition: str
    target_location_id: Optional[int] = None
    completion_predicate: Optional[Dict[str, Any]] = None

    @field_validator("target_location_id", mode="before")
    @classmethod
    def _coerce_location_id(cls, v):
        """Handle LLM returning 'L41', 'Gallery', etc. instead of plain int."""
        if v is None:
            return None
        if isinstance(v, int):
            return v
        if isinstance(v, str):
            # Strip "L" prefix: "L41" → "41"
            cleaned = re.sub(r'^[Ll]', '', v.strip())
            try:
                return int(cleaned)
            except ValueError:
                # Non-numeric string like "Gallery" — ignore
                return None
        return None

    @field_validator("completion_predicate", mode="before")
    @classmethod
    def _sanitize_predicate(cls, v):
        """Keep only well-formed predicates of a known type; else None (LLM fallback)."""
        if not isinstance(v, dict):
            return None
        ptype = v.get("type")
        if ptype not in _PREDICATE_TYPES:
            return None
        arg_name, coerce = _PREDICATE_TYPES[ptype]
        raw = v.get(arg_name)
        if raw is None:
            return None
        try:
            if coerce is int and isinstance(raw, str):
                raw = re.sub(r'^[Ll]', '', raw.strip())
            return {"type": ptype, arg_name: coerce(raw)}
        except (ValueError, TypeError):
            return None


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
                 web_search_manager=None, streaming_server=None, logger=None,
                 treasure_registry=None, procedure_store=None):
        self.config = config
        self.gs = game_state
        self.knowledge = knowledge_manager
        self.map_manager = map_manager
        self.memory = memory_manager
        self.context = context_manager
        self.walkthrough = walkthrough_manager
        self.streaming = streaming_server
        self.web_search = web_search_manager
        self.treasures = treasure_registry
        self.procedures = procedure_store
        self.logger = logger

        # LLM clients
        self.reasoner_client = llm_client or LLMClient(
            config=config, base_url=config.base_url_for("reasoner"),
            api_key=config.api_key_for("reasoner"),
        )
        self.review_client = review_client or self.reasoner_client
        self._reasoner_fail_turn = -999  # Cooldown after failures
        self._reasoner_consecutive_failures = 0

        # Event-driven replanning
        self._pending_events: set = set()
        self._trigger_reason: str = ""

    def notify_event(self, event: str):
        """Record a replanning-worthy event (completion, blockage, theft, score, new room)."""
        self._pending_events.add(event)

    def should_run_reasoner(self) -> bool:
        """Event-driven: run on meaningful events, with a max-gap fallback and
        a min-gap floor to prevent thrash. (Was: flat every-N-turns timer.)"""
        turn = self.gs.turn_count
        if turn == 0: return False
        # Cooldown after failure: wait at least 5 turns before retrying
        if (turn - self._reasoner_fail_turn) < 5:
            return False
        gap = turn - self.gs.objective_update_turn
        if gap < self.config.objective_update_min_gap:
            return False
        if gap >= self.config.objective_update_interval:
            self._trigger_reason = f"Max-gap review ({self.config.objective_update_interval} turns)"
            return True
        if self._pending_events:
            self._trigger_reason = "Events: " + ", ".join(sorted(self._pending_events))
            return True
        return False

    def run_reasoner(self, game_text: str = "") -> Optional[ReasonerResponse]:
        """Call reasoner to generate/update objectives."""
        if not self.config.enable_deep_reasoning:
            return None

        # Build context for reasoner
        max_score = self.gs.max_score or 350
        parts = [f"Turn: {self.gs.turn_count}",
                 f"Score: {self.gs.previous_score} / {max_score}",
                 f"Location: {self.gs.current_room_name} (L{self.gs.current_room_id})",
                 f"Inventory: {', '.join(self.gs.current_inventory) if self.gs.current_inventory else 'empty'}"]

        # Treasure tracker (deterministic, from Z-machine object tree)
        if self.treasures:
            parts.append("\n" + self.treasures.format_for_reasoner())
        elif self.gs.trophy_case:
            deposited = sorted(self.gs.trophy_case)
            parts.append(f"Trophy case ({len(deposited)} deposited): {', '.join(deposited)}")

        # Theft awareness — alert reasoner about stolen items
        if self.gs.theft_events:
            parts.append("\n🚨 THEFT ALERT — Items stolen by the Thief:")
            for evt in self.gs.theft_events:
                parts.append(f"  • {evt['item']} stolen at {evt['location']} (T{evt['turn']})")
            parts.append("These items are in the Thief's possession or his Treasure Room. "
                         "The agent must find and defeat the Thief to recover them. "
                         "Do NOT create objectives to search rooms where the items were lost — they are NOT there.")

        # Score stagnation awareness
        turns_since_score = self.gs.turn_count - self.gs.last_scoring_turn
        if turns_since_score > 15:
            # Check if agent is trapped (can't reach scoring location)
            trapped_msg = ""
            if self.map_manager:
                gm = self.map_manager.game_map
                current = self.gs.current_room_id
                # Try to find path to any room in the "house" area (typical scoring rooms)
                scoring_rooms = [rid for rid, name in gm.room_names.items()
                                if any(kw in name.lower() for kw in ["living room", "kitchen", "attic"])]
                can_reach_scoring = False
                for sr in scoring_rooms:
                    if gm.find_path_bfs(current, sr) is not None:
                        can_reach_scoring = True
                        break
                if scoring_rooms and not can_reach_scoring:
                    trapped_msg = (
                        "\n🚨 TRAPPED: Agent cannot reach the Living Room (trophy case) from current location! "
                        "The trap door is likely one-way. Create objectives to find ALTERNATE ROUTES "
                        "to the surface: look for chimneys, gratings, passages up, or unexplored exits "
                        "that might connect to the house area. This is the TOP PRIORITY.")

            parts.append(f"\n⚠️ SCORE STAGNATION: No score increase in {turns_since_score} turns. "
                        f"Current strategy is not working. Prioritize EXPLORATION objectives "
                        f"to discover new areas and items.{trapped_msg}")

        # Unreachable rooms (from failed pathfinder targets)
        from pathfinder import Pathfinder
        pf = None
        # Find pathfinder from context_manager injection
        if self.context and hasattr(self.context, 'pathfinder') and self.context.pathfinder:
            pf = self.context.pathfinder
        if pf and pf._failed_targets:
            failed_names = []
            for tid, fail_turn in pf._failed_targets.items():
                rname = self.map_manager.game_map.room_names.get(tid, f"Room#{tid}") if self.map_manager else f"L{tid}"
                failed_names.append(f"{rname} (L{tid}, failed T{fail_turn})")
            if failed_names:
                parts.append(f"\n🚫 UNREACHABLE ROOMS (no path from current location):")
                for fn in failed_names:
                    parts.append(f"  • {fn}")
                parts.append("The agent CANNOT reach these rooms via known connections. "
                            "Create objectives to explore NEW exits and discover alternate routes.")

        # Current objectives
        active = self.gs.active_objectives
        if active:
            parts.append("\nCURRENT OBJECTIVES:")
            for o in active:
                parts.append(f"  [{o.id}] {o.name}: {o.text} (condition: {o.completion_condition})")

        # Blocked objectives — the operator hit a hard wall on these
        blocked = self.gs.blocked_objectives
        if blocked:
            parts.append("\n⛔ BLOCKED OBJECTIVES (target unreachable — abandon them, or "
                         "create objectives that clear the blocker / find another route):")
            for o in blocked:
                parts.append(f"  [{o.id}] {o.name}: {o.blocked_reason or 'no path to target'}")

        completed = self.gs.completed_objectives_list[-5:]
        if completed:
            parts.append("\nRECENTLY COMPLETED:")
            for c in completed:
                parts.append(f"  ✓ {c['objective']} (turn {c['completed_turn']})")

        # Recent actions (extended window for better frontier awareness)
        recent = self.gs.action_history[-50:]
        if recent:
            parts.append(f"\nRECENT ACTIONS (last {len(recent)} turns):")
            for e in recent:
                parts.append(f"  T{e.turn}: {e.action} @ {e.location_name} → {e.response[:80]}")

        # Map info + exploration frontier
        if self.map_manager:
            metrics = self.map_manager.get_quality_metrics()
            parts.append(f"\nMap: {metrics.get('room_count', 0)} rooms, {metrics.get('connection_count', 0)} connections")

            # Exploration frontier: rooms with untried exits (from Z-machine ground truth)
            frontier = self.map_manager.game_map.get_exploration_frontier()
            if frontier:
                parts.append(f"\n{frontier}")

        # Knowledgebase (strategic analysis from gameplay)
        if self.knowledge:
            try:
                kb_path = self.knowledge.output_file
                from pathlib import Path
                kb_text = Path(kb_path).read_text().strip()
                if kb_text:
                    parts.append(f"\nKNOWLEDGEBASE (learned from gameplay):\n{kb_text[:3000]}")
            except (FileNotFoundError, AttributeError):
                pass

        # Verified procedures (canonical — outrank memories and walkthrough)
        if self.procedures:
            proc_text = self.procedures.format()
            if proc_text:
                parts.append(f"\n{proc_text}")

        # Key puzzle memories (PUZZLE and DISCOVERY, non-superseded)
        if self.memory:
            puzzle_mems = self.memory.get_puzzle_summary(max_entries=30)
            if puzzle_mems:
                parts.append(f"\nKEY PUZZLE KNOWLEDGE:\n{puzzle_mems}")

        # Walkthrough
        if self.walkthrough:
            wt = self.walkthrough.get_content()
            if wt:
                budget = self.config.reasoner_walkthrough_chars
                wt_slice = wt[:budget]
                if len(wt) > budget:
                    wt_slice += f"\n[... walkthrough truncated: {len(wt) - budget} of {len(wt)} chars omitted ...]"
                    if self.logger:
                        self.logger.info(f"Reasoner walkthrough truncated to {budget}/{len(wt)} chars")
                parts.append(f"\nWALKTHROUGH GUIDE:\n{wt_slice}")

        context = "\n".join(parts)

        prompt = f"""You are a strategic advisor for a text adventure game agent playing Zork I. Analyze the game state and output STRUCTURED objectives as JSON. Objectives should be well thought out and reasoned with a focus on high-level goals and milestones that would help the agent beat the game.

=== VICTORY CONTRACT ===
Zork I is won at {max_score} points: ALL 19 treasures (see TREASURE TRACKER) must be deposited in the Living Room trophy case. Points come from acquiring treasures AND from depositing them (deposit is usually worth more). When all 19 are deposited, a map appears in the trophy case leading to the Stone Barrow endgame — that is the true finish, not merely "some treasures in the case." Use the tracker to target specific missing treasures and their prerequisites instead of re-harvesting familiar early points.

=== SYSTEM ARCHITECTURE (what you DON'T need to plan for) ===
The agent has automated systems that handle navigation and mapping. Your job is STRATEGY, not logistics.

**Automatic Mapping**: Every time the agent moves between rooms, the map system records the connection using Z-machine ground truth. Room exits are detected automatically. The map grows deterministically as the agent explores — you do NOT need to create objectives like "map the area" or "discover connections."

**Automatic Navigation**: When the agent has an in_progress objective with a target_location_id, the system automatically computes and displays the shortest path (BFS). The agent sees step-by-step directions in its context. You do NOT need to include routes, paths, or navigation instructions in objectives — just set the target_location_id and the system handles the rest.

**Navigation Limitations**: The pathfinder can ONLY route through rooms the agent has already visited, using connections it has already traversed. The map is a directed graph — if the agent walked north from A to B, it knows A→north→B, but does NOT know B→south→A until it actually walks south from B. Zork's world is treacherous: doors lock behind you, trap doors close, one-way passages exist. A path that worked before may be blocked now. When the pathfinder fails, the agent must explore manually to discover new routes. Keep this in mind when setting target_location_id — if the agent has never been to a room from its current area, the pathfinder won't help.

**Exploration Frontier**: The map tracks which room exits have been tried vs untried. This data is shown above. Use it to identify where NEW discoveries are most likely.

Your role: decide WHAT the agent should do and WHERE, not HOW to get there.

{context}

=== YOUR TASK ===
Create or update objectives for the agent. You have FULL CONTROL over objectives:
- You can CREATE new objectives
- You can REMOVE objectives that are stale, completed, or no longer relevant

**CRITICAL: PUZZLE PREREQUISITES** - Study the KEY PUZZLE KNOWLEDGE above carefully. Many puzzles require specific items from other locations. Create PREREQUISITE objectives BEFORE the puzzle objective. Example: if a ritual needs a matchbook, create "Get Matchbook from Dam Lobby" BEFORE "Perform Hades Exorcism". The agent will waste turns attempting puzzles without required items if you don't plan the full sequence.

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
- **NEVER create an objective that contradicts KEY PUZZLE KNOWLEDGE or memories above.** Those are ground truth learned from the actual game. If memory says an item is NOT at a location (e.g. "Attic contains no nest"), or that an item's real source is elsewhere (e.g. the jeweled egg is up the tree in the forest, NOT the attic), do NOT create an objective pointing to the wrong place. Trust observed game facts over your own assumptions about Zork.
- If an existing objective contradicts what memory now proves (wrong item location, already-tried-and-failed), ADD its id to abandon_objective_ids.
- Only reference location IDs that exist in the map data above
- Do NOT suggest acquiring items already in inventory: {self.gs.current_inventory}
- Do NOT create objectives to collect or deposit items already in the trophy case: {sorted(self.gs.trophy_case) if self.gs.trophy_case else 'none yet'}
- Keep objectives SHORT and CLEAN (no long paragraphs)
- Aim for 3-7 total active objectives
- **NEVER create objectives that duplicate existing active ones** (check CURRENT OBJECTIVES above)
- **ONE LOCATION PER OBJECTIVE (MANDATORY)**: Each objective MUST target exactly ONE location. Multi-location objectives break pathfinder navigation. Split multi-location tasks into separate objectives.
  * BAD: "Collect sword from Living Room, rope from Attic, and sack from Kitchen"
  * GOOD: "Collect Sword" at L193, "Collect Rope" at L201, "Collect Sack" at L203
- **NEVER include navigation paths in objectives or suggested_approach** — navigation is automatic. Only specify the DESTINATION (target_location_id), not the route.
- **ALWAYS include target_location_id** for ACTION objectives

=== OUTPUT FORMAT ===
For each objective, provide:
- category: "exploration" or "action"
- name: A CONCISE 3-5 word TITLE (e.g., "Collect Brass Lantern", "Open Trap Door")
- text: Short, clean description (1 sentence)
- completion_condition: EXPLICIT, VERIFIABLE condition (e.g., "egg is in inventory", "score increased", "current location is Kitchen")
- target_location_id: For ACTION objectives, the target location number (e.g., 42 for L42)
- completion_predicate: a TYPED condition the system verifies IN CODE every turn (no LLM judgment — far more reliable than prose). Provide one whenever the objective fits a type; omit only for genuinely fuzzy goals. Types:
    {{"type": "inventory_contains", "item": "sword"}}          — item name fragment in inventory
    {{"type": "trophy_contains", "item": "painting"}}          — treasure deposited in trophy case
    {{"type": "room_id_equals", "room_id": 107}}               — agent standing in that room
    {{"type": "score_delta_at_least", "amount": 5}}            — score gained since objective created
    {{"type": "new_rooms_since_created", "count": 3}}          — new rooms discovered since created

Also provide:
- suggested_approach: A CONCISE paragraph (2-3 sentences) explaining how to accomplish the objectives you are creating. ONLY discuss the objectives you are outputting — do not speculate beyond them. The reasoner runs again automatically when these objectives are completed.
- reasoning: Brief summary of the strategic plan

First, THINK DEEPLY about the game state. Then output JSON in ```json fences:
```json
{{
  "reasoning": "Agent needs equipment. Each objective targets ONE location.",
  "suggested_approach": "Collect the sword from the Living Room — it is needed to defeat the troll blocking underground access.",
  "new_objectives": [
    {{"category": "action", "name": "Collect Sword", "text": "Take the elvish sword from the Living Room", "completion_condition": "sword is in inventory", "target_location_id": 193, "completion_predicate": {{"type": "inventory_contains", "item": "sword"}}}}
  ],
  "abandon_objective_ids": ["A001"]
}}
```"""

        # Determine trigger reason (needed before broadcast)
        if self.gs.turn_count == 0:
            trigger = "Episode start"
        else:
            trigger = self._trigger_reason or "Objective review"
        self._pending_events.clear()
        self._trigger_reason = ""

        try:
            rs = self.config.reasoner_sampling
            # Use agent model as fallback after 3 consecutive reasoner failures
            model = self.config.reasoner_model
            if (self._reasoner_consecutive_failures >= 3
                    and self.config.agent_model != self.config.reasoner_model):
                model = self.config.agent_model
                if self.logger:
                    self.logger.info(f"Reasoner using fallback model: {model}")

            # Streaming: broadcast chunks to viewer in real-time
            if self.streaming:
                self.streaming.broadcast_reasoner_start(self.gs.turn_count, trigger)
                def on_reasoner_chunk(accumulated):
                    self.streaming.broadcast_reasoner_chunk(self.gs.turn_count, accumulated)
                resp = self.reasoner_client.chat.completions.create_streaming(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=rs.get("temperature", 0.7),
                    max_tokens=rs.get("max_tokens", 16000),
                    enable_thinking=rs.get("enable_thinking"),
                    name="ObjectiveReasoner",
                    on_chunk=on_reasoner_chunk,
                )
            else:
                resp = self.reasoner_client.chat.completions.create(
                    model=model,
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
            self._reasoner_consecutive_failures = 0  # Reset on success
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

            # Update trigger reason if objectives were abandoned
            if result.abandon_objective_ids:
                trigger = f"Periodic review (abandoned {len(result.abandon_objective_ids)})"

            # Broadcast completion to viewer
            if self.streaming:
                self.streaming.broadcast_reasoner_complete(
                    self.gs.turn_count, result.reasoning, result.suggested_approach,
                    new_obj_data, result.abandon_objective_ids,
                )

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
            self._reasoner_fail_turn = self.gs.turn_count
            self._reasoner_consecutive_failures += 1
            # Fallback: create basic objectives if none exist
            if not self.gs.active_objectives:
                self._create_fallback_objectives()
            return None

    def _create_fallback_objectives(self):
        """Create basic objectives when the reasoner LLM is unavailable."""
        existing_names = {o.name.lower().strip() for o in self.gs.objectives
                         if o.status in ("pending", "in_progress")}
        fallbacks = [
            ObjectiveDefinition(
                category="exploration", name="Explore New Areas",
                text="Discover at least 3 new rooms by exploring unexplored exits",
                completion_condition="3 new rooms discovered"),
            ObjectiveDefinition(
                category="action", name="Collect Valuables",
                text="Pick up any treasures or useful items found during exploration",
                completion_condition="new valuable item in inventory"),
            ObjectiveDefinition(
                category="action", name="Deposit Treasures",
                text="Return treasures to the Living Room trophy case to score points",
                completion_condition="score increased from depositing treasure"),
        ]
        for defn in fallbacks:
            if defn.name.lower().strip() in existing_names:
                continue
            obj_id = self.gs.next_objective_id(defn.category)
            from state import Objective
            obj = Objective(
                id=obj_id, category=defn.category, name=defn.name,
                text=defn.text, completion_condition=defn.completion_condition,
                created_turn=self.gs.turn_count,
            )
            self.gs.objectives.append(obj)
            existing_names.add(defn.name.lower().strip())
        if self.logger:
            self.logger.info("Created fallback objectives (reasoner unavailable)")

    def _apply_reasoner_result(self, result: ReasonerResponse):
        # Abandon objectives (blocked ones may be abandoned too)
        for obj_id in result.abandon_objective_ids:
            obj = self.gs.get_objective(obj_id)
            if obj and obj.status in ("pending", "in_progress", "blocked"):
                obj.status = "abandoned"

        # Add new objectives (skip duplicates)
        existing_names = {o.name.lower().strip() for o in self.gs.objectives
                         if o.status in ("pending", "in_progress")}
        room_count = len(self.map_manager.game_map.rooms) if self.map_manager else 0
        for defn in result.new_objectives[:3]:
            if defn.name.lower().strip() in existing_names:
                continue  # Skip duplicate
            target_id, note = self._resolve_target(
                defn.name, defn.text, defn.target_location_id)
            if note and self.logger:
                self.logger.warning(f"Objective target validation ({defn.name!r}): {note}")
            obj_id = self.gs.next_objective_id(defn.category)
            obj = Objective(
                id=obj_id, category=defn.category, name=defn.name,
                text=defn.text, completion_condition=defn.completion_condition,
                created_turn=self.gs.turn_count,
                target_location_id=target_id,
                completion_predicate=defn.completion_predicate,
                created_score=self.gs.previous_score,
                created_room_count=room_count,
            )
            self.gs.objectives.append(obj)
            existing_names.add(defn.name.lower().strip())

        # Update approach
        if result.suggested_approach:
            self.gs.current_approach = result.suggested_approach

    def _resolve_target(self, name: str, text: str, target_id: Optional[int]):
        """Validate a reasoner-supplied target_location_id against the room registry.

        Fixes mechanical mismatches like "Reach Round Room" pointed at
        East-West Passage (L41) instead of Round Room (L107). Conservative:
        only corrects when the objective names a known room and the target is
        a DIFFERENT known room not mentioned anywhere in the objective.
        Returns (resolved_id_or_None, note_or_None).
        """
        if not self.map_manager:
            return target_id, None
        rooms = self.map_manager.game_map.room_names  # {id: name}

        def mentioned(s: str):
            s_low = (s or "").lower()
            hits = [(rid, rn) for rid, rn in rooms.items()
                    if len(rn) >= 4 and rn.lower() in s_low]
            hits.sort(key=lambda x: -len(x[1]))  # prefer the most specific name
            return hits

        hits = mentioned(name) or mentioned(text)

        if target_id is not None and target_id in rooms:
            tgt_name = rooms[target_id].lower()
            if hits:
                best_id, best_name = hits[0]
                target_mentioned = any(rn.lower() == tgt_name for _, rn in hits)
                # "Reservoir" mentioned only as part of "Reservoir South" is not
                # a real mention of the target — the more specific room wins.
                shadowed = (target_mentioned and best_id != target_id
                            and tgt_name in best_name.lower()
                            and len(best_name) > len(tgt_name))
                if (not target_mentioned or shadowed) and best_id != target_id:
                    return best_id, (f"corrected target L{target_id} ({rooms[target_id]}) → "
                                     f"L{best_id} ({best_name}) — objective names that room")
            return target_id, None

        # Target missing or not a known room: try to resolve from the wording
        if hits:
            best_id, best_name = hits[0]
            return best_id, f"resolved target from wording → L{best_id} ({best_name})"
        if target_id is not None:
            return None, f"dropped unknown room id L{target_id} (not in map registry)"
        return target_id, None

    def mark_blocked(self, obj_id: str, reason: str) -> bool:
        """Mark an objective blocked (target unreachable). Reasoner decides its fate."""
        obj = self.gs.get_objective(obj_id)
        if not obj or obj.status not in ("pending", "in_progress"):
            return False
        obj.status = "blocked"
        obj.blocked_reason = reason
        if self.logger:
            self.logger.warning(f"Objective [{obj_id}] {obj.name} BLOCKED: {reason}")
        self.notify_event("objective_blocked")
        return True

    # ── Deterministic completion (typed predicates, no LLM) ──

    def check_completions_deterministic(self) -> List[str]:
        """Evaluate typed completion predicates in code. Runs every turn; free.
        Blocked objectives are included — reaching the goal unblocks/completes."""
        completed = []
        for o in self.gs.objectives:
            if o.status not in ("pending", "in_progress", "blocked"):
                continue
            if not o.completion_predicate:
                continue
            try:
                if self._eval_predicate(o, o.completion_predicate):
                    o.status = "completed"
                    o.completed_turn = self.gs.turn_count
                    completed.append(o.id)
                    if self.logger:
                        self.logger.info(
                            f"Objective [{o.id}] {o.name} completed by predicate "
                            f"{o.completion_predicate}")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Predicate eval failed for [{o.id}]: {e}")
        if completed:
            self.notify_event("objective_completed")
            self.gs.objective_review_results[self.gs.turn_count] = {
                "content": "\n".join(
                    f"[{oid}] completed: typed predicate satisfied" for oid in completed),
                "completed_objectives": [
                    {"id": oid, "name": (self.gs.get_objective(oid).name
                                          if self.gs.get_objective(oid) else oid)}
                    for oid in completed],
                "updates": [],
            }
        return completed

    def _eval_predicate(self, obj: Objective, pred: Dict[str, Any]) -> bool:
        ptype = pred.get("type")
        if ptype == "inventory_contains":
            frag = str(pred.get("item", "")).lower()
            return bool(frag) and any(frag in i.lower() for i in self.gs.current_inventory)
        if ptype == "trophy_contains":
            frag = str(pred.get("item", "")).lower()
            if not frag:
                return False
            if self.treasures and self.treasures.is_deposited(frag):
                return True
            return any(frag in i.lower() for i in self.gs.trophy_case)
        if ptype == "room_id_equals":
            return self.gs.current_room_id == int(pred.get("room_id", -1))
        if ptype == "score_delta_at_least":
            return (self.gs.previous_score - obj.created_score) >= int(pred.get("amount", 0))
        if ptype == "new_rooms_since_created":
            if not self.map_manager:
                return False
            grown = len(self.map_manager.game_map.rooms) - obj.created_room_count
            return grown >= int(pred.get("count", 0))
        return False

    def needs_llm_completion_check(self) -> bool:
        """True if any active objective lacks a typed predicate (LLM must judge)."""
        return any(not o.completion_predicate for o in self.gs.active_objectives)

    def check_completions(self, action: str, response: str) -> List[str]:
        """LLM review of PROSE completion conditions. Objectives with typed
        predicates are handled in code by check_completions_deterministic and
        are excluded here. Returns completed IDs."""
        if not self.config.enable_objective_completion_llm_check:
            return []

        active = [o for o in self.gs.active_objectives if not o.completion_predicate]
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
            # Streaming: broadcast chunks to viewer in real-time
            if self.streaming:
                self.streaming.broadcast_objective_review_start(self.gs.turn_count, len(active))
                def on_review_chunk(accumulated):
                    self.streaming.broadcast_objective_review_chunk(self.gs.turn_count, accumulated)
                resp = self.review_client.chat.completions.create_streaming(
                    model=self.config.agent_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.turn_review_sampling.get("temperature", 0.1),
                    max_tokens=self.config.turn_review_sampling.get("max_tokens", 2000),
                    enable_thinking=self.config.turn_review_sampling.get("enable_thinking"),
                    name="ObjectiveReview",
                    on_chunk=on_review_chunk,
                )
            else:
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

            review_content = "\n".join(review_lines) if review_lines else "No objective changes."
            update_dicts = [u.model_dump() for u in result.updates]

            # Merge with any deterministic-predicate result already stored this turn
            prior = self.gs.objective_review_results.get(self.gs.turn_count)
            if prior:
                review_content = prior["content"] + "\n" + review_content
                completed_objs = prior["completed_objectives"] + completed_objs
                update_dicts = prior.get("updates", []) + update_dicts
            self.gs.objective_review_results[self.gs.turn_count] = {
                "content": review_content,
                "completed_objectives": completed_objs,
                "updates": update_dicts,
            }

            # Broadcast completion to viewer
            if self.streaming:
                self.streaming.broadcast_objective_review_complete(
                    self.gs.turn_count, review_content, completed_objs, update_dicts,
                )

            if completed_ids:
                self.notify_event("objective_completed")
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
