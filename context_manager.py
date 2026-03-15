"""
Context manager - assembles agent prompts from game state, memories, objectives, map.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime

from config import Config
from state import GameState


class ContextManager:
    """Builds formatted context for agent prompts."""

    def __init__(self, config: Config, game_state: GameState, logger=None):
        self.config = config
        self.game_state = game_state
        self.logger = logger
        self.memory_manager = None  # Injected
        self.pathfinder = None      # Injected

    def build_agent_context(self, game_text: str, extracted=None, map_manager=None) -> Dict[str, Any]:
        """Build context dict from all sources."""
        gs = self.game_state
        ctx = {
            "game_text": game_text,
            "turn": gs.turn_count,
            "score": gs.previous_score,
            "location": gs.current_room_name,
            "location_id": gs.current_room_id,
            "inventory": gs.current_inventory,
            "prev_room": gs.prev_room_name,
            "action_to_current": gs.action_to_current_room,
        }

        # Extracted info
        if extracted:
            ctx["exits"] = extracted.exits
            ctx["visible_objects"] = extracted.visible_objects
            ctx["visible_characters"] = extracted.visible_characters

        # Objectives
        active = gs.active_objectives
        completed = gs.completed_objectives_list
        ctx["objectives"] = active
        ctx["completed_objectives"] = completed[-5:]
        ctx["approach"] = gs.current_approach

        # Map context
        if map_manager:
            ctx["map_mermaid"] = map_manager.game_map.render_mermaid()

        return ctx

    def format_prompt(self, ctx: Dict[str, Any], game_text: str = "") -> str:
        """Format context dict into the text prompt sent to agent."""
        parts = []

        # Game state header
        parts.append(f"=== TURN {ctx.get('turn', 0)} | Score: {ctx.get('score', 0)} ===")
        parts.append(f"Location: {ctx.get('location', '?')} (L{ctx.get('location_id', '?')})")

        if ctx.get('prev_room'):
            parts.append(f"Previous: {ctx['prev_room']} → {ctx.get('action_to_current', '?')}")

        # Inventory
        inv = ctx.get('inventory', [])
        parts.append(f"Inventory: {', '.join(inv) if inv else 'empty'}")

        # Exits
        exits = ctx.get('exits', [])
        if exits:
            parts.append(f"Exits: {', '.join(exits)}")

        # Visible objects/characters
        objs = ctx.get('visible_objects', [])
        if objs:
            parts.append(f"Visible: {', '.join(objs)}")
        chars = ctx.get('visible_characters', [])
        if chars:
            parts.append(f"Characters: {', '.join(chars)}")

        parts.append("")

        # Room description (if recent enough)
        gs = self.game_state
        desc_age = gs.turn_count - gs.last_room_description_turn
        if (gs.last_room_description
            and desc_age <= self.config.room_description_age_window
            and gs.last_room_description_location_id == gs.current_room_id):
            parts.append(f"Room description: {gs.last_room_description}")
            parts.append("")

        # Game text
        if game_text:
            parts.append(f"Game response:\n{game_text}")
            parts.append("")

        # Location memories
        if self.memory_manager:
            mem_text = self.memory_manager.get_location_memory(gs.current_room_id)
            if mem_text:
                parts.append(f"=== LOCATION MEMORIES ===\n{mem_text}")
                parts.append("")

        # Navigation context
        if self.pathfinder:
            nav_ctx = self.pathfinder.get_context()
            if nav_ctx:
                parts.append(nav_ctx)
                parts.append("")

        # Objectives
        active_objs = ctx.get('objectives', [])
        if active_objs:
            parts.append("=== OBJECTIVES ===")
            for obj in active_objs:
                marker = "●" if obj.status == "in_progress" else "○"
                parts.append(f"  {marker} [{obj.id}] {obj.name}: {obj.text}")
            parts.append("")

        approach = ctx.get('approach', '')
        if approach:
            parts.append(f"Suggested approach: {approach}")
            parts.append("")

        completed = ctx.get('completed_objectives', [])
        if completed:
            parts.append("Recently completed:")
            for c in completed[-3:]:
                parts.append(f"  ✓ {c.get('objective', '?')} (turn {c.get('completed_turn', '?')})")
            parts.append("")

        # Recent action history (last 5)
        recent = gs.action_history[-5:]
        if recent:
            parts.append("=== RECENT ACTIONS ===")
            for entry in recent:
                resp_preview = entry.response[:100] + "..." if len(entry.response) > 100 else entry.response
                parts.append(f"  T{entry.turn}: {entry.action} → {resp_preview}")
            parts.append("")

        # Oscillation detection — warn if agent is bouncing between same rooms
        if len(gs.action_history) >= 6:
            recent_locs = [e.location_name for e in gs.action_history[-6:]]
            unique = set(recent_locs)
            if len(unique) <= 2:
                parts.append(f"🚨 OSCILLATION DETECTED: You have been bouncing between {' and '.join(unique)} "
                            f"for the last {len(recent_locs)} turns. STOP and try a completely different direction "
                            f"or approach. Check your map for unexplored exits.")
                parts.append("")

        # Navigation failure message
        if gs.navigation_failure_msg:
            parts.append(f"⚠️ {gs.navigation_failure_msg}")
            gs.navigation_failure_msg = None

        # Map
        mermaid = ctx.get('map_mermaid', '')
        if mermaid and len(mermaid) < 5000:
            parts.append(f"=== WORLD MAP ===\n```mermaid\n{mermaid}\n```")

        return "\n".join(parts)

    def add_action_to_history(self, action: str, response: str, location_id: int, location_name: str):
        """Add completed action to history."""
        from state import ActionEntry
        entry = ActionEntry(
            action=action, response=response,
            location_id=location_id, location_name=location_name,
            turn=self.game_state.turn_count,
        )
        self.game_state.action_history.append(entry)
        self.game_state.action_counts[action.lower()] += 1

    def get_critic_context(self, current_state: str, proposed_action: str,
                           location: str, location_id: int,
                           available_exits: List[str], failed_actions: List[str],
                           agent_reasoning: str = "") -> Dict[str, Any]:
        return {
            "current_state": current_state,
            "proposed_action": proposed_action,
            "location": location,
            "location_id": location_id,
            "available_exits": available_exits,
            "failed_actions": failed_actions,
            "agent_reasoning": agent_reasoning,
        }
