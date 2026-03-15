"""
Central game state - single dataclass holding all mutable state.
All managers read/write this directly. No getters/setters ceremony.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Any, Literal
from datetime import datetime
from collections import Counter
from pydantic import BaseModel, Field


class ActionEntry(BaseModel):
    """Single entry in action history."""
    action: str
    response: str
    location_id: int
    location_name: str
    turn: int = 0
    status: Literal["pending_ai", "complete"] = "complete"


class Objective(BaseModel):
    """Structured objective with lifecycle tracking."""
    id: str
    category: Literal["exploration", "action"]
    name: str
    text: str
    completion_condition: str
    status: Literal["pending", "in_progress", "completed", "abandoned"] = "pending"
    created_turn: int = 0
    completed_turn: Optional[int] = None
    target_location_id: Optional[int] = None
    progress: Optional[str] = None


@dataclass
class GameState:
    """All mutable state for one episode. Managers access fields directly."""

    # Core
    episode_id: str = ""
    turn_count: int = 0
    current_room_id: int = 0
    current_room_name: str = ""
    current_inventory: List[str] = field(default_factory=list)
    previous_score: int = 0
    game_over: bool = False

    # Initial state (for viewer first card)
    initial_response: str = ""
    initial_location_id: Optional[int] = None
    initial_location_name: str = ""

    # Room descriptions
    last_room_description: str = ""
    last_room_description_turn: int = 0
    last_room_description_location_id: Optional[int] = None

    # Navigation
    prev_room_name: Optional[str] = None
    action_to_current_room: Optional[str] = None
    visited_locations: Set[str] = field(default_factory=set)
    navigation_failure_msg: Optional[str] = None
    last_action_moved: bool = False

    # Last turn deltas (for context display)
    last_score_delta: int = 0
    last_items_gained: List[str] = field(default_factory=list)
    last_items_lost: List[str] = field(default_factory=list)
    last_scoring_turn: int = 0  # Turn when score last increased

    # Action tracking
    action_counts: Counter = field(default_factory=Counter)
    action_history: List[ActionEntry] = field(default_factory=list)
    reasoning_history: List[Dict[str, Any]] = field(default_factory=list)
    memory_log_history: List[Dict[str, Any]] = field(default_factory=list)
    failed_actions_by_location: Dict[str, List[str]] = field(default_factory=dict)

    # Critic / extraction history (for viewer)
    critic_history: List[Dict[str, Any]] = field(default_factory=list)
    extracted_info_history: List[Dict[str, Any]] = field(default_factory=list)

    # Rejections
    rejected_actions_per_turn: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict)

    # Objectives
    objectives: List[Objective] = field(default_factory=list)
    objective_update_turn: int = 0
    objective_id_counter: int = 0
    current_approach: str = ""
    objective_staleness: Dict[str, int] = field(default_factory=dict)

    # Knowledge tracking
    last_knowledge_update_turn: int = 0

    # Tool invocation (for viewer)
    active_tool: Optional[str] = None
    active_tool_data: Optional[Dict[str, Any]] = None
    tool_history: List[Dict[str, Any]] = field(default_factory=list)

    # Reasoner events / post-action results (for viewer persistence)
    reasoner_events: List[Dict[str, Any]] = field(default_factory=list)
    memory_synthesis_results: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    objective_review_results: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    # Progressive rendering
    pending_response: Optional[str] = None
    pending_location_id: Optional[int] = None
    pending_location_name: Optional[str] = None
    pending_timestamp: Optional[str] = None

    # Session-persistent
    death_count: int = 0
    death_counted_this_episode: bool = False

    def reset_episode(self, episode_id: str) -> None:
        """Reset all episode-specific state. death_count persists."""
        self.episode_id = episode_id
        self.turn_count = 0
        self.current_room_id = 0
        self.current_room_name = ""
        self.current_inventory.clear()
        self.previous_score = 0
        self.game_over = False
        self.initial_response = ""
        self.initial_location_id = None
        self.initial_location_name = ""
        self.last_room_description = ""
        self.last_room_description_turn = 0
        self.last_room_description_location_id = None
        self.prev_room_name = None
        self.action_to_current_room = None
        self.visited_locations.clear()
        self.navigation_failure_msg = None
        self.last_action_moved = False
        self.last_score_delta = 0
        self.last_items_gained.clear()
        self.last_items_lost.clear()
        self.last_scoring_turn = 0
        self.action_counts.clear()
        self.action_history.clear()
        self.reasoning_history.clear()
        self.memory_log_history.clear()
        self.failed_actions_by_location.clear()
        self.critic_history.clear()
        self.extracted_info_history.clear()
        self.rejected_actions_per_turn.clear()
        self.objectives.clear()
        self.objective_update_turn = 0
        self.objective_id_counter = 0
        self.current_approach = ""
        self.objective_staleness.clear()
        self.last_knowledge_update_turn = 0
        self.active_tool = None
        self.active_tool_data = None
        self.tool_history.clear()
        self.reasoner_events.clear()
        self.memory_synthesis_results.clear()
        self.objective_review_results.clear()
        self.pending_response = None
        self.pending_location_id = None
        self.pending_location_name = None
        self.pending_timestamp = None
        self.death_counted_this_episode = False

    # ── Objective helpers ──

    @property
    def active_objectives(self) -> List[Objective]:
        return [o for o in self.objectives if o.status in ("pending", "in_progress")]

    @property
    def completed_objectives_list(self) -> List[Dict[str, Any]]:
        return [
            {"objective": o.text, "completed_turn": o.completed_turn, "id": o.id, "category": o.category}
            for o in self.objectives if o.status == "completed"
        ]

    @property
    def discovered_objective_texts(self) -> List[str]:
        return [o.text for o in self.active_objectives]

    def next_objective_id(self, category: str) -> str:
        self.objective_id_counter += 1
        prefix = "E" if category == "exploration" else "A"
        return f"{prefix}{self.objective_id_counter:03d}"

    def get_objective(self, obj_id: str) -> Optional[Objective]:
        for o in self.objectives:
            if o.id == obj_id:
                return o
        return None

    def get_export_data(self) -> Dict[str, Any]:
        return {
            "metadata": {
                "episode_id": self.episode_id, "turn_count": self.turn_count,
                "export_timestamp": datetime.now().isoformat(), "game_over": self.game_over,
            },
            "game_state": {
                "current_room_id": self.current_room_id, "current_room": self.current_room_name,
                "current_inventory": self.current_inventory, "current_score": self.previous_score,
                "visited_locations": list(self.visited_locations), "death_count": self.death_count,
            },
            "objectives": [o.model_dump() for o in self.objectives],
            "suggested_approach": self.current_approach,
        }
