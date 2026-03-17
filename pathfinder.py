"""
Pathfinder - BFS navigation through known map.
LLM decides WHERE to go, BFS computes HOW to get there.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from config import Config
from state import GameState


@dataclass
class NavState:
    """Active navigation session."""
    target_id: int
    target_name: str
    path: List[str]
    waypoints: List[Dict[str, Any]]
    step: int = 0
    start_id: int = 0
    start_turn: int = 0
    active: bool = True


class Pathfinder:
    def __init__(self, config: Config, game_state: GameState, map_manager=None, logger=None):
        self.config = config
        self.game_state = game_state
        self.map_manager = map_manager
        self.logger = logger
        self.nav: Optional[NavState] = None
        self.max_path = getattr(config, 'pathfinder_max_path_length', 20)
        self._failed_targets: Dict[int, int] = {}  # target_id -> turn when last failed

    def find_path(self, target_id: int, target_name: str) -> Optional[Dict]:
        """Find path from current location to target using BFS."""
        current = self.game_state.current_room_id
        if current is None:
            return None
        if current == target_id:
            return {"found": True, "directions": [], "waypoints": [], "reason": "Already there."}

        game_map = self.map_manager.game_map
        if target_id not in game_map.rooms:
            result = {"found": False, "directions": [], "waypoints": [],
                    "reason": f"L{target_id} not in known map."}
            self._failed_targets[target_id] = self.game_state.turn_count
            return result

        path_result = game_map.find_path_bfs(current, target_id)
        if path_result is None:
            self._failed_targets[target_id] = self.game_state.turn_count
            return {"found": False, "directions": [], "waypoints": [],
                    "reason": f"No path from L{current} to L{target_id}."}

        dirs, wps = path_result
        if len(dirs) > self.max_path:
            return {"found": False, "directions": [], "waypoints": [],
                    "reason": f"Path too long ({len(dirs)} steps, max {self.max_path})."}

        # Probe each hop via Z-machine save/restore to catch impassable connections
        # (e.g., trap door closed, gate locked). wps[i] is the room before dirs[i].
        if self.map_manager and hasattr(self.map_manager, 'validate_path'):
            hops = [
                (dirs[i], wps[i]["room_id"], wps[i + 1]["room_id"])
                for i in range(len(dirs))
            ]
            probe = self.map_manager.validate_path(hops)
            if not probe["valid"]:
                bh = probe["blocked_hop"]
                self._failed_targets[target_id] = self.game_state.turn_count
                return {
                    "found": False,
                    "directions": [],
                    "waypoints": [],
                    "reason": (
                        f"Path blocked at {bh['room_name']}(L{bh['room_id']}) "
                        f"-> {bh['direction']} (connection is currently impassable)."
                    ),
                    "blocked_hop": bh,
                }

        # Clear failure record on success
        self._failed_targets.pop(target_id, None)
        return {"found": True, "directions": dirs, "waypoints": wps,
                "reason": f"Found {len(dirs)}-step path."}

    def recently_failed(self, target_id: int, window: int = 10) -> bool:
        """Check if pathfinding to this target failed recently."""
        fail_turn = self._failed_targets.get(target_id)
        if fail_turn is None:
            return False
        return (self.game_state.turn_count - fail_turn) < window

    def start_navigation(self, target_id: int, target_name: str) -> bool:
        if self.nav: self.cancel()
        result = self.find_path(target_id, target_name)
        if not result or not result["found"] or not result["directions"]:
            return False
        self.nav = NavState(
            target_id=target_id, target_name=target_name,
            path=result["directions"], waypoints=result["waypoints"],
            start_id=self.game_state.current_room_id,
            start_turn=self.game_state.turn_count,
        )
        return True

    def current_direction(self) -> Optional[str]:
        if not self.nav or not self.nav.active: return None
        if self.nav.step >= len(self.nav.path): return None
        return self.nav.path[self.nav.step]

    def advance(self) -> bool:
        if not self.nav or not self.nav.active: return False
        self.nav.step += 1
        if self.nav.step >= len(self.nav.path):
            self.nav.active = False
            return False
        return True

    def cancel(self):
        self.nav = None

    def get_context(self) -> Optional[str]:
        if not self.nav or not self.nav.active: return None
        d = self.current_direction()
        if d is None: return None
        total = len(self.nav.path)
        step = self.nav.step + 1
        return (f"NAVIGATION ACTIVE: Traveling to {self.nav.target_name} (L{self.nav.target_id})\n"
                f"Full path: {' -> '.join(self.nav.path)}\n"
                f"Progress: Step {step} of {total}\n"
                f">>> NEXT DIRECTION: {d} <<<")

    @property
    def is_active(self) -> bool:
        return self.nav is not None and self.nav.active

    def reset(self):
        self.nav = None
        self._failed_targets.clear()
