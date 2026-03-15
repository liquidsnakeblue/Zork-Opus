"""
Map graph - room/connection tracking with Mermaid rendering.
Uses integer room IDs (Z-machine) as primary keys.
"""

import json
import os
from collections import deque
from typing import Dict, Set, List, Optional, Any, Tuple
from datetime import datetime, timezone
from pathlib import Path


CANONICAL_DIRECTIONS = {
    "north", "south", "east", "west", "up", "down",
    "northeast", "northwest", "southeast", "southwest", "in", "out",
}

# Direction normalization (for action text → canonical)
DIRECTION_MAP = {
    "n": "north", "s": "south", "e": "east", "w": "west",
    "u": "up", "d": "down", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest",
    "north": "north", "south": "south", "east": "east", "west": "west",
    "up": "up", "down": "down", "northeast": "northeast", "northwest": "northwest",
    "southeast": "southeast", "southwest": "southwest",
    "in": "in", "out": "out",
    "northward": "north", "southward": "south", "eastward": "east", "westward": "west",
    "climb up": "up", "go up": "up", "climb down": "down", "go down": "down",
}

NON_MOVEMENT = {
    "look", "l", "inventory", "i", "examine", "read", "take", "get", "drop",
    "open", "close", "attack", "kill", "hit", "eat", "drink", "wait", "save",
    "restore", "quit", "score", "diagnose", "pray", "verbose", "brief",
}


def is_non_movement(action: str) -> bool:
    if not action: return True
    first = action.lower().strip().split()[0]
    return first in NON_MOVEMENT


class Room:
    __slots__ = ("id", "name", "exits")

    def __init__(self, room_id: int, name: str):
        self.id = room_id
        self.name = name
        self.exits: Set[str] = set()


class MapGraph:
    def __init__(self, logger=None):
        self.rooms: Dict[int, Room] = {}
        self.room_names: Dict[int, str] = {}
        self.connections: Dict[int, Dict[str, int]] = {}  # {from_id: {direction: to_id}}
        self.exit_failures: Dict[int, Dict[str, int]] = {}
        self.logger = logger

    def add_room(self, room_id: int, name: str) -> bool:
        if room_id in self.rooms:
            return False
        self.rooms[room_id] = Room(room_id, name)
        self.room_names[room_id] = name
        return True

    def add_connection(self, from_id: int, exit_taken: str, to_id: int, confidence: float = 0.9) -> bool:
        """Add a directional connection. Returns True if new."""
        if from_id not in self.connections:
            self.connections[from_id] = {}
        existing = self.connections[from_id].get(exit_taken)
        if existing == to_id:
            return False
        self.connections[from_id][exit_taken] = to_id
        # Update room exits
        if from_id in self.rooms:
            self.rooms[from_id].exits.add(exit_taken)
        return True

    def track_exit_failure(self, room_id: int, direction: str) -> int:
        if room_id not in self.exit_failures:
            self.exit_failures[room_id] = {}
        self.exit_failures[room_id][direction] = self.exit_failures[room_id].get(direction, 0) + 1
        return self.exit_failures[room_id][direction]

    def prune_invalid_exits(self, room_id: int, min_failures: int = 3) -> int:
        failures = self.exit_failures.get(room_id, {})
        pruned = 0
        for direction, count in list(failures.items()):
            if count >= min_failures:
                if room_id in self.connections and direction in self.connections[room_id]:
                    del self.connections[room_id][direction]
                    pruned += 1
                if room_id in self.rooms:
                    self.rooms[room_id].exits.discard(direction)
        return pruned

    def find_path_bfs(self, start_id: int, target_id: int) -> Optional[Tuple[List[str], List[Dict]]]:
        """BFS shortest path. Returns (directions, waypoints) or None."""
        if start_id == target_id: return [], []
        if start_id not in self.rooms or target_id not in self.rooms: return None

        queue = deque([(start_id, [])])
        visited = {start_id}

        while queue:
            current, path = queue.popleft()
            for direction, next_id in self.connections.get(current, {}).items():
                if next_id in visited: continue
                new_path = path + [(direction, next_id)]
                if next_id == target_id:
                    dirs = [d for d, _ in new_path]
                    waypoints = [{"room_id": start_id, "room_name": self.room_names.get(start_id, "?"),
                                  "direction_to_next": new_path[0][0]}]
                    for i, (d, rid) in enumerate(new_path):
                        waypoints.append({
                            "room_id": rid, "room_name": self.room_names.get(rid, "?"),
                            "direction_to_next": new_path[i+1][0] if i < len(new_path)-1 else None
                        })
                    return dirs, waypoints
                visited.add(next_id)
                queue.append((next_id, new_path))
        return None

    def render_mermaid(self) -> str:
        if not self.rooms: return "graph LR\n    Empty[No rooms discovered]"
        lines = ["graph LR"]
        for rid, room in self.rooms.items():
            safe = room.name.replace('"', "'")
            lines.append(f'    R{rid}["{safe}"]')
        for from_id, conns in self.connections.items():
            for direction, to_id in conns.items():
                lines.append(f"    R{from_id} -->|{direction}| R{to_id}")
        return "\n".join(lines)

    def get_quality_metrics(self) -> Dict[str, Any]:
        total_conns = sum(len(c) for c in self.connections.values())
        return {
            "room_count": len(self.rooms),
            "connection_count": total_conns,
            "avg_exits": total_conns / max(1, len(self.rooms)),
        }

    def save_to_json(self, filepath: str) -> bool:
        try:
            data = {
                "rooms": {str(rid): {"name": r.name, "exits": sorted(r.exits)} for rid, r in self.rooms.items()},
                "connections": {str(fid): {d: tid for d, tid in conns.items()} for fid, conns in self.connections.items()},
                "exit_failures": {str(rid): failures for rid, failures in self.exit_failures.items()},
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            if self.logger: self.logger.error(f"Failed to save map: {e}")
            return False

    @classmethod
    def load_from_json(cls, filepath: str, logger=None) -> Optional['MapGraph']:
        try:
            with open(filepath) as f:
                data = json.load(f)
            g = cls(logger=logger)
            for rid_str, info in data.get("rooms", {}).items():
                rid = int(rid_str)
                g.rooms[rid] = Room(rid, info["name"])
                g.rooms[rid].exits = set(info.get("exits", []))
                g.room_names[rid] = info["name"]
            for fid_str, conns in data.get("connections", {}).items():
                fid = int(fid_str)
                g.connections[fid] = {d: int(tid) for d, tid in conns.items()}
            for rid_str, fails in data.get("exit_failures", {}).items():
                g.exit_failures[int(rid_str)] = fails
            return g
        except Exception as e:
            if logger: logger.warning(f"Failed to load map from {filepath}: {e}")
            return None

    def get_exit_failure_stats(self) -> Dict:
        return {str(k): v for k, v in self.exit_failures.items()}

    def render_exit_failure_report(self) -> str:
        if not self.exit_failures: return "No exit failures tracked."
        lines = []
        for rid, fails in self.exit_failures.items():
            name = self.room_names.get(rid, f"Room#{rid}")
            for d, count in fails.items():
                lines.append(f"  {name} -> {d}: {count} failures")
        return "\n".join(lines)

    def render_confidence_report(self) -> str:
        return f"Map: {len(self.rooms)} rooms, {sum(len(c) for c in self.connections.values())} connections"
