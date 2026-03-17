"""
Map manager - handles map building, movement tracking, and direction detection.
Uses Z-machine ground truth for direction verification.
"""

import os
import re
from typing import Dict, Any, Optional
from pathlib import Path

from config import Config
from state import GameState
from map_graph import MapGraph, DIRECTION_MAP, is_non_movement
from llm_client import LLMClient, extract_json


class MapManager:
    def __init__(self, config: Config, game_state: GameState,
                 jericho=None, llm_client=None, logger=None):
        self.config = config
        self.game_state = game_state
        self.jericho = jericho
        self.llm_client = llm_client
        self.logger = logger

        # Load existing map or start fresh
        map_path = Path(config.game_workdir) / config.map_state_file
        if os.path.exists(map_path):
            loaded = MapGraph.load_from_json(str(map_path), logger)
            self.game_map = loaded or MapGraph(logger)
        else:
            self.game_map = MapGraph(logger)

    def add_initial_room(self, room_id: int, room_name: str):
        if room_name and room_id is not None:
            self.game_map.add_room(room_id, room_name)
            self.game_state.current_room_id = room_id
            self.game_state.current_room_name = room_name

    def update_from_movement(self, action: str, new_id: int, new_name: str,
                             prev_id: int = None, prev_name: str = None,
                             game_response: str = "", pre_state=None):
        """Update map after confirmed movement."""
        try:
            if not new_name or new_id is None: return
            prev_id = prev_id if prev_id is not None else self.game_state.current_room_id
            prev_name = prev_name or self.game_state.current_room_name

            self.game_map.add_room(new_id, new_name)

            if prev_id is not None and prev_id != new_id:
                self._record_connection(action, prev_id, prev_name, new_id, new_name,
                                       game_response, pre_state)

            self.game_state.prev_room_name = self.game_state.current_room_name
            self.game_state.action_to_current_room = action
            self.game_state.current_room_id = new_id
            self.game_state.current_room_name = new_name
        except Exception as e:
            if self.logger: self.logger.error(f"Map update failed: {e}")

    def _record_connection(self, action, from_id, from_name, to_id, to_name,
                           response="", pre_state=None):
        # 1. Z-machine canonical direction test (most reliable)
        direction = self._find_canonical_direction(from_id, to_id, pre_state)
        # 2. Fallback: extract from action text
        if not direction:
            direction = self._extract_direction(action)
        # 3. Last resort: LLM classification
        if not direction:
            direction = self._classify_movement(action, response, from_id, from_name, to_id, to_name)
        if not direction:
            return

        is_new = self.game_map.add_connection(from_id, direction, to_id)
        if is_new:
            self.save_map()

    def _find_canonical_direction(self, from_id, to_id, pre_state=None) -> Optional[str]:
        if not self.jericho or pre_state is None:
            return None
        try:
            current = self.jericho.save_state()
            canonical = ["north", "south", "east", "west", "up", "down",
                        "northeast", "northwest", "southeast", "southwest", "in", "out"]
            found = None
            for d in canonical:
                try:
                    self.jericho.restore_state(pre_state)
                    self.jericho.send_command(d)
                    loc = self.jericho.get_location()
                    if loc and loc.num == to_id:
                        found = d; break
                except Exception:
                    continue
            self.jericho.restore_state(current)
            return found
        except Exception:
            return None

    def _extract_direction(self, action: str) -> Optional[str]:
        if not action: return None
        a = action.lower().strip()
        if a in DIRECTION_MAP: return DIRECTION_MAP[a]
        for prefix in ("go ", "move ", "walk ", "climb ", "run "):
            if a.startswith(prefix):
                rest = a[len(prefix):].strip()
                if rest in DIRECTION_MAP: return DIRECTION_MAP[rest]
        for word in a.split():
            if len(word) > 1 and word in DIRECTION_MAP:
                return DIRECTION_MAP[word]
        return None

    def _classify_movement(self, action, response, from_id, from_name, to_id, to_name) -> Optional[str]:
        if not self.llm_client: return None
        prompt = (f'A Zork player did "{action}" and moved from {from_name} to {to_name}.\n'
                  f'Response: "{response[:300]}"\n'
                  f'Is this a repeatable navigation edge or a side-effect (death/teleport)?\n'
                  f'Reply JSON: {{"navigable": true/false, "edge_label": "short action"}}')
        try:
            resp = self.llm_client.chat.completions.create(
                model=self.config.extractor_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=100, name="MapEdgeClassify",
            )
            import json
            result = json.loads(extract_json(resp.content))
            if result.get("navigable"):
                return result.get("edge_label", action.lower().strip())
        except Exception:
            pass
        return None

    def validate_path(self, hops, num_probes: int = 3) -> dict:
        """Probe each hop in a BFS path using the Z-machine save/restore.

        hops: list of (direction, from_room_id, to_room_id) tuples
        Returns dict with keys:
          - "valid": bool — True if all hops passed all probes
          - "blocked_hop": dict or None — first hop that failed
        """
        if not self.jericho or not hops:
            return {"valid": True, "blocked_hop": None}

        try:
            origin_state = self.jericho.save_state()
        except Exception as e:
            if self.logger:
                self.logger.warning(f"validate_path: could not save state: {e}")
            return {"valid": True, "blocked_hop": None}

        try:
            for hop_index, (direction, from_room_id, to_room_id) in enumerate(hops):
                # Preceding directions needed to reach this hop's start room
                preceding_dirs = [d for d, _, _ in hops[:hop_index]]

                failures = 0
                for _ in range(num_probes):
                    try:
                        self.jericho.restore_state(origin_state)
                        for pre_dir in preceding_dirs:
                            self.jericho.send_command(pre_dir)
                        self.jericho.send_command(direction)
                        loc = self.jericho.get_location()
                        arrived = loc.num if loc else None
                        if arrived != to_room_id:
                            failures += 1
                    except Exception:
                        failures += 1

                if failures == num_probes:
                    from_name = self.game_map.room_names.get(from_room_id, f"Room#{from_room_id}")
                    if self.logger:
                        self.logger.warning(
                            f"validate_path: hop BLOCKED — "
                            f"{from_name}(L{from_room_id}) -{direction}-> "
                            f"(expected L{to_room_id}, all {num_probes} probes failed)"
                        )
                    return {
                        "valid": False,
                        "blocked_hop": {
                            "direction": direction,
                            "room_id": from_room_id,
                            "room_name": from_name,
                            "to_room_id": to_room_id,
                        },
                    }

            return {"valid": True, "blocked_hop": None}
        finally:
            try:
                self.jericho.restore_state(origin_state)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"validate_path: FAILED to restore state: {e}")

    def record_valid_exits(self, room_id: int, exits: list):
        """Store Z-machine ground-truth exits on the room's exit set."""
        if room_id not in self.game_map.rooms:
            return
        room = self.game_map.rooms[room_id]
        before = len(room.exits)
        room.exits.update(exits)
        if len(room.exits) > before and self.logger:
            new = set(exits) - (room.exits - set(exits))
            self.logger.debug(f"Recorded new exits for {room.name} (L{room_id}): {sorted(set(exits) - (room.exits - set(exits)))}")

    def track_failed_action(self, action: str, location_id: int, location_name: str):
        self.game_state.failed_actions_by_location.setdefault(location_name, []).append(action)
        count = self.game_map.track_exit_failure(location_id, action)
        threshold = self.config.exit_failure_threshold
        if count >= threshold:
            self.game_map.prune_invalid_exits(location_id, min_failures=threshold)

    def save_map(self) -> bool:
        path = str(Path(self.config.game_workdir) / self.config.map_state_file)
        return self.game_map.save_to_json(path)

    def get_quality_metrics(self) -> Dict[str, Any]:
        return self.game_map.get_quality_metrics()

    def get_export_data(self) -> Dict[str, Any]:
        return {
            "mermaid_diagram": self.game_map.render_mermaid(),
            "current_room": self.game_state.current_room_name,
            "current_room_id": self.game_state.current_room_id,
            "total_rooms": len(self.game_map.rooms),
            "total_connections": sum(len(c) for c in self.game_map.connections.values()),
            "quality_metrics": self.game_map.get_quality_metrics(),
            "confidence_report": self.game_map.render_confidence_report(),
            "exit_failure_stats": self.game_map.get_exit_failure_stats(),
            "exit_failure_report": self.game_map.render_exit_failure_report(),
            "raw_data": {
                "rooms": {
                    str(rid): {"name": room.name, "exits": sorted(room.exits)}
                    for rid, room in self.game_map.rooms.items()
                },
                "connections": {
                    str(fid): {d: tid for d, tid in conns.items()}
                    for fid, conns in self.game_map.connections.items()
                },
            },
        }
