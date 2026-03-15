"""
Jericho interface - wraps the Z-machine for structured game access.
Provides inventory, location, objects, exits, save/restore, and score.
"""

import csv
import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict
from jericho import FrotzEnv
from jericho.util import clean


class LocationProxy:
    """Wraps a Jericho ZObject to allow overriding the room name from CSV."""
    def __init__(self, z_object, name_override: str):
        self._z = z_object
        self._name = name_override

    @property
    def name(self): return self._name

    @property
    def num(self): return self._z.num

    def __getattr__(self, attr): return getattr(self._z, attr)


class JerichoInterface:
    """Interface for Zork I via Jericho library."""

    def __init__(self, game_file: str, logger=None):
        if not game_file.endswith(".z5"):
            raise ValueError(f"Game file must be .z5, got: {game_file}")
        self.game_file = game_file
        self.logger = logger
        self.env: Optional[FrotzEnv] = None
        self._room_names: Dict[int, str] = {}

    def __enter__(self): return self
    def __exit__(self, *args): self.close(); return False

    def start(self) -> str:
        if not os.path.exists(self.game_file):
            raise FileNotFoundError(f"Game file not found: {self.game_file}")
        self.env = FrotzEnv(self.game_file)
        intro, _ = self.env.reset()
        self._load_room_names()
        return clean(intro)

    def _load_room_names(self):
        for candidate in [Path(self.game_file).parent / "zorkRooms.csv", Path("game_files") / "zorkRooms.csv"]:
            if candidate.exists():
                try:
                    with open(candidate, encoding="utf-8") as f:
                        for row in csv.DictReader(f):
                            self._room_names[int(row["ZorkID"])] = row["Room Name"]
                except Exception:
                    pass
                break

    def room_name(self, room_id: int) -> Optional[str]:
        return self._room_names.get(room_id)

    def send_command(self, cmd: str) -> str:
        if not self.env: raise RuntimeError("Not started")
        obs, _, _, _ = self.env.step(cmd)
        return clean(obs)

    def get_inventory(self) -> List[str]:
        if not self.env: raise RuntimeError("Not started")
        return [obj.name for obj in self.env.get_inventory()]

    def get_inventory_structured(self) -> List[Any]:
        if not self.env: raise RuntimeError("Not started")
        return self.env.get_inventory()

    def get_location(self) -> Any:
        if not self.env: raise RuntimeError("Not started")
        loc = self.env.get_player_location()
        if loc and self._room_names:
            csv_name = self._room_names.get(loc.num)
            if csv_name:
                return LocationProxy(loc, csv_name)
        return loc

    def get_location_name(self) -> str:
        loc = self.get_location()
        return loc.name if loc else ""

    def get_score(self) -> Tuple[int, int]:
        if not self.env: raise RuntimeError("Not started")
        return (self.env.get_score(), self.env.get_max_score())

    def save_state(self) -> Any:
        if not self.env: raise RuntimeError("Not started")
        return self.env.get_state()

    def restore_state(self, state) -> None:
        if not self.env: raise RuntimeError("Not started")
        self.env.set_state(state)

    def get_visible_objects(self) -> List[Any]:
        """Get objects visible in current location (excludes player)."""
        if not self.env: raise RuntimeError("Not started")
        try:
            loc = self.get_location()
            if not loc: return []
            player = self.env.get_player_object()
            pid = player.num if player else None
            return [o for o in self.env.get_world_objects()
                    if o.parent == loc.num and (pid is None or o.num != pid)]
        except Exception:
            return []

    def get_all_objects(self) -> List[Any]:
        if not self.env: raise RuntimeError("Not started")
        return self.env.get_world_objects()

    def check_attribute(self, obj: Any, bit: int) -> bool:
        if obj is None or not hasattr(obj, 'attr'): return False
        try:
            arr = obj.attr
            return bool(arr[bit] != 0) if arr is not None and 0 <= bit < len(arr) else False
        except (AttributeError, IndexError, TypeError):
            return False

    def get_object_attributes(self, obj: Any) -> Dict[str, bool]:
        if obj is None: return {}
        return {
            'touched': self.check_attribute(obj, 3),
            'container': self.check_attribute(obj, 13),
            'openable': self.check_attribute(obj, 14),
            'takeable': self.check_attribute(obj, 26),
        }

    def get_valid_exits(self) -> List[str]:
        """Test every direction from Z-machine dictionary. 100% accurate."""
        if not self.env: return []

        normalize = {
            "n": "north", "s": "south", "e": "east", "w": "west",
            "u": "up", "d": "down", "ne": "northeast", "nw": "northwest",
            "se": "southeast", "sw": "southwest",
            "northe": "northeast", "northw": "northwest",
            "southe": "southeast", "southw": "southwest",
            "north": "north", "south": "south", "east": "east", "west": "west",
            "up": "up", "down": "down", "northeast": "northeast", "northwest": "northwest",
            "southeast": "southeast", "southwest": "southwest",
        }

        try:
            state = self.env.get_state()
            loc = self.env.get_player_location()
            vocab = self.env.get_dictionary()
            dirs = [w.word for w in vocab if w.is_dir]
            exits = set()

            for d in dirs:
                try:
                    self.env.set_state(state)
                    self.env.step(d)
                    new_loc = self.env.get_player_location()
                    if new_loc and new_loc.num != loc.num:
                        exits.add(normalize.get(d.lower(), d))
                except Exception:
                    continue

            self.env.set_state(state)
            return sorted(exits)
        except Exception:
            return []

    def is_game_over(self, text: str) -> Tuple[bool, Optional[str]]:
        t = text.lower()
        for phrase, reason in [("you have died", "death"), ("you are dead", "death"),
                               ("game over", "game_over"), ("****  you have won  ****", "victory")]:
            if phrase in t:
                return True, reason
        return False, None

    def save_to_file(self, path: str) -> bool:
        try:
            state = self.save_state()
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(state, f)
            return True
        except Exception:
            return False

    def load_from_file(self, path: str) -> bool:
        try:
            with open(path, "rb") as f:
                self.restore_state(pickle.load(f))
            return True
        except Exception:
            return False

    def close(self):
        if self.env:
            try: self.env.close()
            except Exception: pass
            finally: self.env = None
