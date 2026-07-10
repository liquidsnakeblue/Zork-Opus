"""
Treasure registry - deterministic tracking of Zork I's 19 treasures.

State comes from the Z-machine object tree, scoped to what the agent could
legitimately know (fog of war preserved): a treasure is only revealed when it
is actually possessed (inventory / trophy case), lying directly in the room
the agent occupies, has been touched, or was stolen from the agent by the
Thief. Unseen treasures stay "unknown" — the registry never leaks positions
the agent hasn't discovered.

Cross-episode knowledge (where a treasure was first found) persists in
game_files/treasure_state.json so later generations start with a hint,
mirroring what the memory system would have recorded anyway.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# (key, display name, lowercase name fragments matched against Z-object names)
TREASURES = [
    ("egg",          "jewel-encrusted egg",          ["egg"]),
    ("canary",       "golden clockwork canary",      ["canary"]),
    ("bauble",       "brass bauble",                 ["bauble"]),
    ("painting",     "painting",                     ["painting"]),
    ("platinum-bar", "platinum bar",                 ["platinum"]),
    ("torch",        "ivory torch",                  ["torch"]),
    ("coffin",       "gold coffin",                  ["coffin"]),
    ("sceptre",      "sceptre",                      ["sceptre", "scepter"]),
    ("trunk",        "trunk of jewels",              ["trunk"]),
    ("trident",      "crystal trident",              ["trident"]),
    ("figurine",     "jade figurine",                ["figurine"]),
    ("bracelet",     "sapphire-encrusted bracelet",  ["bracelet"]),
    ("diamond",      "huge diamond",                 ["diamond"]),
    ("coins",        "bag of coins",                 ["coins"]),
    ("skull",        "crystal skull",                ["skull"]),
    ("scarab",       "jeweled scarab",               ["scarab"]),
    ("emerald",      "large emerald",                ["emerald"]),
    ("chalice",      "silver chalice",               ["chalice"]),
    ("pot-of-gold",  "pot of gold",                  ["pot of gold"]),
]

# Episode-scoped treasure statuses
UNKNOWN, LOCATED, CARRIED, DEPOSITED, STOLEN = (
    "unknown", "located", "carried", "deposited", "stolen")


class TreasureRegistry:
    """Tracks the 19 treasures deterministically from the Z-machine object tree."""

    def __init__(self, config, logger=None):
        self.logger = logger
        self.path = Path(config.game_workdir) / "treasure_state.json"
        self.knowledge: Dict[str, Dict[str, Any]] = {}
        self._load()
        self.episode_state: Dict[str, Dict[str, Any]] = {}
        self.reset_episode()

    # ── Persistence ──

    def _load(self):
        try:
            if self.path.exists():
                self.knowledge = json.loads(self.path.read_text())
        except Exception as e:
            if self.logger:
                self.logger.warning(f"TreasureRegistry: could not load {self.path}: {e}")
            self.knowledge = {}

    def _save(self):
        try:
            self.path.write_text(json.dumps(self.knowledge, indent=2))
        except Exception as e:
            if self.logger:
                self.logger.warning(f"TreasureRegistry: could not save: {e}")

    def reset_episode(self):
        self.episode_state = {
            key: {"status": UNKNOWN, "room_id": None, "room_name": None, "turn": 0}
            for key, _, _ in TREASURES
        }

    # ── Update from Z-machine ──

    def update(self, jericho, current_room_id: int, current_room_name: str, turn: int):
        """Refresh treasure states from the object tree. Cheap; call once per turn."""
        try:
            objs = jericho.get_all_objects()
            player = jericho.get_player_object()
        except Exception:
            return
        if not objs or player is None:
            return

        by_num = {o.num: o for o in objs}
        case = self._find_object(objs, ["trophy case"])
        thief = self._find_object(objs, ["thief"])
        changed = False

        for key, display, frags in TREASURES:
            obj = self._find_object(objs, frags, exclude=["trophy case"])
            if obj is None:
                continue
            chain = self._parent_chain(obj, by_num)
            state = self.episode_state[key]

            if player.num in chain:
                new = (CARRIED, None, None)
            elif case is not None and case.num in chain:
                new = (DEPOSITED, None, None)
            elif thief is not None and thief.num in chain:
                # Only mark stolen if the agent already knew about this treasure —
                # the Thief's unseen stash must not leak through the registry.
                if state["status"] in (CARRIED, LOCATED, STOLEN):
                    new = (STOLEN, None, None)
                else:
                    continue
            elif chain and chain[-1] == current_room_id and (
                    obj.parent == current_room_id or self._touched(jericho, obj)):
                # Lying directly in the agent's current room (or touched before):
                # legitimately observable.
                new = (LOCATED, current_room_id, current_room_name)
            elif self._touched(jericho, obj) and chain:
                # Agent has interacted with it before; track where it ended up.
                room_id = chain[-1]
                new = (LOCATED, room_id, jericho.room_name(room_id) or f"L{room_id}")
            else:
                continue  # unseen — keep fog of war

            if (state["status"], state["room_id"]) != (new[0], new[1]):
                state.update({"status": new[0], "room_id": new[1],
                              "room_name": new[2], "turn": turn})
                changed |= self._record_knowledge(key, display, new, turn)

        if changed:
            self._save()

    def _record_knowledge(self, key: str, display: str, new, turn: int) -> bool:
        """Update cross-episode knowledge; returns True if anything changed."""
        k = self.knowledge.setdefault(key, {"name": display})
        changed = False
        status, room_id, room_name = new
        if status == LOCATED and room_id and not k.get("usual_room_id"):
            k["usual_room_id"] = room_id
            k["usual_room_name"] = room_name
            changed = True
        if status == DEPOSITED and not k.get("ever_deposited"):
            k["ever_deposited"] = True
            changed = True
        if status in (CARRIED, DEPOSITED) and not k.get("ever_acquired"):
            k["ever_acquired"] = True
            changed = True
        return changed

    # ── Queries ──

    def deposited(self) -> List[str]:
        return [self._display(k) for k, s in self.episode_state.items()
                if s["status"] == DEPOSITED]

    def is_deposited(self, name_fragment: str) -> bool:
        frag = (name_fragment or "").lower()
        for key, display, frags in TREASURES:
            if frag in display.lower() or any(f in frag or frag in f for f in frags):
                if self.episode_state[key]["status"] == DEPOSITED:
                    return True
        return False

    def format_for_reasoner(self) -> str:
        dep = [k for k, s in self.episode_state.items() if s["status"] == DEPOSITED]
        lines = [
            f"=== TREASURE TRACKER ({len(dep)}/19 deposited — deterministic, from game engine) ===",
            "Victory requires ALL 19 treasures in the trophy case (350 points), which",
            "reveals a map to the Stone Barrow endgame.",
        ]
        for key, display, _ in TREASURES:
            s = self.episode_state[key]
            k = self.knowledge.get(key, {})
            if s["status"] == DEPOSITED:
                lines.append(f"  ✓ {display}: IN TROPHY CASE")
            elif s["status"] == CARRIED:
                lines.append(f"  ● {display}: carried (deposit it!)")
            elif s["status"] == STOLEN:
                lines.append(f"  ✗ {display}: STOLEN by Thief (recover from his Treasure Room)")
            elif s["status"] == LOCATED:
                lines.append(f"  ○ {display}: seen at {s['room_name']} (L{s['room_id']})")
            elif k.get("usual_room_id"):
                lines.append(f"  ? {display}: not yet found this episode "
                             f"(past episodes: {k['usual_room_name']} (L{k['usual_room_id']}))")
            else:
                lines.append(f"  ? {display}: never located in any episode")
        return "\n".join(lines)

    def get_export_data(self) -> Dict[str, Any]:
        return {
            "episode_state": self.episode_state,
            "deposited_count": len([1 for s in self.episode_state.values()
                                    if s["status"] == DEPOSITED]),
        }

    # ── Internals ──

    @staticmethod
    def _display(key: str) -> str:
        for k, display, _ in TREASURES:
            if k == key:
                return display
        return key

    @staticmethod
    def _find_object(objs, frags, exclude=None):
        for o in objs:
            name = (getattr(o, "name", "") or "").lower()
            if not name:
                continue
            if exclude and any(x in name for x in exclude):
                continue
            if any(f in name for f in frags):
                return o
        return None

    @staticmethod
    def _parent_chain(obj, by_num) -> List[int]:
        """Ancestor object numbers from immediate parent up to the topmost holder."""
        chain = []
        cur = getattr(obj, "parent", 0)
        for _ in range(15):  # cycle guard
            if not cur or cur not in by_num:
                if cur:
                    chain.append(cur)
                break
            chain.append(cur)
            cur = getattr(by_num[cur], "parent", 0)
        return chain

    @staticmethod
    def _touched(jericho, obj) -> bool:
        try:
            return jericho.check_attribute(obj, 3)
        except Exception:
            return False
