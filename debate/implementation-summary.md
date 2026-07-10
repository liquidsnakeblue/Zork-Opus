# Z-Machine Path Probing — Implementation Summary

## Problem Solved
BFS finds paths through connections that are currently impassable (e.g. Cellar→up→Living Room when the trap door is closed). The agent enters dead-end loops for 50+ turns because the map stored a valid connection that no longer works in the current game state.

## What Was Implemented

### 1. `MapManager.validate_path()` — `/home/liquidsnakeblue/Zork-Opus/map_manager.py:134`

New method that probes each hop in a BFS path using the Z-machine's save/restore mechanism (same pattern as `_find_canonical_direction`).

**Signature:**
```python
def validate_path(self, hops, num_probes: int = 3) -> dict
```

**hops format:** `list of (direction, from_room_id, to_room_id)`

**Algorithm:**
- Saves origin Z-machine state at the start
- For each hop at index `i`:
  - Re-walks all preceding hops (hops[0..i-1]) from origin state to reach that hop's start room
  - Probes the hop's direction `num_probes` times (default: 3)
  - If ALL probes fail (arrived at wrong room), marks that hop as blocked
- `try/finally` guarantees state is restored even if an exception occurs
- Returns `{"valid": True, "blocked_hop": None}` on success
- Returns `{"valid": False, "blocked_hop": {"direction", "room_id", "room_name", "to_room_id"}}` on failure

**Graceful degradation:** Returns `{"valid": True}` (pass-through) when jericho is unavailable — no false negatives.

### 2. `Pathfinder.find_path()` integration — `/home/liquidsnakeblue/Zork-Opus/pathfinder.py:63`

After BFS returns a valid path (and after length check), builds hops from the waypoints array and calls `map_manager.validate_path()`.

BFS waypoints conveniently encode hop boundaries: `wps[i]` is the room before `dirs[i]`, and `wps[i+1]` is the expected destination. The hops list is built as:
```python
hops = [(dirs[i], wps[i]["room_id"], wps[i+1]["room_id"]) for i in range(len(dirs))]
```

On blocked result, returns:
```python
{
    "found": False,
    "directions": [],
    "waypoints": [],
    "reason": "Path blocked at Cellar(L72) -> up (connection is currently impassable).",
    "blocked_hop": {"direction": "up", "room_id": 72, "room_name": "Cellar", "to_room_id": ...}
}
```

The blocked target is also recorded in `_failed_targets` so `recently_failed()` suppresses future attempts.

### 3. `_handle_pathfinder()` update — `/home/liquidsnakeblue/Zork-Opus/orchestrator.py:648`

Added a new elif branch for `result.get("blocked_hop")` between the success and generic-failure cases. The blocked-path message tells the agent:
- Which specific hop is blocked (room name, room ID, direction)
- Why: "trap door closed, gate locked" examples given
- Four concrete suggested actions (open the passage, find alternate route, use teleport mechanism, pick a different objective)

## Design Decisions

- **Validation lives on MapManager** (not Pathfinder) because jericho lives there and the save/restore pattern is already established there.
- **BFS is unchanged** — probe is a post-BFS validation step only.
- **3 probes per hop** filters random events (thief encounters, etc.) — a hop must fail ALL probes to be considered blocked.
- **Re-walk from origin per probe** rather than saving per-hop states, to keep the implementation simple and avoid N nested save states.
- **Graceful degradation** when jericho is None (e.g. unit tests or offline analysis).

## Files Modified
- `/home/liquidsnakeblue/Zork-Opus/map_manager.py` — added `validate_path()` at line 134
- `/home/liquidsnakeblue/Zork-Opus/pathfinder.py` — added probe call in `find_path()` at line 63
- `/home/liquidsnakeblue/Zork-Opus/orchestrator.py` — added blocked-hop branch in `_handle_pathfinder()` at ~line 656

All three files pass `ast.parse()` syntax check.
