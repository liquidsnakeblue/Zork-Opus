# Devil's Advocate: Grounded in Code and Data

## What I Verified

I read map_graph.py, map_manager.py, pathfinder.py, orchestrator.py in full, and queried map_state.json directly. Here is what the data actually shows.

### The four verified-connections-with-failures in the live map:

| Room | Direction | Target | Failures | Dijkstra cost |
|------|-----------|--------|----------|---------------|
| Dome Room (133) | down | Torch Room (105) | 3 | 3 |
| Cellar (72) | up | Living Room (193) | 10 | 10 |
| Living Room (193) | down | Cellar (72) | 12 | 12 |
| Altar (212) | down | Cave Tiny (46) | 2 | 2 |

### Do alternatives exist for any of them?

No. I ran Dijkstra with each problem edge removed. For all four, no alternative path exists. The trap door pair is the only underground↔surface connection in the current map. Dome Room→down is the only path to Torch Room. Altar→down is the only path to Cave Tiny.

**Failure-weighted Dijkstra routes through all four of these edges anyway.** The cost becomes 10 or 12 instead of 1, but the path is identical. The BFS loop the agent is stuck in does not change.

### Where get_valid_exits data flows — confirmed:

`extractor.py:40`: `exits = self.jericho.get_valid_exits()`
`orchestrator.py:306`: `self.map_mgr.record_valid_exits(self.gs.current_room_id, info.exits)`
`map_manager.py:134`: `record_valid_exits` does `room.exits.update(exits)` — updates the hint set only, never touches `connections`

Z-machine ground truth is computed every turn via save/restore probing. It flows into `room.exits`. `connections` — what BFS actually traverses — is never touched by it.

### The Dome Room failure reveals something else:

Dome Room has `exit_failures: {'climb down': 5, 'down': 3}`. The `climb down` entries are from a different action text normalization — the agent tried "climb down" (mapped separately) AND "down." The Z-machine probing only uses canonical directions, so `get_valid_exits()` returns "down" if it works. The failure tracker records both. This suggests the Dome Room→down connection may also be a state-gated passage (the shoring timbers puzzle in Zork I), not a permanent one-way issue.

---

## What This Means for Each Proposed Fix

### Failure-weighted Dijkstra (Pragmatist's proposal)

Does not change routing for any of the four problem edges. With no alternatives, Dijkstra must use them regardless of cost. What it does provide: a warning flag in path output. The agent gets told "this path passes through 10-failure edge." Whether the agent acts on that warning is a reasoning question the map system can't control.

This is not useless — the warning is genuinely valuable signal. But it does not prevent the stuck loop. It informs about the stuck loop.

### `blocked_by` condition labels (Architect's proposal)

Requires populating `blocked_conditions` at pathfinding call time. From the code, nothing currently populates this. `_classify_blockage` would need to be added and called on every movement failure that hits a verified connection. If it misfires — and it will, because Zork's failure responses are inconsistent ("You can't go that way" vs. "The trap door is closed") — the edge gets no label and the problem is unchanged.

Even when it works: the pathfinder still has no alternative. It knows the edge is conditionally blocked but must use it anyway. It can surface that condition name to the agent ("trap_door_closed"), which is more actionable than a generic warning. This is the architect's strongest real advantage.

### `_closed_connections` from Z-machine (my proposal)

When the agent visits Cellar and `get_valid_exits()` returns `['north', 'south']` (no 'up'), `connections[72]['up']` gets masked as closed. BFS skips it. The pathfinder returns "no path from Cellar to Living Room."

With no alternative path, the agent gets told: you cannot navigate there right now. This is accurate. The agent must reason about why and what to do next. Whether it succeeds at that reasoning depends on the agent and the memory/knowledge context it has.

**The honest limitation:** `_closed_connections` is per-visit, session-scoped. It only masks edges for rooms the agent is currently standing in. If the agent is pathfinding from Troll Room to Kitchen, it hasn't visited Cellar yet this turn. The mask isn't set. BFS happily routes through Cellar→up. The agent arrives at Cellar, tries to go up, fails, and the failure is tracked. The mask helps for the next pathfinding call FROM Cellar, not for pathfinding calls that route THROUGH Cellar without visiting it.

---

## Revised Assessment: What Actually Fixes the Bug

None of the three proposals fully prevents "agent gets routed through a currently-impassable edge it hasn't visited yet." This is the real failure mode.

The only mechanism that catches this without visiting first is one that probes the Z-machine along the proposed path before committing to it. This exists: `_find_canonical_direction` in `map_manager.py:79` already does save/restore probing to identify which direction led to a given room. That same mechanism could be used in the pathfinder to validate a proposed path before executing it.

**Concrete fix, grounded in existing code:**

In `Pathfinder.find_path` (pathfinder.py:37), after BFS returns a path, optionally probe each edge using `jericho.save_state()` + `jericho.send_command(direction)` + state restore, same as `_find_canonical_direction`. Any edge where the probe fails to reach the expected room gets flagged. If the path contains a failing edge with no alternative, return `found=False` with reason "path blocked at [room]→[direction]."

This is the same Z-machine probing already in the codebase, repurposed to validate paths rather than discover them. Zero new infrastructure. The map stays untouched. And it works whether or not the agent has visited the rooms in the path recently.

Cost: one save/restore cycle per edge in the proposed path, called once at path planning time. For a 3-hop path, 3 Z-machine state operations. That's negligible.

---

## What I'm Conceding

My earlier "one-line BFS guard" framing was wrong — failure counts don't reliably indicate current state, and even if they did, all four problem edges have no alternatives so the guard wouldn't help anyway.

My `_closed_connections` proposal is correct but incomplete — it helps once you're standing in the room, not for pathfinding through rooms ahead.

The Pragmatist is right that Dijkstra + warning flag is a real improvement even if it doesn't stop the stuck loop — it gives the agent information it can act on.

The Architect is right that semantic condition labels are more actionable than raw failure counts, but wrong that this solves navigation correctness rather than planning quality.

The actual fix is path validation via Z-machine probing at plan time, using infrastructure already in the codebase.
