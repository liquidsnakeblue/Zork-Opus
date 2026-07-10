# Systems Pragmatist: The Devil's Advocate Is Almost Right, and the Architect Is Solving a Future Problem

---

## Where I Now Stand

I've read both positions and the data. The Devil's Advocate made the strongest argument. My Dijkstra proposal is worse than their reconciliation proposal for reasons I'll explain. The Architect's `blocked_by` struct is solving a real problem, but it's the wrong problem for right now. Here's my reasoning.

---

## The Devil's Advocate Wins The Technical Argument

Their core observation: `get_valid_exits()` is called every single turn in `extractor.extract()` (`extractor.py:40`), returns Z-machine ground truth about which directions actually work from the current room, and that data flows into `record_valid_exits`. The system already has the right data. The bug is that `record_valid_exits` updates `room.exits` but nothing compares that ground truth against `connections` — the dict BFS actually traverses.

I verified this. `extractor.extract()` calls `jericho.get_valid_exits()` which does save/restore testing of every direction against live game state. It's called at `orchestrator.py:302` at the start of every turn. `info.exits` then feeds `record_valid_exits` at line 306. The reconciliation step the Devil's Advocate proposes adds ~15 lines and zero new abstractions.

My Dijkstra proposal is inferior to this for a specific reason: **cost weighting is a heuristic; reconciliation against ground truth is correct.** Dijkstra with failure-weighted edges *prefers* not routing through the trap door, but when no alternative exists, it routes through it anyway — and now the agent gets sent on a path the Z-machine will reject. The Devil's Advocate's approach *never* routes through a currently-invalid edge, because the Z-machine says it's invalid. Preference vs. correctness. Correctness wins.

---

## Where The Devil's Advocate Understates Their Own Solution

Their proposed `_closed_connections` is a transient in-memory set that resets each session. That's right — game state resets at episode start, so cross-episode persistence of "closed" status would be wrong. But they frame this as a limitation. It's actually the correct design: the map persists topology (permanent), world state is re-validated each episode from Z-machine truth (ephemeral). Two layers, correctly separated.

The one concrete addition I'd make to their proposal: when reconcile identifies a direction in `connections` that `get_valid_exits()` doesn't list, log it at DEBUG level with the failure count. This costs nothing and gives 40 generations worth of diagnostic signal without changing behavior.

---

## Where The Architect Is Right But Premature

The Architect correctly identifies that failure counts can't distinguish between "structurally broken" and "conditionally blocked." That's true. Their `blocked_by` string approach is elegant — store a tag from the response text, let the pathfinder filter on it, separate topology from traversability.

But it's solving the problem that the Devil's Advocate's fix already handles. Once `_closed_connections` is reconciling against Z-machine ground truth, the BFS automatically skips currently-invalid edges. The agent doesn't need to know *why* the trap door is closed — it just needs to not be routed into a dead end. The memory and reasoning layer handles "why" and "what to do about it."

The `blocked_by` approach adds real value in one specific scenario: **the agent is trying to plan a multi-step sequence and needs to know in advance which edge requires a prerequisite before it gets there.** That's a planning problem. The current system doesn't do lookahead planning — the pathfinder finds a route and the agent follows it one step at a time. When it hits a closed edge, navigation cancels (`orchestrator.py:444–447`) and the agent re-plans. That's not great, but it's not the stuck-loop problem either. The stuck loop is BFS routing through a known-closed edge and the agent trying the same direction repeatedly. The reconciliation fix stops that.

The Architect's proposal is the right next step *after* this fix is in. Not instead of it.

---

## Concrete Disagreement With The Architect's Rebuttal

The rebuttal says the one-liner (skip high-failure edges) has no recovery path — once failures hit threshold the edge is skipped forever even after the puzzle is solved. That's a real flaw, and it's why I abandoned my own Dijkstra proposal. But the reconciliation fix doesn't have this problem. When the agent opens the trap door and `get_valid_exits()` returns "up" from Cellar, `reconcile_with_ground_truth` removes "up" from `_closed_connections` that turn. Recovery is immediate, not threshold-based.

The Architect's rebuttal targets the "one-liner skip" approach. It doesn't land on the reconciliation approach.

---

## What I'd Actually Ship

```python
# map_graph.py — add to __init__:
self._closed_connections: Dict[int, Set[str]] = {}

# map_graph.py — new method (15 lines):
def reconcile_with_ground_truth(self, room_id: int, valid_exits: list):
    valid_set = set(valid_exits)
    room_conns = self.connections.get(room_id, {})
    closed = self._closed_connections.setdefault(room_id, set())
    for direction in room_conns:
        if direction not in valid_set:
            if direction not in closed and self.logger:
                fc = self.exit_failures.get(room_id, {}).get(direction, 0)
                self.logger.debug(
                    f"Closing connection {room_id}->{direction} "
                    f"(not in valid exits, {fc} prior failures)")
            closed.add(direction)
        else:
            closed.discard(direction)

# map_graph.py — find_path_bfs, one guard:
if direction in self._closed_connections.get(current, set()):
    continue

# map_manager.py — record_valid_exits, one added call:
self.game_map.reconcile_with_ground_truth(room_id, exits)
```

That's the entire fix. ~25 lines. No new abstractions. No schema changes. No migration. Correct behavior on first visit to any room with conditional connections. Immediate recovery when conditions change.

---

## The Meta-Point

We have 40+ generations of accumulated map data and a running system. The failure modes of changes are asymmetric: a 25-line reconciliation step using data that's already flowing has a narrow blast radius. A schema migration to ConnectionEdge structs touches serialization, persistence, BFS, the pathfinder API, and every place that reads connections. Both might be correct. Ship the small one first, observe generation 45, then decide if `blocked_by` is needed.
