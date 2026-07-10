# Devil's Advocate: You're Solving the Wrong Problem

## The Premise Is Slightly Wrong — Which Matters

The debate setup says the bad connection "gets RE-ADDED every time the agent goes Living Room→down→Cellar." I read the code. That's not what happens.

`add_connection` in `map_graph.py` only records the direction actually traversed. It does NOT auto-create reverse edges. `connections[72]["up"] = 193` was added ONCE — the first time the agent successfully went up from Cellar to Living Room, probably in generation 1 or 2 when the trap door was still open. Since then, `prune_invalid_exits` (lines 88–114) explicitly skips deletion of verified connections. That connection has been immortal for 40+ generations.

This means the problem is not "re-addition." It's **permanent false memory from a conditionally-valid observation.**

## Why "Failure-Weighted BFS" Is Also Wrong

My first instinct was: make BFS skip edges with high failure counts. On reflection, that has a correctness trap. The trap door connection is valid under ONE condition: the agent just came through it from above (the door is momentarily open). After that first downward trip, Cellar→up→Living Room genuinely works — for one move. If the agent goes down and immediately goes back up, it succeeds. So `exit_failures[72]["up"]` might legitimately be 0 or 1.

Failure count is a lagging indicator. A failure-weighted BFS routes freely through this edge whenever the agent hasn't recently failed it — exactly the wrong time to trust it. The failure count doesn't encode the door's state. It encodes the agent's history.

## Why "Map Data Model Changes" Are Also Wrong

The other camp wants to flag connections as one-way, conditional, or verified. But the map doesn't know WHY a connection failed. It doesn't know about trap doors, puzzle gates, or timed events. Adding those concepts to `MapGraph` means the map needs to model Zork's game logic. Now you have two systems that need to agree on the same truth, and that's the architecture that breaks at generation 50 instead of generation 40.

## The Real Root Cause

The system has no concept of **state-conditional validity**. The Z-machine's `get_valid_exits()` already tests every direction against the CURRENT game state via save/restore (lines 153–173 in `game_interface.py`). It's perfect, real-time, zero-inference ground truth. It's called every turn. The result goes into `room.exits`. But `connections` — the dict BFS actually traverses — is never updated from it.

The map has TWO truth stores that diverge:
- `room.exits`: refreshed each visit from Z-machine ground truth
- `connections`: permanent, append-only, never corrected

BFS uses `connections`. The Z-machine truth goes into `room.exits`. Nothing reconciles them. The system already has the right data; it's just flowing into the wrong place.

## The Minimal Correct Fix

When `get_valid_exits()` returns results for the current room, compare against `connections`. Any direction in `connections[room_id]` that `get_valid_exits()` does NOT list is currently invalid. Mark it closed — not permanently deleted, since the door could reopen.

This requires a `_closed_connections` transient set (resets per session) and two small changes:

```python
# In MapGraph.__init__:
self._closed_connections: Dict[int, Set[str]] = {}

# New method called from record_valid_exits or a reconcile step:
def reconcile_with_ground_truth(self, room_id: int, valid_exits: list):
    valid_set = set(valid_exits)
    for direction in list(self.connections.get(room_id, {}).keys()):
        if direction not in valid_set:
            self._closed_connections.setdefault(room_id, set()).add(direction)
        else:
            self._closed_connections.get(room_id, set()).discard(direction)

# In find_path_bfs, one guard added:
if direction in self._closed_connections.get(current, set()):
    continue
```

## What This Doesn't Break

- `prune_invalid_exits` still protects verified connections from permanent deletion — correct
- Cross-episode map persistence is untouched — `_closed_connections` resets each session start, so the map loads clean and is re-validated from Z-machine truth as rooms are visited
- The trap door case works bidirectionally: agent goes down, `up` vanishes from Z-machine exits, BFS skips it. Agent opens the trap door, `up` reappears, BFS routes through it. Ground truth wins both ways
- No game-semantics reasoning in the map layer. The map stays dumb; the Z-machine stays authoritative

## The Uncomfortable Conclusion

The system already HAS the right data — `get_valid_exits()` is called every turn and it's exactly what's needed. The bug is a data pipeline gap: ground truth flows into `room.exits` but never reconciles with `connections`. This is neither a map model problem nor a pathfinder problem. It's a missing synchronization step between two structures that should agree but don't.
