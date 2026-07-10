# Architect Rebuttal: Against "Dumb Map + Smart Pathfinder"

## Conceding the Setup, Rejecting the Conclusion

The devil's advocate is right that separation of concerns is a good principle. They're wrong about which layer the concern belongs to.

The argument is: "Let the map be a dumb traversal log. Let the pathfinder interpret reliability by reading `exit_failures`." But this only works if the pathfinder can act correctly on failure counts alone. It can't — and the reason reveals the flaw in the entire framing.

## The Pathfinder Can't Distinguish Why a Connection Failed

`exit_failures` is an integer. The pathfinder proposed one-liner would be something like:

```python
if self.game_map.exit_failures.get(current, {}).get(direction, 0) >= threshold:
    continue
```

This skips an edge because it failed N times. But consider what the pathfinder now cannot know:
- Did it fail because the trap door was closed? (Conditional — openable)
- Did it fail because the troll was blocking? (Conditional — killable)
- Did it fail because there's actually no passage there? (Structural — delete it)
- Did it fail because the agent's lantern was out? (Conditional — relight it)

All four produce the same failure count. The pathfinder treats them identically: route around it forever. For the conditional cases — the majority of failures in Zork — this permanently degrades pathfinding quality without recovery. The agent opens the trap door and goes through successfully, but the next time it needs to route through Cellar→Living Room, BFS still avoids the edge because failures >= threshold and there's no mechanism to decrement that count or clear the skip.

The "dumb map + smart pathfinder" decomposition requires the pathfinder to be smart about *why* edges fail. But the pathfinder doesn't have that information. The only place in the system that has failure reasons is the game response text — which is processed at the time of failure, in `track_failed_action`, not at BFS query time. The reason has to be stored somewhere. That somewhere is the edge.

## "The Map Shouldn't Understand Game Semantics"

This is the devil's advocate's best line, and it contains a real truth. But `blocked_by: Optional[str]` doesn't ask the map to understand game semantics — it asks the map to store a string that was extracted from game text.

Compare:
- **Bad (game semantics in the map)**: `if room_id == 72 and direction == "up": skip`
- **Bad (game semantics in the map)**: `if traversal_count == 0 and failure_count > 3: mark one_way`
- **Fine (just storage)**: `blocked_by = "trap_door_closed"` — a string the map holds but doesn't interpret

The map doesn't know what "trap_door_closed" means. It doesn't know how to open trap doors. It doesn't know what inventory the agent needs. It just holds the tag. The pathfinder holds the `blocked_conditions` set — that's where game semantics live, because the pathfinder knows the current world state. The map stores; the pathfinder reasons. That IS separation of concerns.

## The One-Liner Fix Has No Recovery Path

Here's the concrete failure mode of the devil's advocate's proposal:

1. Agent tries Cellar→up, fails 3 times. BFS now skips this edge permanently.
2. Agent finds the trap door, opens it (a puzzle solution involving the screwdriver).
3. Agent is now in Cellar with the trap door open. It needs to reach Living Room.
4. BFS still skips Cellar→up because `exit_failures[72]["up"] >= 3`.
5. BFS finds no path. Agent explores randomly, eventually stumbles into Living Room via a 15-step detour through the forest.
6. The pathfinder has become *less capable* after the agent solved a puzzle.

This gets worse across episodes. The trap door bug re-manifests every episode because the failure count in a shared map persists but the game state resets. Generation 41 starts with a map that tells BFS "never route through Cellar→up" — but the trap door is open at game start.

The `blocked_by` approach doesn't have this problem: when the agent successfully traverses Cellar→up, the edge's `blocked_by` is cleared. The map updates to reflect current world state. Recovery is built in.

## Where I Agree With the Pragmatist (and Disagree With the Devil's Advocate)

The pragmatist and I share the core insight: failure counts and connection existence need to interact. Their proposed `one_way: bool` flag is essentially `blocked_by: Optional[str]` with the condition hardcoded to "permanently blocked." That's fine for truly one-way passages but wrong for conditional ones.

The devil's advocate goes further and says even that is too much — keep the map totally passive. I think that undershoots. A passive map that can't record "this edge is currently blocked" forces all conditional-passage intelligence into a component (the pathfinder) that doesn't have access to the information (game response text) needed to reason about it. You end up with either a broken pathfinder or a pathfinder that secretly becomes a map — re-deriving edge state from counts it can't fully interpret.

## The Scaling Argument Cuts Both Ways

The devil's advocate says `blocked_by` doesn't scale because Zork has puzzle-gated, time-conditional, and inventory-dependent connections — you'd need to encode all that in the map.

But the "dumb map" approach also doesn't scale to these cases. With only failure counts:
- Puzzle-gated connections: treated as broken, never re-enabled
- Time-conditional connections: oscillate between working and failing, confusion ensues
- Inventory-dependent connections: permanent avoidance after first failure

At least with `blocked_by`, the map can record what type of blockage it is. Knowing "this needs `brass_lantern_lit`" is more useful than knowing "this failed 7 times." The agent can plan around a known condition. It cannot plan around an opaque failure count.

The scaling concern is real but it argues *for* the richer model, not against it.

## Summary

The devil's advocate's separation of concerns argument is right in principle but applied to the wrong layer. The information needed to make pathfinding decisions about conditional connections (why did this fail?) is available only at failure-recording time, not at BFS-query time. It has to be stored in the edge. The map stores a string; the pathfinder interprets it. That is correct separation of concerns. The one-liner fix doesn't separate concerns — it makes the pathfinder guess, badly, from incomplete data.
