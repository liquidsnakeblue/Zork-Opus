# Devil's Advocate: Second Rebuttal

## The Framing Is Outdated

The "bomb" the team-lead describes is the pragmatist's *first draft*, not their current position. I read the revised file. The pragmatist already discovered what I verified in the actual map_state.json: removing Cellar→up disconnects 19 rooms — Kitchen, the entire above-ground area, the Attic, and the Land of the Dead. They walked back the deletion proposal explicitly.

So "let prune_invalid_exits delete connections" is a position nobody in this debate currently holds. The pragmatist's live position is failure-weighted Dijkstra, which has more in common with my approach than with theirs.

## The Map Is NOT Holding Provably Wrong Data

The team-lead says the map is "holding onto provably wrong data." Let me push back on "provably wrong."

Cellar→up→Living Room worked. The Z-machine confirmed it. It has worked every time the trap door was open. It will work again the next time the agent opens the trap door. The connection is not wrong — it is *conditionally inaccessible*. These are different claims with different implications for how you fix it.

If you delete a conditionally-inaccessible connection, you get a map that is permanently wrong instead of temporarily incomplete. The actual data shows this: deleting that one edge makes Kitchen unreachable from the underground cluster. That's not fixing a data integrity problem — that's introducing a worse one.

The `prune_invalid_exits` guard was written correctly. The guard's comment says exactly this: "Some valid connections fail intermittently (e.g., trap doors that require setup, one-way doors)." Someone in generation 20-something learned this the hard way. The recent fix preserved that lesson.

## What The Data Actually Shows

From the live map_state.json:

- Room 72 (Cellar) → up: **10 failures**
- Room 193 (Living Room) → down: **12 failures**

Both directions are failing. This is not a one-way edge — the trap door blocks both directions when it is shut. The agent can't go down from Living Room OR up from Cellar unless it has previously opened the trap door. The problem is state-dependent traversability, not a bad edge direction.

This actually undercuts the deletion argument more severely: you'd need to delete BOTH directions to "fix" the routing. But Living Room→down→Cellar being deleted means the Cellar is only reachable from the north or south, and the surface world loses its underground connection entirely.

## Does This Undercut "Don't Touch The Map"?

Partially. My original framing that the map data is clean and the pathfinder is solely responsible was too strong. The real issue is that the map has no mechanism to represent current-state traversability, and the pathfinder has no input for it. That's a genuine design gap, not just a pathfinder bug.

But the conclusion from that gap is NOT "delete the connection." It's "add a current-state layer that doesn't corrupt the topological layer." The `_closed_connections` proposal does this. Failure-weighted Dijkstra does this too, in a less direct way.

## The Actual Choice Is Between Two Correct Approaches

The pragmatist's revised Dijkstra and my `_closed_connections` from `get_valid_exits()` are both sound. The real difference:

**Dijkstra with failure costs**: routes around bad edges when alternatives exist, warns the agent when forced through. Keeps the map as-is. The trap door with 10 failures costs 11 hops — BFS effectively treats it as a last resort. This is correct behavior when the agent is navigating from underground to surface and the trap door is the only path. It gets a warning and can plan around it.

**`_closed_connections` from Z-machine**: completely blocks routing through edges the Z-machine says are currently closed. Accurate at instant of visit. Silently re-enables edges when they become traversable again. The trap door costs infinity when shut, zero when open.

Both preserve the topological map intact. Both avoid corrupting 40 generations of verified connection data. Neither requires LLM inference. The difference is precision: Z-machine ground truth is exact; failure costs are a proxy.

The failure-cost approach has one genuine advantage the pragmatist identified: it handles edges you haven't visited recently. If the agent is pathfinding through a room it visited 50 turns ago, `_closed_connections` won't have that room's current Z-machine state. Dijkstra degrades gracefully; `_closed_connections` has a staleness blind spot.

## What I Actually Recommend Now

Implement both, in order of cost:

1. **First**: failure-weighted Dijkstra (30 lines per pragmatist's estimate). Immediate improvement, no new data structures, works even for rooms not currently being visited.

2. **Then**: when the agent visits a room, call `get_valid_exits()` and reconcile against connections. Zero-inference, exact-state override for the current room. This handles the "known bad edge, just visited" case with certainty rather than heuristic.

The map does not need to be touched. The topological facts stay intact. The routing layer gets two complementary signals: accumulated failure history (Dijkstra cost) and real-time ground truth (closed-connections mask). Neither corrupts the other, and removing one doesn't break the system.
