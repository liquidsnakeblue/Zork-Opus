# Architect Defense: Why ConnectionEdge — And When to Ship It

## Acknowledging What Changed

The pragmatist's revised position is substantially better than their first draft, and I need to engage with it honestly rather than defending my proposal reflexively.

Their key data point: both Cellar→up AND Living Room→down are failing (10 and 12 failures respectively). This is a conditional state problem, not a one-way problem. Deleting either edge disconnects 18+ surface rooms from the underground cluster. Their revised proposal — weighted Dijkstra where cost = `1 + failure_count` — correctly avoids deletion while deprioritizing the failing edges. It's 30 lines. It fits the existing architecture at `find_path_bfs`. It doesn't require new data structures.

So let me answer the actual question: what does the full `ConnectionEdge` approach add that weighted Dijkstra doesn't, and is it worth the cost?

## Where Weighted Dijkstra Still Fails

The pragmatist's approach handles the immediate symptom correctly: BFS routes around the trap door when alternatives exist, warns the agent when they don't. But it has a structural weakness that will reassert itself.

**The failure count never resets.**

After the agent opens the trap door and successfully traverses Cellar→up, `exit_failures[72]["up"]` is still 10. The edge costs 11. Next time the agent is in Cellar and needs to reach the surface, BFS will still prefer any alternative route that costs less than 11 steps — and if no such alternative exists, it'll issue the "high failure path" warning even though the trap door is now open and the passage is perfectly safe. The warning becomes noise. The agent has solved the puzzle, but the pathfinder treats this passage as suspect indefinitely.

Now multiply this across generations. The map is shared. Every episode that opens the trap door successfully traverses this edge — but the traversal success is not recorded in `exit_failures`, only the failures are. The cost accumulates asymmetrically. By generation 50, the edge costs 45. It will never be preferred even when the trap door has been open for 20 turns.

This is the same asymmetry problem as before, just slower. The devil's advocate's one-liner hard-blocks. Weighted Dijkstra soft-blocks. Both leave no recovery path.

## The Minimal Fix That Actually Recovers

Here's what I should have said in my first draft instead of the full `ConnectionEdge` struct: **track successful traversals per edge, not just failures.**

The current `add_connection` path records that an edge exists but doesn't count how many times it's been successfully used. `exit_failures` counts failures but there's no matching `exit_successes`. Add one:

```python
self.exit_successes: Dict[int, Dict[str, int]] = {}  # {room_id: {direction: count}}
```

Record a success every time `update_from_movement` confirms traversal of a known edge (not first discovery — subsequent use). Then in weighted Dijkstra, edge cost becomes:

```python
failures = exit_failures.get(room_id, {}).get(direction, 0)
successes = exit_successes.get(room_id, {}).get(direction, 0)
cost = 1 + max(0, failures - successes)
```

When the agent opens the trap door and goes up successfully, `exit_successes[72]["up"]` increments. If it had 10 failures and now has 3 successes, net cost = 1 + max(0, 7) = 8 — still somewhat penalized, but recovering. After using the path a few times, cost approaches 1. This is symmetric: failures raise cost, successes lower it. The map learns from both directions.

This is ~15 lines on top of the pragmatist's 30-line weighted BFS change. No new data structures beyond one more dict that mirrors the existing `exit_failures` pattern. No `blocked_by`, no condition strings, no `blocked_conditions` set in the pathfinder.

## So Why Did I Propose ConnectionEdge?

Because the `exit_successes` fix, while correct, still treats all blockages identically. The agent knows "this edge has 7 net failures" but not "this edge is blocked because the trap door is closed." The difference matters for planning: knowing *why* an edge is hard to traverse lets the reasoner generate targeted sub-objectives ("find out how to open the trap door") rather than just "this path is risky."

The pragmatist correctly identifies that the reasoning layer should generate those objectives. But the reasoner needs the signal. Right now the signal is a warning in `pf_ctx`: "This path passes through an edge with 10 failures." That's weak — the agent may not know what to do with it. "This path requires `trap_door_open`" is actionable. That's what `blocked_by` buys.

## The Honest Answer to "Is It Worth The Complexity?"

For the immediate bug: **no.** The `exit_successes` dict + weighted Dijkstra fixes the stuck-loop problem, fixes the no-recovery problem, and fits the existing architecture. That's what should ship.

For long-term agent capability: **yes, but as a second layer.** The `blocked_by` approach is the right eventual data model because it gives the reasoner actionable condition names instead of opaque failure counts. But it requires the pathfinder to maintain a `blocked_conditions` set reflecting current world state — non-trivial state management that deserves its own focused design.

The mistake in my original proposal was packaging the long-term architectural improvement and the immediate bug fix together. These are separable:

1. **Ship now**: weighted Dijkstra + `exit_successes` for symmetric cost recovery. ~45 lines total, no data model changes.
2. **Ship later**: `blocked_by` on connection edges, fed by response text classification. Requires pathfinder `blocked_conditions` state management. Unlocks reasoner planning around specific conditions.

The pragmatist wins the immediate argument. My position was right about the direction but wrong about the urgency.

## What I Concede and What I Hold

**I concede**: The full `ConnectionEdge` struct is not the right first move. The pragmatist's 30-line weighted BFS change is the correct tactical answer to the immediate bug. My proposal conflated a tactical fix with a strategic improvement.

**I hold**: Weighted Dijkstra without symmetric success tracking will re-accumulate the problem over generations. The `exit_successes` addition is necessary to make the approach correct, not just palatable. And `blocked_by` is the right long-term model even if it's not the right next commit.

**What either light approach misses**: The devil's advocate's one-liner is still wrong — for the reasons in my previous rebuttal. Hard-blocking on failure count with no recovery is worse than weighted routing, not better.
