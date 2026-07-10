# Architect Position: Grounded in Actual Data

*All claims verified against game_files/map_state.json and the actual source files.*

## What the Live Map Actually Shows

**The trap door pair:**
- `connections["72"]["up"] = 193` — Cellar→up→Living Room exists, 10 failures
- `connections["193"]["down"] = 72` — Living Room→down→Cellar exists, 12 failures

**The pragmatist's connectivity claim is correct and verified.** Cellar (72)→up→Living Room (193) is the *only* bridge between the surface cluster and the underground cluster in the current map. Every other connection from the underground side (Troll Room→Cellar, East of Chasm→Cellar, Slide Room→Cellar) is within the underground cluster. Deleting either trap door edge disconnects ~30 underground rooms from the surface. The pragmatist's topology argument stands.

**One claim I made was wrong.** I said "the connection gets RE-ADDED every time the agent successfully goes down from Living Room." That's incorrect. `map_graph.py:73`: `if existing == to_id: return False` — `add_connection` is idempotent. The connection is stored once and never overwritten. The re-add problem I described doesn't exist.

## What the Data Reveals About the Real Problem

The actual issue is more specific than "bad edge." Look at `exit_failures["193"]`:

```
"193": {
  "down": 12,
  "south": 2,
  "north": 2,
  "up": 2
}
```

Living Room has 12 failures going DOWN — the direction that physically works. South, north, and up have 2 failures each — those are structural walls (Zork doesn't connect Living Room those ways). The agent is trying directions that don't exist AND failing on directions that do exist. Both patterns produce identical failure counts.

And `exit_failures["94"]["up"] = 28` — Studio has 28 "up" failures with no corresponding connection entry and no "up" in its room exits. The existing prune logic already handled this correctly (the hint was pruned). The failure tracking works fine for structural nonexistent exits.

The trap door is different because it has a connection entry, and the `prune_invalid_exits` guard at `map_graph.py:102-108` explicitly skips pruning any direction that has a connection entry, regardless of failure count. This is intentional and correct — you don't want to delete the only surface↔underground bridge.

## What Needs to Change

**The pathfinder currently ignores exit_failures entirely.** `find_path_bfs` at `map_graph.py:116-141` iterates `self.connections.get(current, {}).items()` with no reference to `self.exit_failures`. All edges are treated as equally traversable. The 10-failure Cellar→up edge and a 0-failure corridor are indistinguishable to the pathfinder.

**The pragmatist's weighted Dijkstra is the right fix for this.** Replace the BFS deque with a priority queue, use `1 + failure_count` as edge cost. The trap door edges cost 11 and 13 respectively — heavily deprioritized, but not excluded. When no alternative exists (which is the case here — the surface cluster is only reachable via this bridge), the path is still returned.

## The Asymmetry Problem I Claimed — And Whether It's Real

I argued weighted Dijkstra has an asymmetry: failures accumulate but successes never reduce cost. The map has no `exit_successes` structure — confirmed, it's not in the JSON. So after the agent opens the trap door and uses Cellar→up successfully 10 times, that edge still costs 11.

**But does this actually matter for the immediate bug?** Probably not much. Here's why:

The current bug is that BFS routes *through* the failing edge because it has no concept of failure at all. Weighted Dijkstra fixes that — the agent will prefer other routes when they exist. After the agent opens the trap door and traverses the edge, it *works*, and the pathfinder follows the path successfully. The warning annotation tells the agent the edge has failures. The agent traverses it successfully anyway and advances. The cost being 11 vs 1 on a successful traversal is a mild inefficiency, not a hard failure.

The asymmetry becomes meaningful at large failure counts over many generations. At 10 failures, `exit_successes` would meaningfully improve the cost function. But this is a generation-41 optimization, not a generation-42 crasher.

**So I'm withdrawing the `exit_successes` requirement as a blocker.** It's a correct improvement, but it's not necessary to fix the immediate bug. The pragmatist is right that weighted Dijkstra alone solves what's broken today.

## Where I Stand After Reading the Data

**The pragmatist's proposal (weighted Dijkstra, ~30 lines) is the correct immediate fix.** I was overcomplicating it.

**The `blocked_by` approach remains the correct long-term model**, but for a different reason than I originally argued. The live data shows failures on multiple edge types:
- Structural failures (south/north/up on Living Room — walls)
- Conditional failures (down on Living Room, up on Cellar — trap door)
- High-count anomalies (up on Studio, 28 failures — already pruned correctly)

Weighted Dijkstra treats all of these identically by failure count. The system currently cannot tell the pathfinder "this edge is temporarily blocked by a known condition" vs "this edge is a wall." That distinction enables the reasoner to generate targeted objectives. But building that layer requires a `blocked_conditions` tracker in the pathfinder that doesn't exist yet.

**The actual recommendation, in priority order:**

1. **Ship now**: Weighted Dijkstra in `find_path_bfs`. Edge cost = `1 + exit_failures.get(room_id, {}).get(direction, 0)`. ~30 lines. Fixes the immediate stuck-loop.

2. **Ship alongside it**: Annotate pathfinder output when the chosen path crosses any edge with failures > threshold. The orchestrator at `map_graph.py:652-654` already constructs path context — add a single warning line. The agent then has signal to reason about prerequisites.

3. **Ship later**: `blocked_by` on edges + `exit_successes` for symmetric cost recovery. When the reasoner is ready to act on condition-specific information rather than just "risky path" warnings.

The devil's advocate one-liner (hard-skip at threshold) remains wrong: it permanently excludes the only surface↔underground bridge in the current map, making 30+ rooms unreachable. That's not a theoretical concern — it's what would happen to this specific map file today.
