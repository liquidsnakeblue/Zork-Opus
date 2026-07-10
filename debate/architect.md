# Map Architect Position: Connection Metadata & Weighted Pathfinding (Revised)

## Stepping Back: What Is the Map Actually For?

The bug report frames this as a data quality problem — a bad edge needs to be pruned. But I want to challenge that framing.

The Cellar→up→Living Room edge is NOT wrong. It was recorded because the Z-machine confirmed it: at some point, going "up" from Cellar actually landed in Living Room. The data is accurate. The problem is that the map encodes *physical topology* when what the pathfinder needs is *reachability under current world state*. Those are different things in Zork, and conflating them is the actual bug.

## The Wrong Frame: "Bad Connection"

Every "simple fix" so far has treated this as a bad-data problem:
- Prune after N failures — fails because the edge gets re-added legitimately
- Confidence weighting — still wrong: the edge IS traversable, just not always

The trap door is the clearest example, but Zork has many state-gated passages: the grating (needs the golden key), the coal mine shaft (needs the shoring timbers), various locked doors. Every one of them has connections that are *conditionally valid*. Pruning or downweighting them doesn't solve the problem — it makes the map less useful when the agent actually has the prerequisites.

## The Right Frame: Preconditions

A connection has two orthogonal properties:
1. **Physical existence** — does this edge exist in the world?
2. **Current traversability** — is it passable right now, given world state?

The existing system only tracks #1. We need to track #2 without losing #1.

## The Proposed Data Model

Add a `blocked_by` field to each connection edge — a short string naming the condition that blocks it, or `None` for unconditionally open:

```python
ConnectionEdge = {
    "to_id": int,
    "blocked_by": Optional[str],   # None = open, "trap_door_closed" = conditional
    "traversal_count": int,
    "last_failed_turn": Optional[int],
}
```

`blocked_by` is populated not by hardcoding room IDs, but by a heuristic applied when `track_exit_failure` is called on a known connection: if the failure response text mentions a physical obstruction ("The trap door is closed", "The grating is locked"), extract a canonical condition name and set `blocked_by`.

The existing `_classify_movement` LLM call already handles response text. A parallel `_classify_blockage` call costs nothing extra.

## How the Pathfinder Uses This

`find_path_bfs` gets a `blocked_conditions: Set[str]` parameter — the set of known-blocked conditions given current inventory and world state. Edges where `blocked_by in blocked_conditions` are excluded from the search. Edges where `blocked_by` is not `None` but not in `blocked_conditions` are included but flagged as *risky* (agent may need to verify first).

When the agent successfully traverses a previously-blocked edge (e.g., opens the trap door and goes up), `blocked_by` is cleared to `None` for that session (but preserved in long-term storage as `"conditionally_blocked"` so the agent knows to check on future visits).

This means:
- Agent in Cellar can't route through "up" while trap door is closed — correct
- Agent knows the trap door exists and which condition blocks it — can plan to open it
- Once opened, the edge is immediately usable — no confidence decay to recover from
- Across episodes, the condition knowledge persists — agent learns "this passage needs setup"

## Auto-Detection Without Hardcoding

The system learns `blocked_by` from failure response text, not from room ID tables. A lightweight text check handles most cases:

```python
BLOCKAGE_PATTERNS = [
    (r"trap door.{0,20}closed", "trap_door_closed"),
    (r"grating.{0,20}locked", "grating_locked"),
    (r"door.{0,20}locked", "door_locked"),
    (r"can.t go that way", None),   # structural wall, not conditional
]
```

For ambiguous responses the existing LLM classifier handles it. No Zork-specific room IDs anywhere.

## Why Confidence Weighting Gets This Wrong

A confidence-based approach treats repeated failures as evidence the edge is *wrong*. But:
- Agent encounters trap door closed, fails, confidence drops
- Agent opens the trap door, takes the edge successfully, confidence partially recovers
- Agent goes back down, trap door closes again, failures resume
- Confidence oscillates forever on a real, valid edge

The underlying issue is that confidence conflates two different signals: *"this edge is structurally broken"* and *"this edge is currently blocked by world state."* Those need different responses. `blocked_by` separates them. Physical topology is stable; world state is volatile. Model them with different mechanisms.

## Migration Path

1. `load_from_json()`: existing connections load as `{"to_id": tid, "blocked_by": None, "traversal_count": 1, "last_failed_turn": None}` — backward compatible.
2. Existing `exit_failures` entries with high failure counts on known connections: set `blocked_by = "unknown_condition"` as a conservative starting state. Pathfinder treats these as conditionally blocked until a successful traversal clears them.
3. No separate migration script. Old format is detected by value type (bare int vs dict) and wrapped on first load.

## The Deeper Point

The map has been running for 40+ generations because it tracks *verified physical facts*. Don't make physical facts fuzzy with confidence decay — that destroys the one thing the system does reliably. Instead, add a *separate layer for transient state*. One layer says "these rooms are connected." A second layer says "that edge is currently passable." The pathfinder combines both. The map keeps its integrity.
