#!/usr/bin/env python3
"""
Offline evidence miner — rebuilds structured claims from raw episode traces.

Scans every episode_log.jsonl, aggregates outcomes per (location, action), and
flags CONDITION-DEPENDENT actions: ones that both succeed and fail at the same
location. For those, it diffs the "notable" actions taken in the lookback
window before successes vs failures — surfacing candidate preconditions like
"press yellow button" for "turn bolt with wrench".

This is deterministic mining over provenance-carrying data. Promote confirmed
findings into game_files/procedures.json by hand (or by future tooling) —
never import prose memories as truth.

Usage:
    python evidence_miner.py                # summary of condition-dependent actions
    python evidence_miner.py --all          # include always-succeed/always-fail stats
    python evidence_miner.py --lookback 20  # precondition search window (turns)

Writes game_files/evidence/action_stats.json as a side effect.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

EPISODES_DIR = Path("game_files/episodes")
OUT_PATH = Path("game_files/evidence/action_stats.json")

# Verbs whose earlier occurrence plausibly changes world state (precondition candidates)
_NOTABLE = re.compile(
    r"^(press|push|pull|turn|open|close|unlock|lock|tie|untie|light|extinguish|"
    r"pray|ring|read|wave|rub|touch|move|lift|raise|lower|pour|fill|inflate|"
    r"dig|burn|melt|give|throw|put|drop|wind|squeeze|knock|say|shout|echo)\b")

_FAILURE_MARKERS = (
    "you can't", "you cannot", "won't budge", "won't turn", "doesn't budge",
    "i don't know the word", "don't understand", "you used the word",
    "you don't see", "there is a wall", "too narrow", "nothing happens",
    "not here", "isn't here", "can't go that way", "can't do that",
    "already open", "already closed", "already have",
)


def _judge(response, score_delta, moved):
    if score_delta > 0 or moved:
        return True
    r = (response or "").lower()
    if any(m in r for m in _FAILURE_MARKERS):
        return False
    return None


def load_episode(path: Path):
    """Yield per-turn dicts with derived score_delta/moved/success."""
    turns = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if '"turn_completed"' not in line and '"turn_executed"' not in line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("event_type") not in ("turn_completed", "turn_executed"):
                    continue
                turns.append(e)
    except OSError:
        return []
    # turn_executed + turn_completed may both exist for a turn — dedupe by turn no.
    by_turn = {}
    for e in turns:
        by_turn[e.get("turn", 0)] = e
    seq = [by_turn[t] for t in sorted(by_turn)]

    out, prev_score, prev_loc = [], 0, None
    for e in seq:
        score = e.get("score", 0) or 0
        loc = e.get("location", "")
        # source location = where the action was taken (previous turn's destination)
        src = prev_loc if prev_loc is not None else loc
        moved = prev_loc is not None and loc != prev_loc
        out.append({
            "turn": e.get("turn", 0),
            "action": (e.get("action", "") or "").strip().lower(),
            "location": src,
            "score_delta": score - prev_score,
            "moved": moved,
            "response": e.get("response", ""),
            "success": _judge(e.get("response", ""), score - prev_score, moved),
        })
        prev_score, prev_loc = score, loc
    return out


def notable_actions_before(turns, index, lookback):
    """Set of notable actions in the window before turns[index]."""
    lo = max(0, index - lookback)
    return {t["action"] for t in turns[lo:index] if _NOTABLE.match(t["action"])}


def mine(lookback: int):
    stats = defaultdict(lambda: {"success": [], "failure": [], "unknown": []})
    pre_success = defaultdict(list)   # (loc, action) -> [set(preceding notable), ...]
    pre_failure = defaultdict(list)
    pre_unknown = defaultdict(list)

    episodes = sorted(EPISODES_DIR.glob("*/episode_log.jsonl")) if EPISODES_DIR.exists() else []
    for log in episodes:
        ep = log.parent.name
        turns = load_episode(log)
        for i, t in enumerate(turns):
            if not t["action"] or t["action"] in ("look", "inventory", "i"):
                continue
            key = (t["location"], t["action"])
            bucket = ("success" if t["success"] is True
                      else "failure" if t["success"] is False else "unknown")
            stats[key][bucket].append(f"{ep} T{t['turn']}")
            if t["success"] is True:
                pre_success[key].append(notable_actions_before(turns, i, lookback))
            elif t["success"] is False:
                pre_failure[key].append(notable_actions_before(turns, i, lookback))
            else:
                pre_unknown[key].append(notable_actions_before(turns, i, lookback))

    findings = []
    for key, s in sorted(stats.items()):
        # Condition-dependent = explicit failures alongside non-failure outcomes.
        # "unknown" outcomes count as possible successes: many real successes
        # (e.g. "The sluice gates open") change neither score nor location, so
        # the generic verdict can't confirm them — but a failure marker beside
        # them still proves the action is state-gated.
        if not s["failure"] or not (s["success"] or s["unknown"]):
            continue
        succ_sets = pre_success.get(key, []) + pre_unknown.get(key, [])
        fail_sets = pre_failure.get(key, [])
        # Candidates need a real sample — 1-vs-1 intersections are pure noise
        if len(succ_sets) >= 3 and len(fail_sets) >= 2:
            always_before_success = set.intersection(*succ_sets) if succ_sets else set()
            ever_before_failure = set.union(*fail_sets) if fail_sets else set()
            candidates = sorted(always_before_success - ever_before_failure)
        else:
            candidates = []
        findings.append({
            "location": key[0], "action": key[1],
            "successes": s["success"], "possible_successes": s["unknown"],
            "failures": s["failure"],
            "precondition_candidates": candidates,
        })
    return stats, findings


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--lookback", type=int, default=20,
                    help="turns to search before an outcome for preconditions")
    ap.add_argument("--all", action="store_true",
                    help="also dump non-condition-dependent action stats")
    args = ap.parse_args()

    stats, findings = mine(args.lookback)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "lookback": args.lookback,
        "condition_dependent": findings,
        "all_actions": {f"{loc} :: {act}": s for (loc, act), s in stats.items()} if args.all else None,
    }, indent=2))

    print(f"Scanned {len(stats)} (location, action) pairs "
          f"across {len(list(EPISODES_DIR.glob('*/episode_log.jsonl')))} episodes.\n")
    if not findings:
        print("No condition-dependent actions found.")
        return
    print(f"CONDITION-DEPENDENT ACTIONS ({len(findings)}) — candidates for procedures.json:\n")
    for f in findings:
        ok = f["successes"] + f["possible_successes"]
        print(f"▶ '{f['action']}' @ {f['location']}: "
              f"{len(f['successes'])} ok + {len(f['possible_successes'])} possible / "
              f"{len(f['failures'])} failed")
        if f["precondition_candidates"]:
            print(f"    likely preconditions: {', '.join(f['precondition_candidates'])}")
        print(f"    ok:     {', '.join(ok[:6])}")
        print(f"    failed: {', '.join(f['failures'][:6])}")
    print(f"\nFull stats: {OUT_PATH}")


if __name__ == "__main__":
    main()
