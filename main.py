#!/usr/bin/env python3
"""
Zork-Opus: AI plays Zork I
Clean rewrite leveraging Opus 4.6's 1M context window.

Usage:
    python main.py                          # Single generation (interactive)
    python main.py --continuous             # Run indefinitely
    python main.py --episodes 5             # Run 5 generations
    python main.py --fresh --max-turns 200  # Fresh start, 200 turns max
"""

import time
import argparse
import signal
import shutil
from datetime import datetime
from pathlib import Path

from session import SessionTracker
from orchestrator import Orchestrator


import json
import time
import argparse
import signal
import shutil
from datetime import datetime
from pathlib import Path

from session import SessionTracker
from orchestrator import Orchestrator


# ── Endpoint management ──

ENDPOINTS_FILE = Path("endpoints.json")

# Embedded defaults (used if endpoints.json is missing)
_DEFAULT_PRESETS = {
    "1": {"name": "Claude Opus 4.6",
          "url": "http://192.168.4.245:8317/v1", "model": "claude-opus-4-6"},
    "2": {"name": "Claude Sonnet 4.6",
          "url": "http://192.168.4.245:8317/v1", "model": "claude-sonnet-4-6"},
    "3": {"name": "Claude Haiku 4.5",
          "url": "http://192.168.4.245:8317/v1", "model": "claude-haiku-4-5-20251001"},
    "4": {"name": "Qwen 3.5 27B QuantTrio (5090)",
          "url": "http://192.168.4.245:8888/v1", "model": "/root/models/Qwen3.5-27B-AWQ-QuantTrio"},
    "5": {"name": "MiniMax M2.5 (local)",
          "url": "https://api.schuyler.ai/v1", "model": "MiniMax-M2.5-UD-Q3_K_XL-00001-of-00004.gguf"},
    "6": {"name": "DeepSeek R1 0528 (OpenRouter)",
          "url": "https://openrouter.ai/api/v1", "model": "deepseek/deepseek-r1-0528"},
    "7": {"name": "Gemini 3 Flash (OpenRouter)",
          "url": "https://openrouter.ai/api/v1", "model": "google/gemini-3-flash-preview"},
    "8": {"name": "Gemini 2.5 Flash (OpenRouter)",
          "url": "https://openrouter.ai/api/v1", "model": "google/gemini-2.5-flash"},
    "9": {"name": "Qwen 3.6 Plus (OpenRouter free)",
          "url": "https://openrouter.ai/api/v1", "model": "qwen/qwen3.6-plus:free"},
    "10": {"name": "Gemma 4 31B IT (local vLLM)",
           "url": "http://192.168.4.245:30002/v1", "model": "unsloth/gemma-4-31b-it"},
    "11": {"name": "Qwen 3.5 27B DFlash (5090)",
           "url": "http://192.168.4.245:8888/v1", "model": "/root/models/Qwen3.5-27B-AWQ-4bit"},
    "12": {"name": "Qwen 3.6 27B (LM Studio)",
           "url": "http://192.168.4.245:30002/v1", "model": "qwen3.6-27b"},
}


def _load_endpoints() -> dict:
    """Load presets from endpoints.json, falling back to embedded defaults."""
    if ENDPOINTS_FILE.exists():
        try:
            with open(ENDPOINTS_FILE) as f:
                data = json.load(f)
            presets = data.get("presets", {})
            if presets:
                return presets
        except (json.JSONDecodeError, KeyError) as e:
            print(f"⚠️  endpoints.json is malformed ({e}), using defaults")
    # Write defaults if file doesn't exist
    if not ENDPOINTS_FILE.exists():
        _save_endpoints(_DEFAULT_PRESETS)
        print(f"✨ Created {ENDPOINTS_FILE} with default presets")
    return _DEFAULT_PRESETS.copy()


def _save_endpoints(presets: dict) -> None:
    """Save presets to endpoints.json."""
    with open(ENDPOINTS_FILE, "w") as f:
        json.dump({"presets": presets}, f, indent=4)


def _next_key(presets: dict) -> str:
    """Return the next numeric key."""
    return str(max((int(k) for k in presets if k.isdigit()), default=0) + 1)


def _compact_keys(presets: dict) -> dict:
    """Re-number keys 1..N preserving insertion order."""
    return {str(i + 1): v for i, v in enumerate(presets.values())}


def _validate_url(url: str) -> bool:
    """Basic URL format check."""
    return url.startswith("http://") or url.startswith("https://")


def _add_endpoint(presets: dict) -> None:
    print("\n── Add Endpoint ──")
    name = input("  Name: ").strip()
    if not name:
        print("  ⚠️  Name required")
        return
    url = input("  URL (e.g., http://host:port/v1): ").strip()
    if not _validate_url(url):
        print("  ⚠️  URL must start with http:// or https://")
        return
    model = input("  Model: ").strip()
    if not model:
        print("  ⚠️  Model required")
        return
    key = _next_key(presets)
    presets[key] = {"name": name, "url": url, "model": model}
    _save_endpoints(presets)
    print(f"  ✅ Added as [{key}] {name}")


def _delete_endpoint(presets: dict) -> None:
    if not presets:
        print("  No endpoints to delete.")
        return
    keys = "/".join(presets.keys())
    key = input(f"  Key to delete [{keys}]: ").strip()
    if key not in presets:
        print(f"  ⚠️  '{key}' not found")
        return
    removed = presets.pop(key)
    presets = _compact_keys(presets)
    _save_endpoints(presets)
    print(f"  ✅ Removed '{removed['name']}'")


def _rename_endpoint(presets: dict) -> None:
    if not presets:
        print("  No endpoints to rename.")
        return
    keys = "/".join(presets.keys())
    key = input(f"  Key to rename [{keys}]: ").strip()
    if key not in presets:
        print(f"  ⚠️  '{key}' not found")
        return
    old_name = presets[key]["name"]
    new_name = input(f"  New name (current: {old_name}): ").strip()
    if not new_name:
        print("  ⚠️  Name required")
        return
    presets[key]["name"] = new_name
    _save_endpoints(presets)
    print(f"  ✅ Renamed to '{new_name}'")

GAME_FILES = [
    "game_files/Memories.md", "game_files/knowledgebase.md",
    "game_files/map_state.json", "game_files/session_stats.json",
    "current_state.json", "Zork Walkthrough.md",
]

_model_overrides: dict = {}
session = SessionTracker()


def backup_and_reset():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = Path(f"backups/{ts}")
    backup.mkdir(parents=True, exist_ok=True)
    backed = []

    for fp in GAME_FILES:
        src = Path(fp)
        if src.exists():
            shutil.copy2(src, backup / src.name)
            backed.append(src.name)
            if src.suffix == ".json": src.unlink()
            elif src.suffix == ".md": src.write_text("")

    # Backup episodes directory
    episodes = Path("game_files/episodes")
    if episodes.exists() and any(episodes.iterdir()):
        shutil.copytree(episodes, backup / "episodes")
        shutil.rmtree(episodes)
        episodes.mkdir(parents=True, exist_ok=True)
        backed.append(f"episodes/ ({len(list((backup / 'episodes').iterdir()))} items)")

    # Cleanup old walkthrough copies
    wt_copies = list(Path(".").glob("Zork Walkthrough.*.md"))
    if wt_copies:
        wt_dir = backup / "walkthrough_history"
        wt_dir.mkdir(exist_ok=True)
        for wt in wt_copies:
            shutil.move(str(wt), str(wt_dir / wt.name))
        backed.append(f"walkthrough_history/ ({len(wt_copies)} copies)")

    if backed:
        print(f"📦 Backed up {len(backed)} items to: {backup}")
    else:
        print("📦 No existing files to backup")


def select_models():
    global _model_overrides
    print("\n" + "-" * 60 + "\n🤖 MODEL SELECTION\n" + "-" * 60)

    presets = _load_endpoints()

    for role in ["General", "Reasoner"]:
        while True:
            print(f"\n{role} model:")
            for k, p in presets.items():
                print(f"  [{k}] {p['name']}  ({p['url']} → {p['model']})")
            print(f"  [A] Add  [D] Delete  [R] Rename")

            choice = None
            while not choice:
                try:
                    choice = input(f"\n{role} [{'/'.join(list(presets.keys()) + ['A', 'D', 'R'])}]: ").strip()
                except (KeyboardInterrupt, EOFError):
                    print("\nExiting..."); exit(0)

            if choice.upper() == "A":
                _add_endpoint(presets)
                continue
            elif choice.upper() == "D":
                _delete_endpoint(presets)
                presets = _load_endpoints()
                continue
            elif choice.upper() == "R":
                _rename_endpoint(presets)
                presets = _load_endpoints()
                continue
            elif choice not in presets:
                print(f"  ⚠️  '{choice}' not found")
                continue
            break

        preset = presets[choice]
        if role == "General":
            _model_overrides.update({
                "client_base_url": preset["url"],
                "agent_model": preset["model"], "critic_model": preset["model"],
                "extractor_model": preset["model"], "memory_model": preset["model"],
                "analysis_model": preset["model"],
            })
        else:
            _model_overrides.update({
                "reasoner_base_url": preset["url"], "reasoner_model": preset["model"],
            })

    print(f"\n✅ Models configured\n" + "-" * 60)


def run_episode(max_turns=None, generation=None):
    if generation is None:
        generation = session.start_generation()

    episode_id = f"gen_{generation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    stats = {
        'generation': generation, 'high_score': session.stats.high_score,
        'total_deaths': session.stats.total_deaths, 'best_generation': session.stats.best_generation,
    }

    orch = Orchestrator(episode_id, max_turns, stats, _model_overrides)

    print(session.header(), flush=True)
    print(f"🎮 GENERATION {generation} STARTING", flush=True)
    print(f"📏 Max turns: {orch.config.max_turns_per_episode}", flush=True)

    interrupted = False
    score = None

    try:
        score = orch.play_episode()
    except KeyboardInterrupt:
        interrupted = True
        print("\n🛑 Interrupted. [1] Save & exit  [2] Exit now")
        choice = None
        while choice not in ("1", "2"):
            try: choice = input("Choice: ").strip()
            except (KeyboardInterrupt, EOFError): choice = "2"

        if choice == "1":
            print("📦 Saving...")
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                score = orch.gs.previous_score
                orch._finalize_episode(score)
                orch._export_state()
                orch.map_mgr.save_map()
                print(f"✅ Saved. Score: {score}")
            except Exception as e:
                print(f"❌ Save error: {e}")
            finally:
                signal.signal(signal.SIGINT, signal.default_int_handler)
        raise KeyboardInterrupt()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback; traceback.print_exc()
        return

    if not interrupted and score is not None:
        died = orch.gs.game_over and score < 350
        is_high = session.end_generation(score, orch.gs.turn_count, died)

        print(f"\n{'='*60}")
        if is_high: print("🏆🏆🏆 NEW HIGH SCORE! 🏆🏆🏆")
        print(f"🎯 GENERATION {generation} COMPLETE!")
        print(f"  Score: {score}" + (" 🏆" if is_high else f" (High: {session.stats.high_score})"))
        print(f"  Turns: {orch.gs.turn_count}")
        if died: print(f"  💀 DEATH (Total: {session.stats.total_deaths})")
        print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Zork-Opus: AI plays Zork I")
    parser.add_argument("--continuous", action="store_true")
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--continue-run", action="store_true")
    args = parser.parse_args()

    # Startup mode
    if args.fresh:
        backup_and_reset()
        session.reset()
        print("✨ Fresh start")
    elif not args.continue_run:
        print(f"\n{'='*60}\n🎮 ZORK-OPUS\n{'='*60}")
        print(f"📊 {session.display()}\n")
        print("[1] 🆕 Fresh Start  [2] ▶️  Continue")
        choice = None
        while choice not in ("1", "2"):
            try: choice = input("Choice: ").strip()
            except (KeyboardInterrupt, EOFError): print(); exit(0)
        if choice == "1":
            backup_and_reset()
            session.reset()
            print("✨ Fresh start")
        else:
            print(f"▶️  Continuing from Gen {session.stats.generation + 1}")

    select_models()

    if args.continuous:
        print("🔄 CONTINUOUS MODE")
        try:
            while True:
                try: run_episode(args.max_turns)
                except KeyboardInterrupt: raise
                except Exception as e:
                    print(f"❌ {e}"); time.sleep(5)
        except KeyboardInterrupt:
            print(f"\n📊 Final: {session.display()}")
    elif args.episodes > 1:
        try:
            for i in range(args.episodes):
                run_episode(args.max_turns)
        except KeyboardInterrupt:
            print(f"\n📊 Final: {session.display()}")
    else:
        try: run_episode(args.max_turns)
        except KeyboardInterrupt:
            print(f"\n📊 {session.display()}")


if __name__ == "__main__":
    main()
