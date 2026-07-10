#!/usr/bin/env bash
# Zork-Opus launcher.
# Defaults: General=#18 GPT 5.6 Sol, Reasoner=#17 Claude Fable 5
#   (#17 claude-fable-5 rejects temperature/top_p, client strips them)
# Also available: #15 Qwen 27B Liquid, #16 Claude Opus 4.8
# Override without editing:
#   GENERAL=15 REASONER=16 ./run_game.sh   # previous duo (Qwen general + Opus 4.8 reasoner)
#   FRESH=1 ./run_game.sh                  # clean run: back up + wipe learned state, reset to Gen 1
#                                          # (procedures.json survives — canonical, git-tracked)
cd /home/liquidsnakeblue/Zork-Opus
source .venv/bin/activate
if [[ -n "${FRESH:-}" ]]; then
  MODE="--fresh"
else
  MODE="--continue-run"
fi
exec python main.py "$MODE" --continuous --general-preset "${GENERAL:-18}" --reasoner-preset "${REASONER:-17}"
