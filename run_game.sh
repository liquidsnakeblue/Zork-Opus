#!/usr/bin/env bash
# Zork-Opus launcher.
# Defaults: General=#15 Qwen 27B Liquid, Reasoner=#16 Claude Opus 4.8
# Also available (via 192.168.1.41:8317 router):
#   #17 Claude Fable 5   (claude-fable-5 — rejects temperature/top_p, client strips them)
#   #18 GPT 5.6 Sol      (gpt-5.6-sol)
# Override without editing:
#   REASONER=17 ./run_game.sh              # Fable as reasoner
#   GENERAL=18 REASONER=17 ./run_game.sh   # GPT 5.6 Sol general + Fable reasoner
cd /home/liquidsnakeblue/Zork-Opus
source .venv/bin/activate
exec python main.py --continue-run --continuous --general-preset "${GENERAL:-15}" --reasoner-preset "${REASONER:-16}"
