#!/usr/bin/env bash
# Zork-Opus launcher: General=#15 Qwen 27B Liquid, Reasoner=#16 Opus 4.8
cd /home/liquidsnakeblue/Zork-Opus
source .venv/bin/activate
exec python main.py --continue-run --continuous --general-preset 15 --reasoner-preset 16
