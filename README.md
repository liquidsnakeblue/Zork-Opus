# Zork-Opus

> An autonomous AI agent that plays [Zork I](https://en.wikipedia.org/wiki/Zork_I:_The_Great_Underground_Empire), the classic 1980 text adventure game. Uses LLMs to explore, solve puzzles, collect treasures, and beat the game — without human intervention.

Clean rewrite optimized for large-context models (Claude Opus 4.6, Qwen 3.5, etc.) with a modular architecture spanning ~6,600 lines of Python across 22 modules.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Orchestrator                             │
│              (Game loop coordinator — 1,100 lines)              │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │  Agent   │→ │  Critic  │→ │Extractor │  │Jericho (Z5)   │  │
│  │(action   │  │(eval +   │  │(pure     │  │  game engine   │  │
│  │ gen)     │  │ validate)│  │ Z-machine│  │  via Frotz    │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └───────┬───────┘  │
│       │              │             │                │           │
│  ┌────▼──────────────▼─────────────▼────────────────▼───┐      │
│  │                    GameState                          │      │
│  │         (Single source of truth dataclass)            │      │
│  └──────────────────────────────────────────────────────┘      │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │  Memory  │  │   Map    │  │Knowledge │  │ Objectives    │  │
│  │(location │  │(BFS graph│  │(universal│  │(reasoner-     │  │
│  │-based +  │  │ + Mermaid│  │strategy) │  │driven goals)  │  │
│  │ LLM syn) │  │  render) │  │          │  │               │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────────────┘  │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ Pathfinder│ │ Context  │  │Walkthrough│ │ Web Search    │  │
│  │(BFS nav) │  │Manager   │  │(puzzle    │ │(SearXNG +    │  │
│  │          │  │(prompt   │  │ guide)    │ │ Crawl4AI)     │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────────────┘  │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                      │
│  │Streaming │  │ Prompt   │  │ Session  │                      │
│  │(WebSocket│  │ Logger   │  │(stats    │                      │
│  │ → HTML   │  │(JSONL    │  │ persistence)                    │
│  │ viewer)  │  │ debug)   │  │          │                      │
│  └──────────┘  └──────────┘  └──────────┘                      │
└─────────────────────────────────────────────────────────────────┘
```

## Components

| Module | Lines | Purpose |
|--------|-------|---------|
| `orchestrator.py` | 1,121 | Main game loop — coordinates all components through turn-based play |
| `memory.py` | 960 | Location-based memory system with dual cache (persistent + ephemeral), LLM synthesis, spawn item detection |
| `objectives.py` | 621 | Strategic objective generation via reasoner LLM, completion checking, lifecycle management |
| `config.py` | 282 | Pydantic settings loaded from `pyproject.toml` — single source of truth for all configuration |
| `llm_client.py` | 405 | Direct HTTP LLM client with retry, circuit breaker, streaming, and `extract_json()` for robust parsing |
| `context_manager.py` | 307 | Assembles agent prompts from game state, memories, objectives, map, navigation, and item tracker |
| `critic.py` | 204 | Action evaluation with Z-machine object tree pre-check and LLM scoring |
| `agent.py` | 271 | JSON-structured action generation from LLM with thinking/reasoning mode support |
| `map_manager.py` | 243 | Map building from Z-machine ground truth, direction detection, path validation via save/restore probing |
| `map_graph.py` | 251 | Room/connection graph with BFS pathfinding, Mermaid rendering, exit failure tracking, exploration frontier |
| `game_interface.py` | 205 | Jericho/Frotz wrapper — inventory, location, objects, exits, save/restore, score |
| `knowledge.py` | 331 | Periodic knowledge base updates and cross-episode synthesis via LLM analysis |
| `state.py` | 301 | Central `GameState` dataclass — all mutable state for an episode |
| `streaming.py` | 154 | WebSocket server broadcasting real-time reasoning, memory synthesis, objective review to HTML viewer |
| `walkthrough.py` | 129 | Incremental puzzle-solving guide generation from memories |
| `prompt_logger.py` | 117 | Local JSONL prompt/response logging (replaces Langfuse) |
| `pathfinder.py` | 143 | BFS navigation through known map; LLM decides where, BFS computes how |
| `main.py` | 247 | CLI entry point — model selection, episode management, backup and reset |
| `logger.py` | 140 | Dual-format logging (console + JSON file) with event-type filtering |
| `web_search.py` | 83 | MCP-based web search (SearXNG) and page crawling (Crawl4AI) for puzzle hints |
| `session.py` | 71 | Persistent session stats (generation count, high score, deaths) |
| `extractor.py` | 67 | Pure Z-machine data extraction — no LLM calls |

## How It Works

### The Game Loop (per turn)

1. **Extract** — Jericho reads Z-machine state (location, inventory, visible objects, exits, score)
2. **Context** — ContextManager assembles a rich prompt: current state, memories at this location, objectives, map, navigation guidance, item tracker, recent actions
3. **Agent** — LLM receives the context and outputs `{"thinking": "...", "action": "..."}`
4. **Special Actions** — If the action is `Pathfinder: 79`, `Objective: A001`, `Search: ...`, or `Crawl: ...`, the orchestrator intercepts and dispatches to the appropriate handler
5. **Critic** (optional) — LLM evaluates the proposed action with Z-machine ground truth; rejects if score < threshold
6. **Execute** — Command sent to Z-machine via Jericho
7. **Sync** — Score, location, inventory read from Z-machine (authoritative)
8. **Map Update** — New rooms and connections recorded; direction verified by testing all canonical directions via save/restore
9. **Memory Synthesis** — LLM decides if this turn's outcome is worth remembering (PUZZLE, SUCCESS, DISCOVERY, DANGER, etc.)
10. **Objective Check** — LLM reviews if any objectives were completed
11. **Reasoner** (periodic) — Strategic LLM generates/updates objectives every N turns, considering full game state, map frontier, knowledge base, and memories
12. **Knowledge Update** (periodic) — LLM analyzes episode data to build universal strategic knowledge

### Cross-Episode Persistence

Between game resets ("generations"), knowledge accumulates:

- **Memories** (`game_files/Memories.md`) — Location-tagged discoveries persist; ephemeral memories cleared each episode
- **Knowledge Base** (`game_files/knowledgebase.md`) — Universal strategic wisdom distilled from gameplay
- **Map** (`game_files/map_state.json`) — Room graph with verified connections (directed, Z-machine proven)
- **Walkthrough** (`Zork Walkthrough.md`) — Auto-generated puzzle-solving guide, incrementally updated
- **Session Stats** (`game_files/session_stats.json`) — Generation count, high score, total deaths

### Anti-Stuck Systems

- **Score stagnation detection** — Episode terminates if no score change for N turns
- **Location revisit penalty** — Context penalizes returning to recently visited locations
- **Action novelty window** — Recent action history prevents exact repetition
- **Oscillation detection** — Warns when agent bounces between same 2-3 rooms
- **Maze detection** — Suggests item-dropping markers when many turns spent in same area
- **Pathfinder failure tracking** — Recently failed navigation targets are blocked

## Quick Start

### Prerequisites

- **Python 3.11+**
- **Zork I** (`.z5` file) in `jericho-game-suite/zork1.z5`
- **LLM API endpoint** (OpenAI-compatible chat completions)

```bash
# Clone and set up
cd Zork-Opus
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test,s3]"

# Configure your LLM in pyproject.toml [tool.zorkopus.llm]
# Or use the interactive model selector at runtime
```

### Running

```bash
# Interactive mode (single generation, choose model)
python main.py

# Fresh start (backups existing state)
python main.py --fresh

# Continuous mode (run forever, new generation on death/game over)
python main.py --continuous

# Fixed number of episodes
python main.py --episodes 5

# Fresh start with turn limit
python main.py --fresh --max-turns 200

# Continue from last generation without prompts
python main.py --continue-run
```

### Model Selection

At startup, you'll be prompted to select a **General** model (used for agent, critic, memory, extractor, analysis) and a **Reasoner** model (used for objectives and walkthrough). Presets are loaded from `endpoints.json`.

**Managing Endpoints:**

Endpoints are stored in `endpoints.json` (not committed to git — add your own). Defaults are embedded in `main.py` and auto-generated on first run. During model selection, you can:

- **[A] Add** — Prompt for name, URL, model → appends as next key
- **[D] Delete** — Remove an endpoint by key → keys renumbered 1..N
- **[R] Rename** — Change the display name of an endpoint

You can also edit `endpoints.json` directly in your editor:

```json
{
  "presets": {
    "1": { "name": "My Model", "url": "http://...", "model": "model-id" }
  }
}
```

Default presets available:

| # | Model | Endpoint |
|---|-------|----------|
| 1 | Claude Opus 4.6 | Local (Ollama/LM Studio) |
| 2 | Claude Sonnet 4.6 | Local |
| 3 | Claude Haiku 4.5 | Local |
| 4 | Qwen 3.5 27B QuantTrio | Local vLLM (5090) |
| 5 | MiniMax M2.5 | schuyler.ai |
| 6 | DeepSeek R1 0528 | OpenRouter |
| 7 | Gemini 3 Flash | OpenRouter |
| 8 | Gemini 2.5 Flash | OpenRouter |
| 9 | Qwen 3.6 Plus | OpenRouter (free) |
| 10 | Gemma 4 31B IT | Local vLLM |
| 11 | Qwen 3.5 27B DFlash | Local vLLM (5090) |
| 12 | Qwen 3.6 27B | LM Studio |

### Configuration

All settings live in `pyproject.toml` under `[tool.zorkopus.*]`. Key sections:

- `[tool.zorkopus.llm]` — Model names, base URLs, per-role URL overrides
- `[tool.zorkopus.agent_sampling]` — Temperature, top_p, max_tokens, thinking mode
- `[tool.zorkopus.orchestrator]` — Max turns, knowledge/objective update intervals
- `[tool.zorkopus.loop_break]` — Anti-stuck thresholds and penalties
- `[tool.zorkopus.pathfinder]` — Enable/disable, max path length
- `[tool.zorkopus.streaming]` — WebSocket host/port for live viewer
- `[tool.zorkopus.web_search]` — SearXNG/Crawl4AI URLs, enabled/disabled
- `[tool.zorkopus.prompt_logger]` — JSONL debug logging directory

### Live Viewer

Open `viewer.html` in a browser (use a local server, e.g., `python -m http.server`). Connects to the WebSocket server at `ws://localhost:8765` to stream:

- Real-time agent reasoning as it's generated
- Memory synthesis decisions
- Objective reasoner output
- Pathfinder navigation progress
- Interactive game map (Cytoscape + dagre)
- Turn-by-turn card timeline with expandable details

## Project Structure

```
Zork-Opus/
├── main.py                 # CLI entry point
├── orchestrator.py         # Game loop coordinator
├── agent.py                # Action generation from LLM
├── critic.py               # Action evaluation + Z-machine validation
├── extractor.py            # Pure Z-machine data extraction
├── memory.py               # Memory system (synthesis, cache, file I/O)
├── map_graph.py            # Room/connection graph + BFS pathfinding
├── map_manager.py          # Map building from movement + direction detection
├── pathfinder.py           # BFS navigation to room targets
├── context_manager.py      # Prompt assembly from all subsystems
├── knowledge.py            # Knowledge base generation + cross-episode synthesis
├── objectives.py           # Objective lifecycle + reasoner integration
├── walkthrough.py          # Puzzle-solving guide generation
├── config.py               # Pydantic settings from pyproject.toml
├── llm_client.py           # HTTP client with retry/circuit breaker/streaming
├── game_interface.py       # Jericho/Frotz wrapper
├── state.py                # Central GameState dataclass
├── streaming.py            # WebSocket server for live viewer
├── web_search.py           # MCP-based search + crawl
├── prompt_logger.py        # Local JSONL debug logging
├── logger.py               # Console + file logging
├── session.py              # Persistent session stats
├── viewer.html             # Live WebSocket viewer (4,700+ lines)
├── pyproject.toml          # Dependencies + all configuration
├── prompts/
│   ├── agent.md            # Agent system prompt + game guide
│   └── critic.md           # Critic evaluation guidelines
├── room_images/            # AI-generated room images (110 PNGs + 110 TXTs)
├── game_files/             # Runtime data directory
│   ├── episodes/           # Per-episode logs
│   ├── prompt_logs/        # Per-episode LLM call logs (JSONL)
│   ├── Memories.md         # Location-tagged memories (persists)
│   ├── knowledgebase.md    # Strategic knowledge (persists)
│   └── map_state.json      # World map graph (persists)
├── jericho-game-suite/     # Zork I .z5 game file
├── backups/                # Timestamped state backups
├── Zork Walkthrough.md     # Auto-generated puzzle guide (persists)
└── .gitignore
```

## Technical Details

### Z-Machine Interaction

The game is played through [Jericho](https://github.com/dfmller/jericho), a Python library wrapping the Frotz Z-machine interpreter. Key capabilities:

- **Ground-truth exits**: Every direction is tested via save/restore for 100% accuracy
- **Object tree access**: Full Z-machine object hierarchy (visible objects, containers, attributes)
- **State save/restore**: Used for direction verification, spawn item detection, and path validation
- **Room name CSV**: Custom room names from `zorkRooms.csv` override Z-machine defaults

### LLM Client

Custom HTTP client (no OpenAI SDK dependency) supporting:

- OpenAI-compatible `chat/completions` endpoint
- Extended parameters: `top_k`, `min_p`, `repetition_penalty`, `enable_thinking`
- Streaming with chunk callbacks for real-time viewer updates
- Exponential backoff with jitter (configurable)
- Circuit breaker pattern (opens after N failures, recovers after timeout)
- Robust JSON extraction: handles markdown fences, thinking blocks, format tokens, unescaped control characters, and unbalanced braces

### Memory System

- **Location-tagged**: Memories are stored at the source location where the event occurred
- **Dual cache**: Persistent (core/permanent) survives episode reset; ephemeral (agent-caused state) is cleared each episode
- **LLM synthesis**: After each turn, the memory LLM decides if the outcome is worth remembering, categorizes it, and checks for contradictions with existing memories
- **Supersession**: New memories can supersede or invalidate old ones (with persistence rules preventing permanent→ephemeral downgrade)
- **Spawn detection**: Items are tested for takeability on first visit via save/restore; starting locations recorded as core memories

### Map System

- **Directed graph**: Connections are recorded only in the direction traversed (Zork has one-way passages, closing doors, etc.)
- **Z-machine verified**: Every connection is confirmed by actual traversal; directions are verified by testing all canonical directions
- **Exit failure tracking**: Failed directions are counted and pruned from room exit hints after threshold (but never from verified connections)
- **Exploration frontier**: Untried exits per room computed from Z-machine ground truth, fed to reasoner for strategic planning
- **BFS pathfinding**: Shortest path through known connections; path validation via multi-probe save/restore to catch intermittently blocked hops

### Objective System

- **Reasoner-driven**: A separate LLM call (every N turns or when all objectives complete) generates 3-7 strategic objectives
- **Two types**: ACTION (specific task with target location) or EXPLORATION (discovery when blocked)
- **Auto-navigation**: When an in_progress objective has a target_location_id, BFS path is auto-computed and shown as navigation guidance
- **Completion checking**: LLM verifies objective completion conditions against current state after each turn
- **Fallback**: Basic objectives created if reasoner is unavailable

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENROUTER_API_KEY` | API key for OpenRouter endpoints |
| `CLIENT_API_KEY` | API key for other endpoints |
| `ZORK_S3_BUCKET` | Optional S3 bucket for state export |

## Optional Features

- **Web Search** (`enable_web_search = true`): Agent can search the web via SearXNG MCP and crawl pages via Crawl4AI when stuck on puzzles. Requires a running SearXNG instance and Crawl4AI MCP endpoint.
- **AWS S3 Export** (`s3_bucket = "..."`): State exports are also pushed to S3. Requires `pip install -e ".[s3]"`.
- **Critic** (`enable_critic = true`): LLM-based action evaluation before execution. Adds latency but can prevent bad actions.
- **Streaming** (`enable_streaming = true`): Real-time WebSocket broadcasting to `viewer.html`. Requires `websockets` package.

## Contributing

All source modules live in the project root. Configuration is centralized in `pyproject.toml`. The architecture favors flat files over deep nesting — each module is self-contained and communicates through the shared `GameState` dataclass.

## License

This project uses the [Jericho](https://github.com/dfmller/jericho) library (MIT License) for Z-machine interaction.
