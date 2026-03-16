"""
Memory system - consolidated from 6 files into 1.
Location-based memories with dual cache (persistent + ephemeral),
file persistence, LLM synthesis, and spawn item detection.
"""

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Literal
from pydantic import BaseModel, Field, model_validator

from config import Config
from state import GameState, ActionEntry
from llm_client import LLMClient, extract_json


# ── Data Models ──

MemoryStatusType = Literal["ACTIVE", "TENTATIVE", "SUPERSEDED"]

class MemoryStatus:
    ACTIVE: MemoryStatusType = "ACTIVE"
    TENTATIVE: MemoryStatusType = "TENTATIVE"
    SUPERSEDED: MemoryStatusType = "SUPERSEDED"

INVALIDATION_MARKER = "INVALIDATED"


@dataclass
class Memory:
    category: str
    title: str
    episode: int
    turns: str
    score_change: Optional[int]
    text: str
    persistence: str  # "core" | "permanent" | "ephemeral"
    status: MemoryStatusType = MemoryStatus.ACTIVE
    superseded_by: Optional[str] = None
    superseded_at_turn: Optional[int] = None
    invalidation_reason: Optional[str] = None
    goal: Optional[str] = None

    def __post_init__(self):
        if self.persistence not in ("core", "permanent", "ephemeral"):
            raise ValueError(f"Invalid persistence: {self.persistence}")


class SynthesisResponse(BaseModel):
    model_config = {"strict": True}
    reasoning: str = Field(default="", max_length=8000)
    should_remember: bool
    category: Optional[str] = None
    memory_title: Optional[str] = None
    memory_text: Optional[str] = None
    persistence: Optional[str] = None
    status: MemoryStatusType = Field(default=MemoryStatus.ACTIVE)
    supersedes_memory_titles: Set[str] = Field(default_factory=set, max_length=10)
    invalidate_memory_titles: Set[str] = Field(default_factory=set, max_length=10)
    invalidation_reason: Optional[str] = None
    goal: Optional[str] = None

    @model_validator(mode='after')
    def _validate(self) -> 'SynthesisResponse':
        if self.should_remember:
            if not self.category:
                self.category = "NOTE"  # Default instead of crashing
            if not self.memory_title:
                self.should_remember = False
                return self
            if not self.memory_text:
                self.should_remember = False
                return self
            if self.persistence not in ("core", "permanent", "ephemeral"):
                self.persistence = "permanent"  # Default instead of crashing
        if self.invalidate_memory_titles and not (self.invalidation_reason or "").strip():
            self.invalidation_reason = "Contradicted by new observation"
        overlap = self.supersedes_memory_titles & self.invalidate_memory_titles
        if overlap:
            raise ValueError(f"Can't both supersede and invalidate: {overlap}")
        return self


# ── Cache Manager ──

class MemoryCache:
    """Dual cache: persistent (core/permanent) + ephemeral."""

    def __init__(self):
        self.persistent: Dict[int, List[Memory]] = {}
        self.ephemeral: Dict[int, List[Memory]] = {}

    def add(self, location_id: int, memory: Memory):
        cache = self.ephemeral if memory.persistence == "ephemeral" else self.persistent
        cache.setdefault(location_id, []).append(memory)

    def get(self, location_id: int, *, include_superseded=False, persistent_only=False,
            ephemeral_only=False) -> List[Memory]:
        result = []
        if not ephemeral_only:
            result.extend(self.persistent.get(location_id, []))
        if not persistent_only:
            result.extend(self.ephemeral.get(location_id, []))
        if not include_superseded:
            result = [m for m in result if m.status != MemoryStatus.SUPERSEDED]
        return result

    def supersede(self, location_id: int, title: str, by_title: str, turn: int) -> bool:
        for cache in (self.persistent, self.ephemeral):
            for m in cache.get(location_id, []):
                if m.title.strip().lower() == title.strip().lower():
                    m.status = MemoryStatus.SUPERSEDED
                    m.superseded_by = by_title
                    m.superseded_at_turn = turn
                    return True
        return False

    def invalidate(self, location_id: int, title: str, reason: str, turn: int) -> bool:
        for cache in (self.persistent, self.ephemeral):
            for m in cache.get(location_id, []):
                if m.title.strip().lower() == title.strip().lower():
                    m.status = MemoryStatus.SUPERSEDED
                    m.superseded_by = INVALIDATION_MARKER
                    m.superseded_at_turn = turn
                    m.invalidation_reason = reason
                    return True
        return False

    def clear_ephemeral(self) -> int:
        count = sum(len(v) for v in self.ephemeral.values())
        self.ephemeral.clear()
        return count

    @property
    def location_count(self) -> int:
        return len(self.persistent)

    @property
    def total_persistent(self) -> int:
        return sum(len(v) for v in self.persistent.values())


# ── File Parser ──

_LOC_RE = re.compile(r"^## Location (\d+): (.*)$")
_MEM_RE = re.compile(r"^\*\*\[(\w+)(?: - (\w+))?(?: - (\w+))?\] (.+?)\*\* \*\((.*?)\)\*$")
_INV_RE = re.compile(r'^\[Invalidated at T(\d+): "([^"]*)"\]$')


def parse_memories_file(path: Path, cache: MemoryCache, logger=None):
    """Parse Memories.md into the cache."""
    if not path.exists():
        return

    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        if logger: logger.error(f"Failed to read {path}: {e}")
        return

    location_id = None
    header = None
    text_lines = []
    inv_info = None

    def flush():
        nonlocal header, text_lines, inv_info
        if header and location_id is not None:
            _add_parsed_memory(cache, location_id, header, text_lines, inv_info, logger)
        header = None
        text_lines = []
        inv_info = None

    for line in content.split("\n"):
        line = line.rstrip()

        if line.startswith("## Location"):
            flush()
            m = _LOC_RE.match(line)
            if m:
                location_id = int(m.group(1))
            else:
                location_id = None
            continue

        m = _MEM_RE.match(line)
        if m:
            flush()
            if location_id is not None:
                cat, f2, f3, title, meta = m.groups()
                persistence = "permanent"
                status = MemoryStatus.ACTIVE
                if f3:
                    if f2 and f2.upper() in ("CORE", "PERMANENT", "EPHEMERAL"):
                        persistence = f2.lower()
                    if f3 in (MemoryStatus.ACTIVE, MemoryStatus.TENTATIVE, MemoryStatus.SUPERSEDED):
                        status = f3
                elif f2:
                    u = f2.upper()
                    if u in ("CORE", "PERMANENT", "EPHEMERAL"):
                        persistence = f2.lower()
                    elif f2 in (MemoryStatus.ACTIVE, MemoryStatus.TENTATIVE, MemoryStatus.SUPERSEDED):
                        status = f2
                header = (cat, persistence, status, title.strip(), meta.strip())
            continue

        if header and line.strip():
            if line.strip() in ("---", "###", "### Memories"):
                continue
            if line.startswith("**Visits:**") or line.startswith("**Episodes:**"):
                continue
            if line.startswith("#"):
                continue
            if line.strip().startswith("[Superseded at"):
                continue
            if line.strip().startswith("[Invalidated at"):
                im = _INV_RE.match(line.strip())
                if im:
                    inv_info = (int(im.group(1)), im.group(2))
                continue
            text_lines.append(line.strip())

    flush()

    if logger:
        logger.info(f"Loaded {cache.total_persistent} memories from {cache.location_count} locations")


def _add_parsed_memory(cache, loc_id, header, text_lines, inv_info, logger):
    cat, persistence, status, title, meta = header
    try:
        ep, turns, score, goal = _parse_meta(meta)
        text = " ".join(text_lines)
        if status == MemoryStatus.SUPERSEDED and text.startswith("~~") and text.endswith("~~"):
            text = text[2:-2]
        inv_reason = inv_info[1] if inv_info else None
        m = Memory(
            category=cat, title=title, episode=ep, turns=turns,
            score_change=score, text=text, persistence=persistence, status=status,
            superseded_by=INVALIDATION_MARKER if inv_reason else None,
            invalidation_reason=inv_reason, goal=goal,
        )
        cache.add(loc_id, m)
    except Exception as e:
        if logger: logger.warning(f"Skipping malformed memory [{cat}] {title}: {e}")


def _parse_meta(meta: str) -> Tuple[int, str, Optional[int], Optional[str]]:
    parts = [p.strip() for p in meta.split(",")]
    ep = int(parts[0][2:]) if parts[0].startswith("Ep") else 1
    turns = parts[1][1:] if len(parts) > 1 and parts[1].startswith("T") else ""
    score = None
    goal = None
    for p in parts[2:]:
        p = p.strip()
        if p.startswith("Goal:"): goal = p[5:].strip()
        elif p.startswith("+") or p.startswith("-"):
            try: score = int(p)
            except ValueError: pass
    return ep, turns, score, goal


# ── File Writer ──

def write_memory(config: Config, memory: Memory, location_id: int, location_name: str, logger=None) -> bool:
    """Write a memory to Memories.md with backup."""
    path = Path(config.game_workdir) / "Memories.md"
    try:
        if path.exists():
            shutil.copy(path, str(path) + ".backup")
            content = path.read_text(encoding="utf-8")
        else:
            content = "# Location Memories\n\n"

        sections = _find_sections(content)
        if location_id in sections:
            content = _append_to_section(content, sections[location_id], memory)
        else:
            content = _new_section(content, location_id, location_name, memory)

        path.write_text(content, encoding="utf-8")
        return True
    except Exception as e:
        if logger: logger.error(f"Failed to write memory: {e}")
        return False


def update_memory_status(config: Config, location_id: int, title: str, new_status: str,
                         superseded_by: str = None, turn: int = None,
                         invalidation_reason: str = None, logger=None) -> bool:
    """Update status of existing memory in file."""
    path = Path(config.game_workdir) / "Memories.md"
    if not path.exists():
        return False

    try:
        shutil.copy(path, str(path) + ".backup")
        content = path.read_text(encoding="utf-8")
        lines = content.split("\n")

        if invalidation_reason:
            ref_line = f'[Invalidated at T{turn}: "{invalidation_reason}"]'
        else:
            ref_line = f'[Superseded at T{turn} by "{superseded_by}"]'

        in_loc = False
        found = False
        result = []
        i = 0

        while i < len(lines):
            line = lines[i]
            lm = _LOC_RE.match(line)
            if lm:
                in_loc = (int(lm.group(1)) == location_id)
                result.append(line); i += 1; continue

            if in_loc:
                mm = _MEM_RE.match(line)
                if mm:
                    cat, f2, f3, t, meta = mm.groups()
                    if title.strip().lower() == t.strip().lower():
                        found = True
                        # Determine persistence marker
                        pm = None
                        if f3 and f2 and f2.upper() in ("CORE", "PERMANENT", "EPHEMERAL"):
                            pm = f2.upper()
                        elif f2 and f2.upper() in ("CORE", "PERMANENT", "EPHEMERAL"):
                            pm = f2.upper()

                        if pm:
                            result.append(f"**[{cat} - {pm} - {new_status}] {t}** *({meta})*")
                        else:
                            result.append(f"**[{cat} - {new_status}] {t}** *({meta})*")
                        result.append(ref_line)
                        i += 1

                        # Wrap text in strikethrough
                        while i < len(lines):
                            tl = lines[i]
                            if tl.startswith("**[") or tl.strip() == "---" or tl.startswith("##"):
                                break
                            if tl.strip().startswith("[Superseded at") or tl.strip().startswith("[Invalidated at"):
                                i += 1; continue
                            if tl.strip() and not tl.strip().startswith("~~"):
                                tl = f"~~{tl}~~"
                            result.append(tl); i += 1
                        continue

            result.append(line); i += 1

        if not found:
            return False

        path.write_text("\n".join(result), encoding="utf-8")
        return True
    except Exception as e:
        if logger: logger.error(f"Failed to update memory status: {e}")
        return False


def _find_sections(content: str) -> Dict[int, Dict[str, Any]]:
    sections = {}
    lines = content.split("\n")
    cur_id = None
    cur_start = None
    for i, line in enumerate(lines):
        if line.startswith("## Location"):
            if cur_id is not None:
                sections[cur_id]["end"] = i - 1
            m = _LOC_RE.match(line)
            if m:
                cur_id = int(m.group(1))
                cur_start = i
                sections[cur_id] = {"start": i, "end": None, "name": m.group(2).strip()}
    if cur_id is not None:
        sections[cur_id]["end"] = len(lines) - 1
    return sections


def _format_entry(m: Memory) -> str:
    meta = [f"Ep{m.episode}", f"T{m.turns}"]
    if m.score_change is not None:
        meta.append(f"+{m.score_change}" if m.score_change >= 0 else str(m.score_change))
    if m.goal:
        meta.append(f"Goal: {m.goal}")

    cat_str = m.category
    if m.persistence in ("core", "permanent"):
        cat_str = f"{m.category} - {m.persistence.upper()}"

    if m.status == MemoryStatus.ACTIVE:
        header = f"**[{cat_str}] {m.title}** *({', '.join(meta)})*"
    else:
        header = f"**[{cat_str} - {m.status}] {m.title}** *({', '.join(meta)})*"

    text = f"~~{m.text}~~" if m.status == MemoryStatus.SUPERSEDED else m.text
    lines = [header]
    if m.status == MemoryStatus.SUPERSEDED:
        if m.invalidation_reason:
            lines.append(f'[Invalidated at T{m.superseded_at_turn}: "{m.invalidation_reason}"]')
        elif m.superseded_by:
            lines.append(f'[Superseded at T{m.superseded_at_turn} by "{m.superseded_by}"]')
    lines.append(text)
    return "\n".join(lines)


def _append_to_section(content: str, section: Dict, memory: Memory) -> str:
    lines = content.split("\n")
    insert = section["end"]
    for i in range(section["start"], section["end"] + 1):
        if lines[i].strip() == "---":
            insert = i; break

    # Update visit count
    vi = section["start"] + 1
    if vi < len(lines):
        vm = re.search(r"\*\*Visits:\*\* (\d+)", lines[vi])
        visits = int(vm.group(1)) + 1 if vm else 1
        em = re.search(r"\*\*Episodes:\*\* ([\d, ]+)", lines[vi])
        episodes = set()
        if em:
            episodes = {int(e.strip()) for e in em.group(1).split(",") if e.strip()}
        episodes.add(memory.episode)
        lines[vi] = f"**Visits:** {visits} | **Episodes:** {', '.join(str(e) for e in sorted(episodes))}"

    lines.insert(insert, _format_entry(memory))
    lines.insert(insert + 1, "")
    return "\n".join(lines)


def _new_section(content: str, loc_id: int, loc_name: str, memory: Memory) -> str:
    content = content.rstrip()
    if not content.endswith("\n"): content += "\n"
    if not content.endswith("\n\n"): content += "\n"

    section = [
        f"## Location {loc_id}: {loc_name}",
        f"**Visits:** 1 | **Episodes:** {memory.episode}",
        "", "### Memories", "",
        _format_entry(memory), "",
        "---", "",
    ]
    return content + "\n".join(section)


# ── Spawn Item Detector ──

EPHEMERAL_IDS = {114, 115, 113}  # thief, bag, stiletto
EPHEMERAL_PARENTS = {114}        # thief


@dataclass
class SpawnItem:
    name: str
    obj_id: int
    location_id: int
    location_name: str
    surface_name: Optional[str] = None


class SpawnDetector:
    def __init__(self, logger=None):
        self.logger = logger
        self._memorized: set = set()

    def detect(self, jericho, location_id: int, location_name: str) -> List[SpawnItem]:
        items = []
        try:
            visible = jericho.get_visible_objects()
            if not visible: return []
            all_objs = jericho.get_all_objects()

            for obj in visible:
                item = self._test(jericho, obj, location_id, location_name)
                if item: items.append(item)
                if obj.num not in EPHEMERAL_PARENTS:
                    for child in all_objs:
                        if child.parent == obj.num:
                            ci = self._test(jericho, child, location_id, location_name,
                                          surface=obj.name, parent_id=obj.num)
                            if ci: items.append(ci)
        except Exception as e:
            if self.logger: self.logger.error(f"Spawn detection failed: {e}")
        return items

    def _test(self, jericho, obj, loc_id, loc_name, surface=None, parent_id=None) -> Optional[SpawnItem]:
        if obj.num in EPHEMERAL_IDS: return None
        if parent_id and parent_id in EPHEMERAL_PARENTS: return None
        attrs = jericho.get_object_attributes(obj)
        if attrs.get('touched', False): return None

        state = jericho.save_state()
        try:
            resp = jericho.send_command(f"take {obj.name}")
            is_takeable = "taken" in resp.lower()
        finally:
            jericho.restore_state(state)

        if is_takeable:
            return SpawnItem(obj.name, obj.num, loc_id, loc_name, surface)
        return None

    def filter_new(self, items: List[SpawnItem]) -> List[SpawnItem]:
        return [i for i in items if i.obj_id not in self._memorized]

    def mark_memorized(self, items: List[SpawnItem]):
        for i in items: self._memorized.add(i.obj_id)

    def reset(self):
        self._memorized.clear()

    def create_memories(self, items: List[SpawnItem], episode: int, turn: int) -> List[Memory]:
        return [
            Memory(
                category="DISCOVERY", title=f"{i.name} at spawn", episode=episode,
                turns=str(turn), score_change=0,
                text=f"{i.name} (ID: {i.obj_id}) is present at this location at game start."
                     + (f" Found on {i.surface_name}." if i.surface_name else ""),
                persistence="core", status=MemoryStatus.ACTIVE,
            )
            for i in items
        ]


# ── Memory Manager (orchestrates cache, file I/O, and LLM synthesis) ──

class MemoryManager:
    """Manages location-based memories with dual cache, file persistence, and LLM synthesis."""

    def __init__(self, config: Config, game_state: GameState,
                 llm_client: Optional[LLMClient] = None, logger=None, streaming_server=None):
        self.config = config
        self.game_state = game_state
        self.logger = logger
        self.streaming_server = streaming_server
        self.cache = MemoryCache()
        self._llm = llm_client
        self._llm_ready = llm_client is not None

        # Parse existing memories
        parse_memories_file(
            Path(config.game_workdir) / "Memories.md",
            self.cache, logger
        )

    @property
    def llm_client(self):
        if not self._llm_ready:
            self._llm = LLMClient(config=self.config, logger=self.logger)
            self._llm_ready = True
        return self._llm

    def reset_episode(self):
        count = self.cache.clear_ephemeral()
        if self.logger:
            self.logger.info(f"Cleared {count} ephemeral memories")

    def add_memory(self, location_id: int, location_name: str, memory: Memory) -> bool:
        # Duplicate check
        existing = self.cache.get(location_id, include_superseded=True)
        if any(e.title == memory.title for e in existing):
            return False

        if memory.persistence == "ephemeral":
            self.cache.add(location_id, memory)
            return True

        if write_memory(self.config, memory, location_id, location_name, self.logger):
            self.cache.add(location_id, memory)
            return True
        return False

    def get_location_memory(self, location_id: int) -> str:
        memories = self.cache.get(location_id)
        if not memories: return ""

        active = [m for m in memories if m.status == MemoryStatus.ACTIVE]
        tentative = [m for m in memories if m.status == MemoryStatus.TENTATIVE]

        lines = []
        for m in active:
            lines.append(f"[{m.category}] {m.title}: {m.text}")
        if tentative:
            if active: lines.append("")
            lines.append("⚠️  TENTATIVE (unconfirmed):")
            for m in tentative:
                lines.append(f"  [{m.category}] {m.title}: {m.text}")
        return "\n".join(lines)

    def get_puzzle_summary(self, max_entries: int = 40) -> str:
        """Get a summary of key PUZZLE and DISCOVERY memories for the reasoner, prioritized by importance."""
        all_mems = []
        for loc_id, memories in self.cache.persistent.items():
            for m in memories:
                if m.status != MemoryStatus.ACTIVE:
                    continue
                if m.category not in ("PUZZLE", "DISCOVERY", "SUCCESS", "FAILURE"):
                    continue
                # Score by importance
                score = 0
                if m.category == "PUZZLE":
                    score += 3
                elif m.category == "SUCCESS":
                    score += 2
                elif m.category == "FAILURE":
                    score += 1
                text_lower = (m.title + " " + m.text).lower()
                # Boost memories about key game mechanics
                if any(kw in text_lower for kw in ["score", "treasure", "trophy", "ritual", "exorcism",
                                                     "match", "candle", "bell", "book", "prayer",
                                                     "egg", "painting", "torch", "jewel", "coffin",
                                                     "dam", "rainbow", "pot of gold", "scarab",
                                                     "chalice", "trident", "bauble", "diamond"]):
                    score += 3
                if m.persistence == "core":
                    score += 1
                all_mems.append((score, loc_id, m))

        # Sort by importance (highest first)
        all_mems.sort(key=lambda x: -x[0])
        entries = []
        for _score, loc_id, m in all_mems[:max_entries]:
            entries.append(f"- [{m.category}] {m.title} (L{loc_id}): {m.text}")
        return "\n".join(entries) if entries else ""

    def record_action_outcome(self, location_id: int, location_name: str,
                              action: str, response: str, z_context: Dict) -> None:
        """Main entry: synthesize memory after each action."""
        synthesis = self._synthesize(location_id, location_name, action, response, z_context)
        if not synthesis: return

        # Process supersessions
        turn = self.game_state.turn_count
        for old_title in (synthesis.supersedes_memory_titles or set()):
            if synthesis.persistence == "ephemeral":
                old = None
                for m in self.cache.get(location_id, persistent_only=True, include_superseded=True):
                    if m.title == old_title: old = m; break
                if old and old.persistence in ("core", "permanent"):
                    continue  # Don't let ephemeral supersede persistent
            update_memory_status(self.config, location_id, old_title, MemoryStatus.SUPERSEDED,
                               superseded_by=synthesis.memory_title, turn=turn, logger=self.logger)
            self.cache.supersede(location_id, old_title, synthesis.memory_title, turn)

        # Process invalidations
        for title in (synthesis.invalidate_memory_titles or set()):
            update_memory_status(self.config, location_id, title, MemoryStatus.SUPERSEDED,
                               invalidation_reason=synthesis.invalidation_reason, turn=turn, logger=self.logger)
            self.cache.invalidate(location_id, title, synthesis.invalidation_reason, turn)

        if not synthesis.should_remember:
            return

        # Extract episode number
        episode = 1
        import re as _re
        digits = _re.findall(r'\d+', str(self.game_state.episode_id))
        if digits: episode = int(digits[0])

        memory = Memory(
            category=synthesis.category, title=synthesis.memory_title, episode=episode,
            turns=str(turn), score_change=z_context.get('score_delta'),
            text=synthesis.memory_text, persistence=synthesis.persistence,
            status=synthesis.status, goal=synthesis.goal,
        )
        self.add_memory(location_id, location_name, memory)

    def _synthesize(self, loc_id, loc_name, action, response, z_ctx) -> Optional[SynthesisResponse]:
        try:
            existing = self.cache.get(loc_id)
            window = self.config.memory_history_window()
            current_turn = self.game_state.turn_count

            # Format recent actions
            history = self.game_state.action_history[:-1] if self.game_state.action_history else []
            recent = history[-window:] if history else []
            actions_fmt = "\n".join(
                f"Turn {i}: {e.action}\nResponse: {e.response}"
                for i, e in enumerate(recent, max(1, current_turn - len(recent)))
            )

            # Format existing memories
            if existing:
                mem_list = "\n".join(f"  • [{m.category}] {m.title}" for m in existing)
                existing_section = f"\nEXISTING MEMORIES AT THIS LOCATION:\n{mem_list}\n"
            else:
                existing_section = "\nEXISTING MEMORIES: None\n"

            # Goal context
            goal_ctx = ""
            active = [o for o in self.game_state.objectives if o.status == "in_progress"]
            if active:
                goal_ctx = f"GOAL: {active[0].name} — {active[0].text}"

            visit_status = "FIRST VISIT" if z_ctx.get('first_visit') else "RETURN VISIT"
            move_warn = (f"\n⚠️ MOVEMENT TURN: You moved FROM {loc_name}. "
                        f"Only remember events AT {loc_name}.\n") if z_ctx.get('location_changed') else ""

            prompt = f"""# Memory Synthesis Prompt (Condensed)

Location: {loc_name} (ID: {loc_id})
Visit Status: {visit_status}
{move_warn}{existing_section}

---

## DECISION FLOWCHART

```
1. DUPLICATE CHECK
   ├─ Same title exists? → should_remember: false
   ├─ Semantic match? (reveals=provides=shows) → should_remember: false
   ├─ Re-observing same object on return visit? → should_remember: false
   └─ "X is takeable" when CORE discovery exists for X? → should_remember: false

2. NAVIGATION CHECK (MapGraph handles these - DO NOT remember)
   ├─ Exits/directions ("path leads north")
   ├─ Location discovery ("found Forest")
   ├─ Room connections ("forest connects to clearing")
   └─ Movement success ("went north")
   ⚠️ EXCEPTION: If existing memories have PUZZLE/DANGER entries for this
      action and location changed = True, this is a PUZZLE SOLVE, not navigation.
      Record as PUZZLE (e.g., "Chimney - accepts climb with painting + lantern").

3. CONTRADICTION CHECK
   ├─ Contradicts existing memory? → supersede it
   ├─ Reveals delayed consequence? → supersede optimistic memory
   └─ Clarifies TENTATIVE memory? → supersede it

4. IF NEW ACTIONABLE INFO → should_remember: true
   ✓ Object interactions, dangers, puzzle mechanics
   ✓ Item discoveries, score actions, item placement
   ✓ **Dropped items (critical for retrieval)**
```

### Implicit Takeability Rule
If a CORE discovery already notes an item's presence (e.g., "sword at spawn"),
do NOT create a separate "X is takeable" memory. Takeability is assumed unless:
- Taking it grants points (record as SUCCESS with score)
- Taking it fails or has conditions (record as FAILURE)
- Taking it triggers danger (record as DANGER)

### Dropped Item Rule
When agent drops an item, ALWAYS create an ephemeral memory:
- Category: NOTE
- Persistence: ephemeral
- Title format: "Dropped [item] here"
- Text: "[Item] was dropped at [location] for later retrieval"

### PUZZLE Memory Type
Use PUZZLE category for any interaction that reveals how a puzzle works.

**Structure:**
- Title format: "[Object/Mechanism] - [outcome verb]"
- Text: What was tried + result + implication

**Outcome verbs:**
- "requires" → missing prerequisite (failed attempt)
- "accepts" → correct input/action (success)
- "rejects" → wrong input/action (won't work)
- "responds" → partial progress or hint
- "solves" → puzzle completed

**Persistence:** permanent (puzzle mechanics don't change)
**Status:** ACTIVE unless outcome unclear

---

## QUICK REFERENCE TABLES

### Persistence (REQUIRED field)
| Type | When | Examples |
|------|------|----------|
| **core** | First visit ONLY + room description observation | "Sword here" (from room text) |
| **permanent** | Game mechanics, dangers, rules (any visit) | "Troll attacks", "Window can open" |
| **ephemeral** | Agent-caused state changes (any visit) | "Dropped sword here", "Opened window" |

### Status (defaults to ACTIVE)
| Status | When |
|--------|------|
| **ACTIVE** | Outcome certain, fully understood |
| **TENTATIVE** | Success unclear, might have hidden downside |

### Category (REQUIRED for should_remember=true)
SUCCESS | FAILURE | DISCOVERY | DANGER | NOTE | **PUZZLE**

---

## SUPERSESSION RULES

**Allowed:** ephemeral→ephemeral, ephemeral→permanent, permanent→permanent, core→core, core→permanent

**Forbidden:** permanent→ephemeral, core→ephemeral (causes data loss on reset)

⚠️ Copy titles EXACTLY from existing memories. Max 3 titles per array.

---

## CURRENT GOAL
{goal_ctx or "(No active goal)"}

When creating a memory, set the "goal" field to the objective name if this action
was an attempt at the current goal. This links the memory to the goal for future
reference.

When a PUZZLE memory succeeds (category SUCCESS or PUZZLE with "accepts"/"solves"
outcome) AND existing PUZZLE memories at this location share the same goal with
failure outcomes ("requires"/"rejects"), consider superseding the failures —
the successful approach has been found.

## RECENT ACTIONS
{actions_fmt or "(No recent actions available - this is one of the first turns)"}

## CURRENT ACTION
Action: {action}
Response: {response}

State: Score {z_ctx.get('score_delta', 0):+d} | Location changed: {z_ctx.get('location_changed', False)} | Inventory changed: {z_ctx.get('inventory_changed', False)} | Died: {z_ctx.get('died', False)} | First visit: {z_ctx.get('first_visit', False)}

---

## OUTPUT FORMAT

**If NOT remembering:**
```json
{{"reasoning": "brief explanation of why this is not worth remembering", "should_remember": false}}
```

**If remembering:**
```json
{{
  "reasoning": "brief explanation of why this should be remembered and what category/persistence to use",
  "should_remember": true,
  "category": "DISCOVERY",
  "memory_title": "3-6 words evergreen",
  "memory_text": "1-2 sentences actionable",
  "persistence": "permanent",
  "status": "ACTIVE",
  "goal": "Get past the troll",
  "supersedes_memory_titles": []
}}
```

Optional fields: `goal`, `invalidate_memory_titles`, `invalidation_reason`"""

            # Streaming: broadcast chunks to viewer in real-time
            if self.streaming_server:
                self.streaming_server.broadcast_memory_synthesis_start(
                    current_turn, loc_name, action)
                def on_memory_chunk(accumulated):
                    self.streaming_server.broadcast_memory_synthesis_chunk(current_turn, accumulated)
                resp = self.llm_client.chat.completions.create_streaming(
                    model=self.config.memory_model, messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.memory_sampling.get('temperature', 0.3),
                    max_tokens=self.config.memory_sampling.get('max_tokens', 4000),
                    name="MemorySynthesis",
                    enable_thinking=self.config.memory_sampling.get('enable_thinking'),
                    on_chunk=on_memory_chunk,
                )
            else:
                resp = self.llm_client.chat.completions.create(
                    model=self.config.memory_model, messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.memory_sampling.get('temperature', 0.3),
                    max_tokens=self.config.memory_sampling.get('max_tokens', 4000),
                    name="MemorySynthesis",
                    enable_thinking=self.config.memory_sampling.get('enable_thinking'),
                )

            content = resp.content
            if not content and resp.reasoning_content:
                content = resp.reasoning_content

            json_str = extract_json(content)
            # Handle array wrapper: some LLMs return [{...}] instead of {...}
            import json as _json
            try:
                test = _json.loads(json_str)
                if isinstance(test, list) and len(test) > 0:
                    json_str = _json.dumps(test[0])
            except _json.JSONDecodeError:
                pass
            parsed = SynthesisResponse.model_validate_json(json_str)

            # Hallucination caps
            if len(parsed.supersedes_memory_titles) > 3:
                parsed.supersedes_memory_titles = set()
            if len(parsed.invalidate_memory_titles) > 3:
                parsed.invalidate_memory_titles = set()

            # Store for viewer
            self.game_state.memory_synthesis_results[current_turn] = {
                "content": parsed.reasoning,
                "memory_created": parsed.should_remember,
                "memory_title": parsed.memory_title,
                "memory_category": parsed.category,
            }

            # Broadcast completion to viewer
            if self.streaming_server:
                self.streaming_server.broadcast_memory_synthesis_complete(
                    current_turn, parsed.reasoning, parsed.should_remember,
                    parsed.memory_title, parsed.category,
                )

            if not parsed.should_remember:
                return parsed  # Still return for supersession/invalidation processing
            return parsed

        except Exception as e:
            if self.logger: self.logger.error(f"Memory synthesis failed: {e}")
            self.game_state.memory_synthesis_results[self.game_state.turn_count] = {
                "content": "", "memory_created": False, "memory_title": None, "memory_category": None
            }
            return None
