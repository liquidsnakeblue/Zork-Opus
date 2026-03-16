"""
Orchestrator - coordinates all components through the game loop.
The original 2000+ line file, consolidated and cleaned.
"""

import re
import time
import json
import logging
import pickle
from collections import deque
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

from config import Config
from state import GameState, ActionEntry
from game_interface import JerichoInterface
from agent import ZorkAgent
from critic import ZorkCritic, CriticResponse
from extractor import Extractor
from memory import MemoryManager, SpawnDetector, Memory, MemoryStatus
from map_manager import MapManager
from pathfinder import Pathfinder
from context_manager import ContextManager
from knowledge import KnowledgeManager
from objectives import ObjectiveManager
from walkthrough import WalkthroughManager
from session import SessionTracker
from prompt_logger import PromptLogger
from logger import setup_logging, setup_episode_logging
from llm_client import LLMClient

try:
    from streaming import StreamServer
    STREAMING_OK = True
except ImportError:
    STREAMING_OK = False
    StreamServer = None

try:
    from web_search import WebSearchManager
    WEB_SEARCH_OK = True
except ImportError:
    WEB_SEARCH_OK = False
    WebSearchManager = None


class Orchestrator:
    """Coordinates all game components through the turn-based game loop."""

    def __init__(self, episode_id: str, max_turns: int = None,
                 session_stats: dict = None, model_overrides: dict = None):
        self.session_stats = session_stats or {}

        # Load config
        self.config = Config.from_toml()
        if max_turns is not None:
            self.config.max_turns_per_episode = max_turns
        if model_overrides:
            for k, v in model_overrides.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)

        # Logger
        self.logger = setup_logging(
            self.config.episode_log_file, self.config.json_log_file, logging.DEBUG)

        # Game state
        self.gs = GameState()
        self.gs.episode_id = episode_id

        # Prompt logger (replaces Langfuse)
        self.prompt_logger = PromptLogger(
            log_dir=self.config.prompt_log_dir,
            max_content=self.config.prompt_log_max_content,
            enabled=self.config.enable_prompt_logger,
        )
        self.prompt_logger.set_episode(episode_id)

        # Episode logging
        self.episode_log = setup_episode_logging(episode_id, self.config.game_workdir)

        # Game interface
        self.jericho = JerichoInterface(self.config.game_file_path, self.logger)

        # Streaming
        self.streaming: Optional[StreamServer] = None
        if STREAMING_OK and self.config.enable_streaming:
            self.streaming = StreamServer(self.config.websocket_host, self.config.websocket_port, self.logger)

        # LLM client (shared)
        self.llm = LLMClient(
            config=self.config, base_url=self.config.base_url_for("agent"),
            api_key=self.config.api_key_for("agent"), logger=self.logger,
            prompt_logger=self.prompt_logger,
        )

        # Core components
        self.agent = ZorkAgent(self.config, client=self.llm, logger=self.logger, episode_id=episode_id)
        self.critic = ZorkCritic(self.config, client=self.llm, logger=self.logger, episode_id=episode_id)
        self.extractor = Extractor(self.jericho, self.config, self.logger, episode_id)

        # Managers
        self.map_mgr = MapManager(self.config, self.gs, self.jericho, self.llm, self.logger)
        self.ctx = ContextManager(self.config, self.gs, self.logger)
        self.memory = MemoryManager(self.config, self.gs, self.llm, self.logger, self.streaming)
        self.spawn_detector = SpawnDetector(self.logger)
        self.walkthrough = WalkthroughManager(self.config, self.gs, self.logger)
        self.pathfinder = Pathfinder(self.config, self.gs, self.map_mgr, self.logger)

        # Web search
        self.web_search: Optional[WebSearchManager] = None
        if WEB_SEARCH_OK and self.config.enable_web_search:
            self.web_search = WebSearchManager(self.config, self.gs, self.logger)

        # State export
        self._s3 = None
        if self.config.s3_bucket:
            try:
                import boto3
                self._s3 = boto3.client("s3")
            except Exception:
                pass

        # Knowledge
        self.knowledge = KnowledgeManager(
            self.config, self.agent, self.map_mgr, self.llm, self.logger)

        # Reasoner client (separate if different base URL from agent)
        reasoner_url = self.config.base_url_for("reasoner")
        agent_url = self.config.base_url_for("agent")
        if reasoner_url != agent_url or self.config.reasoner_model != self.config.agent_model:
            self.reasoner_llm = LLMClient(
                config=self.config, base_url=reasoner_url,
                api_key=self.config.api_key_for("reasoner"),
                logger=self.logger, prompt_logger=self.prompt_logger,
            )
        else:
            self.reasoner_llm = self.llm

        # Objectives
        self.objectives = ObjectiveManager(
            self.config, self.gs, self.knowledge, self.map_mgr, self.memory,
            self.ctx, self.walkthrough, self.reasoner_llm, self.llm,
            self.web_search, self.streaming, self.logger,
        )

        # Inject cross-references
        self.ctx.memory_manager = self.memory
        self.ctx.pathfinder = self.pathfinder

        # Tracking
        self.critic_confidence_history: List[float] = []
        self._action_history: deque = deque(maxlen=30)
        self._location_history: deque = deque(maxlen=20)
        self._last_score_turn = 0
        self._last_tracked_score = 0

        # Rejection state
        self._rejected_this_turn: List[str] = []
        self._trust_level = 0.8

    def play_episode(self) -> int:
        """Play a complete Zork episode. Returns final score."""
        try:
            if self.streaming:
                self.streaming.start()
                if self.session_stats:
                    self.streaming.broadcast_generation_start(
                        self.session_stats.get('generation', 1),
                        self.session_stats.get('high_score', 0),
                        self.session_stats.get('total_deaths', 0),
                        self.session_stats.get('best_generation', 0),
                        self.gs.episode_id,
                    )

            # Initialize episode
            self.gs.reset_episode(self.gs.episode_id)
            self.agent.update_episode_id(self.gs.episode_id)
            self.extractor.update_episode_id(self.gs.episode_id)
            self.critic.update_episode_id(self.gs.episode_id)
            self.memory.reset_episode()
            self.spawn_detector.reset()
            self.knowledge.reset()
            self.pathfinder.reset()

            # Start Jericho
            intro = self.jericho.start()
            self.jericho.send_command("verbose")

            # Capture initial state
            loc = self.jericho.get_location()
            self.gs.initial_response = self.extractor.get_clean_text(intro)
            self.gs.initial_location_id = loc.num if loc else 0
            self.gs.initial_location_name = loc.name if loc else "Unknown"

            # Process initial extraction
            info = self.extractor.extract(intro)
            self._process_extraction(info, "", intro)

            # Generate initial objectives
            self._generate_turn1_objectives()

            # Export initial state
            self._export_state()

            # Game loop
            score = self._game_loop(intro)

            # Finalize
            self._finalize_episode(score)
            self._export_state()
            self.map_mgr.save_map()

            return score

        except Exception as e:
            self.logger.error(f"Episode failed: {e}", exc_info=True)
            return self.gs.previous_score
        finally:
            if self.streaming:
                try: self.streaming.stop()
                except Exception: pass
            try: self.jericho.close()
            except Exception: pass

    def _game_loop(self, initial_state: str) -> int:
        current = initial_state

        while not self.gs.game_over and self.gs.turn_count < self.config.max_turns_per_episode:
            self.gs.turn_count += 1
            self.gs.tool_history = []

            if self.config.turn_delay_seconds > 0:
                time.sleep(self.config.turn_delay_seconds)

            try:
                action, current = self._run_turn(current)
            except Exception as e:
                self.logger.error(f"Turn error, using 'look': {e}")
                action, current = "look", current

            # Progress tracking
            self._track_progress()
            if self.gs.turn_count % self.config.stuck_check_interval == 0:
                stuck = self.gs.turn_count - self._last_score_turn
                if stuck >= self.config.max_turns_stuck:
                    self.logger.warning(f"Terminating: no progress for {stuck} turns")
                    self.gs.game_over = True
                    return self.gs.previous_score

            # Periodic knowledge update
            if self.knowledge.should_update(self.gs.turn_count):
                self.knowledge.update_from_episode(self.gs.episode_id, self.gs.turn_count)

            # Clear pending state and export complete turn
            self.gs.pending_response = None
            self.gs.pending_location_id = None
            self.gs.pending_location_name = None
            self.gs.pending_timestamp = None
            self._export_state(include_pending=False)

        return self.gs.previous_score

    def _run_turn(self, current_state: str) -> Tuple[str, str]:
        """Execute a single turn: agent → critic → execute → extract → memory."""
        span = self.prompt_logger.log_span_start(
            f"turn-{self.gs.turn_count}", turn=self.gs.turn_count,
            metadata={"score": self.gs.previous_score, "location": self.gs.current_room_name}
        )

        # Build context
        info = self.extractor.extract(current_state)
        ctx = self.ctx.build_agent_context(current_state, info, self.map_mgr)
        formatted = self.ctx.format_prompt(ctx, current_state)

        # Add stuck warning
        stuck_warning = self._build_stuck_warning()
        if stuck_warning:
            formatted = stuck_warning + "\n" + formatted

        # Store for handler re-generation
        self._turn_ctx = ctx
        self._turn_state = current_state

        # Export pending "thinking" card for progressive rendering
        # Shows the game response with action/reasoning = null (viewer shows animation)
        try:
            ah = self.gs.action_history
            if len(ah) == 0:
                incoming = self.gs.initial_response
                prev_action = None
                prev_loc_id = None
                prev_loc_name = None
            else:
                prev = ah[-1]
                incoming = prev.response
                prev_action = prev.action
                prev_loc_id = prev.location_id
                prev_loc_name = prev.location_name
            # Current location is where the agent IS now (authoritative from game state)
            self.gs.pending_response = incoming
            self.gs.pending_location_id = self.gs.current_room_id
            self.gs.pending_location_name = self.gs.current_room_name
            self.gs.pending_prev_location_id = prev_loc_id
            self.gs.pending_prev_location_name = prev_loc_name
            self.gs.pending_prev_action = prev_action
            self.gs.pending_timestamp = datetime.now().isoformat()
            self._export_state(include_pending=True)
        except Exception:
            pass

        # Broadcast turn start
        if self.streaming:
            self.streaming.broadcast_turn_start(self.gs.turn_count, self.gs.current_room_name, self.gs.previous_score)

        # Get action from agent (streaming if available)
        if self.streaming:
            def on_chunk(reasoning, action=None):
                self.streaming.broadcast_reasoning_chunk(self.gs.turn_count, reasoning, action)
            result = self.agent.get_action_streaming(
                game_state_text="", relevant_memories=formatted, on_chunk=on_chunk)
        else:
            result = self.agent.get_action(game_state_text="", relevant_memories=formatted)

        action = result["action"]
        reasoning = result.get("reasoning", "")

        # Handle special action patterns
        action, reasoning = self._handle_pathfinder(action, reasoning, formatted)
        action, reasoning, obj_id, obj_text = self._handle_objective(action, reasoning, formatted)
        action, reasoning = self._handle_search(action, reasoning, formatted)

        # Critic evaluation (if enabled)
        if self.config.enable_critic:
            action, _ = self._critic_loop(current_state, action, ctx, formatted, reasoning)
        else:
            # No critic, no blocking — let Zork's parser handle invalid actions naturally.
            # The agent sees the game's response ("You can't go that way") and adapts.
            pass

        # ── Phase 0: Capture state BEFORE action ──
        # Memories stored at SOURCE location (where action taken), not destination
        pre_state = self.jericho.save_state()
        score_before, _ = self.jericho.get_score()
        loc_before = self.jericho.get_location()
        loc_id_before = loc_before.num if loc_before else 0
        loc_name_before = loc_before.name if loc_before else "Unknown"
        inv_before = self.jericho.get_inventory_structured()
        inv_names_before = set(o.name for o in inv_before)

        # ── Execute action ──
        response = self.jericho.send_command(action)

        # ── Sync state from Z-machine (authoritative source of truth) ──
        score_after, _ = self.jericho.get_score()
        loc_after = self.jericho.get_location()
        loc_id_after = loc_after.num if loc_after else 0
        loc_name_after = loc_after.name if loc_after else "Unknown"
        inv_after = self.jericho.get_inventory_structured()
        inv_names_after = set(o.name for o in inv_after)

        # Update game state from Z-machine
        self.gs.previous_score = score_after
        self.gs.current_room_id = loc_id_after
        self.gs.current_room_name = loc_name_after
        self.gs.current_inventory = [o.name for o in inv_after]

        # Track action history
        self._action_history.append(action.lower())

        # Add to action history (uses source location — where action was taken)
        self.ctx.add_action_to_history(action, response, loc_id_before, loc_name_before)

        # Record reasoning
        self.gs.reasoning_history.append({
            "turn": self.gs.turn_count, "reasoning": reasoning,
            "action": action, "timestamp": datetime.now().isoformat(),
            "objective_selection": {"id": obj_id, "text": obj_text} if obj_id else None,
        })

        # Room description tracking
        if response and len(response) > 50:
            self.gs.last_room_description = response
            self.gs.last_room_description_turn = self.gs.turn_count
            self.gs.last_room_description_location_id = loc_id_after

        # ── Movement detection and map update ──
        moved = loc_id_before != loc_id_after
        self.gs.last_action_moved = moved

        # Skip map connection recording on death (respawn teleports are NOT real edges)
        if moved and not self.gs.game_over:
            self.map_mgr.update_from_movement(
                action, loc_id_after, loc_name_after,
                loc_id_before, loc_name_before, response, pre_state)

            # Pathfinder: validate waypoint before advancing
            if self.pathfinder.is_active:
                nav = self.pathfinder.nav
                if loc_id_after == nav.target_id:
                    self.pathfinder.cancel()  # Reached target
                elif nav.step + 1 < len(nav.waypoints):
                    expected = nav.waypoints[nav.step + 1]
                    if loc_id_after == expected.get("room_id"):
                        self.pathfinder.advance()
                    else:
                        # Off-path — cancel navigation
                        self.gs.navigation_failure_msg = (
                            f"NAVIGATION OFF-PATH: Expected {expected.get('room_name')}, "
                            f"arrived elsewhere. Navigation cancelled."
                        )
                        self.pathfinder.cancel()
                else:
                    self.pathfinder.advance()
        elif not moved:
            from map_graph import is_non_movement, DIRECTION_MAP
            if not is_non_movement(action) and action.lower().strip() in DIRECTION_MAP:
                self.map_mgr.track_failed_action(action, loc_id_before, loc_name_before)

        # ── Game over check ──
        game_over, death_reason = self.jericho.is_game_over(response)
        if game_over:
            self.gs.game_over = True
            if death_reason and "death" in death_reason.lower() and not self.gs.death_counted_this_episode:
                self.gs.death_count += 1
                self.gs.death_counted_this_episode = True

        # ── Track deltas for context display ──
        score_delta = score_after - score_before
        inv_changed = inv_names_before != inv_names_after
        self.gs.last_score_delta = score_delta
        self.gs.last_items_gained = sorted(inv_names_after - inv_names_before)
        self.gs.last_items_lost = sorted(inv_names_before - inv_names_after)
        if score_delta > 0:
            self.gs.last_scoring_turn = self.gs.turn_count
        self.gs.update_item_locations()
        # first_visit: check memory cache (cross-episode), not visited_locations (episode-scoped)
        first_visit = loc_id_after not in self.memory.cache.persistent

        z_ctx = {
            "score_delta": score_delta,
            "location_changed": moved,
            "inventory_changed": inv_changed,
            "died": game_over and death_reason and "death" in death_reason.lower(),
            "first_visit": first_visit,
        }
        self.gs.visited_locations.add(str(loc_id_after))

        # ── Programmatic drop/take detection (bypasses LLM for reliability) ──
        items_dropped = inv_names_before - inv_names_after
        items_taken = inv_names_after - inv_names_before

        if items_dropped and loc_id_before:
            episode_num = 1
            digits = re.findall(r'\d+', str(self.gs.episode_id))
            if digits: episode_num = int(digits[0])
            for item in items_dropped:
                drop_mem = Memory(
                    category="NOTE", title=f"Dropped {item} here",
                    episode=episode_num, turns=str(self.gs.turn_count), score_change=0,
                    text=f"Agent dropped {item} at this location for later retrieval.",
                    persistence="ephemeral", status=MemoryStatus.ACTIVE,
                )
                self.memory.add_memory(loc_id_before, loc_name_before, drop_mem)

        # ── Spawn item detection (runs every location, not just first visit) ──
        if loc_id_after:
            self._detect_spawn_items()

        # ── Memory synthesis (uses SOURCE location) ──
        self.memory.record_action_outcome(
            loc_id_before, loc_name_before, action, response, z_ctx)

        # ── Object event tracking ──
        self.knowledge.detect_object_events(
            list(inv_names_before), list(inv_names_after), self.jericho, action, self.gs.turn_count)

        # ── Objective completion check ──
        if self.gs.turn_count % self.config.completion_check_interval == 0:
            self.objectives.check_completions(action, response)

        # ── Reasoner update (periodic + ensure objectives exist) ──
        if self.objectives.should_run_reasoner():
            self.objectives.run_reasoner(response)
        elif not self.gs.active_objectives and self.gs.turn_count > 0:
            # All objectives completed/abandoned — trigger reasoner for new ones
            # (bypasses interval check but still respects failure cooldown)
            if (self.gs.turn_count - self.objectives._reasoner_fail_turn) >= 5:
                self.objectives.run_reasoner(response)

        # Log turn completion
        self.logger.info(
            f"Turn {self.gs.turn_count}: '{action}'",
            extra={
                "event_type": "turn_completed", "turn": self.gs.turn_count,
                "action": action, "score": self.gs.previous_score,
                "location": self.gs.current_room_name,
                "confidence": 0.0,
            })

        # Broadcast turn complete
        if self.streaming:
            self.streaming.broadcast_turn_complete(
                self.gs.turn_count, action, self.gs.previous_score, self.gs.current_room_name)

        self.prompt_logger.log_span_end(span, {"action": action, "score": self.gs.previous_score})

        return action, response

    def _process_extraction(self, info, action: str, response: str):
        """Update game state from extracted info."""
        if not info: return
        if info.location_name:
            self.gs.current_room_name = info.location_name
        loc = self.jericho.get_location()
        if loc:
            self.gs.current_room_id = loc.num
            if not self.map_mgr.game_map.rooms:
                self.map_mgr.add_initial_room(loc.num, loc.name)
        self.gs.current_inventory = info.inventory
        if info.score is not None:
            self.gs.previous_score = info.score

        # Update room description tracking
        if response and len(response) > 50 and self.gs.current_room_id:
            self.gs.last_room_description = response
            self.gs.last_room_description_turn = self.gs.turn_count
            self.gs.last_room_description_location_id = self.gs.current_room_id

    def _detect_spawn_items(self):
        items = self.spawn_detector.detect(self.jericho, self.gs.current_room_id, self.gs.current_room_name)
        new_items = self.spawn_detector.filter_new(items)
        if not new_items: return

        episode = 1
        digits = re.findall(r'\d+', str(self.gs.episode_id))
        if digits: episode = int(digits[0])

        memories = self.spawn_detector.create_memories(new_items, episode, self.gs.turn_count)
        for mem in memories:
            self.memory.add_memory(self.gs.current_room_id, self.gs.current_room_name, mem)
        self.spawn_detector.mark_memorized(new_items)

    def _generate_turn1_objectives(self):
        """Generate initial objectives before game loop (only if resuming from prior session)."""
        # Check for cross-episode data (memories from PREVIOUS episodes, not current init)
        # A fresh start has at most 1 room (the starting room added during init)
        # and only spawn-detection "core" memories. Prior sessions have many more.
        has_prior_memories = self.memory.cache.total_persistent > 1
        has_prior_map = len(self.map_mgr.game_map.rooms) > 1
        if has_prior_memories or has_prior_map:
            self.logger.info("Resuming session — generating initial objectives")
            self.objectives.run_reasoner("")
        else:
            self.logger.info("Fresh run — skipping initial objectives")

    # ── Handler chain ──

    def _handle_pathfinder(self, action, reasoning, context) -> Tuple[str, str]:
        m = re.match(r'^[Pp]athfinder:\s*(\d+)$', action.strip())
        if not m: return action, reasoning

        target_id = int(m.group(1))
        room_name = self.map_mgr.game_map.room_names.get(target_id, f"Room#{target_id}")

        # Check if this target recently failed — warn the agent immediately
        if self.pathfinder.recently_failed(target_id, window=15):
            pf_ctx = (f"\n=== PATHFINDER BLOCKED ===\nTarget: {room_name} (L{target_id})\n"
                     f"⛔ This target ALREADY FAILED recently. NO PATH EXISTS from your current area.\n"
                     f"STOP retrying Pathfinder to this room. Instead:\n"
                     f"1. Explore DIFFERENT unexplored exits to discover new connections\n"
                     f"2. Select a DIFFERENT objective with Objective: <id>\n"
                     f"3. Look for alternate routes (chimneys, gates, passages)\n")
            agent_result = self.agent.get_action("", context + pf_ctx)
            return agent_result["action"], agent_result.get("reasoning", "")

        self.gs.active_tool = "pathfinder"
        self.gs.active_tool_data = {"target_room_id": target_id, "target_name": room_name}
        if self.streaming: self.streaming.broadcast_tool_status(self.gs.active_tool, self.gs.active_tool_data)

        result = self.pathfinder.find_path(target_id, room_name)
        if result and result["found"] and result["directions"]:
            dirs = result["directions"]
            steps = "\n".join(f"  {'>>>' if i==0 else '   '} Step {i+1}: {d}" for i, d in enumerate(dirs))
            pf_ctx = (f"\n=== PATHFINDER RESULT ===\nTarget: {room_name} (L{target_id})\n"
                     f"PATH FOUND ({len(dirs)} steps):\n{steps}\n"
                     f'>>> THIS TURN: Execute "{dirs[0]}" <<<\n')
            self.pathfinder.start_navigation(target_id, room_name)
        else:
            reason = result["reason"] if result else "Pathfinder error"
            pf_ctx = (f"\n=== PATHFINDER RESULT ===\nTarget: {room_name}\n"
                     f"NO PATH: {reason}\n"
                     f"⚠️ Do NOT retry Pathfinder to this room — it will fail again.\n"
                     f"ACTION REQUIRED: Explore NEW unexplored exits to discover connections "
                     f"toward your target. Pick a direction you haven't tried.\n")

        self.gs.tool_history.append({"tool": "pathfinder", "target": room_name, "success": bool(result and result["found"])})
        self.gs.active_tool = None
        if self.streaming: self.streaming.broadcast_tool_status()

        agent_result = self.agent.get_action("", context + pf_ctx)
        return agent_result["action"], agent_result.get("reasoning", "")

    def _handle_objective(self, action, reasoning, context):
        m = re.match(r'^[Oo]bjective:\s*([A-Za-z]\d+)$', action.strip())
        if not m: return action, reasoning, None, None

        obj_id = m.group(1).upper()
        obj = self.gs.get_objective(obj_id)
        obj_text = obj.text if obj else obj_id

        self.gs.active_tool = "objective"
        self.gs.active_tool_data = {"objective_id": obj_id, "objective_text": obj_text}
        if self.streaming: self.streaming.broadcast_tool_status(self.gs.active_tool, self.gs.active_tool_data)

        success = self.objectives.mark_in_progress(obj_id)
        status = "Now IN PROGRESS" if success else "NOT FOUND"
        obj_ctx = f"\n=== OBJECTIVE SELECTED ===\n{obj_id} - {obj_text}\nStatus: {status}\nNow provide your game command.\n"

        self.gs.tool_history.append({"tool": "objective", "objective_id": obj_id, "status": "in_progress" if success else "not_found"})

        # Regenerate context with updated objective status
        ctx = self.ctx.build_agent_context(self._turn_state, None, self.map_mgr)
        updated = self.ctx.format_prompt(ctx, self._turn_state) + obj_ctx
        stuck = self._build_stuck_warning()
        if stuck: updated = stuck + "\n" + updated

        if self.streaming:
            def on_chunk(r, a=None):
                self.streaming.broadcast_objective_followup_chunk(self.gs.turn_count, r, a, obj_id, obj_text)
            result = self.agent.get_action_streaming("", updated, on_chunk)
        else:
            result = self.agent.get_action("", updated)

        self.gs.active_tool = None
        if self.streaming: self.streaming.broadcast_tool_status()
        return result["action"], result.get("reasoning", ""), obj_id, obj_text

    def _handle_search(self, action, reasoning, context) -> Tuple[str, str]:
        MAX_ITERS = 5
        current_action, current_reasoning = action, reasoning
        accumulated = context

        for iteration in range(MAX_ITERS):
            search_m = re.match(r'^[Ss]earch:\s*(.+)$', current_action.strip())
            crawl_m = re.match(r'^[Cc]rawl:\s*(.+?)\s*\|\s*(.+)$', current_action.strip())

            if not search_m and not crawl_m:
                return current_action, current_reasoning
            if not self.web_search:
                # Web search is disabled — tell the agent and get a new action
                no_search_ctx = ("\n=== WEB SEARCH UNAVAILABLE ===\n"
                                "Web search is disabled. You must solve puzzles using "
                                "observation, experimentation, and your memory. "
                                "Try a different approach or select a different objective.\n")
                accumulated += no_search_ctx
                result = self.agent.get_action("", accumulated)
                current_action = result["action"]
                current_reasoning = result.get("reasoning", "")
                continue  # Check if the new action is also a search

            if search_m:
                query = search_m.group(1).strip()
                results = self.web_search.search(query)
                if results:
                    lines = [f"\n=== SEARCH RESULTS for '{query}' ==="]
                    for i, r in enumerate(results, 1):
                        lines.extend([f"Result {i}: {r['title']}", f"  URL: {r['url']}",
                                     f"  Snippet: {r['snippet']}", ""])
                    lines.append("=== END ===")
                    result_ctx = "\n".join(lines)
                else:
                    result_ctx = f"\n=== SEARCH: No results for '{query}' ===\n"
            elif crawl_m:
                urls, question = crawl_m.group(1).strip(), crawl_m.group(2).strip()
                answer = self.web_search.crawl_and_ask(urls, question)
                result_ctx = (f"\n=== CRAWL RESULT ===\n{answer or 'No answer'}\n=== END ===\n")

            accumulated += result_ctx
            result = self.agent.get_action("", accumulated)
            current_action, current_reasoning = result["action"], result.get("reasoning", "")

        return current_action, current_reasoning

    def _critic_loop(self, state, action, ctx, formatted, reasoning) -> Tuple[str, float]:
        """Critic evaluation with rejection loop."""
        self._rejected_this_turn = []

        for attempt in range(3):
            result = self.critic.evaluate_action(
                state, action,
                available_exits=self.jericho.get_valid_exits(),
                action_counts=self.gs.action_counts,
                current_location_name=self.gs.current_room_name,
                failed_actions_by_location=self.gs.failed_actions_by_location,
                previous_actions_and_responses=self.gs.action_history[-3:],
                jericho_interface=self.jericho,
                inventory=self.gs.current_inventory,
                agent_reasoning=reasoning,
            )

            threshold = self.config.critic_rejection_threshold * self._trust_level
            if result.score >= threshold:
                break

            # Reject and get new action
            self._rejected_this_turn.append(action)
            feedback = f"\n[Action '{action}' rejected: {result.justification}]"
            new_result = self.agent.get_action(state + feedback, formatted)
            action = new_result["action"]
            reasoning = new_result.get("reasoning", "")

        self.critic_confidence_history.append(result.confidence)
        return action, result.score

    # ── Progress / stuck detection ──

    def _track_progress(self):
        if not hasattr(self, '_progress_initialized'):
            self._last_score_turn = self.gs.turn_count
            self._last_tracked_score = self.gs.previous_score
            self._progress_initialized = True
            return

        if self.gs.previous_score != self._last_tracked_score:
            # Real score increase — full reset
            self._last_score_turn = self.gs.turn_count
            self._last_tracked_score = self.gs.previous_score
        elif self.config.enable_objective_based_progress and self.gs.completed_objectives_list:
            last_turn = max(c["completed_turn"] for c in self.gs.completed_objectives_list)
            if (self.gs.turn_count - last_turn) <= 1:
                # Objective completion — partial reprieve (3 turns), not full reset
                # This prevents exploration objectives from indefinitely masking score stagnation
                stuck_so_far = self.gs.turn_count - self._last_score_turn
                if stuck_so_far > 3:
                    self._last_score_turn = max(self._last_score_turn, self.gs.turn_count - (stuck_so_far - 3))

    def _build_stuck_warning(self) -> str:
        if not self.config.enable_stuck_warnings: return ""
        stuck = self.gs.turn_count - self._last_score_turn
        if stuck < self.config.stuck_warning_threshold: return ""
        remaining = self.config.max_turns_stuck - stuck
        urgency = "🚨 CRITICAL" if remaining <= 5 else "⚠️ WARNING" if remaining <= 10 else "⚠️ STAGNATION"
        advice = ""
        if remaining <= 15:
            advice = ("\nSTRATEGY TO SCORE:\n"
                     "• If carrying treasures: find ANY route to Living Room trophy case\n"
                     "• Explore exits you haven't tried — look for chimneys, gates, passages\n"
                     "• Try 'up' or 'climb' in unusual locations\n"
                     "• Switch to a completely different area of the map\n"
                     "• Select a different objective with Objective: <id>\n")
        return (f"{'='*60}\n{urgency}\nNo progress for {stuck} turns. "
                f"EPISODE ENDS in {remaining} turns without score increase.{advice}\n{'='*60}")

    # ── State export ──

    def _export_state(self, include_pending: bool = False):
        if not self.config.enable_state_export: return
        try:
            high = max(self.session_stats.get('high_score', 0), self.gs.previous_score)
            recent_log = self._build_recent_log()

            # Add pending entry for progressive rendering if requested
            if include_pending and self.gs.pending_response:
                prev_loc = getattr(self.gs, 'pending_prev_location_id', None)
                cur_loc = self.gs.pending_location_id
                recent_log.append({
                    "turn": self.gs.turn_count, "action": None,
                    "zork_response": self.gs.pending_response,
                    "reasoning": None,
                    "location_id": self.gs.pending_location_id,
                    "location_name": self.gs.pending_location_name,
                    "prev_location_id": prev_loc,
                    "prev_location_name": getattr(self.gs, 'pending_prev_location_name', None),
                    "prev_action": getattr(self.gs, 'pending_prev_action', None),
                    "moved": prev_loc is not None and prev_loc != cur_loc,
                    "status": "pending_ai",
                    "timestamp": self.gs.pending_timestamp,
                })

            data = {
                "metadata": {
                    "episode_id": self.gs.episode_id,
                    "timestamp": datetime.now().isoformat(),
                    "turn_count": self.gs.turn_count,
                    "game_over": self.gs.game_over,
                    "score": self.gs.previous_score,
                    "max_turns": self.config.max_turns_per_episode,
                    "models": {"agent": self.config.agent_model, "critic": self.config.critic_model},
                    "generation": self.session_stats.get('generation', 1),
                    "high_score": high,
                    "best_generation": self.session_stats.get('best_generation', 0),
                    "total_deaths": self.session_stats.get('total_deaths', 0) + self.gs.death_count,
                },
                "current_state": {
                    "location": self.gs.current_room_name,
                    "inventory": self.gs.current_inventory,
                    "in_combat": False,
                    "death_count": self.gs.death_count,
                    "objectives": [o.model_dump() for o in self.gs.objectives],
                    "objective_update_turn": self.gs.objective_update_turn,
                    "discovered_objectives": self.gs.discovered_objective_texts,
                    "completed_objectives": self.gs.completed_objectives_list,
                },
                "recent_log": recent_log,
                "tool_status": {"active": self.gs.active_tool, "data": self.gs.active_tool_data,
                                "history": self.gs.tool_history},
                "reasoner_events": self.gs.reasoner_events,
                "map": self.map_mgr.get_export_data(),
                "knowledge_base": self.knowledge.get_export_data(),
                "navigation": {
                    "navigation_active": self.pathfinder.is_active,
                    "target": self.pathfinder.nav.target_name if self.pathfinder.nav else None,
                    "path": self.pathfinder.nav.path if self.pathfinder.nav else None,
                    "step": self.pathfinder.nav.step if self.pathfinder.nav else None,
                },
                "performance": {
                    "memory_entries": len(self.gs.memory_log_history),
                },
                "context_management": {
                    "memory_entries": len(self.gs.memory_log_history),
                },
            }
            with open(self.config.state_export_file, "w") as f:
                json.dump(data, f, indent=2)

            if self._s3 and self.config.s3_bucket:
                content = json.dumps(data, indent=2)
                self._s3.put_object(
                    Bucket=self.config.s3_bucket,
                    Key=f"{self.config.s3_key_prefix}current_state.json",
                    Body=content, ContentType="application/json",
                )
        except Exception as e:
            self.logger.error(f"State export failed: {e}")

    def _build_recent_log(self) -> List[Dict]:
        log = []
        actions = self.gs.action_history
        reasonings = self.gs.reasoning_history

        for i, entry in enumerate(actions):
            # Game response: what the agent sees upon arrival (result of previous action)
            if i == 0:
                incoming = self.gs.initial_response
            else:
                incoming = actions[i - 1].response

            # Current location: where the agent IS this turn (where entry.action is taken)
            cur_loc = entry.location_id
            cur_name = entry.location_name

            # Previous location + action that brought us here
            if i == 0:
                prev_loc = None
                prev_name = None
                prev_action = None
                moved = False
            else:
                prev_loc = actions[i - 1].location_id
                prev_name = actions[i - 1].location_name
                prev_action = actions[i - 1].action
                moved = prev_loc != cur_loc

            item = {
                "turn": i + 1, "action": entry.action,
                "zork_response": incoming,
                "location_id": cur_loc, "location_name": cur_name,
                "prev_location_id": prev_loc, "prev_location_name": prev_name,
                "prev_action": prev_action, "moved": moved,
                "reasoning": reasonings[i].get("reasoning", "") if i < len(reasonings) else "",
                "status": "complete",
            }

            # Objective selection data from reasoning history
            if i < len(reasonings) and reasonings[i].get("objective_selection"):
                obj_sel = reasonings[i]["objective_selection"]
                if obj_sel.get("id"):
                    item["objective_selection"] = {
                        "objective_id": obj_sel["id"],
                        "objective_text": obj_sel.get("text", ""),
                        "followup_reasoning": reasonings[i].get("followup_reasoning", ""),
                    }

            if i < len(self.gs.critic_history):
                item.update(self.gs.critic_history[i])
            if (i + 1) in self.gs.memory_synthesis_results:
                item["memory_synthesis"] = self.gs.memory_synthesis_results[i + 1]
            if (i + 1) in self.gs.objective_review_results:
                item["objective_review"] = self.gs.objective_review_results[i + 1]
            log.append(item)

        return log

    def _finalize_episode(self, score: int):
        """End-of-episode synthesis."""
        self.logger.info(f"Finalizing episode {self.gs.episode_id}, score={score}")

        # Knowledge update
        self.knowledge.update_from_episode(self.gs.episode_id, self.gs.turn_count, is_final=True)

        # Cross-episode synthesis
        if self.config.enable_inter_episode_synthesis:
            self.knowledge.synthesize_cross_episode({
                "episode_id": self.gs.episode_id,
                "final_score": score,
                "turn_count": self.gs.turn_count,
                "episode_ended_in_death": self.gs.death_counted_this_episode,
                "completed_objectives": [c["objective"] for c in self.gs.completed_objectives_list],
            })

        # Walkthrough generation
        if self.config.enable_walkthrough_generation:
            try:
                self.walkthrough.generate()
            except Exception as e:
                self.logger.error(f"Walkthrough generation failed: {e}")

    def get_status(self) -> Dict:
        return {
            "episode": self.gs.episode_id, "turn": self.gs.turn_count,
            "score": self.gs.previous_score, "location": self.gs.current_room_name,
            "game_over": self.gs.game_over, "deaths": self.gs.death_count,
            "objectives": len(self.gs.active_objectives),
            "map_rooms": len(self.map_mgr.game_map.rooms),
        }
