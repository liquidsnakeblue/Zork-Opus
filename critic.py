"""
ZorkCritic - evaluates proposed actions using LLM + Z-machine object tree validation.
Provides confidence scores and failure detection.
"""

import json
import re
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel
from config import Config
from llm_client import LLMClient, extract_json
from game_interface import JerichoInterface


class CriticResponse(BaseModel):
    justification: str
    score: float
    confidence: float = 0.8


class FailureDetection(BaseModel):
    reason: str
    action_failed: bool


class ValidationResult(BaseModel):
    valid: bool
    reason: str
    confidence: float = 0.9


class CriticTrust:
    """Tracks critic performance and adjusts trust."""
    def __init__(self):
        self.trust = 0.8
        self.correct = 0
        self.incorrect = 0
        self.total = 0
        self.recent: List[bool] = []

    def update(self, was_correct: bool):
        self.total += 1
        self.recent.append(was_correct)
        if len(self.recent) > 20:
            self.recent.pop(0)
        if was_correct:
            self.correct += 1
        else:
            self.incorrect += 1
        if len(self.recent) >= 5:
            self.trust = min(0.95, max(0.3, sum(self.recent) / len(self.recent)))


class ZorkCritic:
    def __init__(self, config: Config, client: Optional[LLMClient] = None,
                 logger=None, episode_id: str = "unknown"):
        self.config = config
        self.model = config.critic_model
        self.logger = logger
        self.episode_id = episode_id
        self.trust = CriticTrust()

        s = config.critic_sampling
        self.temperature = s.get("temperature", 0.1)
        self.max_tokens = s.get("max_tokens", 4096)
        self.enable_thinking = s.get("enable_thinking")

        self.client = client or LLMClient(
            config=config, base_url=config.base_url_for("critic"),
            api_key=config.api_key_for("critic"),
        )

        try:
            with open("prompts/critic.md") as f:
                self.system_prompt = f.read()
        except FileNotFoundError:
            try:
                with open("critic.md") as f:
                    self.system_prompt = f.read()
            except FileNotFoundError:
                self.system_prompt = "You are a Zork action critic. Score actions 0-1."

    def evaluate_action(
        self, game_state_text: str, proposed_action: str,
        available_exits: List[str] = None, action_counts: Dict = None,
        current_location_name: str = "", failed_actions_by_location: Dict = None,
        previous_actions_and_responses=None, jericho_interface=None,
        inventory: List[str] = None, agent_reasoning: str = "",
    ) -> CriticResponse:
        """Full LLM-based critic evaluation with object tree pre-check."""
        # Object tree validation first (fast, no LLM)
        if jericho_interface:
            validation = self.validate_against_object_tree(proposed_action, jericho_interface)
            if not validation.valid:
                return CriticResponse(
                    score=0.0, justification=f"[Object Tree] {validation.reason}",
                    confidence=validation.confidence
                )

        # Build context
        context_parts = [f"Location: {current_location_name}"]
        if available_exits:
            context_parts.append(f"Valid exits: {', '.join(available_exits)}")
        if inventory:
            context_parts.append(f"Inventory: {', '.join(inventory)}")
        if failed_actions_by_location and current_location_name in failed_actions_by_location:
            failed = failed_actions_by_location[current_location_name]
            if failed:
                context_parts.append(f"Failed here: {', '.join(failed[-5:])}")

        context = "\n".join(context_parts)
        prompt = (f"{context}\n\nProposed action: {proposed_action}\n"
                  f"Agent reasoning: {agent_reasoning}\n\n"
                  f"Evaluate. Respond JSON: {{\"justification\": \"...\", \"score\": 0.0-1.0, \"confidence\": 0.0-1.0}}")

        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages,
                temperature=self.temperature, max_tokens=self.max_tokens,
                enable_thinking=self.enable_thinking, name="Critic",
            )
            raw = resp.content or ""
            native_reasoning = resp.reasoning_content
            if not raw and native_reasoning:
                raw = native_reasoning

            cleaned = extract_json(raw)
            return CriticResponse.model_validate_json(cleaned)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Critic eval failed: {e}")
            return CriticResponse(score=0.5, justification="[Critic error - defaulting]", confidence=0.3)

    def validate_against_object_tree(self, action: str, jericho: JerichoInterface) -> ValidationResult:
        """Fast Z-machine validation without LLM."""
        action_lower = action.strip().lower()

        # Direction validation
        directions = {"north", "south", "east", "west", "up", "down",
                      "northeast", "northwest", "southeast", "southwest",
                      "n", "s", "e", "w", "u", "d", "ne", "nw", "se", "sw", "in", "out"}
        direction_prefixes = ["go ", "walk ", "run "]

        is_direction = action_lower in directions
        for p in direction_prefixes:
            if action_lower.startswith(p):
                is_direction = True
                break

        if is_direction:
            exits = jericho.get_valid_exits()
            # Normalize action to canonical
            normalize = {"n": "north", "s": "south", "e": "east", "w": "west",
                         "u": "up", "d": "down", "ne": "northeast", "nw": "northwest",
                         "se": "southeast", "sw": "southwest"}
            canonical = action_lower
            for p in direction_prefixes:
                if canonical.startswith(p):
                    canonical = canonical[len(p):].strip()
                    break
            canonical = normalize.get(canonical, canonical)

            if canonical not in exits and exits:
                return ValidationResult(valid=False, reason=f"'{canonical}' not a valid exit. Valid: {exits}")

        # Object interaction validation
        take_match = re.match(r'^(?:take|get|pick up)\s+(.+)$', action_lower)
        if take_match:
            target = take_match.group(1).strip()
            visible = jericho.get_visible_objects()
            inv = jericho.get_inventory_structured()
            visible_names = [o.name.lower() for o in visible]
            inv_names = [o.name.lower() for o in inv]

            if target in inv_names:
                return ValidationResult(valid=False, reason=f"'{target}' is already in inventory")

            # Check children of visible containers too
            all_objs = jericho.get_all_objects()
            for v in visible:
                for child in all_objs:
                    if child.parent == v.num:
                        visible_names.append(child.name.lower())

        return ValidationResult(valid=True, reason="passed")

    def detect_failure(self, action: str, response: str, jericho: JerichoInterface) -> FailureDetection:
        """Detect if an action failed based on game response and Z-machine state."""
        resp_lower = response.lower()
        fail_phrases = [
            "i don't understand", "you can't", "that's not", "what do you want",
            "i beg your pardon", "there's nothing", "you don't see", "that doesn't",
        ]
        for phrase in fail_phrases:
            if phrase in resp_lower:
                return FailureDetection(reason=f"Parser failure: '{phrase}'", action_failed=True)
        return FailureDetection(reason="no failure detected", action_failed=False)

    def update_episode_id(self, eid: str):
        self.episode_id = eid
