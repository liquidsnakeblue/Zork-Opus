"""
ZorkAgent - generates game actions from LLM with structured JSON output.
Supports both standard and native-thinking model modes.
"""

import re
import json
import time
from typing import Optional, Dict
from pathlib import Path
from pydantic import BaseModel, Field
from config import Config
from llm_client import LLMClient, extract_json, REASONING_MODEL_MARKERS


class AgentResponse(BaseModel):
    thinking: str = Field(description="Reasoning about what to do")
    action: str = Field(description="Game command to execute")


class AgentActionOnly(BaseModel):
    action: str = Field(description="Game command to execute")


class ZorkAgent:
    def __init__(self, config: Config, client: Optional[LLMClient] = None,
                 logger=None, episode_id: str = "unknown"):
        self.config = config
        self.model = config.agent_model
        self.logger = logger
        self.episode_id = episode_id

        # Sampling params
        s = config.agent_sampling
        self.temperature = s.get("temperature")
        self.top_p = s.get("top_p")
        self.top_k = s.get("top_k")
        self.min_p = s.get("min_p")
        self.max_tokens = s.get("max_tokens")
        self.presence_penalty = s.get("presence_penalty")
        self.repetition_penalty = s.get("repetition_penalty")
        self.enable_thinking = s.get("enable_thinking")

        # Thinking mode
        self.enable_thinking_mode = s.get("enable_thinking_mode", False)
        self.is_thinking_model = any(m in self.model.lower() for m in REASONING_MODEL_MARKERS)
        self.use_native_thinking = self.enable_thinking_mode

        # LLM client
        self.client = client or LLMClient(
            config=config, base_url=config.base_url_for("agent"),
            api_key=config.api_key_for("agent"),
        )

        # Load system prompt
        self.system_prompt = self._load_prompt()

    def _load_prompt(self) -> str:
        try:
            with open("prompts/agent.md") as f:
                base = f.read()
        except FileNotFoundError:
            try:
                with open("agent.md") as f:
                    base = f.read()
            except FileNotFoundError:
                return "You are a Zork I player. Output JSON with thinking and action fields."

        return self._enhance_with_knowledge(base)

    def _enhance_with_knowledge(self, base: str) -> str:
        kb_path = Path(self.config.game_workdir) / self.config.knowledge_file
        if not kb_path.exists():
            return base

        try:
            content = kb_path.read_text(encoding="utf-8")
            # Strip map section (provided dynamically)
            content = re.sub(r"## CURRENT WORLD MAP\s*\n\s*```mermaid\s*\n.*?\n```", "", content, flags=re.DOTALL).strip()
            if not content:
                return base

            section = f"\n\n**STRATEGIC GUIDE FROM PREVIOUS EPISODES:**\n\n{content}\n\n**END OF STRATEGIC GUIDE**\n\n"
            if "**Output Format" in base:
                idx = base.find("**Output Format")
                return base[:idx] + section + base[idx:]
            return base + section
        except Exception:
            return base

    def get_action(self, game_state_text: str, relevant_memories: Optional[str] = None) -> Dict[str, str]:
        """Get action from agent with reasoning."""
        messages = self._build_messages(game_state_text, relevant_memories)

        try:
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages,
                temperature=self.temperature, top_p=self.top_p, top_k=self.top_k,
                min_p=self.min_p, max_tokens=self.max_tokens,
                presence_penalty=self.presence_penalty,
                repetition_penalty=self.repetition_penalty,
                enable_thinking=self.enable_thinking, name="Agent",
            )

            raw = resp.content.strip() if resp.content else ""
            native_reasoning = resp.reasoning_content
            if not raw and native_reasoning:
                raw = native_reasoning

            cleaned = extract_json(raw)
            try:
                if self.use_native_thinking:
                    parsed = AgentActionOnly.model_validate_json(cleaned)
                    return {"action": self._clean(parsed.action), "reasoning": native_reasoning, "raw": raw}
                else:
                    parsed = AgentResponse.model_validate_json(cleaned)
                    return {"action": self._clean(parsed.action), "reasoning": parsed.thinking, "raw": raw}
            except Exception:
                return {"action": "look", "reasoning": native_reasoning or "[parse error]", "raw": raw}

        except Exception as e:
            if self.logger:
                self.logger.error(f"Agent error: {e}")
            return {"action": "look", "reasoning": None, "raw": None}

    def get_action_streaming(self, game_state_text: str, relevant_memories: Optional[str] = None,
                             on_chunk=None) -> Dict[str, str]:
        """Streaming variant of get_action."""
        messages = self._build_messages(game_state_text, relevant_memories)

        last_reasoning_len = 0
        last_action_len = 0

        def handle_chunk(content: str):
            nonlocal last_reasoning_len, last_action_len
            stripped = content.strip()
            if stripped.startswith('```'):
                fence_end = stripped.find('\n')
                if fence_end != -1: stripped = stripped[fence_end + 1:]
                if stripped.rstrip().endswith('```'): stripped = stripped.rstrip()[:-3]
                stripped = stripped.strip()

            is_json = stripped.startswith('{')
            if is_json:
                fields = self._extract_partial_fields(stripped)
                r, a = fields.get("thinking"), fields.get("action")
            elif self.use_native_thinking or self.is_thinking_model:
                r, a = content, None
            else:
                return

            r_changed = r and len(r) > last_reasoning_len
            a_changed = a and len(a) > last_action_len
            if r_changed or a_changed:
                if r: last_reasoning_len = len(r)
                if a: last_action_len = len(a)
                if on_chunk:
                    on_chunk(r, a)

        # Retry with backoff on connection/streaming errors.
        # The game pauses here until the LLM responds — no wasted turns on "look".
        max_retries = 10
        base_delay = 5.0
        max_delay = 120.0

        for attempt in range(max_retries + 1):
            try:
                resp = self.client.chat.completions.create_streaming(
                    model=self.model, messages=messages,
                    temperature=self.temperature, top_p=self.top_p, top_k=self.top_k,
                    min_p=self.min_p, max_tokens=self.max_tokens,
                    presence_penalty=self.presence_penalty,
                    repetition_penalty=self.repetition_penalty,
                    enable_thinking=self.enable_thinking, name="Agent-Streaming",
                    on_chunk=handle_chunk,
                )

                raw = resp.content.strip() if resp.content else ""
                native_reasoning = resp.reasoning_content
                if not raw and native_reasoning:
                    raw = native_reasoning

                cleaned = extract_json(raw)
                try:
                    if self.use_native_thinking:
                        parsed = AgentActionOnly.model_validate_json(cleaned)
                        return {"action": self._clean(parsed.action), "reasoning": native_reasoning, "raw": raw}
                    else:
                        parsed = AgentResponse.model_validate_json(cleaned)
                        return {"action": self._clean(parsed.action), "reasoning": parsed.thinking, "raw": raw}
                except Exception:
                    return {"action": "look", "reasoning": native_reasoning or "[parse error]", "raw": raw}

            except Exception as e:
                if attempt < max_retries:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    if self.logger:
                        self.logger.warning(
                            f"Agent streaming error (attempt {attempt + 1}/{max_retries + 1}), "
                            f"retrying in {delay:.0f}s: {e}")
                    time.sleep(delay)
                else:
                    if self.logger:
                        self.logger.error(
                            f"Agent streaming failed after {max_retries + 1} attempts: {e}")
                    return {"action": "look", "reasoning": None, "raw": None}

    def _build_messages(self, game_state_text: str, relevant_memories: Optional[str]) -> list:
        if "o1" in self.model:
            messages = [{"role": "user", "content": self.system_prompt, "cache_control": {"type": "ephemeral"}}]
        else:
            messages = [{"role": "system", "content": self.system_prompt, "cache_control": {"type": "ephemeral"}}]

        user_content = game_state_text
        if relevant_memories:
            user_content = f"{user_content}\n\n{relevant_memories}" if user_content else relevant_memories

        if self.use_native_thinking:
            user_content += ('\n\n**RESPONSE FORMAT:** You MUST respond with valid JSON containing only the action:\n'
                           '{"action": "the command to execute"}\n'
                           'Example: {"action": "take lamp"}')
        elif self.is_thinking_model:
            user_content += ('\n\n**RESPONSE FORMAT:** You MUST respond with valid JSON:\n'
                           '{"thinking": "your reasoning here", "action": "the command to execute"}\n'
                           'Example: {"thinking": "I see a lamp.", "action": "take lamp"}')

        messages.append({"role": "user", "content": user_content})
        return messages

    def _clean(self, action: str) -> str:
        a = re.sub(r"```.*?```", "", action.strip(), flags=re.DOTALL)
        a = re.sub(r"[`\"']", "", a).strip().lower().strip(".,!?;:")
        return a or "look"

    def _extract_partial_fields(self, partial: str) -> dict:
        result = {}
        for field_name in ("thinking", "action"):
            m = re.search(rf'"{field_name}"\s*:\s*"', partial)
            if not m: continue
            content_start = m.end()
            chars = []
            i = 0
            remaining = partial[content_start:]
            while i < len(remaining):
                c = remaining[i]
                if c == '\\' and i + 1 < len(remaining):
                    chars.append(remaining[i:i+2]); i += 2; continue
                if c == '"': break
                chars.append(c); i += 1
            extracted = ''.join(chars)
            try:
                result[field_name] = json.loads(f'"{extracted}"')
            except Exception:
                result[field_name] = extracted
        return result

    def reload_knowledge(self) -> bool:
        try:
            with open("prompts/agent.md") as f:
                base = f.read()
        except FileNotFoundError:
            try:
                with open("agent.md") as f:
                    base = f.read()
            except FileNotFoundError:
                return False
        self.system_prompt = self._enhance_with_knowledge(base)
        return True

    def update_episode_id(self, eid: str):
        self.episode_id = eid
