"""
Game configuration - loads from pyproject.toml and environment.
Single source of truth for all settings.
"""

import os
import tomllib
from typing import Optional
from pathlib import Path
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv


def _default_retry() -> dict:
    return {
        "max_retries": 5, "initial_delay": 1.0, "max_delay": 60.0,
        "exponential_base": 2.0, "jitter_factor": 0.1, "timeout_seconds": 120.0,
        "circuit_breaker_failure_threshold": 10, "circuit_breaker_recovery_timeout": 300.0,
    }


class Config(BaseSettings):
    # Core
    max_turns_per_episode: int = 500
    turn_delay_seconds: float = 0
    turn_window_size: int = 100

    # Files
    episode_log_file: str = "zork_episode_log.txt"
    json_log_file: str = "zork_episode_log.jsonl"
    state_export_file: str = "current_state.json"
    map_state_file: str = "map_state.json"
    knowledge_file: str = "knowledgebase.md"
    game_workdir: str = "game_files"
    game_file_path: str = "jericho-game-suite/zork1.z5"

    # LLM
    client_base_url: str = "https://openrouter.ai/api/v1"
    client_api_key: Optional[str] = None
    agent_model: str = "openrouter/pony-alpha"
    critic_model: str = "openrouter/pony-alpha"
    extractor_model: str = "openrouter/pony-alpha"
    analysis_model: str = "openrouter/pony-alpha"
    memory_model: str = "openrouter/pony-alpha"
    reasoner_model: str = "openrouter/pony-alpha"

    # Per-model base URLs (optional overrides)
    agent_base_url: Optional[str] = None
    critic_base_url: Optional[str] = None
    extractor_base_url: Optional[str] = None
    analysis_base_url: Optional[str] = None
    memory_base_url: Optional[str] = None
    reasoner_base_url: Optional[str] = None

    # Retry
    retry: dict = Field(default_factory=_default_retry)

    # Intervals
    knowledge_update_interval: int = 100
    objective_update_interval: int = 10

    # Objectives
    enable_objective_refinement: bool = False
    objective_refinement_interval: int = 50
    max_objectives_before_forced_refinement: int = 15
    refined_objectives_target_count: int = 8
    enable_objective_completion_llm_check: bool = True
    completion_check_interval: int = 1
    completion_history_window: int = 3

    # State export
    enable_state_export: bool = True
    s3_bucket: Optional[str] = None
    s3_key_prefix: str = "zorkgpt/"

    # Gameplay
    critic_rejection_threshold: float = -0.2
    enable_critic: bool = False
    enable_exit_pruning: bool = True
    exit_failure_threshold: int = 2
    enable_inter_episode_synthesis: bool = True
    enable_walkthrough_generation: bool = True

    # Memory
    memory_file: str = "Memories.md"
    max_memories_shown: int = 15
    room_description_age_window: int = 10

    # Loop break
    max_turns_stuck: int = 80
    stuck_check_interval: int = 10
    enable_objective_based_progress: bool = True
    enable_location_penalty: bool = True
    location_revisit_penalty: float = -0.2
    location_revisit_window: int = 5
    enable_exploration_hints: bool = True
    action_novelty_window: int = 15
    enable_stuck_warnings: bool = False
    stuck_warning_threshold: int = 20

    # Deep reasoning
    enable_deep_reasoning: bool = True
    reasoner_effort: str = "high"

    # Pathfinder
    enable_pathfinder: bool = True
    pathfinder_max_path_length: int = 20

    # Streaming
    enable_streaming: bool = False
    websocket_host: str = "0.0.0.0"
    websocket_port: int = 8765

    # Web search
    enable_web_search: bool = False
    web_search_searxng_url: str = "http://192.168.4.26:8090/searxng/mcp"
    web_search_crawl4ai_url: str = "https://mcp.schuyler.ai/crawl4ai/mcp"
    web_search_max_results: int = 3
    web_search_max_crawl_length: int = 2000
    web_search_timeout: float = 15.0
    web_search_stuck_threshold: int = 10

    # Prompt logger (replaces Langfuse)
    enable_prompt_logger: bool = True
    prompt_log_dir: str = "game_files/prompt_logs"
    prompt_log_max_content: int = 50000

    # Sampling parameter dicts (loaded from TOML sections)
    agent_sampling: dict = Field(default_factory=dict)
    critic_sampling: dict = Field(default_factory=dict)
    memory_sampling: dict = Field(default_factory=dict)
    analysis_sampling: dict = Field(default_factory=dict)
    reasoner_sampling: dict = Field(default_factory=dict)
    turn_review_sampling: dict = Field(default_factory=dict)
    pathfinder_sampling: dict = Field(default_factory=dict)
    extractor_sampling: dict = Field(default_factory=dict)

    model_config = SettingsConfigDict(
        env_prefix="ZORKOPUS_", env_file=None, case_sensitive=False, extra="forbid"
    )

    @model_validator(mode='after')
    def _validate_stuck(self) -> 'Config':
        if self.max_turns_stuck < self.stuck_check_interval:
            raise ValueError("max_turns_stuck must be >= stuck_check_interval")
        if self.enable_stuck_warnings and self.stuck_warning_threshold >= self.max_turns_stuck:
            raise ValueError("stuck_warning_threshold must be < max_turns_stuck")
        return self

    def model_post_init(self, __context) -> None:
        if self.client_api_key is None:
            if "openrouter" in (self.client_base_url or "").lower():
                self.client_api_key = os.environ.get("OPENROUTER_API_KEY")
            else:
                self.client_api_key = os.environ.get("CLIENT_API_KEY")
        if self.s3_bucket is None:
            self.s3_bucket = os.environ.get("ZORK_S3_BUCKET")
        Path(self.game_workdir).mkdir(parents=True, exist_ok=True)

    def base_url_for(self, role: str) -> str:
        """Get effective base URL for a model role."""
        override = getattr(self, f"{role}_base_url", None)
        return override or self.client_base_url

    def api_key_for(self, role: str) -> Optional[str]:
        """Get effective API key for a model role."""
        url = self.base_url_for(role).lower()
        if "openrouter" in url:
            return os.environ.get("OPENROUTER_API_KEY") or self.client_api_key
        elif "moonshot" in url:
            return os.environ.get("MOONSHOT_API_KEY") or self.client_api_key
        return self.client_api_key

    def memory_history_window(self) -> int:
        return max(1, self.memory_sampling.get("memory_history_window", 3))

    @classmethod
    def from_toml(cls, path: Optional[Path] = None) -> "Config":
        load_dotenv()
        path = path or Path("pyproject.toml")
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {path}")

        with open(path, "rb") as f:
            raw = tomllib.load(f)

        cfg = raw.get("tool", {}).get("zorkopus", {})
        llm = cfg.get("llm", {})
        orch = cfg.get("orchestrator", {})
        files = cfg.get("files", {})
        gp = cfg.get("gameplay", {})
        retry = cfg.get("retry", {})
        mem = cfg.get("memory", {})
        lb = cfg.get("loop_break", {})
        pf = cfg.get("pathfinder", {})
        st = cfg.get("streaming", {})
        ws = cfg.get("web_search", {})
        aws = cfg.get("aws", {})
        pl = cfg.get("prompt_logger", {})
        rs = cfg.get("reasoner_sampling", {})

        d = {
            "max_turns_per_episode": orch.get("max_turns_per_episode"),
            "turn_delay_seconds": gp.get("turn_delay_seconds"),
            "turn_window_size": gp.get("turn_window_size"),
            "episode_log_file": files.get("episode_log_file"),
            "json_log_file": files.get("json_log_file"),
            "state_export_file": files.get("state_export_file"),
            "map_state_file": files.get("map_state_file"),
            "knowledge_file": files.get("knowledge_file"),
            "game_workdir": gp.get("game_workdir"),
            "game_file_path": files.get("game_file_path"),
            "client_base_url": llm.get("client_base_url"),
            "agent_model": llm.get("agent_model"),
            "critic_model": llm.get("critic_model"),
            "extractor_model": llm.get("extractor_model"),
            "analysis_model": llm.get("analysis_model"),
            "memory_model": llm.get("memory_model"),
            "reasoner_model": llm.get("reasoner_model"),
            "agent_base_url": llm.get("agent_base_url"),
            "critic_base_url": llm.get("critic_base_url"),
            "extractor_base_url": llm.get("extractor_base_url"),
            "analysis_base_url": llm.get("analysis_base_url"),
            "memory_base_url": llm.get("memory_base_url"),
            "reasoner_base_url": llm.get("reasoner_base_url"),
            "retry": retry or _default_retry(),
            "knowledge_update_interval": orch.get("knowledge_update_interval"),
            "objective_update_interval": orch.get("objective_update_interval"),
            "enable_objective_refinement": orch.get("enable_objective_refinement"),
            "enable_state_export": orch.get("enable_state_export"),
            "s3_key_prefix": aws.get("s3_key_prefix"),
            "critic_rejection_threshold": gp.get("critic_rejection_threshold"),
            "enable_critic": gp.get("enable_critic"),
            "enable_exit_pruning": gp.get("enable_exit_pruning"),
            "exit_failure_threshold": gp.get("exit_failure_threshold"),
            "enable_inter_episode_synthesis": orch.get("enable_inter_episode_synthesis"),
            "enable_walkthrough_generation": orch.get("enable_walkthrough_generation"),
            "memory_file": mem.get("memory_file"),
            "max_memories_shown": mem.get("max_memories_shown"),
            "enable_deep_reasoning": rs.get("enable_deep_reasoning", True),
            "reasoner_effort": rs.get("reasoner_effort", "high"),
            "enable_pathfinder": pf.get("enable_pathfinder"),
            "pathfinder_max_path_length": pf.get("max_path_length"),
            "enable_streaming": st.get("enable_streaming"),
            "websocket_host": st.get("websocket_host"),
            "websocket_port": st.get("websocket_port"),
            "enable_web_search": ws.get("enable_web_search"),
            "web_search_searxng_url": ws.get("searxng_url"),
            "web_search_crawl4ai_url": ws.get("crawl4ai_url"),
            "web_search_max_results": ws.get("max_results"),
            "web_search_max_crawl_length": ws.get("max_crawl_length"),
            "web_search_timeout": ws.get("timeout"),
            "web_search_stuck_threshold": ws.get("stuck_threshold"),
            "enable_prompt_logger": pl.get("enable", True),
            "prompt_log_dir": pl.get("log_dir", "game_files/prompt_logs"),
            "prompt_log_max_content": pl.get("max_content_length", 50000),
            "agent_sampling": cfg.get("agent_sampling", {}),
            "critic_sampling": cfg.get("critic_sampling", {}),
            "memory_sampling": cfg.get("memory_sampling", {}),
            "analysis_sampling": cfg.get("analysis_sampling", {}),
            "reasoner_sampling": rs,
            "turn_review_sampling": cfg.get("turn_review_sampling", {}),
            "pathfinder_sampling": cfg.get("pathfinder_sampling", {}),
            "extractor_sampling": cfg.get("extractor_sampling", {}),
            "enable_objective_completion_llm_check": cfg.get("objective_completion", {}).get("enable_llm_check", True),
            "completion_check_interval": cfg.get("objective_completion", {}).get("check_interval", 1),
            "completion_history_window": cfg.get("objective_completion", {}).get("history_window", 3),
        }

        # Loop break settings (only if present)
        for key in ["max_turns_stuck", "stuck_check_interval", "enable_objective_based_progress",
                     "enable_location_penalty", "location_revisit_penalty", "location_revisit_window",
                     "enable_exploration_hints", "action_novelty_window",
                     "enable_stuck_warnings", "stuck_warning_threshold"]:
            val = lb.get(key)
            if val is not None:
                d[key] = val

        # Filter None values so Pydantic uses defaults
        d = {k: v for k, v in d.items() if v is not None}
        return cls.model_validate(d)
