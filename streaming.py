"""
WebSocket streaming server - broadcasts AI reasoning in real-time to the viewer.
"""

import json
import asyncio
import threading
from typing import Optional, Dict, Any, List

try:
    import websockets
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False


class StreamServer:
    """WebSocket server for streaming AI reasoning to connected viewers."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765, logger=None):
        self.host = host
        self.port = port
        self.logger = logger
        self._clients: set = set()
        self._server = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if not WS_AVAILABLE:
            if self.logger: self.logger.warning("websockets not installed, streaming disabled")
            return

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self):
        async def handler(ws, path=None):
            self._clients.add(ws)
            try:
                async for _ in ws:
                    pass
            finally:
                self._clients.discard(ws)

        self._server = await websockets.serve(handler, self.host, self.port)
        if self.logger:
            self.logger.info(f"Streaming server started on ws://{self.host}:{self.port}")
        await self._server.wait_closed()

    def stop(self):
        if self._server and self._loop:
            self._loop.call_soon_threadsafe(self._server.close)

    def broadcast(self, msg_type: str, data: Dict[str, Any]):
        if not self._clients or not self._loop:
            return
        message = json.dumps({"type": msg_type, **data})
        for client in list(self._clients):
            try:
                asyncio.run_coroutine_threadsafe(client.send(message), self._loop)
            except Exception:
                self._clients.discard(client)

    # Convenience broadcast methods
    def broadcast_turn_start(self, turn: int, location: str, score: int):
        self.broadcast("turn_start", {"turn": turn, "location": location, "score": score})

    def broadcast_reasoning_chunk(self, turn: int, reasoning: str, action: str = None):
        self.broadcast("reasoning_chunk", {"turn": turn, "reasoning": reasoning, "action": action})

    def broadcast_turn_complete(self, turn: int, action: str, score: int, location: str):
        self.broadcast("turn_complete", {"turn": turn, "action": action, "score": score, "location": location})

    def broadcast_tool_status(self, active_tool=None, tool_data=None, tool_history=None):
        self.broadcast("tool_status", {"active": active_tool, "data": tool_data, "history": tool_history})

    def broadcast_generation_start(self, generation: int, high_score: int, total_deaths: int,
                                    best_generation: int, episode_id: str):
        self.broadcast("generation_start", {
            "generation": generation, "high_score": high_score,
            "total_deaths": total_deaths, "best_generation": best_generation,
            "episode_id": episode_id,
        })

    def broadcast_memory_synthesis_start(self, turn: int, location_name: str, action: str):
        self.broadcast("memory_synthesis_start", {"turn": turn, "location_name": location_name, "action": action})

    def broadcast_memory_synthesis_chunk(self, turn: int, content: str):
        self.broadcast("memory_synthesis_chunk", {"turn": turn, "content": content})

    def broadcast_memory_synthesis_complete(self, turn: int, content: str, memory_created: bool,
                                            memory_title: str = None, memory_category: str = None):
        self.broadcast("memory_synthesis_complete", {
            "turn": turn, "content": content, "memory_created": memory_created,
            "memory_title": memory_title, "memory_category": memory_category,
        })

    def broadcast_web_search_start(self, turn: int, query: str):
        self.broadcast("web_search_start", {"turn": turn, "query": query})

    def broadcast_web_search_complete(self, turn: int, query: str, results_summary: str, success: bool):
        self.broadcast("web_search_complete", {"turn": turn, "query": query, "summary": results_summary, "success": success})

    def broadcast_objective_followup_chunk(self, turn: int, reasoning: str, action: str = None,
                                            objective_id: str = None, objective_text: str = None):
        self.broadcast("objective_followup_chunk", {
            "turn": turn, "reasoning": reasoning, "action": action,
            "objective_id": objective_id, "objective_text": objective_text,
        })

    def broadcast_objective_followup(self, turn: int, reasoning: str, action: str,
                                      objective_id: str, objective_text: str):
        self.broadcast("objective_followup", {
            "turn": turn, "reasoning": reasoning, "action": action,
            "objective_id": objective_id, "objective_text": objective_text,
        })

    # Reasoner broadcasts
    def broadcast_reasoner_start(self, turn: int, trigger_reason: str):
        self.broadcast("reasoner_start", {"turn": turn, "trigger_reason": trigger_reason})

    def broadcast_reasoner_chunk(self, turn: int, reasoning: str):
        self.broadcast("reasoner_chunk", {"turn": turn, "reasoning": reasoning})

    def broadcast_reasoner_complete(self, turn: int, reasoning: str, suggested_approach: str = None,
                                     objectives: list = None, abandoned: list = None):
        self.broadcast("reasoner_complete", {
            "turn": turn, "reasoning": reasoning,
            "suggested_approach": suggested_approach,
            "objectives": objectives or [], "abandoned": abandoned or [],
        })

    # Objective review broadcasts
    def broadcast_objective_review_start(self, turn: int, objective_count: int):
        self.broadcast("objective_review_start", {"turn": turn, "objective_count": objective_count})

    def broadcast_objective_review_chunk(self, turn: int, content: str):
        self.broadcast("objective_review_chunk", {"turn": turn, "content": content})

    def broadcast_objective_review_complete(self, turn: int, content: str,
                                             completed_objectives: list = None, updates: list = None):
        self.broadcast("objective_review_complete", {
            "turn": turn, "content": content,
            "completed_objectives": completed_objectives or [],
            "updates": updates or [],
        })
