"""
Web search manager - MCP-based search and crawl via JSON-RPC 2.0.
Uses SearXNG for search and Crawl4AI for page analysis.
"""

import json
from typing import List, Dict, Any
import requests

from config import Config
from state import GameState


class WebSearchManager:
    def __init__(self, config: Config, game_state: GameState, logger=None):
        self.config = config
        self.gs = game_state
        self.logger = logger
        self.searxng_url = config.web_search_searxng_url
        self.crawl4ai_url = config.web_search_crawl4ai_url
        self.max_results = config.web_search_max_results
        self.max_crawl = config.web_search_max_crawl_length
        self.timeout = config.web_search_timeout
        self._req_id = 0

    def search(self, query: str) -> List[Dict[str, str]]:
        result = self._mcp_call(self.searxng_url, "searx_search",
                                {"q": query, "engines": "google,duckduckgo,bing,startpage,wikipedia"})
        if "error" in result: return []
        return [{"url": r.get("url", ""), "title": r.get("title", ""), "snippet": r.get("content", "")}
                for r in result.get("results", [])[:self.max_results]]

    def crawl_and_ask(self, urls: str, question: str) -> str:
        result = self._mcp_call(self.crawl4ai_url, "crawl4ai_ask", {"urls": urls, "question": question})
        if "error" in result: return ""
        if "text" in result: return result["text"][:self.max_crawl]
        results = result.get("results", [])
        if results and results[0].get("success"):
            return results[0].get("answer", "")[:self.max_crawl]
        return ""

    def _mcp_call(self, url: str, tool: str, args: Dict) -> Dict:
        self._req_id += 1
        payload = {"jsonrpc": "2.0", "method": "tools/call",
                    "params": {"name": tool, "arguments": args}, "id": self._req_id}
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout,
                                headers={"Content-Type": "application/json", "Accept": "application/json, text/event-stream"})
            resp.raise_for_status()
        except requests.Timeout:
            return {"error": f"timeout ({self.timeout}s)"}
        except requests.ConnectionError:
            return {"error": f"cannot reach {tool}"}
        except requests.HTTPError as e:
            return {"error": str(e)}

        ct = resp.headers.get("Content-Type", "")
        try:
            rpc = self._parse_sse(resp.text) if "text/event-stream" in ct else resp.json()
        except (json.JSONDecodeError, ValueError):
            return {"error": "invalid response"}

        if "error" in rpc:
            return {"error": rpc["error"].get("message", str(rpc["error"]))}

        try:
            text = rpc.get("result", {}).get("content", [{}])[0].get("text", "")
            return json.loads(text)
        except (json.JSONDecodeError, KeyError, IndexError):
            return {"text": text} if text else {"error": "malformed response"}

    def _parse_sse(self, text: str) -> Dict:
        for line in text.split("\n"):
            if line.strip().startswith("data:"):
                try:
                    data = json.loads(line[5:].strip())
                    if "result" in data or "error" in data: return data
                except json.JSONDecodeError:
                    continue
        return {"error": "no valid result in SSE"}

    def reset(self):
        self._req_id = 0
