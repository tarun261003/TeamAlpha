# truth_seeker_env/server/web_search_tool.py
"""
Real web search tool using Google Search API via Serper.dev.

Ported from the OpenEnv `web_search/server/web_search_tool.py` reference
implementation. Adapted to work standalone (no openenv_core dependency).

Two modes:
  snippet_only=True  — fast: returns Google snippets only (default)
  snippet_only=False — deep: fetches full page content and expands snippets

Usage:
    tool = WebSearchTool(api_key="your-serper-key", snippet_only=True)
    results = tool.search("What is the capital of France?")
    # returns formatted string of search results
"""

from __future__ import annotations
import random
import logging
from dataclasses import dataclass, field
from typing import Optional

import requests
import chardet

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Data models (standalone — no openenv_core dependency)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class WebContent:
    """A single web search result."""
    title: str = ""
    content: str = ""
    url: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# WebSearchTool
# ──────────────────────────────────────────────────────────────────────────────

class WebSearchTool:
    """
    Searches the web using Google Search API via Serper.dev.

    Ported from OpenEnv web_search reference. Key features:
      - Calls Serper.dev POST /search endpoint
      - snippet_only mode: returns Google snippets (fast, cheap)
      - deep mode: fetches full pages and expands snippets to paragraphs
      - Formats results into a readable string for the RL agent
    """

    def __init__(
        self,
        api_key: str,
        top_k: int = 5,
        timeout: int = 60,
        snippet_only: bool = True,
        proxy: Optional[str] = None,
    ):
        self.api_key = api_key
        self.top_k = top_k
        self.timeout = timeout
        self.snippet_only = snippet_only
        self.proxy = proxy

    def search(self, query: str) -> str:
        """
        Execute a web search and return formatted results as a string.

        This is the main entry point used by WebSearchProxy.
        Returns a formatted multi-result string, or an error message.
        """
        query = query.strip()
        if not query:
            return "No results: empty query."

        try:
            web_contents = self._google_search(query)
            if web_contents:
                return self._format_web_contents(web_contents, query)
            else:
                return f"No search results found for query: {query}"
        except Exception as e:
            logger.error(f"WebSearchTool.search failed: {e}", exc_info=True)
            return f"Web search error: {e}"

    # ------------------------------------------------------------------
    # Serper.dev API
    # ------------------------------------------------------------------

    def _google_search(self, query: str) -> list[WebContent]:
        """
        Call Serper.dev Google Search API and return structured results.
        """
        proxies = {"http": self.proxy, "https": self.proxy} if self.proxy else None

        resp = requests.post(
            "https://google.serper.dev/search",
            json={
                "q": query,
                "num": self.top_k,
                "gl": "us",
                "hl": "en",
            },
            headers={
                "Content-Type": "application/json",
                "X-API-KEY": self.api_key,
            },
            timeout=self.timeout,
            proxies=proxies,
        )
        resp.raise_for_status()
        response = resp.json()
        items = response.get("organic", [])

        web_contents = []
        if self.snippet_only:
            # Quick mode: just use snippets
            for item in items:
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                context = " ".join(self._parse_search_snippet(snippet))

                if title or context:
                    title = title or "No title."
                    context = context or "No snippet available."
                    web_contents.append(WebContent(
                        title=title,
                        content=context,
                        url=item.get("link", ""),
                    ))
        else:
            # Deep mode: fetch full page content
            links = [item.get("link", "") for item in items if "link" in item]
            raw_contents = self._fetch_web_contents(links)

            for i, item in enumerate(items):
                title = item.get("title", "")
                snippet = item.get("snippet", "")

                # Extract relevant context from the full page
                context = (
                    self._expand_search_snippet(snippet, raw_contents[i])
                    if i < len(raw_contents) and raw_contents[i]
                    else snippet
                )

                if title or context:
                    title = title or "No title."
                    context = context or "No content available."
                    web_contents.append(WebContent(
                        title=title,
                        content=context,
                        url=item.get("link", ""),
                    ))

        return web_contents

    # ------------------------------------------------------------------
    # Page fetching
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_web_contents(urls: list[str], limit: int = 8) -> list[str]:
        """
        Fetch multiple web page contents sequentially.
        Returns list of page text (empty string for failures).
        """

        def _fetch(url: str) -> str:
            if not url:
                return ""

            user_agents = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Mozilla/5.0 (compatible; Googlebot/2.1; "
                "+https://www.google.com/bot.html)",
            ]
            headers = {"User-Agent": random.choice(user_agents)}

            try:
                response = requests.get(url, headers=headers, timeout=10)
                raw = response.content
                detected = chardet.detect(raw)
                encoding = detected.get("encoding") or "utf-8"
                return raw.decode(encoding, errors="ignore")
            except Exception:
                return ""

        return [_fetch(url) for url in urls[:limit]]

    # ------------------------------------------------------------------
    # Snippet processing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_search_snippet(snippet: str) -> list[str]:
        """Parse a search snippet into meaningful segments (>5 words)."""
        segments = snippet.split("...")
        return [s.strip() for s in segments if len(s.strip().split()) > 5]

    @staticmethod
    def _expand_search_snippet(snippet: str, web_content: str) -> str:
        """
        Find snippet segments in the full web content and expand them
        to full paragraphs for richer context.
        """
        snippets = WebSearchTool._parse_search_snippet(snippet)
        ctx_paras = []

        for s in snippets:
            pos = web_content.replace("\n", " ").find(s)
            if pos == -1:
                continue

            # Expand to paragraph boundaries
            sta = pos
            while sta > 0 and web_content[sta] != "\n":
                sta -= 1

            end = pos + len(s)
            while end < len(web_content) and web_content[end] != "\n":
                end += 1

            para = web_content[sta:end].strip()
            if para and para not in ctx_paras:
                ctx_paras.append(para)

        return "\n".join(ctx_paras)

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_web_contents(web_contents: list[WebContent], query: str) -> str:
        """Format search results into a readable string for the agent."""
        lines = [f"Search results for: {query}\n"]

        for i, result in enumerate(web_contents, 1):
            lines.append(f"[{i}] {result.title}")
            lines.append(f"    URL: {result.url or 'N/A'}")
            lines.append(
                f"    {result.content[:500]}"
                f"{'...' if len(result.content) > 500 else ''}"
            )
            lines.append("")

        return "\n".join(lines)
