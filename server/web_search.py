# truth_seeker_env/server/web_search.py
"""
Dual-mode web search proxy for Truth-Seeker's Sandbox.

Phase 2 upgrade: supports BOTH real web search and keyword-overlap fallback.

Mode selection (automatic):
  SERPER_API_KEY is set → real Google search via Serper.dev API
  SERPER_API_KEY empty  → Jaccard keyword-overlap against tasks.jsonl web_results

The public API is STABLE — no changes needed in environment.py, grader.py,
or reward.py:
    load_episode_data(web_results_dict)
    search(query) -> str
    get_query_quality_score(query) -> float
"""

import os
import re
import logging

logger = logging.getLogger(__name__)


class WebSearchProxy:
    """
    Dual-mode web search proxy.

    Production mode (SERPER_API_KEY set):
      - search() calls the real Serper.dev Google Search API via WebSearchTool
      - Returns live web results formatted as readable text

    Fallback mode (no SERPER_API_KEY):
      - search() uses Jaccard keyword-overlap against pre-loaded web_results
      - Identical to Phase 1 behavior

    Both modes:
      - load_episode_data() loads per-episode topic keywords and mock results
      - get_query_quality_score() uses topic keywords for grading (deterministic)
    """

    RELEVANCE_THRESHOLD = 0.3  # Minimum Jaccard overlap for fallback mode

    def __init__(self, serper_api_key: str = ""):
        self._results: list = []
        self._topic_keywords: list = []

        # Determine mode
        api_key = serper_api_key or os.environ.get("SERPER_API_KEY", "")
        self._use_real_search = bool(api_key)

        if self._use_real_search:
            from .web_search_tool import WebSearchTool
            self._tool = WebSearchTool(
                api_key=api_key,
                top_k=5,
                timeout=60,
                snippet_only=True,   # Fast mode — stays within 20min inference limit
                proxy=None,
            )
            logger.info(
                f"WebSearchProxy: REAL SEARCH mode enabled "
                f"(key ends with ...{api_key[-3:]})"
            )
        else:
            self._tool = None
            logger.info(
                "WebSearchProxy: FALLBACK mode (no SERPER_API_KEY). "
                "Using keyword-overlap against tasks.jsonl web_results."
            )

    # ------------------------------------------------------------------
    # Public API (stable — unchanged from Phase 1)
    # ------------------------------------------------------------------

    def load_episode_data(self, web_results: dict):
        """
        Called by environment.reset() to load this episode's web data.

        In BOTH modes, topic_keywords are loaded for get_query_quality_score().
        In fallback mode, pre-loaded results are also used by search().
        """
        self._topic_keywords = web_results.get("keywords", [])
        self._results = web_results.get("results", [])
        logger.debug(
            f"WebSearchProxy: {len(self._results)} fallback results, "
            f"{len(self._topic_keywords)} topic keywords loaded"
        )

    def search(self, query: str) -> str:
        """
        Search for information matching the query.

        Real mode:  calls Serper.dev Google Search API
        Fallback:   Jaccard keyword-overlap against pre-loaded results
        """
        if self._use_real_search:
            return self._real_search(query)
        return self._fallback_search(query)

    def get_query_quality_score(self, query: str) -> float:
        """
        0.0–1.0 measure of how well this query matches topic keywords.
        Used by the Hard grader to reward targeted searching.

        This is ALWAYS deterministic (uses topic_keywords from tasks.jsonl),
        regardless of whether real search or fallback mode is active.

        Both topic keywords and query are tokenized identically using
        word-boundary regex, so compound tokens like 'us-east-1' are
        split into individual words for fair comparison.

        Score = |query_tokens ∩ topic_tokens| / |topic_tokens|
        """
        if not self._topic_keywords:
            return 0.0
        # Tokenize keywords the same way we tokenize the query
        topic_tokens = set()
        for kw in self._topic_keywords:
            topic_tokens.update(_tokenize(kw.lower()))
        if not topic_tokens:
            return 0.0
        query_tokens = set(_tokenize(query.lower()))
        if not query_tokens:
            return 0.0
        return len(query_tokens & topic_tokens) / len(topic_tokens)

    # ------------------------------------------------------------------
    # Real search (Serper.dev)
    # ------------------------------------------------------------------

    def _real_search(self, query: str) -> str:
        """
        Delegate to WebSearchTool for real Google search results.
        Falls back to keyword-overlap if the API call fails.
        """
        try:
            result = self._tool.search(query)
            if result and not result.startswith("Web search error:"):
                return result
            # API returned no results — try fallback
            logger.warning(
                f"Real search returned no results for '{query}', "
                "trying fallback."
            )
            return self._fallback_search(query)
        except Exception as e:
            logger.error(
                f"Real search failed for '{query}': {e}. "
                "Falling back to keyword-overlap.",
                exc_info=True,
            )
            return self._fallback_search(query)

    # ------------------------------------------------------------------
    # Fallback search (Jaccard keyword overlap — Phase 1 logic)
    # ------------------------------------------------------------------

    def _fallback_search(self, query: str) -> str:
        """
        Score query against pre-loaded results using Jaccard keyword overlap.
        Returns only results above RELEVANCE_THRESHOLD, sorted by score.
        """
        if not self._results:
            return "No web results available for this task."

        query_tokens = set(_tokenize(query.lower()))
        if not query_tokens:
            return "No relevant web results found: empty query."

        scored = []
        for result in self._results:
            result_kws = set()
            for kw in result.get("relevance_keywords", []):
                result_kws.update(_tokenize(kw.lower()))
            if not result_kws:
                continue
            union = query_tokens | result_kws
            overlap = len(query_tokens & result_kws)
            score = overlap / len(union) if union else 0.0
            if score >= self.RELEVANCE_THRESHOLD:
                scored.append((score, result))

        if not scored:
            return "No relevant web results found for this query."

        scored.sort(key=lambda x: x[0], reverse=True)

        parts = []
        for _, result in scored[:3]:
            parts.append(
                f"[Source: {result.get('url', 'unknown')}]\n"
                f"Title: {result.get('title', '')}\n"
                f"{result.get('snippet', '')}"
            )
        return "\n\n---\n\n".join(parts)


def _tokenize(text: str) -> list:
    return re.findall(r'\b\w+\b', text)
