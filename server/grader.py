# truth_seeker_env/server/grader.py
"""
Deterministic graders for all three task tiers.

GraderRouter routes to the correct grader based on task_type.
All graders output a float in [0.0, 1.0].

Grading formulas:
  Easy   = 0.6 * fact_match + 0.4 * source_match
  Medium = 0.5 * fact_match + 0.2 * source_match + 0.3 * tool_diversity
  Hard   = 0.4 * fact_match + 0.3 * avg_query_quality + 0.3 * citation_density
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .web_search import WebSearchProxy

logger = logging.getLogger(__name__)


class GraderRouter:
    """
    Routes to the correct tier grader based on state.task_type.
    Holds a reference to WebSearchProxy so HardGrader can score query quality.
    """

    def __init__(self, web_search: "WebSearchProxy"):
        self._easy = EasyGrader()
        self._medium = MediumGrader()
        self._hard = HardGrader(web_search)

    def grade(self, action, state) -> float:
        """Grade the submitted answer. Returns float in [0.0, 1.0]."""
        if state.task_type == "easy":
            return self._easy.grade(action, state)
        elif state.task_type == "medium":
            return self._medium.grade(action, state)
        elif state.task_type == "hard":
            return self._hard.grade(action, state)
        else:
            logger.warning(f"Unknown task_type '{state.task_type}' — grader returns 0.0")
            return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Easy Grader
# ──────────────────────────────────────────────────────────────────────────────

class EasyGrader:
    """
    Score = 0.6 * fact_match_ratio + 0.4 * source_match_ratio

    Agent must use READ_DOC to find facts that exist in the internal docs.
    """

    def grade(self, action, state) -> float:
        fact_score = _fact_match(action.answer, state.ground_truth_key_facts)
        source_score = _source_match(action.citations, state.ground_truth_sources)
        score = 0.6 * fact_score + 0.4 * source_score
        logger.debug(
            f"EasyGrader: fact={fact_score:.3f} source={source_score:.3f} → {score:.3f}"
        )
        return _clamp(score)


# ──────────────────────────────────────────────────────────────────────────────
# Medium Grader
# ──────────────────────────────────────────────────────────────────────────────

class MediumGrader:
    """
    Score = 0.5 * fact_match + 0.2 * source_match + 0.3 * tool_diversity

    tool_diversity:
        1.0 — agent used BOTH READ_DOC and WEB_SEARCH
        0.5 — agent used exactly one of them
        0.0 — agent used neither (jumped straight to SUBMIT_ANSWER)
    """

    def grade(self, action, state) -> float:
        fact_score = _fact_match(action.answer, state.ground_truth_key_facts)
        source_score = _source_match(action.citations, state.ground_truth_sources)

        used = set(state.action_history)
        has_read = "READ_DOC" in used
        has_web = "WEB_SEARCH" in used
        if has_read and has_web:
            diversity = 1.0
        elif has_read or has_web:
            diversity = 0.5
        else:
            diversity = 0.0

        score = 0.5 * fact_score + 0.2 * source_score + 0.3 * diversity
        logger.debug(
            f"MediumGrader: fact={fact_score:.3f} source={source_score:.3f} "
            f"diversity={diversity:.1f} → {score:.3f}"
        )
        return _clamp(score)


# ──────────────────────────────────────────────────────────────────────────────
# Hard Grader
# ──────────────────────────────────────────────────────────────────────────────

class HardGrader:
    """
    Score = 0.4 * fact_match + 0.3 * avg_query_quality + 0.3 * citation_density

    avg_query_quality:
        Average WebSearchProxy.get_query_quality_score() across all WEB_SEARCH
        queries. Rewards targeted, specific queries over garbage queries.
        Agent that never searched → 0.0.

    citation_density:
        (distinct_sources_cited) / (expected_sources_count), capped at 1.0.
        Hard tasks have no internal docs. If no sources expected → 1.0 marks.
    """

    def __init__(self, web_search: "WebSearchProxy"):
        self._web_search = web_search

    def grade(self, action, state) -> float:
        fact_score = _fact_match(action.answer, state.ground_truth_key_facts)

        # Query quality across all WEB_SEARCH actions this episode
        web_queries = [
            entry["query"]
            for entry in _action_details(state)
            if entry["action_type"] == "WEB_SEARCH"
        ]
        if web_queries:
            scores = [self._web_search.get_query_quality_score(q) for q in web_queries]
            avg_quality = sum(scores) / len(scores)
        else:
            avg_quality = 0.0

        # Citation density
        expected = state.ground_truth_sources
        if not expected:
            density = 1.0
        elif action.citations:
            distinct_cited = len({c.lower().strip() for c in action.citations})
            density = min(1.0, distinct_cited / len(expected))
        else:
            density = 0.0

        score = 0.4 * fact_score + 0.3 * avg_quality + 0.3 * density
        logger.debug(
            f"HardGrader: fact={fact_score:.3f} quality={avg_quality:.3f} "
            f"density={density:.3f} → {score:.3f}"
        )
        return _clamp(score)


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def _fact_match(answer: str, key_facts: list) -> float:
    """Proportion of key_facts that appear (case-insensitive) in the answer."""
    if not key_facts:
        return 1.0  # nothing to match → full score
    answer_lower = answer.lower()
    matched = sum(1 for fact in key_facts if fact.lower() in answer_lower)
    return matched / len(key_facts)


def _source_match(citations: list, ground_truth_sources: list) -> float:
    """Proportion of ground_truth_sources referenced in the agent's citations."""
    if not ground_truth_sources:
        return 1.0
    if not citations:
        return 0.0
    citations_lower = [c.lower() for c in citations]
    matched = sum(
        1 for src in ground_truth_sources
        if any(src.lower() in c for c in citations_lower)
    )
    return matched / len(ground_truth_sources)


def _action_details(state) -> list:
    """
    Returns per-step {action_type, query} records.
    Prefers state.action_details (rich); falls back to action_history (type only).
    """
    if getattr(state, "action_details", None):
        return state.action_details
    return [{"action_type": a, "query": ""} for a in getattr(state, "action_history", [])]


def _clamp(score: float) -> float:
    return round(min(max(score, 0.01), 0.99), 4)