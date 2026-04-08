# truth_seeker_env/server/reward.py
"""
Per-step reward calculator for Truth-Seeker's Sandbox.

Design principles:
  1. FORMAT IS A GATE — bad format caps the grader score at 0.3, not a raw penalty.

  2. STEP REWARDS ARE TRAINING-SIGNAL ONLY.
     tool_use_reward() returns a dense per-step reward that the RL trainer receives
     after each READ_DOC / WEB_SEARCH action. This shapes learning during training.
     These are NOT added to the terminal score.

     state.accumulated_step_reward exists for logging and debugging only.
     It is intentionally excluded from compute_terminal_reward() to avoid
     double-counting: the grader already captures outcome quality (fact_match,
     source_match) which indirectly reflects how well the agent retrieved.
     Adding step rewards on top would reward retrieval twice.

  3. TERMINAL REWARD = GRADER SCORE with format gate.
     GraderRouter.grade() produces the submission-facing 0.0–1.0 score.
     compute_terminal_reward() applies the format gate and returns the final value.

  4. Hallucination detection is DISABLED for Phase 1.
     Phase 2: NLI entailment / LLM-as-judge / spaCy NER.
"""

import re
import logging

logger = logging.getLogger(__name__)


class RewardCalculator:

    # ── Per-step tool-use rewards ──────────────────────────────────────────
    RELEVANT_DOC_RETRIEVAL   =  0.2
    IRRELEVANT_DOC_RETRIEVAL = -0.1
    RELEVANT_WEB_SEARCH      =  0.3
    IRRELEVANT_WEB_SEARCH    = -0.15
    WRONG_TOOL_PENALTY       = -0.3
    UNKNOWN_ACTION_PENALTY   = -0.1

    # ── Timeout penalty (not part of grader score) ────────────────────────
    TIMEOUT_PENALTY          = -0.3

    # ── Hallucination: DISABLED for Phase 1 ─────────────────────────────
    # Setting to 0.0 ensures _has_hallucination() returning True is a no-op.
    HALLUCINATION_PENALTY    =  0.0

    # ── Format gate ────────────────────────────────────────────────────────
    FORMAT_CAP_ON_FAILURE    =  0.3

    # ── Per-step normalization bounds ──────────────────────────────────────
    # Best:  3 * RELEVANT_WEB_SEARCH = 0.9
    # Worst: 5 * WRONG_TOOL + TIMEOUT = -1.8
    MAX_POSSIBLE_REWARD =  1.9   # Some headroom above best-case 0.9
    MIN_POSSIBLE_REWARD = -1.8

    # ── Allowed tools per tier ─────────────────────────────────────────────
    _ALLOWED_TOOLS = {
        "easy":   {"READ_DOC", "SUBMIT_ANSWER"},
        "medium": {"READ_DOC", "WEB_SEARCH", "SUBMIT_ANSWER"},
        "hard":   {"WEB_SEARCH", "SUBMIT_ANSWER"},
    }

    # ------------------------------------------------------------------
    # Per-step reward
    # ------------------------------------------------------------------

    def tool_use_reward(self, action, state, retrieved_chunk: str) -> float:
        """
        Raw per-step reward for READ_DOC or WEB_SEARCH.
        Caller (environment.py) accumulates this into state.accumulated_step_reward.
        """
        allowed = self._ALLOWED_TOOLS.get(state.task_type, set())

        if action.action_type not in allowed:
            logger.debug(
                f"Wrong tool '{action.action_type}' for task_type='{state.task_type}'"
            )
            return self.WRONG_TOOL_PENALTY

        is_relevant = self._chunk_is_relevant(
            retrieved_chunk, state.question, state.ground_truth_key_facts
        )

        if action.action_type == "READ_DOC":
            return self.RELEVANT_DOC_RETRIEVAL if is_relevant else self.IRRELEVANT_DOC_RETRIEVAL
        elif action.action_type == "WEB_SEARCH":
            return self.RELEVANT_WEB_SEARCH if is_relevant else self.IRRELEVANT_WEB_SEARCH

        return 0.0

    # ------------------------------------------------------------------
    # Terminal reward (applied to grader score)
    # ------------------------------------------------------------------

    def compute_terminal_reward(self, grader_score: float, action) -> float:
        """
        Apply the format gate to the grader score and return the final episode reward.

        Args:
            grader_score: Raw grader output in [0.0, 1.0] from GraderRouter.grade().
            action: The SUBMIT_ANSWER action (checked for format compliance).

        Returns:
            Final episode reward in [0.0, 1.0].
            Capped at FORMAT_CAP_ON_FAILURE (0.3) if answer or reasoning is empty.

        Note on step rewards:
            state.accumulated_step_reward is NOT included here. Step rewards are
            dense training signals for the RL loop between steps. The grader score
            already captures answer quality that indirectly reflects retrieval quality.
            Adding step rewards would double-count retrieval. See module docstring.
        """
        reward = grader_score

        if not self._check_format(action):
            reward = min(reward, self.FORMAT_CAP_ON_FAILURE)
            logger.debug(f"Format gate applied: grader={grader_score:.3f} capped to {self.FORMAT_CAP_ON_FAILURE}")

        logger.debug(
            f"Terminal reward: grader_score={grader_score:.3f} → final={reward:.3f}"
        )
        return round(reward, 4)

    def timeout_penalty(self) -> float:
        """Raw penalty added to accumulated_step_reward when max_steps exceeded."""
        return self.TIMEOUT_PENALTY

    # ------------------------------------------------------------------
    # Normalization (for per-step rewards, if needed externally)
    # ------------------------------------------------------------------

    def normalize(self, raw_reward: float) -> float:
        """
        Normalize a raw per-step reward to [0.0, 1.0].
        Formula: clamp((raw - MIN) / (MAX - MIN), 0, 1)
        """
        span = self.MAX_POSSIBLE_REWARD - self.MIN_POSSIBLE_REWARD
        if span == 0:
            return 0.5
        return max(0.0, min(1.0, (raw_reward - self.MIN_POSSIBLE_REWARD) / span))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _chunk_is_relevant(self, chunk: str, question: str, key_facts: list) -> bool:
        """
        A chunk is relevant if:
          (a) it contains at least one key fact (direct substring match), OR
          (b) it has significant word overlap with the question (fallback).

        The "No " prefix guard short-circuits on "No results found" type strings.
        """
        if not chunk or chunk.startswith("No "):
            return False

        chunk_lower = chunk.lower()

        for fact in key_facts:
            if fact.lower() in chunk_lower:
                return True

        question_tokens = set(re.findall(r'\b\w+\b', question.lower()))
        chunk_tokens = set(re.findall(r'\b\w+\b', chunk_lower))
        overlap = len(question_tokens & chunk_tokens)
        threshold = max(2, len(question_tokens) * 0.3)
        return overlap >= threshold

    def _has_hallucination(self, answer: str, state) -> bool:
        """
        DISABLED for Phase 1.

        Token-level hallucination detection is too aggressive on paraphrased
        but grounded answers. Will return False unconditionally until Phase 2
        implements NLI entailment or LLM-as-judge scoring.
        """
        return False  # Phase 1: always no hallucination penalty

    def _check_format(self, action) -> bool:
        """Format check: both answer and reasoning must be non-empty strings."""
        return (
            bool(action.answer and action.answer.strip()) and
            bool(action.reasoning and action.reasoning.strip())
        )
