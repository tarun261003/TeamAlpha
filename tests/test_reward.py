"""
Tests for reward.py

Run with: pytest tests/test_reward.py -v
"""

import pytest
from server.reward import RewardCalculator


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ──────────────────────────────────────────────────────────────────────────────

class MockAction:
    def __init__(self, action_type="READ_DOC", query="test query",
                 answer="", reasoning="my reasoning", citations=None):
        self.action_type = action_type
        self.query = query
        self.answer = answer
        self.reasoning = reasoning
        self.citations = citations or []

class MockState:
    def __init__(self, task_type="easy", question="What is the timeout?",
                 key_facts=None, sources=None, retrieved_chunks=None,
                 max_steps=10, step_count=3, accumulated_step_reward=0.0):
        self.task_type = task_type
        self.question = question
        self.ground_truth_key_facts = key_facts or []
        self.ground_truth_sources = sources or []
        self.retrieved_chunks = retrieved_chunks or []
        self.max_steps = max_steps
        self.step_count = step_count
        self.accumulated_step_reward = accumulated_step_reward


@pytest.fixture
def calc():
    return RewardCalculator()


# ──────────────────────────────────────────────────────────────────────────────
# Format gate
# ──────────────────────────────────────────────────────────────────────────────

def test_format_check_passes(calc):
    action = MockAction(answer="some answer", reasoning="some reasoning")
    assert calc._check_format(action) is True

def test_format_check_fails_empty_answer(calc):
    action = MockAction(answer="", reasoning="reasoning")
    assert calc._check_format(action) is False

def test_format_check_fails_empty_reasoning(calc):
    action = MockAction(answer="answer", reasoning="")
    assert calc._check_format(action) is False

def test_format_check_fails_whitespace_only(calc):
    action = MockAction(answer="   ", reasoning="   ")
    assert calc._check_format(action) is False

def test_format_gate_caps_score(calc):
    # grader_score=0.9, bad format → capped at 0.3
    action = MockAction(answer="", reasoning="")
    result = calc.compute_terminal_reward(0.9, action)
    assert result == pytest.approx(0.3, abs=0.001)

def test_format_gate_allows_high_score(calc):
    # grader_score=0.9, good format → 0.9 passes through
    action = MockAction(answer="good answer", reasoning="thorough reasoning")
    result = calc.compute_terminal_reward(0.9, action)
    assert result == pytest.approx(0.9, abs=0.001)


# ──────────────────────────────────────────────────────────────────────────────
# Hallucination disabled
# ──────────────────────────────────────────────────────────────────────────────

def test_hallucination_disabled(calc):
    """Phase 1: _has_hallucination() always returns False."""
    state = MockState(retrieved_chunks=["small context"])
    # Even with an answer full of "unsupported" words
    assert calc._has_hallucination("xyzzy quux frobnicate wibble blargh", state) is False

def test_hallucination_disabled_empty_context(calc):
    state = MockState(retrieved_chunks=[])
    assert calc._has_hallucination("anything", state) is False


# ──────────────────────────────────────────────────────────────────────────────
# Per-step tool-use rewards
# ──────────────────────────────────────────────────────────────────────────────

def test_relevant_read_doc_reward(calc):
    action = MockAction(action_type="READ_DOC", query="timeout")
    state = MockState(task_type="easy", key_facts=["60 seconds"])
    chunk = "The default timeout is 60 seconds for all requests."
    reward = calc.tool_use_reward(action, state, chunk)
    assert reward == calc.RELEVANT_DOC_RETRIEVAL

def test_irrelevant_read_doc_reward(calc):
    action = MockAction(action_type="READ_DOC", query="timeout")
    state = MockState(task_type="easy", key_facts=["unrelated"])
    chunk = "No relevant documents found."
    reward = calc.tool_use_reward(action, state, chunk)
    assert reward == calc.IRRELEVANT_DOC_RETRIEVAL

def test_wrong_tool_penalty_easy(calc):
    """WEB_SEARCH on an easy task → wrong tool penalty."""
    action = MockAction(action_type="WEB_SEARCH", query="fastapi timeout")
    state = MockState(task_type="easy")
    chunk = "some web result"
    reward = calc.tool_use_reward(action, state, chunk)
    assert reward == calc.WRONG_TOOL_PENALTY

def test_wrong_tool_penalty_hard(calc):
    """READ_DOC on a hard task → wrong tool penalty."""
    action = MockAction(action_type="READ_DOC", query="query")
    state = MockState(task_type="hard")
    chunk = "doc content"
    reward = calc.tool_use_reward(action, state, chunk)
    assert reward == calc.WRONG_TOOL_PENALTY

def test_relevant_web_search_reward(calc):
    action = MockAction(action_type="WEB_SEARCH", query="fastapi timeout default")
    state = MockState(task_type="hard", key_facts=["60 seconds"],
                      question="What is the FastAPI default timeout?")
    chunk = "The default timeout is 60 seconds."
    reward = calc.tool_use_reward(action, state, chunk)
    assert reward == calc.RELEVANT_WEB_SEARCH

def test_irrelevant_web_search_reward(calc):
    action = MockAction(action_type="WEB_SEARCH", query="random query")
    state = MockState(task_type="hard", key_facts=["very specific phrase xyz"])
    chunk = "No relevant web results found for this query."
    reward = calc.tool_use_reward(action, state, chunk)
    assert reward == calc.IRRELEVANT_WEB_SEARCH


# ──────────────────────────────────────────────────────────────────────────────
# Timeout penalty
# ──────────────────────────────────────────────────────────────────────────────

def test_timeout_penalty_value(calc):
    assert calc.timeout_penalty() == calc.TIMEOUT_PENALTY
    assert calc.timeout_penalty() < 0

# ──────────────────────────────────────────────────────────────────────────────
# Normalization
# ──────────────────────────────────────────────────────────────────────────────

def test_normalize_min(calc):
    assert calc.normalize(calc.MIN_POSSIBLE_REWARD) == pytest.approx(0.0, abs=0.001)

def test_normalize_max(calc):
    assert calc.normalize(calc.MAX_POSSIBLE_REWARD) == pytest.approx(1.0, abs=0.001)

def test_normalize_clamps_below_zero(calc):
    assert calc.normalize(-999.0) == 0.0

def test_normalize_clamps_above_one(calc):
    assert calc.normalize(999.0) == 1.0

def test_normalize_midpoint(calc):
    mid = (calc.MAX_POSSIBLE_REWARD + calc.MIN_POSSIBLE_REWARD) / 2
    result = calc.normalize(mid)
    assert result == pytest.approx(0.5, abs=0.01)

def test_normalize_deterministic(calc):
    for raw in [-1.5, -0.5, 0.0, 0.5, 1.2]:
        assert calc.normalize(raw) == calc.normalize(raw)

def test_normalize_output_always_in_range(calc):
    for raw in [-10, -2.3, -1.8, 0, 0.5, 1.0, 1.9, 5.0, 100]:
        result = calc.normalize(raw)
        assert 0.0 <= result <= 1.0, f"normalize({raw}) = {result} out of range"
