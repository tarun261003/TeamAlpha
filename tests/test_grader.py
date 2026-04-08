"""
Tests for grader.py

Run with: pytest tests/test_grader.py -v
"""

import pytest
from server.grader import (
    EasyGrader, MediumGrader, HardGrader, GraderRouter,
    _fact_match, _source_match,
)
from server.web_search import WebSearchProxy


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

class MockAction:
    def __init__(self, answer="", citations=None, reasoning="test"):
        self.answer = answer
        self.citations = citations or []
        self.reasoning = reasoning

class MockState:
    def __init__(self, task_type="easy", action_history=None,
                 action_details=None, key_facts=None, sources=None):
        self.task_type = task_type
        self.action_history = action_history or []
        self.action_details = action_details or []
        self.ground_truth_key_facts = key_facts or []
        self.ground_truth_sources = sources or []


@pytest.fixture
def web_search():
    ws = WebSearchProxy()
    ws.load_episode_data({
        "keywords": ["fastapi", "timeout", "default"],
        "results": [
            {
                "url": "https://example.com",
                "title": "FastAPI Config",
                "snippet": "Default timeout is 60 seconds.",
                "relevance_keywords": ["fastapi", "timeout", "default"],
            }
        ],
    })
    return ws


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def test_fact_match_full():
    assert _fact_match("The timeout is 60 seconds.", ["60 seconds", "timeout"]) == 1.0

def test_fact_match_partial():
    score = _fact_match("The timeout is 60 seconds.", ["60 seconds", "default"])
    assert score == 0.5

def test_fact_match_none():
    assert _fact_match("Completely unrelated answer.", ["60 seconds", "timeout"]) == 0.0

def test_fact_match_no_facts():
    assert _fact_match("Any answer.", []) == 1.0

def test_source_match_full():
    assert _source_match(["fastapi-docs"], ["fastapi-docs"]) == 1.0

def test_source_match_substring():
    # citations often contain full URLs while sources are doc IDs
    assert _source_match(["https://example.com/fastapi-docs"], ["fastapi-docs"]) == 1.0

def test_source_match_none():
    assert _source_match([], ["fastapi-docs"]) == 0.0

def test_source_match_no_sources_expected():
    assert _source_match([], []) == 1.0


# ──────────────────────────────────────────────────────────────────────────────
# EasyGrader
# ──────────────────────────────────────────────────────────────────────────────

def test_easy_grader_full_score():
    grader = EasyGrader()
    action = MockAction(
        answer="The default timeout is 60 seconds.",
        citations=["fastapi-docs"],
    )
    state = MockState(
        task_type="easy",
        key_facts=["60 seconds"],
        sources=["fastapi-docs"],
    )
    score = grader.grade(action, state)
    assert score == 1.0, f"Expected 1.0, got {score}"

def test_easy_grader_missing_source():
    grader = EasyGrader()
    action = MockAction(
        answer="The default timeout is 60 seconds.",
        citations=[],
    )
    state = MockState(key_facts=["60 seconds"], sources=["fastapi-docs"])
    score = grader.grade(action, state)
    # fact=1.0, source=0.0 → 0.6*1.0 + 0.4*0.0 = 0.6
    assert score == pytest.approx(0.6, abs=0.01)

def test_easy_grader_wrong_answer():
    grader = EasyGrader()
    action = MockAction(answer="I don't know.", citations=[])
    state = MockState(key_facts=["60 seconds"], sources=["fastapi-docs"])
    score = grader.grade(action, state)
    assert score == 0.0

def test_easy_grader_output_in_range():
    grader = EasyGrader()
    action = MockAction(answer="x", citations=[])
    state = MockState(key_facts=["fact"], sources=["src"])
    score = grader.grade(action, state)
    assert 0.0 <= score <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# MediumGrader
# ──────────────────────────────────────────────────────────────────────────────

def test_medium_grader_full_diversity():
    grader = MediumGrader()
    action = MockAction(
        answer="The answer includes key_fact here.",
        citations=["source_a"],
    )
    state = MockState(
        task_type="medium",
        action_history=["READ_DOC", "WEB_SEARCH", "SUBMIT_ANSWER"],
        key_facts=["key_fact"],
        sources=["source_a"],
    )
    score = grader.grade(action, state)
    # fact=1.0, source=1.0, diversity=1.0 → 0.5+0.2+0.3 = 1.0
    assert score == pytest.approx(1.0, abs=0.01)

def test_medium_grader_no_diversity():
    grader = MediumGrader()
    action = MockAction(answer="key_fact is mentioned.", citations=["source_a"])
    state = MockState(
        task_type="medium",
        action_history=["SUBMIT_ANSWER"],
        key_facts=["key_fact"],
        sources=["source_a"],
    )
    score = grader.grade(action, state)
    # diversity=0 → 0.5*1.0 + 0.2*1.0 + 0.3*0.0 = 0.7
    assert score == pytest.approx(0.7, abs=0.01)

def test_medium_grader_single_tool():
    grader = MediumGrader()
    action = MockAction(answer="key_fact found.", citations=["source_a"])
    state = MockState(
        task_type="medium",
        action_history=["READ_DOC", "SUBMIT_ANSWER"],
        key_facts=["key_fact"],
        sources=["source_a"],
    )
    score = grader.grade(action, state)
    # diversity=0.5 → 0.5 + 0.2 + 0.15 = 0.85
    assert score == pytest.approx(0.85, abs=0.01)


# ──────────────────────────────────────────────────────────────────────────────
# HardGrader
# ──────────────────────────────────────────────────────────────────────────────

def test_hard_grader_good_query(web_search):
    grader = HardGrader(web_search)
    action = MockAction(
        answer="The default timeout is 60 seconds.",
        citations=["https://example.com"],
    )
    state = MockState(
        task_type="hard",
        action_details=[{"action_type": "WEB_SEARCH", "query": "fastapi default timeout"}],
        key_facts=["60 seconds"],
        sources=["https://example.com"],
    )
    score = grader.grade(action, state)
    assert score > 0.5, f"Good query should score >0.5, got {score}"

def test_hard_grader_garbage_query(web_search):
    grader = HardGrader(web_search)
    action = MockAction(
        answer="The timeout is 60 seconds.",
        citations=["https://example.com"],
    )
    state = MockState(
        task_type="hard",
        action_details=[{"action_type": "WEB_SEARCH", "query": "abc xyz random garbage"}],
        key_facts=["60 seconds"],
        sources=["https://example.com"],
    )
    score = grader.grade(action, state)
    # query quality ≈ 0, citation density = 1.0 → 0.4*1.0 + 0.3*0 + 0.3*1.0 = 0.7
    assert score <= 0.8

def test_hard_grader_no_search(web_search):
    grader = HardGrader(web_search)
    action = MockAction(answer="60 seconds.", citations=[])
    state = MockState(
        task_type="hard",
        action_details=[],
        key_facts=["60 seconds"],
        sources=["src"],
    )
    score = grader.grade(action, state)
    # avg_quality=0, density=0, fact=1.0 → 0.4
    assert score == pytest.approx(0.4, abs=0.01)


# ──────────────────────────────────────────────────────────────────────────────
# GraderRouter
# ──────────────────────────────────────────────────────────────────────────────

def test_router_dispatches_correctly(web_search):
    router = GraderRouter(web_search)

    easy_action = MockAction(answer="60 seconds", citations=["doc"])
    easy_state = MockState(task_type="easy", key_facts=["60 seconds"], sources=["doc"])
    assert 0.0 <= router.grade(easy_action, easy_state) <= 1.0

    hard_action = MockAction(answer="60 seconds", citations=[])
    hard_state = MockState(task_type="hard", key_facts=["60 seconds"])
    assert 0.0 <= router.grade(hard_action, hard_state) <= 1.0

def test_router_unknown_type(web_search):
    router = GraderRouter(web_search)
    action = MockAction(answer="anything")
    state = MockState(task_type="unknown")
    assert router.grade(action, state) == 0.0

def test_all_graders_output_in_range(web_search):
    router = GraderRouter(web_search)
    for task_type in ["easy", "medium", "hard"]:
        action = MockAction(answer="some answer containing fact", citations=["src"])
        state = MockState(
            task_type=task_type,
            action_history=["READ_DOC", "WEB_SEARCH"],
            action_details=[
                {"action_type": "WEB_SEARCH", "query": "fastapi timeout"}
            ],
            key_facts=["fact"],
            sources=["src"],
        )
        score = router.grade(action, state)
        assert 0.0 <= score <= 1.0, f"{task_type} grader out of range: {score}"
