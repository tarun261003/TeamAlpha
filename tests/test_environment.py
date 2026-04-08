"""
End-to-end environment tests — runs without Pinecone or any external service.
Uses the local keyword-overlap fallback and the task dataset from tasks.jsonl.

Run with: pytest tests/test_environment.py -v
"""

import os
import sys
import pytest

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.environment import TruthSeekerEnvironment, MAX_CONTEXT_CHARS
from models import TruthSeekerAction

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset", "tasks.jsonl",
)


@pytest.fixture(scope="module")
def env():
    """Single env instance for all tests — no Pinecone key, uses local fallback."""
    return TruthSeekerEnvironment(
        dataset_path=DATASET_PATH,
        qdrant_url="",         # forces local fallback (no Qdrant server)
        qdrant_api_key="",     # forces local fallback
    )


# ──────────────────────────────────────────────────────────────────────────────
# reset()
# ──────────────────────────────────────────────────────────────────────────────

def test_reset_returns_valid_observation(env):
    obs = env.reset()
    assert obs.task_id != ""
    assert obs.task_type in {"easy", "medium", "hard"}
    assert obs.question != ""
    assert len(obs.available_tools) >= 1
    assert obs.step_number == 0
    assert obs.max_steps > 0

def test_reset_easy_sets_correct_tools(env):
    obs = env.reset(task_type="easy")
    assert obs.task_type == "easy"
    assert "READ_DOC" in obs.available_tools
    assert "WEB_SEARCH" not in obs.available_tools

def test_reset_medium_sets_correct_tools(env):
    obs = env.reset(task_type="medium")
    assert obs.task_type == "medium"
    assert "READ_DOC" in obs.available_tools
    assert "WEB_SEARCH" in obs.available_tools

def test_reset_hard_sets_correct_tools(env):
    obs = env.reset(task_type="hard")
    assert obs.task_type == "hard"
    assert "WEB_SEARCH" in obs.available_tools
    assert "READ_DOC" not in obs.available_tools

def test_reset_clears_state(env):
    env.reset(task_type="easy")
    # Do a step to dirty the state
    env.step(TruthSeekerAction(action_type="READ_DOC", query="test"))
    # Reset should clear everything
    obs = env.reset()
    assert obs.step_number == 0
    assert obs.retrieved_context == ""
    assert env.state.step_count == 0
    assert env.state.accumulated_step_reward == 0.0
    assert env.state.action_history == []
    assert env.state.retrieved_chunks == []


# ──────────────────────────────────────────────────────────────────────────────
# step() — tool actions
# ──────────────────────────────────────────────────────────────────────────────

def test_step_read_doc_is_nonterminal(env):
    env.reset(task_type="easy")
    action = TruthSeekerAction(action_type="READ_DOC", query="timeout")
    result = env.step(action)
    assert result.done is False
    assert result.reward is not None
    assert result.observation.step_number == 1

def test_step_web_search_is_nonterminal(env):
    env.reset(task_type="hard")
    action = TruthSeekerAction(action_type="WEB_SEARCH", query="pinecone sla uptime")
    result = env.step(action)
    assert result.done is False
    assert result.observation.step_number == 1

def test_step_builds_context(env):
    env.reset(task_type="easy")
    env.step(TruthSeekerAction(action_type="READ_DOC", query="timeout"))
    obs = env.reset(task_type="easy")   # get fresh obs
    env.step(TruthSeekerAction(action_type="READ_DOC", query="request timeout"))
    # Context should be populated
    assert env.state.retrieved_chunks != []

def test_step_context_never_exceeds_max(env):
    env.reset(task_type="medium")
    for _ in range(6):
        env.step(TruthSeekerAction(action_type="READ_DOC", query="timeout default request"))
        if env.state.episode_done:
            break
    assert len(env.state.context_window) <= MAX_CONTEXT_CHARS


# ──────────────────────────────────────────────────────────────────────────────
# step() — SUBMIT_ANSWER
# ──────────────────────────────────────────────────────────────────────────────

def test_submit_answer_terminates(env):
    env.reset(task_type="easy")
    action = TruthSeekerAction(
        action_type="SUBMIT_ANSWER",
        answer="The timeout is 60 seconds.",
        reasoning="I found this in the docs.",
        citations=["fastapi-internals-v0104"],
    )
    result = env.step(action)
    assert result.done is True
    assert 0.0 <= result.reward <= 1.0

def test_submit_without_search_has_lower_score(env):
    """Submitting immediately (no retrieval) should score lower than after searching."""
    env.reset(task_type="easy")
    action = TruthSeekerAction(
        action_type="SUBMIT_ANSWER",
        answer="I don't know the answer.",
        reasoning="No retrieval done.",
        citations=[],
    )
    result = env.step(action)
    assert result.done is True
    assert result.reward < 0.5

def test_submit_with_correct_facts_scores_higher(env):
    env.reset(task_type="easy")
    # Read ground truth from state so the test works regardless of which easy task is selected
    gt_answer = env.state.ground_truth_answer
    gt_sources = env.state.ground_truth_sources
    gt_question = env.state.question
    # First retrieve using terms from the question
    env.step(TruthSeekerAction(action_type="READ_DOC", query=gt_question))
    # Then submit with the ground truth answer and sources
    action = TruthSeekerAction(
        action_type="SUBMIT_ANSWER",
        answer=gt_answer,
        reasoning="I found this in the internal docs.",
        citations=gt_sources,
    )
    result = env.step(action)
    assert result.done is True
    assert result.reward >= 0.5

def test_reward_in_range_after_submit(env):
    env.reset()
    result = env.step(TruthSeekerAction(
        action_type="SUBMIT_ANSWER",
        answer="Some answer.",
        reasoning="Some reasoning.",
    ))
    assert 0.0 <= result.reward <= 1.0


# ──────────────────────────────────────────────────────────────────────────────
# Step limit
# ──────────────────────────────────────────────────────────────────────────────

def test_step_limit_triggers_done(env):
    obs = env.reset(task_type="easy")
    max_steps = obs.max_steps
    done = False
    for i in range(max_steps + 5):  # extra steps to ensure limit fires
        if done:
            break
        result = env.step(TruthSeekerAction(action_type="READ_DOC", query="test"))
        done = result.done
    assert done is True


# ──────────────────────────────────────────────────────────────────────────────
# Post-episode guard
# ──────────────────────────────────────────────────────────────────────────────

def test_no_action_after_done(env):
    env.reset(task_type="easy")
    env.step(TruthSeekerAction(
        action_type="SUBMIT_ANSWER",
        answer="done",
        reasoning="done",
    ))
    # Episode is now done — any further step should be a no-op
    result = env.step(TruthSeekerAction(action_type="READ_DOC", query="test"))
    assert result.done is True
    assert result.reward == 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Action history tracking
# ──────────────────────────────────────────────────────────────────────────────

def test_action_history_recorded(env):
    env.reset(task_type="medium")
    env.step(TruthSeekerAction(action_type="READ_DOC", query="q1"))
    env.step(TruthSeekerAction(action_type="WEB_SEARCH", query="q2"))
    assert "READ_DOC" in env.state.action_history
    assert "WEB_SEARCH" in env.state.action_history

def test_action_details_recorded(env):
    env.reset(task_type="medium")
    env.step(TruthSeekerAction(action_type="WEB_SEARCH", query="pinecone regions"))
    details = env.state.action_details
    assert len(details) == 1
    assert details[0]["action_type"] == "WEB_SEARCH"
    assert details[0]["query"] == "pinecone regions"
