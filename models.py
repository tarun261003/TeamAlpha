# truth_seeker_env/models.py
"""
Data models for Truth-Seeker's Sandbox.

Uses plain Pydantic BaseModel — no dependency on openenv_core internals.
The openenv-core SDK uses WebSockets and a different module layout than
what the dipg-gym reference assumed. We own both server and client, so
we use FastAPI + requests directly.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TruthSeekerAction:
    """
    Multi-turn action the agent can take each step.

    action_type options:
      "READ_DOC"      — Query the Pinecone vector DB (Easy + Medium tasks)
      "WEB_SEARCH"    — Query the web search proxy  (Medium + Hard tasks)
      "SUBMIT_ANSWER" — Terminal: submit final answer and end episode
    """
    action_type: str = ""
    query: str = ""                               # Search query (READ_DOC / WEB_SEARCH)
    answer: str = ""                              # Final answer (SUBMIT_ANSWER only)
    reasoning: str = ""                           # Agent's chain-of-thought trace
    citations: list = field(default_factory=list) # Source references for grading


@dataclass
class TruthSeekerObservation:
    """
    What the agent sees after each action.

    retrieved_context accumulates across steps via a sliding window
    (capped at MAX_CONTEXT_CHARS=4000). Each chunk is labeled with its
    source: [READ_DOC] or [WEB_SEARCH].
    """
    task_id: str = ""
    task_type: str = ""           # "easy" | "medium" | "hard"
    question: str = ""
    available_tools: list = field(default_factory=list)
    retrieved_context: str = ""   # Sliding window of retrieved chunks
    step_number: int = 0
    max_steps: int = 10
    instructions: str = ""


@dataclass
class TruthSeekerState:
    """
    Internal environment state — hidden from agent, used by grader and reward calc.

    retrieved_chunks — full history for grading (not truncated).
    context_window  — agent-visible sliding window (truncated at MAX_CONTEXT_CHARS).
    action_details  — rich per-step records {action_type, query} for Hard grader.
    accumulated_step_reward — dense RL training signal (NOT added to terminal score).
    """
    task_id: str = ""
    task_type: str = ""
    question: str = ""
    ground_truth_answer: str = ""
    ground_truth_key_facts: list = field(default_factory=list)
    ground_truth_sources: list = field(default_factory=list)
    internal_docs: dict = field(default_factory=dict)       # doc_id -> content
    web_results: dict = field(default_factory=dict)          # proxy data dict
    step_count: int = 0
    max_steps: int = 10
    action_history: list = field(default_factory=list)       # action_type strings
    action_details: list = field(default_factory=list)       # {action_type, query} per step
    retrieved_chunks: list = field(default_factory=list)     # full history for grading
    context_window: str = ""                                 # truncated agent-visible ctx
    accumulated_step_reward: float = 0.0                     # training signal only
    episode_done: bool = False


@dataclass
class StepResult:
    """Returned by TruthSeekerEnv.step() and .reset()."""
    observation: TruthSeekerObservation = field(default_factory=TruthSeekerObservation)
    reward: Optional[float] = None
    done: bool = False
