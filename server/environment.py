# truth_seeker_env/server/environment.py
"""
Core TruthSeekerEnvironment — the heart of the RAG environment.

Multi-turn episode flow:
  1. reset()             → load task, init state, return initial observation
  2. step(READ_DOC)      → query Pinecone RAG, accumulate context, per-step reward
  3. step(WEB_SEARCH)    → query web proxy, accumulate context, per-step reward
  4. step(SUBMIT_ANSWER) → grade answer, apply format gate, done=True

Context accumulation:
  - agent sees a SLIDING WINDOW capped at MAX_CONTEXT_CHARS (FIFO truncation)
  - full retrieved_chunks list preserved in State for grader evaluation

Grader vs Reward separation:
  - GraderRouter.grade() produces the submission-facing 0.0–1.0 score
  - RewardCalculator.tool_use_reward() produces per-step RL training signals
  - compute_terminal_reward() applies the format gate to the grader score

No openenv_core dependency — this is a plain FastAPI server.
"""

import logging
from typing import Optional
from dataclasses import asdict

from models import TruthSeekerAction, TruthSeekerObservation, TruthSeekerState, StepResult
from .task_manager import TaskManager
from .rag_retriever import QdrantRAGRetriever
from .web_search import WebSearchProxy
from .reward import RewardCalculator
from .grader import GraderRouter

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 4000
CONTEXT_SEPARATOR = "\n\n--- [Retrieved Chunk] ---\n\n"

_TOOL_INSTRUCTIONS = {
    "easy": (
        "Use READ_DOC to search the internal documentation. "
        "The answer exists in the docs. When ready, call SUBMIT_ANSWER."
    ),
    "medium": (
        "Use READ_DOC to search the internal documentation and WEB_SEARCH "
        "to fill any gaps. Combine both sources and call SUBMIT_ANSWER when ready."
    ),
    "hard": (
        "Documentation has nothing relevant. Use WEB_SEARCH iteratively to "
        "discover the answer from the web. Call SUBMIT_ANSWER when ready."
    ),
}

_AVAILABLE_TOOLS = {
    "easy":   ["READ_DOC", "SUBMIT_ANSWER"],
    "medium": ["READ_DOC", "WEB_SEARCH", "SUBMIT_ANSWER"],
    "hard":   ["WEB_SEARCH", "SUBMIT_ANSWER"],
}


class TruthSeekerEnvironment:

    def __init__(
        self,
        dataset_path: str,
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: str = "",
        qdrant_collection: str = "truth-seeker",
        embedding_model: str = "all-MiniLM-L6-v2",
        serper_api_key: str = "",
    ):
        self.task_manager = TaskManager(dataset_path)
        self.rag_retriever = QdrantRAGRetriever(
            url=qdrant_url,
            api_key=qdrant_api_key,
            collection_name=qdrant_collection,
            embedding_model=embedding_model,
        )
        self.web_search = WebSearchProxy(serper_api_key=serper_api_key)
        self.reward_calc = RewardCalculator()
        self.grader = GraderRouter(self.web_search)

        self._state = TruthSeekerState()  # blank; replaced on reset()

    # ──────────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────────

    def reset(self, task_type: Optional[str] = None) -> TruthSeekerObservation:
        """
        Load a new task, initialise state, return initial observation.
        """
        if task_type is not None:
            self.task_manager.set_task_type_filter(task_type)

        task = self.task_manager.get_next_task()

        self._state = TruthSeekerState(
            task_id=task["task_id"],
            task_type=task["task_type"],
            question=task["question"],
            ground_truth_answer=task["ground_truth"]["answer"],
            ground_truth_key_facts=task["ground_truth"].get("key_facts", []),
            ground_truth_sources=task["ground_truth"].get("sources", []),
            internal_docs=task.get("internal_docs", {}),
            web_results=task.get("web_results", {}),
            max_steps=task.get("max_steps", 10),
            step_count=0,
            action_history=[],
            action_details=[],
            retrieved_chunks=[],
            context_window="",
            accumulated_step_reward=0.0,
            episode_done=False,
        )

        self.rag_retriever.load_episode_docs(task.get("internal_docs", {}))
        self.web_search.load_episode_data(task.get("web_results", {}))

        logger.info(
            f"Episode reset: task_id={task['task_id']} "
            f"type={task['task_type']} max_steps={self._state.max_steps}"
        )
        return self._build_observation()

    def step(self, action: TruthSeekerAction) -> StepResult:
        """
        Execute one action, return StepResult(observation, reward, done).

        Routing:
          SUBMIT_ANSWER → grade via GraderRouter, apply format gate, done=True
          READ_DOC      → Pinecone RAG, accumulate context, per-step reward
          WEB_SEARCH    → web proxy, accumulate context, per-step reward
          unknown       → UNKNOWN_ACTION_PENALTY, non-terminal
        """
        # Safeguard: reject actions after episode end
        if self._state.episode_done:
            logger.warning("step() called after episode is done — ignoring.")
            return StepResult(
                observation=self._build_observation(),
                reward=0.0,
                done=True,
            )

        self._state.step_count += 1
        self._state.action_history.append(action.action_type)

        # ── Terminal: SUBMIT_ANSWER ────────────────────────────────────
        if action.action_type == "SUBMIT_ANSWER":
            grader_score = self.grader.grade(action, self._state)
            final_reward = self.reward_calc.compute_terminal_reward(
                grader_score, action
            )
            self._state.episode_done = True
            logger.info(
                f"SUBMIT_ANSWER: task={self._state.task_id} "
                f"grader={grader_score:.3f} final={final_reward:.3f} "
                f"steps={self._state.step_count}"
            )
            return StepResult(
                observation=self._build_terminal_observation(),
                reward=final_reward,
                done=True,
            )

        # ── Tool actions ───────────────────────────────────────────────
        elif action.action_type == "READ_DOC":
            new_chunk = self.rag_retriever.search(action.query)
            self._accumulate_context(new_chunk, source="READ_DOC")
            step_reward = self.reward_calc.tool_use_reward(
                action, self._state, new_chunk
            )

        elif action.action_type == "WEB_SEARCH":
            new_chunk = self.web_search.search(action.query)
            self._accumulate_context(new_chunk, source="WEB_SEARCH")
            step_reward = self.reward_calc.tool_use_reward(
                action, self._state, new_chunk
            )

        else:
            logger.warning(f"Unknown action_type: '{action.action_type}'")
            step_reward = self.reward_calc.UNKNOWN_ACTION_PENALTY
            new_chunk = ""

        # Record rich action details for Hard grader's query quality scoring
        self._state.action_details.append({
            "action_type": action.action_type,
            "query": action.query,
        })
        # accumulated_step_reward = dense RL training signal for the trainer;
        # intentionally NOT used in compute_terminal_reward() (see reward.py docstring)
        self._state.accumulated_step_reward += step_reward

        # ── Step limit check ───────────────────────────────────────────
        if self._state.step_count >= self._state.max_steps:
            timeout = self.reward_calc.timeout_penalty()
            self._state.accumulated_step_reward += timeout
            self._state.episode_done = True
            logger.info(
                f"Step limit reached: task={self._state.task_id} "
                f"steps={self._state.step_count}"
            )
            return StepResult(
                observation=self._build_observation(),
                reward=step_reward + timeout,
                done=True,
            )

        return StepResult(
            observation=self._build_observation(),
            reward=step_reward,
            done=False,
        )

    @property
    def state(self) -> TruthSeekerState:
        return self._state

    def set_state(self, state: TruthSeekerState) -> TruthSeekerState:
        self._state = state
        return self._state

    def close(self):
        pass

    # ──────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────

    def _accumulate_context(self, new_chunk: str, source: str):
        """
        Append labeled chunk to full history and rebuild the agent-visible
        sliding window (FIFO, capped at MAX_CONTEXT_CHARS).
        """
        labeled = f"[{source}] {new_chunk}"
        self._state.retrieved_chunks.append(labeled)

        window_parts = []
        total = 0
        for chunk in reversed(self._state.retrieved_chunks):
            if total + len(chunk) > MAX_CONTEXT_CHARS:
                break
            window_parts.insert(0, chunk)
            total += len(chunk)

        self._state.context_window = CONTEXT_SEPARATOR.join(window_parts)

    def _build_observation(self) -> TruthSeekerObservation:
        return TruthSeekerObservation(
            task_id=self._state.task_id,
            task_type=self._state.task_type,
            question=self._state.question,
            available_tools=_AVAILABLE_TOOLS.get(self._state.task_type, []),
            retrieved_context=self._state.context_window,
            step_number=self._state.step_count,
            max_steps=self._state.max_steps,
            instructions=_TOOL_INSTRUCTIONS.get(
                self._state.task_type,
                "Use available tools and SUBMIT_ANSWER when ready."
            ),
        )

    def _build_terminal_observation(self) -> TruthSeekerObservation:
        obs = self._build_observation()
        obs.available_tools = []
        obs.instructions = "Episode complete."
        return obs
