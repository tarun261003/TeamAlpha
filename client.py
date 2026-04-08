# truth_seeker_env/client.py
"""
HTTP client for TruthSeekerEnvironment.

Uses plain `requests` — no dependency on openenv_core SDK.

The actual OpenEnv SDK uses WebSockets (openenv.core.env_client.EnvClient).
We own both the server (FastAPI/HTTP) and the client, so we talk to our
FastAPI endpoints directly with HTTP POST calls. This is simpler, portable,
and avoids the SDK's WebSocket complexity for local testing.

Server endpoints:
  POST /reset        → { "observation": {...} }
  POST /step         → { "observation": {...}, "reward": float, "done": bool }
"""

import json
import requests
from dataclasses import dataclass
from typing import Optional

from models import TruthSeekerAction, TruthSeekerObservation, TruthSeekerState, StepResult


class TruthSeekerEnv:
    """
    HTTP client for the TruthSeekerEnvironment server.

    Usage:
        env = TruthSeekerEnv(base_url="http://localhost:8000")
        obs = env.reset()                         # random task
        obs = env.reset(task_type="easy")         # specific tier
        result = env.step(TruthSeekerAction(
            action_type="READ_DOC",
            query="fastapi timeout configuration",
        ))
        print(result.observation.retrieved_context)
        print(result.reward)
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, task_type: Optional[str] = None) -> TruthSeekerObservation:
        """Reset, optionally filtered to a difficulty tier."""
        params = {}
        if task_type is not None:
            params["task_type"] = task_type

        response = requests.post(
            f"{self.base_url}/reset",
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return self._parse_observation(payload)

    def step(self, action: TruthSeekerAction) -> StepResult:
        """Send an action and return the result."""
        payload = {
            "action_type": action.action_type,
            "query":       action.query,
            "answer":      action.answer,
            "reasoning":   action.reasoning,
            "citations":   action.citations,
        }
        response = requests.post(
            f"{self.base_url}/step",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return self._parse_result(data)

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_observation(self, payload: dict) -> TruthSeekerObservation:
        """Parse /reset or inner observation dict into TruthSeekerObservation."""
        # /reset returns { "observation": {...} }
        obs_data = payload.get("observation", payload)
        if isinstance(obs_data, dict) and "observation" in obs_data:
            obs_data = obs_data["observation"]
        if not isinstance(obs_data, dict):
            obs_data = {}

        return TruthSeekerObservation(
            task_id=obs_data.get("task_id", ""),
            task_type=obs_data.get("task_type", ""),
            question=obs_data.get("question", ""),
            available_tools=obs_data.get("available_tools", []),
            retrieved_context=obs_data.get("retrieved_context", ""),
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data.get("max_steps", 10),
            instructions=obs_data.get("instructions", ""),
        )

    def _parse_result(self, payload: dict) -> StepResult:
        """Parse /step response into StepResult."""
        obs = self._parse_observation(payload)
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )
