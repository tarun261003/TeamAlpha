# truth_seeker_env/server/app.py
"""
FastAPI server entry point for Truth-Seeker's Sandbox.

Architecture note:
  The OpenEnv SDK uses WebSockets internally to manage Docker container
  lifecycles, but our *server* only needs to speak plain HTTP.
  The evaluator pulls our Docker image, starts it, and calls:
    POST /reset  → { "observation": {...} }
    POST /step   → { "observation": {...}, "reward": float, "done": bool }
    GET  /state  → TruthSeekerState dict

  We build the server with plain FastAPI + uvicorn — no openenv_core dependency.

Concurrency: single-instance (one active episode at a time).
For parallel agents at scale, wrap with per-session env instances.
"""

import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import Optional, List
from dataclasses import asdict

from .environment import TruthSeekerEnvironment
from models import TruthSeekerAction, TruthSeekerObservation

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Config from environment variables
# ──────────────────────────────────────────────────────────────────────────────

DATASET_PATH     = os.environ.get("DATASET_PATH",    "dataset/tasks.jsonl")
QDRANT_URL       = os.environ.get("QDRANT_URL",      "https://503671a8-0c76-4019-ab6a-3a238095f135.eu-west-2-0.aws.cloud.qdrant.io:6333")
QDRANT_API_KEY   = os.environ.get("QDRANT_API_KEY",  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6M2QwN2RkYTEtN2Y0MS00YTkxLTgwZTAtZmQ4MWM5ODRjOWRkIn0.qXCI9iSC-mbzUjFDXBbQnNCZQLbvVU3Asvf5798VWJY")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "truth-seeker")
EMBEDDING_MODEL  = os.environ.get("EMBEDDING_MODEL",  "all-MiniLM-L6-v2")
SERPER_API_KEY   = os.environ.get("SERPER_API_KEY",   "")

# ──────────────────────────────────────────────────────────────────────────────
# Environment instance
# ──────────────────────────────────────────────────────────────────────────────

env = TruthSeekerEnvironment(
    dataset_path=DATASET_PATH,
    qdrant_url=QDRANT_URL,
    qdrant_api_key=QDRANT_API_KEY,
    qdrant_collection=QDRANT_COLLECTION,
    embedding_model=EMBEDDING_MODEL,
    serper_api_key=SERPER_API_KEY,
)

# ──────────────────────────────────────────────────────────────────────────────
# Request / Response models (Pydantic, for FastAPI validation)
# ──────────────────────────────────────────────────────────────────────────────

class StepRequest(BaseModel):
    action_type: str
    query: str = ""
    answer: str = ""
    reasoning: str = ""
    citations: List[str] = []


class ObservationResponse(BaseModel):
    task_id: str
    task_type: str
    question: str
    available_tools: List[str]
    retrieved_context: str
    step_number: int
    max_steps: int
    instructions: str


class StepResponse(BaseModel):
    observation: ObservationResponse
    reward: Optional[float]
    done: bool


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Truth-Seeker's Sandbox",
    description=(
        "A multi-turn RAG reasoning environment for RL training. "
        "Agents retrieve documents and submit answers across three difficulty tiers."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _obs_to_dict(obs: TruthSeekerObservation) -> dict:
    return {
        "task_id": obs.task_id,
        "task_type": obs.task_type,
        "question": obs.question,
        "available_tools": obs.available_tools,
        "retrieved_context": obs.retrieved_context,
        "step_number": obs.step_number,
        "max_steps": obs.max_steps,
        "instructions": obs.instructions,
    }


@app.get("/")
def read_root():
    """Redirect home page visitors to the OpenAPI documentation."""
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"status": "healthy", "environment": "truth_seeker_env"}


@app.post("/reset")
def reset(task_type: Optional[str] = None):
    """Start a new episode. Optionally filter by task_type: easy|medium|hard."""
    obs = env.reset(task_type=task_type)
    return {"observation": _obs_to_dict(obs)}


@app.post("/step")
def step(request: StepRequest):
    """Execute one action in the current episode."""
    action = TruthSeekerAction(
        action_type=request.action_type,
        query=request.query,
        answer=request.answer,
        reasoning=request.reasoning,
        citations=request.citations,
    )
    result = env.step(action)
    return {
        "observation": _obs_to_dict(result.observation),
        "reward": result.reward,
        "done": result.done,
    }


@app.get("/state")
def state():
    """Return the full internal episode state (for debuggers and evaluators)."""
    s = env.state
    return asdict(s)


# ──────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
