"""
Baseline inference script for Truth-Seeker's Sandbox.

Mandatory structured logging format (required by OpenEnv submission):
  [START]  — emitted once at the beginning of each episode
  [STEP]   — emitted once per step (action + observation + reward)
  [END]    — emitted once at episode termination

Usage:
    API_BASE_URL=http://localhost:8000 \
    MODEL_NAME=llama-3.3-70b-versatile \
    GROQ_API_KEY=gsk-... \
    python inference.py

Environment variables:
    API_BASE_URL   — URL of the running truth_seeker_env server
    MODEL_NAME     — Groq-compatible model name (e.g., llama-3.3-70b-versatile)
    GROQ_API_KEY   — Your Groq API key
    MAX_EPISODES   — Number of episodes to run (default: 3, one per task type)
"""

import os
import json
import logging
import sys
from openai import OpenAI

from client import TruthSeekerEnv
from models import TruthSeekerAction

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

# Required by OpenEnv Submission Rules
OPENAI_API_KEY = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY") or HF_TOKEN or "dummy-key-for-local"
MAX_EPISODES = int(os.environ.get("MAX_EPISODES", "3"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# LLM Call
# ──────────────────────────────────────────────────────────────────────────────

def call_llm(client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Action Parser
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a technical documentation researcher.
You must answer questions by using the provided tools: READ_DOC, WEB_SEARCH, and SUBMIT_ANSWER.

Always respond with a JSON object in exactly this format:
{
  "action_type": "READ_DOC" | "WEB_SEARCH" | "SUBMIT_ANSWER",
  "query": "<search query if using READ_DOC or WEB_SEARCH, else empty string>",
  "answer": "<complete answer if SUBMIT_ANSWER, else empty string>",
  "reasoning": "<your chain-of-thought reasoning>",
  "citations": ["<source1>", "<source2>"]
}

Rules:
- Use READ_DOC to search internal documentation.
- Use WEB_SEARCH to search the web.
- If you find a specific fact or number that directly answers the question from a retrieved snippet, CALL SUBMIT_ANSWER IMMEDIATELY instead of continuing to search. Do not over-research.
- Always include non-empty reasoning in every response.
- citations should list document IDs or URLs you used.
- IMPORTANT ANTI-LOOP RULE: If a search query returns no results or unhelpful results, DO NOT repeat the exact same query.
- If you are stuck, broaden your keywords, use shorter search phrases (e.g., just 'Python version' instead of a full sentence), or try entirely different search terms.
- If you are failing to find information with one tool, try the other (if both are available).
"""


def parse_action(llm_response: str, available_tools: list) -> TruthSeekerAction:
    """Parse LLM JSON response into TruthSeekerAction. Falls back gracefully on errors."""
    try:
        # Strip markdown code fences if present
        text = llm_response.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

        data = json.loads(text)
        action_type = data.get("action_type", "SUBMIT_ANSWER")

        # Enforce available tools
        if action_type not in available_tools:
            logger.warning(f"LLM returned unavailable tool '{action_type}', defaulting to SUBMIT_ANSWER")
            action_type = "SUBMIT_ANSWER"

        return TruthSeekerAction(
            action_type=action_type,
            query=data.get("query", ""),
            answer=data.get("answer", "I could not determine the answer."),
            reasoning=data.get("reasoning", "No reasoning provided."),
            citations=data.get("citations", []),
        )
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Failed to parse LLM response: {e}. Submitting fallback answer.")
        return TruthSeekerAction(
            action_type="SUBMIT_ANSWER",
            answer="Unable to determine the answer due to a parsing error.",
            reasoning="The model response could not be parsed as valid JSON.",
            citations=[],
        )


# ──────────────────────────────────────────────────────────────────────────────
# Episode Runner
# ──────────────────────────────────────────────────────────────────────────────

def run_episode(env: TruthSeekerEnv, llm: OpenAI, task_type: str = None) -> float:
    """Run a single episode and return the final reward."""

    obs = env.reset(task_type=task_type)

    print(f"[START] task={obs.task_id} env=truth_seeker_env model={MODEL_NAME}")

    step_num = 0
    total_reward = 0.0
    past_queries = []
    episode_rewards = []

    while True:
        # Build user prompt from current observation
        past_queries_str = "\n".join(f"- {q}" for q in past_queries) if past_queries else "None"
        
        user_prompt = f"""Question: {obs.question}

Available tools: {obs.available_tools}
Instructions: {obs.instructions}
Step: {obs.step_number}/{obs.max_steps}

Past search queries you already tried (DO NOT REPEAT THESE):
{past_queries_str}

Retrieved context so far:
{obs.retrieved_context or "(no context retrieved yet)"}

What action will you take next? Respond with a JSON object."""

        llm_response = call_llm(llm, SYSTEM_PROMPT, user_prompt)
        action = parse_action(llm_response, obs.available_tools)
        
        if action.query:
            past_queries.append(action.query)

        result = env.step(action)
        step_num += 1
        total_reward += result.reward or 0.0

        episode_rewards.append(result.reward or 0.0)

        print(
            f"[STEP] step={step_num} "
            f"action={action.action_type} "
            f"reward={(result.reward or 0.0):.2f} "
            f"done={str(result.done).lower()} "
            f"error=null",
            flush=True
        )

        if result.done:
            final_score = max(0.01, min(0.99, result.reward or 0.0))
            print(
                f"[END] success={str(final_score >= 0.5).lower()} "
                f"steps={step_num} "
                f"score={final_score:.2f} "
                f"rewards={','.join(f'{r:.2f}' for r in episode_rewards)}",
                flush=True
            )
            return result.reward

        obs = result.observation


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    if OPENAI_API_KEY == "dummy-key-for-local" and not API_BASE_URL:
        logger.warning("No API key or API_BASE_URL provided. If you get auth errors, set OPENAI_API_KEY, HF_TOKEN, or use Ollama (API_BASE_URL='http://localhost:11434/v1').")

    llm = OpenAI(api_key=OPENAI_API_KEY, base_url=API_BASE_URL)
    env = TruthSeekerEnv(base_url=ENV_BASE_URL)

    task_types = ["easy", "medium", "hard"]
    results = []

    for i, task_type in enumerate(task_types[:MAX_EPISODES]):
        logger.info(f"=== Episode {i + 1}/{MAX_EPISODES} — task_type={task_type} ===")
        try:
            reward = run_episode(env, llm, task_type=task_type)
            results.append({"task_type": task_type, "reward": reward})
        except Exception as e:
            logger.error(f"Episode failed: {e}", exc_info=True)
            results.append({"task_type": task_type, "reward": 0.0, "error": str(e)})

    print("\n=== Inference Summary ===")
    for r in results:
        print(f"  {r['task_type']:8s}  reward={r['reward']:.4f}")

    avg = sum(r["reward"] for r in results) / len(results) if results else 0.0
    print(f"\n  Average reward: {avg:.4f}")


if __name__ == "__main__":
    main()
