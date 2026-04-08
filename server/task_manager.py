# truth_seeker_env/server/task_manager.py
"""
Task manager: loads tasks.jsonl and serves tasks by difficulty tier.

Supports:
  - Random shuffled iteration (default, mirrors dipg-gym)
  - Filtering by task_type for the inference script and reset(task_type=...)
"""

import json
import random
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class TaskManager:
    """
    Loads tasks from a JSONL file and serves one at a time.

    tasks.jsonl schema (required keys per line):
      task_id, task_type, question, ground_truth,
      internal_docs, web_results, max_steps
    """

    VALID_TASK_TYPES = {"easy", "medium", "hard"}

    def __init__(self, dataset_path: str):
        self._all_tasks = self._load(dataset_path)
        if not self._all_tasks:
            raise ValueError(f"No valid tasks loaded from: {dataset_path}")

        self._filtered_tasks = list(self._all_tasks)
        self._shuffled_indices = list(range(len(self._filtered_tasks)))
        random.shuffle(self._shuffled_indices)
        self._index = 0
        logger.info(f"TaskManager loaded {len(self._all_tasks)} tasks from {dataset_path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_task_type_filter(self, task_type: Optional[str]):
        """
        Restrict which tasks are served. Pass None to clear filter.
        Called by environment.reset(task_type=...) and inference.py.
        """
        if task_type is None:
            self._filtered_tasks = list(self._all_tasks)
        else:
            if task_type not in self.VALID_TASK_TYPES:
                raise ValueError(
                    f"Unknown task_type '{task_type}'. Valid: {self.VALID_TASK_TYPES}"
                )
            self._filtered_tasks = [
                t for t in self._all_tasks if t["task_type"] == task_type
            ]
            if not self._filtered_tasks:
                raise ValueError(
                    f"No tasks with task_type='{task_type}' in dataset."
                )

        self._shuffled_indices = list(range(len(self._filtered_tasks)))
        random.shuffle(self._shuffled_indices)
        self._index = 0
        logger.info(
            f"TaskManager filter='{task_type}': {len(self._filtered_tasks)} tasks available"
        )

    def get_next_task(self) -> dict:
        """Cycle through shuffled tasks, reshuffle on exhaustion."""
        if self._index >= len(self._shuffled_indices):
            random.shuffle(self._shuffled_indices)
            self._index = 0

        idx = self._shuffled_indices[self._index]
        self._index += 1
        task = self._filtered_tasks[idx]
        logger.debug(f"Serving task {task['task_id']} ({task['task_type']})")
        return task

    def task_count(self, task_type: Optional[str] = None) -> int:
        if task_type is None:
            return len(self._all_tasks)
        return sum(1 for t in self._all_tasks if t["task_type"] == task_type)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _load(path: str) -> list:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        tasks = []
        required_keys = {
            "task_id", "task_type", "question",
            "ground_truth", "internal_docs", "web_results", "max_steps",
        }
        with open(p, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    task = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed JSON line {line_no}: {e}")
                    continue

                missing = required_keys - set(task.keys())
                if missing:
                    logger.warning(f"Skipping line {line_no} — missing keys: {missing}")
                    continue

                tasks.append(task)

        return tasks
