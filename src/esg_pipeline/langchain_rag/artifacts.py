from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from .results import LangchainRAGTaskResult


class ArtifactWriter:
    """Helper for persisting prompts, contexts, responses, and summary tables."""

    def __init__(self, experiment_dir: Path) -> None:
        self.experiment_dir = experiment_dir
        self.prompts_dir = experiment_dir / "prompts"
        self.responses_dir = experiment_dir / "responses"
        self.contexts_dir = experiment_dir / "contexts"

        for directory in (self.prompts_dir, self.responses_dir, self.contexts_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def write_prompt(self, task_id: str, prompt: str) -> Path:
        path = self.prompts_dir / f"{task_id}.txt"
        path.write_text(prompt, encoding="utf-8")
        return path

    def write_context(self, task_id: str, context: str) -> Path:
        path = self.contexts_dir / f"{task_id}.md"
        path.write_text(context, encoding="utf-8")
        return path

    def write_response_payload(self, task_id: str, payload: Dict[str, object]) -> Path:
        path = self.responses_dir / f"{task_id}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def write_raw_response(self, task_id: str, raw_response: str) -> Path:
        path = self.responses_dir / f"{task_id}.txt"
        path.write_text(raw_response, encoding="utf-8")
        return path

    def persist_summary(self, results: Iterable[LangchainRAGTaskResult]) -> Path:
        table_path = self.experiment_dir / "summary.json"
        rows = [result.to_table_row() for result in results]
        table_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        return table_path


__all__ = ["ArtifactWriter"]
