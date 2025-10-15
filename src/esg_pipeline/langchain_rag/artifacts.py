from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import shutil

from .results import LangchainRAGTaskResult


class ArtifactWriter:
    """Helper for persisting prompts, contexts, responses, and summary tables."""

    def __init__(self, experiment_dir: Path) -> None:
        self.experiment_dir = experiment_dir
        self.prompts_dir = experiment_dir / "prompts"
        self.responses_dir = experiment_dir / "responses"
        self.contexts_dir = experiment_dir / "contexts"
        self.charts_dir = experiment_dir / "charts"

        for directory in (
            self.prompts_dir,
            self.responses_dir,
            self.contexts_dir,
            self.charts_dir,
        ):
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

    def write_chart_summary(self, task_id: str, markdown: str) -> Path:
        path = self.charts_dir / f"{task_id}.md"
        path.write_text(markdown, encoding="utf-8")
        return path

    def store_chart_asset(
        self,
        task_id: str,
        source: Path,
        index: int,
        chart_id: Optional[str] = None,
    ) -> Path:
        task_dir = self.charts_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        prefix = chart_id.replace(":", "_") if chart_id else f"{index:02d}"
        destination = task_dir / f"{prefix}_{source.name}"
        if source.resolve() != destination.resolve():  # avoid self-copy
            shutil.copy2(source, destination)
        return destination

    def write_chart_table_summary(
        self,
        *,
        chart_records: List[Dict[str, object]],
        table_records: List[Dict[str, object]],
    ) -> Path:
        """
        Persist a combined summary of chart and table metadata for the experiment.

        The payload captures the captions, descriptions, and supporting paths so
        downstream analysis can inspect all visual artefacts produced during the run.
        """

        summary_path = self.experiment_dir / "chart_table_summary.json"
        payload = {
            "charts": chart_records,
            "tables": table_records,
        }
        summary_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return summary_path


__all__ = ["ArtifactWriter"]
