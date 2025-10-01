from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class PromptOverrides:
    indicator_synonyms: List[str] = field(default_factory=list)
    extra_instructions: Optional[str] = None


@dataclass
class BenchmarkConfig:
    source: str
    year: int
    unit: str
    value: Optional[float] = None


@dataclass
class TaskConfig:
    id: str
    indicator: str
    page: int
    benchmark: BenchmarkConfig
    expected: Optional[str] = None
    prompt_overrides: PromptOverrides = field(default_factory=PromptOverrides)


@dataclass
class DatasetConfig:
    company: str
    industry: str
    year: int
    document: Path
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    dataset: DatasetConfig
    tasks: List[TaskConfig]

    @classmethod
    def from_json(cls, config_path: Path) -> "ExperimentConfig":
        with config_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        dataset_payload = payload.get("dataset", {})
        dataset = DatasetConfig(
            company=dataset_payload["company"],
            industry=dataset_payload["industry"],
            year=int(dataset_payload["year"]),
            document=Path(dataset_payload["document"]),
            metadata=dict(dataset_payload.get("metadata", {})),
        )

        tasks: List[TaskConfig] = []
        for task_payload in payload.get("tasks", []):
            benchmark_payload = task_payload["benchmark"]
            benchmark = BenchmarkConfig(
                source=benchmark_payload["source"],
                year=int(benchmark_payload["year"]),
                unit=benchmark_payload["unit"],
                value=benchmark_payload.get("value"),
            )
            overrides_payload = task_payload.get("prompt_overrides", {})
            overrides = PromptOverrides(
                indicator_synonyms=list(overrides_payload.get("indicator_synonyms", [])),
                extra_instructions=overrides_payload.get("extra_instructions"),
            )
            task = TaskConfig(
                id=task_payload["id"],
                indicator=task_payload["indicator"],
                page=int(task_payload["page"]),
                benchmark=benchmark,
                expected=task_payload.get("expected"),
                prompt_overrides=overrides,
            )
            tasks.append(task)

        return cls(dataset=dataset, tasks=tasks)


def load_experiment(config_path: Path) -> ExperimentConfig:
    return ExperimentConfig.from_json(config_path)
