from __future__ import annotations

from .benchmarks import BenchmarkRecord
from .config import DatasetConfig, TaskConfig

PROMPT_TEMPLATE = (

"""
In {benchmark_year}, the {industry} industry average for {indicator} was {benchmark_value}.
You are given a page from {company}'s sustainability report.
You must follow this direction:
 1. Look only at the information provided in this context. Do not use external knowledge.
 2. Identify {company}'s reported value for {indicator}. If no value is reported, answer "not found".
 3. Compare the company's value with the industry average:
    - If the company’s value is numerically higher than the industry average, answer "higher".
    - If the company’s value is numerically lower than the industry average, answer "lower".
    - If the company’s value equals the industry average, answer "equal".
 4. Format Your Response: Your entire response must strictly follow the format below. Do not add any introductory text or explanations.
    - output: {{higher | lower | equal | not found}},
    - extracted_value: {{extracted_value or not found}}

"""
)


def format_benchmark_value(value: float, unit: str) -> str:
    unit_key = unit.lower()
    if unit_key in {"percent", "percent_of_total"}:
        percentage = value
        suffix = "%" if unit_key == "percent" else "% of total"
        if percentage.is_integer():
            return f"{int(percentage)}{suffix}"
        return f"{percentage:.2f}{suffix}"
    if unit_key == "m3":
        if value >= 1_000_000:
            scaled = value / 1_000_000
            return f"{scaled:.2f} million m3"
        if value >= 1_000:
            scaled = value / 1_000
            return f"{scaled:.1f}k m3"
    if value.is_integer():
        formatted = f"{int(value)}"
    else:
        formatted = f"{value}" if abs(value) >= 1 else f"{value:.2f}"
    return f"{formatted} {unit}"


def build_prompt(
    dataset: DatasetConfig,
    task: TaskConfig,
    benchmark: BenchmarkRecord,
    add_synonyms: bool = True,
) -> str:
    benchmark_value_text = format_benchmark_value(benchmark.value, benchmark.unit)
    base_prompt = PROMPT_TEMPLATE.format(
        benchmark_year=benchmark.year,
        industry=dataset.industry,
        indicator=task.indicator,
        benchmark_value=benchmark_value_text,
        company=dataset.company,
    )

    extra_parts = []
    if add_synonyms and task.prompt_overrides.indicator_synonyms:
        synonyms = ", ".join(task.prompt_overrides.indicator_synonyms)
        extra_parts.append(
            f"Relevant synonyms for the indicator include: {synonyms}. Treat them as equivalent."
        )
    if task.prompt_overrides.extra_instructions:
        extra_parts.append(task.prompt_overrides.extra_instructions.strip())

    if extra_parts:
        base_prompt = f"{base_prompt}\n\n" + "\n".join(extra_parts)
    return base_prompt


__all__ = ["build_prompt", "format_benchmark_value", "PROMPT_TEMPLATE"]
