#!/usr/bin/env python3
"""Generate Markdown reports summarising experiment artifacts."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parent.parent


def slugify_company(name: str | None) -> str | None:
    if not name:
        return None
    cleaned = re.sub(r"[^0-9A-Za-z\s]", "", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None
    return cleaned.lower().replace(" ", "_")


def load_experiments(directory: Path) -> Dict[str, dict]:
    experiments: Dict[str, dict] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"[warn] Skipping invalid experiment file {path}: {exc}", file=sys.stderr)
            continue
        dataset = data.get("dataset", {}) if isinstance(data, dict) else {}
        company = dataset.get("company") if isinstance(dataset, dict) else None
        year = dataset.get("year") if isinstance(dataset, dict) else None
        company_slug = slugify_company(company)
        slug = None
        if company_slug and year is not None:
            slug = f"{company_slug}_{year}"
        experiments[path.stem] = {
            "path": path,
            "data": data,
            "dataset": dataset,
            "slug": slug,
        }
    return experiments


def match_experiment_id(task_id: str, experiment_ids: Iterable[str]) -> str | None:
    for experiment_id in experiment_ids:
        if task_id.startswith(f"{experiment_id}_"):
            return experiment_id
    return None


def load_runs(directory: Path, experiments: Dict[str, dict]) -> Dict[str, List[dict]]:
    runs_by_experiment: Dict[str, List[dict]] = defaultdict(list)
    experiment_ids = list(experiments.keys())
    for run_dir in sorted(directory.iterdir()):
        if not run_dir.is_dir():
            continue
        summary_path = run_dir / "summary.json"
        if not summary_path.exists():
            continue
        try:
            summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"[warn] Skipping invalid summary {summary_path}: {exc}", file=sys.stderr)
            continue
        if not isinstance(summary_data, list) or not summary_data:
            print(f"[warn] No tasks in summary {summary_path}", file=sys.stderr)
            continue
        experiment_id = None
        for entry in summary_data:
            task_id = str(entry.get("Task", ""))
            experiment_id = match_experiment_id(task_id, experiment_ids)
            if experiment_id:
                break
        if not experiment_id:
            print(f"[warn] Could not match summary {summary_path} to an experiment", file=sys.stderr)
            continue
        runs_by_experiment[experiment_id].append({
            "name": run_dir.name,
            "path": run_dir,
            "summary": summary_data,
        })
    return runs_by_experiment


def compute_run_stats(entries: List[dict]) -> dict:
    total = len(entries)
    match_counts = {"yes": 0, "partial": 0, "no": 0}
    not_found_outputs = 0
    expected_not_found = 0
    for row in entries:
        match_value = str(row.get("Match", "")).strip().lower()
        if match_value in match_counts:
            match_counts[match_value] += 1
        else:
            match_counts["no"] += 1
        model_output = str(row.get("Model Output", "")).strip().lower()
        expected_value = str(row.get("Expected", "")).strip().lower()
        if model_output == "not found":
            not_found_outputs += 1
        if expected_value == "not found":
            expected_not_found += 1
    matches = match_counts["yes"]
    partial = match_counts["partial"]
    mismatches = total - matches - partial
    accuracy = (matches / total) * 100 if total else 0.0
    return {
        "total": total,
        "matches": matches,
        "partial": partial,
        "mismatches": mismatches,
        "not_found_outputs": not_found_outputs,
        "expected_not_found": expected_not_found,
        "accuracy": accuracy,
    }


def escape_md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def format_table(headers: List[str], rows: Iterable[Iterable[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]
    if body_lines:
        return "\n".join([header_line, separator_line, *body_lines])
    empty_row = ['*No data*'] + [''] * (len(headers) - 1)
    return "\n".join([header_line, separator_line, "| " + " | ".join(empty_row) + " |"])

def get_model_label(run_name: str, experiment_slug: str | None) -> str:
    if experiment_slug and run_name.startswith(f"{experiment_slug}_"):
        return run_name[len(experiment_slug) + 1 :]
    return run_name


def build_report(args: argparse.Namespace) -> Tuple[str, Dict[str, dict], Dict[str, List[dict]]]:
    experiments = load_experiments(args.experiments_dir)
    if not experiments:
        raise SystemExit("No experiments found.")
    runs_by_experiment = load_runs(args.artifacts_dir, experiments)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []
    lines.append("# Experiment Report")
    lines.append("")
    lines.append(f"_Generated on {timestamp}_")
    lines.append("")

    summary_rows: List[List[str]] = []
    for experiment_id, runs in sorted(runs_by_experiment.items()):
        experiment_info = experiments.get(experiment_id, {})
        dataset = experiment_info.get("dataset", {})
        exp_label_parts = [dataset.get("company") or experiment_id.replace("_", " ").title()]
        if dataset.get("year") is not None:
            exp_label_parts.append(str(dataset["year"]))
        exp_label = " ".join(exp_label_parts)
        for run in sorted(runs, key=lambda item: item["name"]):
            stats = compute_run_stats(run["summary"])
            model_label = get_model_label(run["name"], experiment_info.get("slug"))
            summary_rows.append([
                escape_md(exp_label),
                escape_md(model_label),
                str(stats["total"]),
                str(stats["matches"]),
                str(stats["mismatches"]),
                f"{stats['accuracy']:.1f}%",
            ])
    if summary_rows:
        lines.append("## Summary")
        lines.append("")
        lines.append(format_table([
            "Experiment",
            "Model",
            "Tasks",
            "Matches",
            "Mismatches",
            "Accuracy",
        ], summary_rows))
        lines.append("")
    else:
        lines.append("No artifact summaries were found under the provided directory.")
        lines.append("")

    for experiment_id in sorted(experiments.keys()):
        experiment_info = experiments[experiment_id]
        dataset = experiment_info.get("dataset", {})
        runs = runs_by_experiment.get(experiment_id, [])
        exp_header_parts = [dataset.get("company") or experiment_id.replace("_", " ").title()]
        if dataset.get("year") is not None:
            exp_header_parts.append(str(dataset["year"]))
        lines.append(f"## {' — '.join(exp_header_parts)}")
        lines.append("")
        if dataset:
            meta_lines = []
            if dataset.get("industry"):
                meta_lines.append(f"- **Industry**: {escape_md(str(dataset['industry']))}")
            if dataset.get("document"):
                doc_path = dataset["document"]
                meta_lines.append(f"- **Source document**: `{escape_md(str(doc_path))}`")
            metadata = dataset.get("metadata")
            if isinstance(metadata, dict):
                extra = [f"{key}: {value}" for key, value in metadata.items()]
                if extra:
                    meta_lines.append(f"- **Metadata**: {escape_md('; '.join(extra))}")
            if meta_lines:
                lines.extend(meta_lines)
                lines.append("")
        if not runs:
            lines.append("No artifact runs were found for this experiment.")
            lines.append("")
            continue
        for run in sorted(runs, key=lambda item: item["name"]):
            stats = compute_run_stats(run["summary"])
            model_label = get_model_label(run["name"], experiment_info.get("slug"))
            lines.append(f"### Model: {escape_md(model_label)}")
            lines.append("")
            lines.append(
                "- **Run folder**: `"
                + escape_md(str(run["path"].relative_to(ROOT)))
                + "`"
            )
            detail_metrics = [
                f"Tasks: {stats['total']}",
                f"Matches: {stats['matches']}",
            ]
            if stats["partial"]:
                detail_metrics.append(f"Partial: {stats['partial']}")
            detail_metrics.append(f"Mismatches: {stats['mismatches']}")
            detail_metrics.append(f"Not found outputs: {stats['not_found_outputs']}")
            lines.append("- **Performance**: " + ", ".join(detail_metrics))
            lines.append(f"- **Accuracy**: {stats['accuracy']:.1f}%")
            if stats["expected_not_found"]:
                lines.append(
                    f"- **Expected not found**: {stats['expected_not_found']} task(s)"
                )
            lines.append("")
            headers = [
                "Task",
                "Indicator",
                "Page",
                "Benchmark",
                "Expected",
                "Model Output",
                "Extracted Value",
                "Match",
            ]
            task_rows = []
            for entry in run["summary"]:
                match_value = str(entry.get("Match", "")).strip()
                match_lower = match_value.lower()
                if match_lower == "yes":
                    match_display = "✅ yes"
                elif match_lower == "partial":
                    match_display = "⚠️ partial"
                else:
                    match_display = f"❌ {match_value or 'no'}"
                task_rows.append([
                    escape_md(str(entry.get("Task", ""))),
                    escape_md(str(entry.get("Indicator", ""))),
                    escape_md(str(entry.get("Page", ""))),
                    escape_md(str(entry.get("Benchmark", ""))),
                    escape_md(str(entry.get("Expected", ""))),
                    escape_md(str(entry.get("Model Output", ""))),
                    escape_md(str(entry.get("Extracted Value", ""))),
                    escape_md(match_display),
                ])
            lines.append(format_table(headers, task_rows))
            lines.append("")
    report = "\n".join(lines).rstrip() + "\n"
    return report, experiments, runs_by_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Markdown report for experiments.")
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=ROOT / "experiments",
        help="Directory containing experiment JSON configuration files.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ROOT / "artifacts",
        help="Directory containing artifact outputs with summaries.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "experiment_report.md",
        help="Path to write the generated Markdown report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, _, _ = build_report(args)
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
