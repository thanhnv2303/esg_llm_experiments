from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional


@dataclass(frozen=True)
class BenchmarkRecord:
    indicator: str
    unit: str
    year: int
    value: float


class BenchmarkRepository:
    def __init__(self, csv_path: Path) -> None:
        self._index: Dict[str, Dict[int, BenchmarkRecord]] = {}
        self._load(csv_path)

    def _load(self, csv_path: Path) -> None:
        with csv_path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                indicator = row["indicator"].strip()
                unit = row["unit"].strip()
                try:
                    year = int(row["year"])
                except ValueError as exc:
                    raise ValueError(f"Invalid year in benchmark file: {row['year']}") from exc
                try:
                    value = float(row["value"])
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid value in benchmark file for {indicator} ({row['value']})"
                    ) from exc
                record = BenchmarkRecord(indicator=indicator, unit=unit, year=year, value=value)
                indicator_key = indicator.lower()
                year_map = self._index.setdefault(indicator_key, {})
                year_map[year] = record

    def available_indicators(self) -> Iterable[str]:
        return sorted(self._index.keys())

    def get(self, indicator: str, year: int) -> Optional[BenchmarkRecord]:
        year_map = self._index.get(indicator.lower())
        if not year_map:
            return None
        return year_map.get(year)

    def require(self, indicator: str, year: int) -> BenchmarkRecord:
        record = self.get(indicator, year)
        if record is None:
            raise KeyError(f"Benchmark not found for indicator '{indicator}' and year {year}")
        return record


__all__ = ["BenchmarkRecord", "BenchmarkRepository"]
