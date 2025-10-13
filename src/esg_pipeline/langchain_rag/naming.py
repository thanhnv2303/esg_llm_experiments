from __future__ import annotations

from typing import Optional

from .config import LangchainRAGConfig


def derive_experiment_id(
    company: str,
    year: int,
    model_name: str,
    config: LangchainRAGConfig,
    suffix: Optional[str],
) -> str:
    base = f"{company}_{year}_{model_name}"
    safe = "_".join(part.lower().replace(" ", "_") for part in base.split("_"))
    safe = "".join(char for char in safe if char.isalnum() or char in {"_", "-"})
    descriptor = config.describe()
    if suffix:
        descriptor = f"{descriptor}-{suffix}"
    return f"{safe}_{descriptor}"


__all__ = ["derive_experiment_id"]
