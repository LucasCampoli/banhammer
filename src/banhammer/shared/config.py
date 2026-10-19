"""YAML config and .env loading."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

ROOT = Path(__file__).resolve().parents[3]


def load_env(path: Path | None = None) -> None:
    load_dotenv(path or ROOT / ".env")


def load_yaml(path: Path | str) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return data or {}


class ModuleSpec(BaseModel):
    """One enabled pipeline module. Extra keys are options for its constructor."""

    model_config = ConfigDict(extra="allow")

    name: str

    def options(self) -> dict[str, Any]:
        return {key: value for key, value in self.model_dump().items() if key != "name"}


def parse_modules(value: Any) -> list[ModuleSpec]:
    """Accept a module name, ``{name, ...options}``, or a list of either."""
    if value is None:
        return []
    if isinstance(value, str):
        return [ModuleSpec(name=value)]
    if isinstance(value, dict):
        return [ModuleSpec.model_validate(value)]
    specs: list[ModuleSpec] = []
    for item in value:
        if isinstance(item, str):
            specs.append(ModuleSpec(name=item))
        else:
            specs.append(ModuleSpec.model_validate(item))
    return specs
