"""Module name to implementation. Stages that phase 1 does not fill stay empty."""
from __future__ import annotations

from collections.abc import Callable

STAGES = ("build_query", "expand_query", "retrieve", "fusion", "filter", "rerank", "generate")

_REGISTRY: dict[tuple[str, str], type] = {}


def register(stage: str, name: str) -> Callable[[type], type]:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")

    def deco(cls: type) -> type:
        _REGISTRY[(stage, name)] = cls
        return cls

    return deco


def get(stage: str, name: str) -> type:
    try:
        return _REGISTRY[(stage, name)]
    except KeyError:
        known = sorted(module for registered, module in _REGISTRY if registered == stage)
        raise KeyError(f"no module {name!r} for stage {stage!r} (registered: {known})") from None


def registered(stage: str) -> list[str]:
    return sorted(module for registered, module in _REGISTRY if registered == stage)
