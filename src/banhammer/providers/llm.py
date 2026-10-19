"""Chat models: openai, huggingface, or a fake one for tests."""
from __future__ import annotations

from typing import Any

from banhammer.models import BalanceReport, PowerAssessment


class FakeChatModel:
    """Returns the same report every time. Enough for with_structured_output."""

    def __init__(self, response: BalanceReport | dict | None = None) -> None:
        self.response = response or BalanceReport(
            power_assessment=PowerAssessment(summary="No opinion (fake model)."),
            risk="ok",
            confidence=0.5,
        )
        self.schema = None

    def with_structured_output(self, schema: type) -> FakeChatModel:
        self.schema = schema
        return self

    def invoke(self, _messages: Any, **_kwargs: Any) -> Any:
        if self.schema is None or isinstance(self.response, self.schema):
            return self.response
        if isinstance(self.response, dict):
            return self.schema.model_validate(self.response)
        return self.schema.model_validate(self.response.model_dump())


def get_llm(cfg: dict) -> Any:
    provider = cfg["provider"]
    if provider == "fake":
        return FakeChatModel(cfg.get("response"))
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs = {"model": cfg.get("model", "gpt-4o-mini")}
        if cfg.get("temperature") is not None:
            kwargs["temperature"] = cfg["temperature"]
        return ChatOpenAI(**kwargs)
    if provider == "huggingface":
        return _huggingface_chat(cfg)
    raise ValueError(f"unknown llm provider {provider!r}")


def _huggingface_chat(cfg: dict) -> Any:
    try:
        from langchain_huggingface import ChatHuggingFace
    except ImportError as exc:
        raise RuntimeError("pip install -e '.[hf]' to use the huggingface provider") from exc
    model = cfg.get("model", "HuggingFaceTB/SmolLM2-360M-Instruct")
    return ChatHuggingFace.from_model_id(model, task="text-generation")
