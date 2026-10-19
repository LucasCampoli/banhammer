"""Turn configs/pipeline.yaml into a LangGraph."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from langgraph.graph import END, START, StateGraph

from banhammer.shared.config import load_env, load_yaml, parse_modules
from banhammer.pipeline.state import PipelineState
from banhammer.providers import StoreFactory, get_embeddings, get_llm
from banhammer.shared.registry import get

# Later phases add modules to these stages. The graph runs whatever the YAML enables.
STAGE_ORDER = ("build_query", "expand_query", "retrieve", "fusion", "filter", "rerank", "generate")
REQUIRED = ("build_query", "retrieve", "generate")


def build_graph(config: dict | Path | str, *, factory: StoreFactory | None = None):
    load_env()
    cfg = config if isinstance(config, dict) else load_yaml(config)
    pipeline = cfg.get("pipeline")
    if not isinstance(pipeline, dict):
        raise ValueError("config has no pipeline section")

    stages = {name: parse_modules(pipeline.get(name)) for name in STAGE_ORDER}
    _check(stages)
    _load_modules()

    embeddings = factory.embeddings if factory is not None else get_embeddings(cfg["embeddings"])
    model = cfg["embeddings"].get("model") or cfg["embeddings"]["provider"]
    stores = factory or StoreFactory(cfg["vector_store"], embeddings, model=model, index_cfg=cfg.get("index"))
    llm = get_llm(cfg["llm"])

    graph = StateGraph(PipelineState)
    previous = START
    n_retrievers = len(stages["retrieve"])
    for stage in STAGE_ORDER:
        for index, spec in enumerate(stages[stage]):
            module = _make(stage, spec.name, spec.options(), llm, stores)
            node_name = f"{stage}_{spec.name}" if index == 0 else f"{stage}_{spec.name}_{index + 1}"
            graph.add_node(node_name, _step(stage, module, index, n_retrievers))
            graph.add_edge(previous, node_name)
            previous = node_name
    graph.add_edge(previous, END)
    return graph.compile()


def _load_modules() -> None:
    # Importing a module registers it.
    from banhammer.pipeline import filters  # noqa: F401
    from banhammer.pipeline.generation import llm_judge  # noqa: F401
    from banhammer.pipeline.query import full_text  # noqa: F401
    from banhammer.pipeline.retrieval import dense  # noqa: F401


def _make(stage: str, name: str, options: dict, llm: Any, stores: Any) -> Any:
    try:
        cls = get(stage, name)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc
    if stage == "retrieve":
        return cls(stores, k=options.get("k", 10))
    if stage == "generate":
        return cls(llm)
    if options:
        return cls(**options)
    return cls()


def _check(stages: dict[str, list]) -> None:
    for stage in REQUIRED:
        if not stages[stage]:
            raise ValueError(f"pipeline requires a {stage} module")
    if len(stages["generate"]) != 1:
        raise ValueError("pipeline requires exactly one generate module")
    retrievers = len(stages["retrieve"])
    fusions = len(stages["fusion"])
    if retrievers > 1 and fusions != 1:
        raise ValueError("fusion is required when more than one retriever is enabled")
    if fusions and retrievers <= 1:
        raise ValueError("fusion is only used with more than one retriever")
    if fusions > 1:
        raise ValueError("pipeline accepts at most one fusion module")
    if len(stages["rerank"]) > 1:
        raise ValueError("pipeline accepts at most one rerank module")


def _step(stage: str, module: Any, index: int, n_retrievers: int) -> Callable[[PipelineState], dict]:
    if stage == "build_query":
        def build(state: PipelineState) -> dict:
            queries = list(state.get("queries") or [])
            queries.extend(module.build(state["card"]))
            return {"queries": queries}
        return build

    if stage == "expand_query":
        def expand(state: PipelineState) -> dict:
            return {"queries": module.expand(state["card"], state.get("queries") or [])}
        return expand

    if stage == "retrieve":
        def retrieve(state: PipelineState) -> dict:
            hits = []
            for query in state.get("queries") or []:
                hits.extend(module.search(query, module.k))
            # One retriever writes the hit list directly. Several keep their own
            # lists so the fusion step can see each ranking.
            if n_retrievers == 1:
                return {"hits": hits}
            grouped = [] if index == 0 else list(state.get("hit_lists") or [])
            grouped.append(hits)
            return {"hit_lists": grouped}
        return retrieve

    if stage == "fusion":
        def fuse(state: PipelineState) -> dict:
            return {"hits": module.fuse(state.get("hit_lists") or [])}
        return fuse

    if stage == "filter":
        def apply_filter(state: PipelineState) -> dict:
            return {"hits": module.apply(state["card"], state.get("hits") or [])}
        return apply_filter

    if stage == "rerank":
        def rerank(state: PipelineState) -> dict:
            return {"hits": module.rerank(state["card"], state.get("hits") or [])}
        return rerank

    def generate(state: PipelineState) -> dict:
        return {"report": module.generate(state["card"], state.get("hits") or [])}
    return generate
