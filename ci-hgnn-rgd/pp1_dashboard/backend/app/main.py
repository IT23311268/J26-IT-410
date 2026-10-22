"""Standalone GET-only dashboard API with no research-pipeline dependencies."""

import logging
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from .adapters.artifacts import (
    EDGE_TYPES,
    NODE_TYPES,
    ArtifactUnavailable,
    DemoReader,
    RealReader,
    synthetic,
)
from .schemas import (
    DEMO_NOTICE,
    Datasets,
    Envelope,
    GraphStats,
    Intents,
    Mode,
    Neighborhood,
    Node,
    NodeDetail,
    Overview,
    Page,
    Progress,
    Topic,
    TopicDetail,
    Validation,
)

PageLimit = Annotated[int, Query(ge=1, le=100)]
Offset = Annotated[int, Query(ge=0, le=100000)]
Search = Annotated[str, Query(max_length=160)]
Identifier = Annotated[str | None, Query(max_length=300)]
Filters = Annotated[str | None, Query(max_length=250)]


def parse_filters(value, allowed):
    if value is None:
        return None
    parts = set(filter(None, value.split(",")))
    if parts - allowed:
        raise HTTPException(422, "Unsupported graph filter.")
    return parts


def create_app(component_root: Path | None = None):
    app = FastAPI(title="CI-HGNN-RGD · PP1 read-only artifact viewer", version="0.1.0")
    root = component_root or Path(__file__).resolve().parents[3]
    readers = {}
    lock = Lock()

    def dispatch(mode, action, **kwargs):
        try:
            with lock:
                if mode not in readers:
                    try:
                        readers[mode] = DemoReader() if mode == "demo" else RealReader(root)
                    except (KeyError, TypeError) as exc:
                        raise ArtifactUnavailable(
                            "Published artifact metadata is incompatible."
                        ) from exc
            reader = readers[mode]
            reader.assert_unchanged()
            if action == "citation_intents" and not reader.intents:
                raise ArtifactUnavailable("No published citation-intent predictions are available.")
            data = getattr(reader, action)(**kwargs)
            if mode == "real" and synthetic(data):
                raise ArtifactUnavailable("Synthetic records were rejected in real mode.")
            return Envelope(
                mode=mode,
                status="available",
                notice=DEMO_NOTICE if mode == "demo" else None,
                graph_version=reader.version,
                data=data,
            )
        except KeyError:
            raise HTTPException(
                404, "Record not found in the selected published artifact."
            ) from None
        except (ArtifactUnavailable, OSError, ValueError, sqlite3.Error) as exc:
            message = (
                str(exc)
                if isinstance(exc, ArtifactUnavailable)
                else (
                    "The required artifact is missing, unreadable or incompatible. No "
                    "demo fallback was selected."
                )
            )
            return Envelope(mode=mode, status="unavailable", notice=message)

    @app.exception_handler(Exception)
    async def safe_error(_request, exc):
        logging.getLogger("pp1_dashboard").error("Viewer error: %s", type(exc).__name__)
        return JSONResponse(
            status_code=500,
            content={"detail": "The artifact viewer could not complete the request."},
        )

    @app.get("/api/health")
    def health():
        return {"status": "ok", "read_only": True, "default_mode": "real"}

    @app.get("/api/overview", response_model=Envelope[Overview])
    def overview(mode: Mode = "real"):
        return dispatch(mode, "overview")

    @app.get("/api/datasets", response_model=Envelope[Datasets])
    def datasets(mode: Mode = "real"):
        return dispatch(mode, "datasets")

    @app.get("/api/citation-intents", response_model=Envelope[Intents])
    def intents(
        mode: Mode = "real",
        q: Search = "",
        label: Literal["background", "method", "result"] | None = None,
        limit: PageLimit = 20,
        offset: Offset = 0,
    ):
        return dispatch(mode, "citation_intents", q=q, label=label, limit=limit, offset=offset)

    @app.get("/api/topics", response_model=Envelope[Page[Topic]])
    def topics(mode: Mode = "real", q: Search = "", limit: PageLimit = 20, offset: Offset = 0):
        return dispatch(mode, "topics", q=q, limit=limit, offset=offset)

    @app.get("/api/topics/{topic_id}", response_model=Envelope[TopicDetail])
    def topic_detail(topic_id: str, mode: Mode = "real"):
        if len(topic_id) > 300:
            raise HTTPException(422, "Identifier exceeds the resource limit.")
        return dispatch(mode, "topic_detail", topic_id=topic_id)

    @app.get("/api/graph/stats", response_model=Envelope[GraphStats])
    def stats(mode: Mode = "real"):
        return dispatch(mode, "graph_stats")

    @app.get("/api/graph/nodes", response_model=Envelope[Page[Node]])
    def nodes(
        mode: Mode = "real",
        q: Search = "",
        node_type: Literal["paper", "author", "topic", "research_field"] | None = None,
        limit: PageLimit = 20,
        offset: Offset = 0,
    ):
        return dispatch(mode, "nodes", q=q, node_type=node_type, limit=limit, offset=offset)

    @app.get("/api/graph/nodes/{node_id}", response_model=Envelope[NodeDetail])
    def node_detail(node_id: str, mode: Mode = "real"):
        if len(node_id) > 300:
            raise HTTPException(422, "Identifier exceeds the resource limit.")
        return dispatch(mode, "node_detail", node_id=node_id)

    @app.get("/api/graph/neighborhood", response_model=Envelope[Neighborhood])
    def neighborhood(
        mode: Mode = "real",
        seed: Identifier = None,
        depth: Annotated[int, Query(ge=1, le=2)] = 1,
        limit: Annotated[int, Query(ge=1, le=150)] = 70,
        node_types: Filters = None,
        edge_types: Filters = None,
    ):
        return dispatch(
            mode,
            "neighborhood",
            seed=seed,
            depth=depth,
            limit=limit,
            node_types=parse_filters(node_types, NODE_TYPES),
            edge_types=parse_filters(edge_types, EDGE_TYPES),
        )

    @app.get("/api/progress", response_model=Envelope[Progress])
    def progress(mode: Mode = "real"):
        return dispatch(mode, "progress")

    @app.get("/api/validation/status", response_model=Envelope[Validation])
    def validation(mode: Mode = "real"):
        return dispatch(mode, "validation")

    return app


app = create_app()
