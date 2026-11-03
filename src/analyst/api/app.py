"""FastAPI app: sources, schema, catalog, read-only queries and agent questions.

Run locally:  uv run uvicorn analyst.api.app:app --reload
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

import anthropic
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel

from analyst import __version__
from analyst.agent import Agent, AgentAnswer
from analyst.agent.providers import ProviderError, make_llm_client
from analyst.catalog import Catalog, CatalogColumn, CatalogService, CatalogStore, CatalogTable
from analyst.config import Settings, load_settings
from analyst.connectors import Connector, QueryError, QueryTimeoutError, SourceRegistry
from analyst.memory import MemoryStore, recall, render
from analyst.models import QueryResult, SchemaInfo
from analyst.semantic import load_semantics
from analyst.sql_guard import UnsafeQueryError


class SourceSummary(BaseModel):
    name: str
    kind: str
    dialect: str


class QueryRequest(BaseModel):
    sql: str
    limit: int | None = None


class AskRequest(BaseModel):
    question: str


class DescriptionUpdate(BaseModel):
    description: str | None


def create_app(settings: Settings | None = None, llm_client: Any = None) -> FastAPI:
    """`llm_client` defaults to the provider in config.yaml, created on the first question."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        cfg = settings or load_settings()
        registry = SourceRegistry(cfg)
        app.state.settings = cfg
        app.state.registry = registry
        app.state.catalog = CatalogService(registry, CatalogStore(cfg.app_db_path), cfg.profiling)
        app.state.semantics = load_semantics(cfg.metrics_path)
        app.state.memory = MemoryStore(cfg.app_db_path)
        app.state.llm_client = llm_client
        try:
            yield
        finally:
            registry.close()

    app = FastAPI(title="DataAnalyst Agent", version=__version__, lifespan=lifespan)
    _add_routes(app)
    return app


def _registry(request: Request) -> SourceRegistry:
    return request.app.state.registry


def _catalog(request: Request) -> CatalogService:
    return request.app.state.catalog


def _source(source: str, registry: Annotated[SourceRegistry, Depends(_registry)]) -> Connector:
    try:
        return registry.get(source)
    except KeyError as e:
        raise HTTPException(404, str(e.args[0])) from e


RegistryDep = Annotated[SourceRegistry, Depends(_registry)]
CatalogDep = Annotated[CatalogService, Depends(_catalog)]
SourceDep = Annotated[Connector, Depends(_source)]


def _add_routes(app: FastAPI) -> None:
    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/sources")
    def list_sources(registry: RegistryDep) -> list[SourceSummary]:
        return [SourceSummary(name=c.name, kind=c.kind, dialect=c.dialect) for c in registry.all()]

    @app.get("/sources/{source}/schema")
    def get_schema(connector: SourceDep, refresh: bool = False) -> SchemaInfo:
        return connector.get_schema(refresh=refresh)

    @app.post("/sources/{source}/query")
    def run_query(body: QueryRequest, connector: SourceDep) -> QueryResult:
        try:
            return connector.run_query(body.sql, limit=body.limit)
        except UnsafeQueryError as e:
            raise HTTPException(400, f"rejected: {e}") from e
        except QueryTimeoutError as e:
            raise HTTPException(408, str(e)) from e
        except QueryError as e:
            raise HTTPException(400, f"query failed: {e}") from e

    @app.post("/sources/{source}/ask")
    def ask(
        body: AskRequest, request: Request, connector: SourceDep, catalog: CatalogDep
    ) -> AgentAnswer:
        built = catalog.get(connector.name)
        if built is None:
            raise HTTPException(
                409, f"build the catalog first: POST /sources/{connector.name}/catalog/refresh"
            )
        state = request.app.state
        try:
            if state.llm_client is None:
                state.llm_client = make_llm_client(state.settings.agent)
        except ProviderError as e:
            raise HTTPException(503, e.message) from e
        agent = Agent(
            state.llm_client,
            connector,
            built,
            state.settings.agent,
            state.semantics.get(connector.name),
        )
        try:
            memory = render(recall(state.memory, connector.name, body.question))
            return agent.ask(body.question, memory=memory)
        except ProviderError as e:
            code = {401: 503, 403: 503, 429: 429}.get(e.status_code or 0, 502)
            raise HTTPException(code, f"model provider error: {e.message}") from e
        except anthropic.AuthenticationError as e:
            raise HTTPException(503, "no valid Claude credentials (set ANTHROPIC_API_KEY)") from e
        except anthropic.RateLimitError as e:
            raise HTTPException(429, "Claude API rate limit; retry shortly") from e
        except anthropic.APIConnectionError as e:
            raise HTTPException(503, "could not reach the Claude API") from e
        except anthropic.APIStatusError as e:
            raise HTTPException(502, f"Claude API error {e.status_code}: {e.message}") from e

    @app.get("/sources/{source}/catalog")
    def get_catalog(connector: SourceDep, catalog: CatalogDep) -> Catalog:
        result = catalog.get(connector.name)
        if result is None:
            raise HTTPException(
                404,
                f"no catalog for {connector.name!r} yet; POST /sources/{connector.name}"
                "/catalog/refresh",
            )
        return result

    @app.post("/sources/{source}/catalog/refresh")
    def refresh_catalog(connector: SourceDep, catalog: CatalogDep) -> Catalog:
        return catalog.refresh(connector.name)

    @app.patch("/sources/{source}/catalog/tables/{table}")
    def describe_table(
        table: str, body: DescriptionUpdate, connector: SourceDep, catalog: CatalogDep
    ) -> CatalogTable:
        try:
            return catalog.describe_table(connector.name, table, body.description)
        except KeyError as e:
            raise HTTPException(404, f"not in catalog: {e.args[0]}") from e

    @app.patch("/sources/{source}/catalog/tables/{table}/columns/{column}")
    def describe_column(
        table: str,
        column: str,
        body: DescriptionUpdate,
        connector: SourceDep,
        catalog: CatalogDep,
    ) -> CatalogColumn:
        try:
            return catalog.describe_column(connector.name, table, column, body.description)
        except KeyError as e:
            raise HTTPException(404, f"not in catalog: {e.args[0]}") from e


app = create_app()
