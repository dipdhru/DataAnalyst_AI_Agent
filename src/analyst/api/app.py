"""FastAPI app: sources, schema, catalog, read-only queries and file uploads.

Run locally:  uv run uvicorn analyst.api.app:app --reload
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from analyst import __version__
from analyst.catalog import Catalog, CatalogColumn, CatalogService, CatalogStore, CatalogTable
from analyst.config import Settings, load_settings
from analyst.connectors import (
    Connector,
    IngestResult,
    QueryError,
    QueryTimeoutError,
    SourceRegistry,
)
from analyst.connectors.files import TableExistsError, UnsupportedFileError
from analyst.models import QueryResult, SchemaInfo
from analyst.sql_guard import UnsafeQueryError


class SourceSummary(BaseModel):
    name: str
    kind: str
    dialect: str


class QueryRequest(BaseModel):
    sql: str
    limit: int | None = None


class DescriptionUpdate(BaseModel):
    description: str | None


class UploadResponse(BaseModel):
    tables: list[IngestResult]
    catalog_refreshed: bool


def create_app(settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        cfg = settings or load_settings()
        registry = SourceRegistry(cfg)
        app.state.settings = cfg
        app.state.registry = registry
        app.state.catalog = CatalogService(registry, CatalogStore(cfg.app_db_path), cfg.profiling)
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

    @app.post("/uploads")
    def upload_file(
        request: Request,
        registry: RegistryDep,
        catalog: CatalogDep,
        file: Annotated[UploadFile, File()],
        table_name: Annotated[str | None, Form()] = None,
        replace: Annotated[bool, Form()] = False,
        profile: Annotated[bool, Form()] = True,
    ) -> UploadResponse:
        settings: Settings = request.app.state.settings
        original = Path(file.filename or "upload").name
        raw_dir = settings.uploads_raw_dir
        raw_dir.mkdir(parents=True, exist_ok=True)
        dest = raw_dir / f"{uuid.uuid4().hex}{Path(original).suffix.lower()}"
        _save_capped(file, dest, settings.uploads.max_upload_mb * 1024 * 1024)
        try:
            tables = registry.uploads.ingest(
                dest, table_name=table_name, replace=replace, original_filename=original
            )
        except UnsupportedFileError as e:
            dest.unlink(missing_ok=True)
            raise HTTPException(415, str(e)) from e
        except TableExistsError as e:
            dest.unlink(missing_ok=True)
            raise HTTPException(409, str(e)) from e
        if profile:
            catalog.refresh(registry.uploads.name)
        return UploadResponse(tables=tables, catalog_refreshed=profile)

    @app.delete("/uploads/tables/{table}", status_code=204)
    def drop_upload(table: str, registry: RegistryDep, catalog: CatalogDep) -> None:
        try:
            registry.uploads.drop_table(table)
        except KeyError as e:
            raise HTTPException(404, f"no uploaded table {table!r}") from e
        if catalog.get(registry.uploads.name) is not None:
            catalog.refresh(registry.uploads.name)


def _save_capped(file: UploadFile, dest: Path, max_bytes: int) -> None:
    written = 0
    with dest.open("wb") as out:
        while chunk := file.file.read(1024 * 1024):
            written += len(chunk)
            if written > max_bytes:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"file larger than {max_bytes // (1024 * 1024)} MB")
            out.write(chunk)
    file.file.close()


app = create_app()
