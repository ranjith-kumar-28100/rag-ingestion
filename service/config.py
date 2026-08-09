"""Service infrastructure settings (env prefix ``RAG_SVC_``).

Only infra lives here — broker URL, DB URL, filesystem locations. The ingestion
knobs stay in ``rag_ingestion.IngestionConfig`` (env prefix ``RAG_INGEST_``), so
the two concerns never bleed into each other.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class ServiceSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_SVC_",
        env_file=".env",
        extra="ignore",
    )

    # message queue / result backend ------------------------------------------
    redis_url: str = "redis://localhost:6379/0"

    # persistence -------------------------------------------------------------
    # SQLAlchemy URL. Default is a local SQLite file; swap for Postgres etc.
    # without touching the ORM models (that is the point of using SQLAlchemy).
    database_url: str = "sqlite:///./data/rag_service.db"

    # filesystem --------------------------------------------------------------
    data_dir: Path = Path("./data")
    upload_dir: Path = Path("./data/uploads")
    output_dir: Path = Path("./data/output")
    max_upload_mb: int = 200

    # UI -> API ----------------------------------------------------------------
    api_base_url: str = "http://localhost:8000"

    def ensure_dirs(self) -> None:
        for directory in (self.data_dir, self.upload_dir, self.output_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> ServiceSettings:
    settings = ServiceSettings()
    settings.ensure_dirs()
    return settings
