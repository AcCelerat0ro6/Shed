"""RAG 存储连接设置。进程环境覆盖显式指定的 dotenv 文件。"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import dotenv_values

_DEFAULT_ES_INDEX = "xwrag_default"


@dataclass(frozen=True, slots=True, kw_only=True)
class PostgresSettings:
    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = field(default="", repr=False)
    database: str = "my_pi_agent"

    def __post_init__(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ValueError("DB_PORT must be between 1 and 65535")
        if not (self.host.strip() and self.user.strip() and self.database.strip()):
            raise ValueError("DB_HOST, DB_USER and DB_NAME must be nonempty")


@dataclass(frozen=True, slots=True, kw_only=True)
class ElasticsearchSettings:
    addr: str = "http://localhost:9200"
    username: str = ""
    password: str = field(default="", repr=False)
    index: str = _DEFAULT_ES_INDEX

    def __post_init__(self) -> None:
        url = urlsplit(self.addr)
        if (url.scheme not in ("http", "https") or not url.hostname or url.username is not None
                or url.query or url.fragment or url.port == 0):
            raise ValueError("ELASTICSEARCH_ADDR must be an HTTP(S) base URL without credentials, query or fragment")
        if self.password and not self.username:
            raise ValueError("ELASTICSEARCH_PASSWORD requires ELASTICSEARCH_USERNAME")
        if (not self.index or self.index != self.index.lower() or self.index in (".", "..")
                or self.index[0] in "-_+" or len(self.index.encode()) > 255
                or re.search(r'[\\/*?"<>|\s,#:]', self.index)):
            raise ValueError("ELASTICSEARCH_INDEX must name one valid lowercase index")


@dataclass(frozen=True, slots=True, kw_only=True)
class RagSettings:
    postgres: PostgresSettings = field(default_factory=PostgresSettings)
    elasticsearch: ElasticsearchSettings = field(default_factory=ElasticsearchSettings)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None, *, env_file: Path | None = None) -> RagSettings:
        values: dict[str, str] = {}
        if env_file is not None:
            with Path(env_file).open(encoding="utf-8") as stream:
                values.update({k: v for k, v in dotenv_values(stream=stream, interpolate=False).items() if v is not None})
        values.update(os.environ if environ is None else environ)
        if values.get("DB_DRIVER", "postgres").strip() != "postgres":
            raise ValueError("DB_DRIVER must be postgres")
        if values.get("RETRIEVE_DRIVER", "elasticsearch_v8").strip() != "elasticsearch_v8":
            raise ValueError("RETRIEVE_DRIVER must be elasticsearch_v8")
        try:
            port = int(values.get("DB_PORT", "5432"))
        except ValueError as exc:
            raise ValueError("DB_PORT must be a TCP port number") from exc
        return cls(
            postgres=PostgresSettings(
                host=values.get("DB_HOST", "localhost"), port=port, user=values.get("DB_USER", "postgres"),
                password=values.get("DB_PASSWORD", ""), database=values.get("DB_NAME", "my_pi_agent"),
            ),
            elasticsearch=ElasticsearchSettings(
                addr=values.get("ELASTICSEARCH_ADDR", "http://localhost:9200"),
                username=values.get("ELASTICSEARCH_USERNAME", ""), password=values.get("ELASTICSEARCH_PASSWORD", ""),
                index=values.get("ELASTICSEARCH_INDEX") or _DEFAULT_ES_INDEX,
            ),
        )
