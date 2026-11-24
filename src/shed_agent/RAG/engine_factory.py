"""从部署配置打开单一 Elasticsearch 8 后端。"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from urllib.parse import quote

import httpx

from .settings import ElasticsearchSettings

_RETRY_STATUSES = (502, 503, 504)
_MAX_RETRIES = 3


async def _request(
    client: httpx.AsyncClient, method: str, path: str, *, json: Mapping[str, object] | None = None,
) -> httpx.Response:
    # elastic-transport-go v8.9.0 在传输错误及这些状态码下重试三次。
    for _ in range(_MAX_RETRIES):
        try:
            response = await client.request(method, path, json=json)
        except httpx.TransportError:
            continue
        if response.status_code not in _RETRY_STATUSES:
            return response
    return await client.request(method, path, json=json)


@dataclass(frozen=True, slots=True)
class ElasticsearchStore:
    client: httpx.AsyncClient
    index: str

    async def search(self, body: Mapping[str, object]) -> dict:
        """在配置的索引上执行 Elasticsearch Query DSL，供检索仓储复用。"""
        response = await _request(self.client, "POST", f"{quote(self.index, safe='')}/_search", json=dict(body))
        response.raise_for_status()
        return response.json()

    async def ensure_index(self, *, number_of_shards: int = 0, number_of_replicas: int = -1) -> None:
        index_path = quote(self.index, safe="")
        response = await _request(self.client, "HEAD", index_path)
        if response.status_code != 404:
            response.raise_for_status()
            return
        settings = {}
        if number_of_shards > 0:
            settings["number_of_shards"] = number_of_shards
        if number_of_replicas >= 0:
            settings["number_of_replicas"] = number_of_replicas
        response = await _request(self.client, "PUT", index_path, json={"settings": settings} if settings else {})
        response.raise_for_status()

    async def get_mapping(self) -> dict:
        response = await _request(self.client, "GET", f"{quote(self.index, safe='')}/_mapping")
        response.raise_for_status()
        return response.json()


@asynccontextmanager
async def open_elasticsearch(settings: ElasticsearchSettings, *,
                             transport: httpx.AsyncBaseTransport | None = None) -> AsyncIterator[ElasticsearchStore]:
    auth = (settings.username, settings.password) if settings.username else None
    timeout = httpx.Timeout(None, connect=30.0)
    async with httpx.AsyncClient(base_url=settings.addr, auth=auth, transport=transport, timeout=timeout) as client:
        response = await _request(client, "GET", "")
        response.raise_for_status()
        version = response.json().get("version", {}).get("number", "")
        if not isinstance(version, str) or version.partition(".")[0] != "8":
            raise ValueError(f"Expected Elasticsearch 8, received version {version!r}")
        yield ElasticsearchStore(client, settings.index)
