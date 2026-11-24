"""租户级检索配置及其数据库 JSON 表示。"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, fields


DEFAULT_RETRIEVAL_TOP_K = 50  # 默认召回深度，也用作混合检索候选池的下限。
MAX_REQUESTED_RESULTS = 200  # knowledge-search 的请求数和请求级 rerank.top_k 的上限。


@dataclass(slots=True, kw_only=True)
class RetrievalConfig:
    """租户级检索配置。字段保存原始值，get_effective_* 方法在读取时应用默认值。"""

    embedding_top_k: int = 0  # 向量召回的分块数。
    vector_threshold: float = 0.0
    keyword_threshold: float = 0.0
    rerank_top_k: int = 0
    rerank_threshold: float = 0.0
    rerank_model_id: str = ""
    rrf_k: int = field(default=0, metadata={"omitempty": True})  # 倒数排名融合的平滑常数。
    rrf_vector_weight: float = field(default=0.0, metadata={"omitempty": True})
    rrf_keyword_weight: float = field(default=0.0, metadata={"omitempty": True})

    def get_effective_embedding_top_k(self) -> int:
        if self is None or self.embedding_top_k <= 0:
            return DEFAULT_RETRIEVAL_TOP_K
        return self.embedding_top_k

    def get_effective_vector_threshold(self) -> float:
        if self is None or self.vector_threshold <= 0:
            return 0.15
        return self.vector_threshold

    def get_effective_keyword_threshold(self) -> float:
        if self is None or self.keyword_threshold <= 0:
            return 0.3
        return self.keyword_threshold

    def get_effective_rerank_top_k(self) -> int:
        if self is None or self.rerank_top_k <= 0:
            return 10
        return self.rerank_top_k

    def get_effective_rerank_threshold(self) -> float:
        if self is None or self.rerank_threshold <= 0:
            return 0.2
        return self.rerank_threshold

    def get_effective_rrf_k(self) -> int:
        if self is None or self.rrf_k <= 0:
            return 60
        return self.rrf_k

    def get_effective_rrf_weights(self) -> tuple[float, float]:
        """按向量、关键词的顺序返回权重，非正值逐项回退，正值按原值使用。"""
        if self is None:
            return 0.7, 0.3
        vector = 0.7 if self.rrf_vector_weight <= 0 else self.rrf_vector_weight
        keyword = 0.3 if self.rrf_keyword_weight <= 0 else self.rrf_keyword_weight
        return vector, keyword

    def value(self) -> bytes:
        """将原始配置编码为 UTF-8 JSON，供数据库写入。"""
        payload = {}
        for item in fields(self):
            value = _config_scalar(item.name, getattr(self, item.name), type(item.default))
            if item.metadata.get("omitempty") and value == 0:
                continue
            payload[item.name] = value
        return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")

    def scan(self, value: object) -> None:
        """从数据库的 JSON 字节更新当前配置。缺失字段和 JSON null 均保留原值。"""
        if not isinstance(value, (bytes, bytearray, memoryview)):
            return
        # 对象保留键值对顺序（包括重复键），JSON 数组仍解码为列表。
        pairs = json.loads(bytes(value).decode("utf-8", errors="replace"),
                           object_pairs_hook=tuple, parse_constant=_reject_json_constant)
        if pairs is None:
            return
        if not isinstance(pairs, tuple):
            raise ValueError("RetrievalConfig requires a JSON object")
        schema = {item.name: type(item.default) for item in fields(self)}
        first_error = None
        for key, raw in pairs:
            name = key.casefold()
            if name not in schema or raw is None:
                continue
            try:
                setattr(self, name, _config_scalar(name, raw, schema[name]))
            except ValueError as error:
                # Go Unmarshal 继续读取其余字段，并在结束后返回第一个字段类型错误。
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


def _reject_json_constant(token: str) -> None:
    raise ValueError(f"invalid JSON number: {token}")


def _config_scalar(name: str, value: object, kind: type) -> int | float | str:
    """按 Go string、64 位 int 或 float64 解码，排除 JSON 布尔值及数值溢出。"""
    if kind is str and isinstance(value, str):
        # Go 将 JSON 字符串中未配对的 UTF-16 代理项解码为替代字符。
        return re.sub(r"[\ud800-\udfff]", "\ufffd", value)
    if kind is int and type(value) is int and -(2**63) <= value < 2**63:
        return value
    if kind is float and isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            number = float(value)
        except OverflowError:
            pass
        else:
            if math.isfinite(number):
                return number
    raise ValueError(f"{name} must be a representable {kind.__name__}")
