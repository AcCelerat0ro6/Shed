"""会话条目的 JSONL 序列化、反序列化及异常定义。

负责 SessionEntry 与单行 JSONL 文本之间的双向转换及行格式校验，
持久化驱动与跨进程文件锁由 storage.py 实现。
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter

from .entries import MessageEntry, SessionEntry


class SessionJsonlError(ValueError):
    """JSONL 序列化、反序列化或行格式损坏时抛出的异常。"""


_ENTRY_ADAPTER: TypeAdapter[SessionEntry] = TypeAdapter(SessionEntry)


def entry_to_json_line(entry: SessionEntry, assets_dir: Path | None = None) -> str:
    """将 SessionEntry 序列化为不含尾随换行符的单行 JSON 字符串。

    若传入 assets_dir 且消息包含内联 Base64 图片，自动将图片转存为磁盘资产并替换为路径引用。
    """
    try:
        if assets_dir is not None and isinstance(entry, MessageEntry) and isinstance(entry.message.content, list):
            from .assets import save_image_asset

            new_content = []
            has_image_to_spill = False
            for part in entry.message.content:
                if isinstance(part, dict) and part.get("type") == "image":
                    raw_data = part.get("data")
                    if raw_data:
                        has_image_to_spill = True
                        meta = save_image_asset(assets_dir, raw_data, str(part.get("mime_type", "image/png")))
                        spilled_block = {
                            "type": "image",
                            "asset_path": meta["asset_path"],
                            "sha256": meta["sha256"],
                            "mime_type": meta["mime_type"],
                            "width": meta["width"],
                            "height": meta["height"],
                            "size_bytes": meta["size_bytes"],
                        }
                        new_content.append(spilled_block)
                        continue
                new_content.append(part)

            if has_image_to_spill:
                entry = entry.model_copy(
                    update={"message": entry.message.model_copy(update={"content": new_content})}
                )
        return entry.model_dump_json(by_alias=True, exclude_none=True)
    except Exception as exc:
        raise SessionJsonlError(f"Failed to serialize entry to JSON line: {exc}") from exc


def entry_from_json_line(line: str) -> SessionEntry:
    """将单行 JSON 字符串解析并校验为具体的 SessionEntry 实例。"""
    stripped = line.strip()
    if not stripped:
        raise SessionJsonlError("JSONL line is empty or blank")

    try:
        raw = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise SessionJsonlError(f"Invalid JSON line: {exc}") from exc

    if not isinstance(raw, dict):
        raise SessionJsonlError(f"Expected JSON object, got {type(raw).__name__}")

    try:
        return _ENTRY_ADAPTER.validate_python(raw)
    except Exception as exc:
        raise SessionJsonlError(f"Cannot parse JSON object as SessionEntry: {exc}") from exc


__all__ = [
    "SessionJsonlError",
    "entry_to_json_line",
    "entry_from_json_line",
]
