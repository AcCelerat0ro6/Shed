"""会话二进制图片资产的内容寻址持久化与加载工具。"""

from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path
from typing import Any

from PIL import Image

_MIME_TO_EXT: dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
}


def save_image_asset(base_dir: Path, data_base64: str, mime_type: str) -> dict[str, Any]:
    """将 Base64 图片数据以内容寻址方式落盘至 <base_dir>/images/<sha256>.<ext>。

    使用 SHA-256 哈希命名实现自动去重，并通过临时文件原子替换避免部分写入。
    """
    raw_bytes = base64.b64decode(data_base64)
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    normalized_mime = mime_type.lower().split(";")[0].strip()
    ext = _MIME_TO_EXT.get(normalized_mime, ".png")

    images_dir = base_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    asset_rel_path = f"images/{sha256}{ext}"
    asset_file = base_dir / asset_rel_path

    if not asset_file.exists():
        temp_file = images_dir / f".tmp_{sha256}{ext}"
        temp_file.write_bytes(raw_bytes)
        temp_file.replace(asset_file)

    width = 0
    height = 0
    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            width, height = img.size
    except Exception:
        pass

    return {
        "type": "image",
        "asset_path": asset_rel_path,
        "sha256": sha256,
        "mime_type": normalized_mime,
        "width": width,
        "height": height,
        "size_bytes": len(raw_bytes),
    }


def load_image_asset(base_dir: Path, asset_path: str) -> str:
    """从磁盘读取图片资产并编码为 Base64 字符串，供模型请求组装使用。"""
    file_path = base_dir / asset_path
    if not file_path.exists():
        raise FileNotFoundError(f"Image asset file not found: {file_path}")
    raw_bytes = file_path.read_bytes()
    return base64.b64encode(raw_bytes).decode("ascii")
