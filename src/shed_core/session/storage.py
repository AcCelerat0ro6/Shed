# pyright: reportUnusedCallResult=false
"""带跨进程文件锁的只追加会话存储协议与实现。

提供 SessionStorage 异步协议、InMemorySessionStorage 内存实现以及 JsonlSessionStorage 文件实现。
底层跨平台文件锁辅助函数位于模块末尾，避免在模块顶层引入条件分支定义。
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import time
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from .entries import SessionEntry
from .jsonl import (
    SessionJsonlError,
    entry_from_json_line,
    entry_to_json_line,
)


@runtime_checkable
class SessionStorage(Protocol):
    """纯异步只追加会话存储协议，采用不可变追加写入模式以保障历史完整性。"""

    async def append(self, entry: SessionEntry) -> None:
        """追加单个条目到存储末尾。"""
        ...

    async def append_batch(self, entries: Sequence[SessionEntry]) -> None:
        """原子追加一批条目到存储末尾。"""
        ...

    async def read_all(self) -> list[SessionEntry]:
        """读取存储中按追加顺序排列的所有历史条目。"""
        ...


class InMemorySessionStorage:
    """基于内存的只追加会话存储实现，协程安全，适用于单元测试及无需持久化的场景。"""

    def __init__(self, initial_entries: Sequence[SessionEntry] | None = None) -> None:
        self._entries: list[SessionEntry] = list(initial_entries) if initial_entries else []
        self._lock = asyncio.Lock()

    async def append(self, entry: SessionEntry) -> None:
        async with self._lock:
            self._entries.append(entry)

    async def append_batch(self, entries: Sequence[SessionEntry]) -> None:
        async with self._lock:
            self._entries.extend(entries)

    async def read_all(self) -> list[SessionEntry]:
        async with self._lock:
            return list(self._entries)


class JsonlSessionStorage:
    """基于单文件行级追加的 SessionStorage 实现。

    支持 `.{name}.lock` 跨进程文件锁，以及异常退出遗留 `.tmp` 碎片的自愈清理。
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.lock_path = self.path.parent / f".{self.path.name}.lock"
        self._async_lock = asyncio.Lock()
        self._remove_incomplete_temp()

    def _remove_incomplete_temp(self) -> list[Path]:
        """清理异常中断遗留的临时碎片文件。"""
        cleaned: list[Path] = []
        parent = self.path.parent
        if not parent.exists():
            return cleaned

        patterns = [
            f".{self.path.name}*.tmp",
            f"{self.path.name}*.tmp",
        ]
        seen: set[Path] = set()
        for pattern in patterns:
            for tmp_file in parent.glob(pattern):
                if tmp_file in seen or not tmp_file.is_file():
                    continue
                seen.add(tmp_file)
                with contextlib.suppress(OSError):
                    tmp_file.unlink(missing_ok=True)
                    cleaned.append(tmp_file)
        return cleaned

    @contextlib.asynccontextmanager
    async def _process_lock(self, timeout: float = 10.0) -> AsyncIterator[None]:
        """基于 `.{name}.lock` 的跨进程文件锁。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        start_time = time.monotonic()
        fd = os.open(str(self.lock_path), os.O_RDWR | os.O_CREAT)
        try:
            acquired = False
            while not acquired:
                try:
                    _lock_file(fd)
                    acquired = True
                except (BlockingIOError, OSError, PermissionError):
                    if time.monotonic() - start_time > timeout:
                        raise SessionJsonlError(f"Timeout ({timeout}s) waiting for lock on {self.lock_path}") from None
                    await asyncio.sleep(0.01)
            try:
                yield
            finally:
                _unlock_file(fd)
        finally:
            os.close(fd)

    @contextlib.asynccontextmanager
    async def _lock(self, timeout: float = 10.0) -> AsyncIterator[None]:
        """协程锁与跨进程锁的双重上下文。"""
        async with self._async_lock, self._process_lock(timeout=timeout):
            yield

    async def append(self, entry: SessionEntry) -> None:
        """追加单个条目到存储末尾。"""
        async with self._lock():
            self._remove_incomplete_temp()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = entry_to_json_line(entry)
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            except OSError as exc:
                raise SessionJsonlError(f"Failed to append to {self.path}: {exc}") from exc

    async def append_batch(self, entries: Sequence[SessionEntry]) -> None:
        """原子追加一批条目到存储末尾。"""
        if not entries:
            return
        async with self._lock():
            self._remove_incomplete_temp()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            lines = [entry_to_json_line(e) for e in entries]
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    for line in lines:
                        f.write(line + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            except OSError as exc:
                raise SessionJsonlError(f"Failed to append batch to {self.path}: {exc}") from exc

    async def read_all(self) -> list[SessionEntry]:
        """读取存储中所有按追加时序排列的历史条目。"""
        async with self._lock():
            self._remove_incomplete_temp()
            if not self.path.exists():
                return []

            try:
                with open(self.path, encoding="utf-8") as f:
                    raw_lines = f.readlines()
            except OSError as exc:
                raise SessionJsonlError(f"Failed to read {self.path}: {exc}") from exc

            lines = [ln.strip() for ln in raw_lines if ln.strip()]
            if not lines:
                return []

            entries: list[SessionEntry] = []
            for i, line in enumerate(lines):
                is_last = i == len(lines) - 1
                try:
                    entry = entry_from_json_line(line)
                    entries.append(entry)
                except SessionJsonlError as exc:
                    # 尾行撕裂容错：若仅最后一行损坏且前面已有有效条目，忽略残缺尾行
                    if _should_tolerate_tail_error(is_last, len(entries)):
                        break
                    raise exc
            return entries


# 跨平台文件锁与尾行容错辅助函数


def _should_tolerate_tail_error(is_last: bool, count: int) -> bool:
    """判断是否应容忍尾行解析失败（仅当为最后一行且前面已解析出有效条目时返回 True）。"""
    return is_last and count > 0


def _lock_file(fd: int) -> None:
    """获取文件描述符的非阻塞排他锁（按操作系统选择 msvcrt 或 fcntl）。"""
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        return

    import fcntl

    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(fd: int) -> None:
    """底层文件解锁适配。"""
    if sys.platform == "win32":
        import msvcrt

        with contextlib.suppress(OSError):
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    with contextlib.suppress(OSError):
        fcntl.flock(fd, fcntl.LOCK_UN)


__all__ = [
    "SessionStorage",
    "InMemorySessionStorage",
    "JsonlSessionStorage",
]
