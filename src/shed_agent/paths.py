"""全局用户目录与项目本地资源的路径管理。"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

__all__ = ["AgentPaths"]


@dataclass(frozen=True, slots=True)
class AgentPaths:
    """集中解析与管理全局用户目录及项目级路径。"""

    home: Path = field(
        default_factory=lambda: Path(os.environ.get("MY_AGENT_HOME") or (Path.home() / ".my-pi-agent")).resolve()
    )
    agents_home: Path = field(default_factory=lambda: (Path.home() / ".agents").resolve())

    # 全局资源路径
    @property
    def auth_path(self) -> Path:
        return self.home / "auth.json"

    @property
    def auth_lock_path(self) -> Path:
        return self.home / "auth.json.lock"

    @property
    def settings_path(self) -> Path:
        return self.home / "settings.json"

    @property
    def sessions_dir(self) -> Path:
        return self.home / "sessions"

    @property
    def skills_dir(self) -> Path:
        return self.home / "skills"

    @property
    def prompts_dir(self) -> Path:
        return self.home / "prompts"

    @property
    def themes_dir(self) -> Path:
        return self.home / "themes"

    @property
    def extensions_dir(self) -> Path:
        return self.home / "extensions"

    @property
    def logs_dir(self) -> Path:
        return self.home / "logs"

    @property
    def memory_dir(self) -> Path:
        return self.home / "memory"

    # 项目局部路径
    def project_agent_dir(self, cwd: Path) -> Path:
        return cwd / ".my-pi-agent"

    def project_settings_path(self, cwd: Path) -> Path:
        return self.project_agent_dir(cwd) / "settings.json"

    def project_skills_dir(self, cwd: Path) -> Path:
        return self.project_agent_dir(cwd) / "skills"

    def project_agents_skills_dir(self, cwd: Path) -> Path:
        return cwd / ".agents" / "skills"

    # 会话与日志分区路径（基于路径 Slug 与哈希前缀）
    def project_session_dir(self, cwd: Path) -> Path:
        """根据项目工作目录计算全局唯一的 sessions/<slug>-<hash> 目录。"""
        resolved = cwd.resolve()
        digest = sha256(str(resolved).encode("utf-8")).hexdigest()[:6]
        slug = self._slugify_path(resolved)
        target = self.sessions_dir / f"{slug}-{digest}"
        target.mkdir(parents=True, exist_ok=True)
        return target

    def default_session_path(self, cwd: Path) -> Path:
        return self.project_session_dir(cwd) / "default.jsonl"

    def project_logs_dir(self, cwd: Path) -> Path:
        """根据项目工作目录计算全局唯一的 logs/<slug>-<hash> 目录。"""
        resolved = cwd.resolve()
        digest = sha256(str(resolved).encode("utf-8")).hexdigest()[:6]
        slug = self._slugify_path(resolved)
        target = self.logs_dir / f"{slug}-{digest}"
        target.mkdir(parents=True, exist_ok=True)
        return target

    def session_log_path(self, cwd: Path, session_id: str) -> Path:
        """获取指定会话的可读调试日志文件路径 (<session-id>.debug.log)。"""
        return self.project_logs_dir(cwd) / f"{session_id}.debug.log"

    def session_events_path(self, cwd: Path, session_id: str) -> Path:
        """获取指定会话的标准机器可读事件流文件路径 (<session-id>.events.jsonl)。"""
        return self.project_logs_dir(cwd) / f"{session_id}.events.jsonl"

    def ensure_directories(self) -> None:
        """初始化全局目录结构。"""
        for directory in (
            self.home,
            self.sessions_dir,
            self.skills_dir,
            self.prompts_dir,
            self.themes_dir,
            self.extensions_dir,
            self.logs_dir,
            self.memory_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _slugify_path(path: Path, max_length: int = 48) -> str:
        parts = [part for part in path.parts if part not in (path.anchor, "")]
        try:
            relative_to_home = path.relative_to(Path.home())
            parts = ["home", *relative_to_home.parts]
        except ValueError:
            pass
        normalized = [clean for part in parts if (clean := re.sub(r"[^a-zA-Z0-9._-]+", "-", part).strip(".-_").lower())]
        slug = "-".join(normalized)
        if len(slug) <= max_length:
            return slug or "project"

        suffix_parts: list[str] = []
        suffix_length = 0
        for part in reversed(normalized):
            if suffix_length + len(part) + 1 > max_length:
                break
            suffix_parts.append(part)
            suffix_length += len(part) + 1
        return "-".join(reversed(suffix_parts)) or slug[-max_length:].strip("-") or "project"
