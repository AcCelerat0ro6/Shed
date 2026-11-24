"""插件包发现、清单解析与组件目录聚合管理。

支持的目录布局约定：
- 清单文件：优先读取 `.claude-plugin/plugin.json`、`.plugin/plugin.json` 或根目录 `plugin.json`，缺失或解析失败时按目录名生成默认清单；
- 组件目录：自动识别 `skills/`（兼容 `commands/` 及根目录单技能 `SKILL.md`）、`agents/` 与 `.mcp.json`；
- 资源分发：由 PluginManager 统一聚合各启用插件的资源目录，供 SkillManager、SubagentManager 及 MCP 客户端消费。
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PLUGIN_MANIFEST_DIRS = [".claude-plugin", ".plugin"]
PLUGIN_MANIFEST_FILE = "plugin.json"


@dataclass(frozen=True)
class PluginAuthor:
    """插件作者信息。支持字符串或字典输入。"""

    name: str
    email: str | None = None
    url: str | None = None

    @classmethod
    def from_value(cls, value: Any) -> PluginAuthor:
        if isinstance(value, str):
            m = re.match(r"^(.*?)(?:\s*<([^>]+)>)?$", value.strip())
            if m:
                name = m.group(1).strip()
                email = m.group(2).strip() if m.group(2) else None
                return cls(name=name, email=email)
            return cls(name=value)
        if isinstance(value, dict):
            return cls(
                name=value.get("name", ""),
                email=value.get("email"),
                url=value.get("url"),
            )
        return cls(name="unknown")


@dataclass(frozen=True)
class PluginManifest:
    """插件元数据清单，对应 plugin.json 配置结构。"""

    name: str
    version: str = "1.0.0"
    description: str = ""
    author: PluginAuthor | None = None
    homepage: str | None = None
    repository: str | None = None
    license: str | None = None
    keywords: list[str] = field(default_factory=list)


@dataclass
class Plugin:
    """表示一个已发现的插件包及其元数据清单。"""

    name: str
    path: Path
    manifest: PluginManifest
    enabled: bool = True

    @classmethod
    def from_directory(cls, plugin_dir: Path | str) -> Plugin:
        """从目录加载插件，优先查找清单文件，缺失或损坏时按目录名兜底推断。"""
        p = Path(plugin_dir).resolve()
        manifest_path = None
        for m_dir in PLUGIN_MANIFEST_DIRS:
            candidate = p / m_dir / PLUGIN_MANIFEST_FILE
            if candidate.is_file():
                manifest_path = candidate
                break
        if manifest_path is None:
            root_candidate = p / PLUGIN_MANIFEST_FILE
            if root_candidate.is_file():
                manifest_path = root_candidate

        if manifest_path:
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
                author = None
                if "author" in data:
                    author = PluginAuthor.from_value(data["author"])
                manifest = PluginManifest(
                    name=data.get("name") or p.name,
                    version=data.get("version", "1.0.0"),
                    description=data.get("description", ""),
                    author=author,
                    homepage=data.get("homepage"),
                    repository=data.get("repository"),
                    license=data.get("license"),
                    keywords=data.get("keywords", []),
                )
                return cls(name=manifest.name, path=p, manifest=manifest)
            except Exception as e:
                logger.warning(f"Failed to parse manifest at {manifest_path}: {e}")

        # 无清单或清单解析失败时按目录名生成默认清单
        fallback_manifest = PluginManifest(
            name=p.name,
            version="1.0.0",
            description=f"Plugin loaded from {p.name}",
        )
        return cls(name=p.name, path=p, manifest=fallback_manifest)

    @property
    def skills_dir(self) -> Path | None:
        """返回技能目录：优先 skills/，次选 commands/，单技能简写直接返回插件根目录。"""
        skills = self.path / "skills"
        if skills.is_dir():
            return skills
        commands = self.path / "commands"
        if commands.is_dir():
            return commands
        # 单技能简写：插件根目录直接包含 SKILL.md
        if (self.path / "SKILL.md").is_file():
            return self.path
        return None

    @property
    def agents_dir(self) -> Path | None:
        """返回子代理目录 agents/。"""
        agents = self.path / "agents"
        return agents if agents.is_dir() else None

    @property
    def mcp_config_path(self) -> Path | None:
        """返回 MCP 配置文件路径 .mcp.json。"""
        mcp = self.path / ".mcp.json"
        return mcp if mcp.is_file() else None


class PluginManager:
    """负责插件目录的发现、清单加载及组件资源目录聚合。"""

    def __init__(self, dirs: Sequence[str | Path] | None = None) -> None:
        """初始化并扫描插件目录；None 探测 `<cwd>/.agents/plugins`，`[]` 显式禁用，非空序列扫描指定路径。"""
        self.plugins: dict[str, Plugin] = {}
        if dirs is None:
            default_dir = Path.cwd() / ".agents" / "plugins"
            if default_dir.is_dir():
                self._discover_from_dir(default_dir)
        elif dirs:
            for d in dirs:
                p = Path(d).resolve()
                if p.is_dir():
                    if self._is_single_plugin(p):
                        self._load_one_plugin(p)
                    else:
                        self._discover_from_dir(p)

    def _is_single_plugin(self, p: Path) -> bool:
        """判断是否为单个插件的根目录。"""
        for m_dir in PLUGIN_MANIFEST_DIRS:
            if (p / m_dir / PLUGIN_MANIFEST_FILE).is_file():
                return True
        if (p / PLUGIN_MANIFEST_FILE).is_file():
            return True
        return (
            (p / "skills").is_dir()
            or (p / "agents").is_dir()
            or (p / "commands").is_dir()
            or (p / "SKILL.md").is_file()
        )

    def _load_one_plugin(self, p: Path) -> None:
        """安全加载单个插件，隔离异常。"""
        try:
            plugin = Plugin.from_directory(p)
            self.plugins[plugin.name] = plugin
        except Exception as e:
            logger.warning(f"Failed to load plugin from {p}: {e}")

    def _discover_from_dir(self, root: Path) -> None:
        """扫描目录下的直接子目录。"""
        try:
            for item in sorted(root.iterdir()):
                if item.is_dir() and not item.name.startswith((".", "_")):
                    self._load_one_plugin(item)
        except Exception as e:
            logger.warning(f"Failed to discover plugins from {root}: {e}")

    def get_skill_dirs(self) -> list[Path]:
        """收集所有启用插件的 skills 目录。"""
        return [p.skills_dir for p in self.plugins.values() if p.enabled and p.skills_dir]

    def get_subagent_dirs(self) -> list[Path]:
        """收集所有启用插件的 agents 目录。"""
        return [p.agents_dir for p in self.plugins.values() if p.enabled and p.agents_dir]

    def get_mcp_config_paths(self) -> list[Path]:
        """收集所有启用插件的 .mcp.json 配置文件路径。"""
        return [p.mcp_config_path for p in self.plugins.values() if p.enabled and p.mcp_config_path]
