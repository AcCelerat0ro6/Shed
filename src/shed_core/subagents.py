"""子代理（Subagent）管理：定义模型、文件发现与系统提示词清单格式化。

SubagentManager 采用仓储模式，在初始化时扫描目录中的 Markdown 定义文件，
按名称建立索引并生成供主代理选择子代理的 XML 清单。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from shed_core.skills import parse_frontmatter


@dataclass(frozen=True)
class Subagent:
    """子代理配置实体。

    name 优先取自 frontmatter（缺省回退为文件名主干），description 供主代理判断是否委派，
    content 为子代理自身的系统提示词正文（不继承父代理系统提示词）。
    """

    name: str
    description: str
    content: str
    file_path: Path
    model: str | None = None  # 为 None 时继承父代理模型配置
    effort: str | None = None  # 当前仅解析保留，暂未向下透传到模型请求参数
    max_turns: int | None = None  # 最大迭代轮次限制，缺省沿用父代理配置
    tools: tuple[str, ...] | None = None  # 允许使用的工具白名单；None 表示继承父代理全部可用工具
    disallowed_tools: tuple[str, ...] = ()  # 禁止使用的工具黑名单
    skills: tuple[str, ...] | None = None  # 注入到子代理系统提示词的技能名称子集


DEFAULT_SUBAGENT_SYSTEM = "You are a subagent. Complete the given task independently, then summarize your findings."
DEFAULT_SUBAGENT = Subagent(
    name="default",
    description="A general-purpose subagent for standalone tasks.",
    content=DEFAULT_SUBAGENT_SYSTEM,
    file_path=Path("<builtin>"),
)


def _split_csv(value: object) -> tuple[str, ...] | None:
    """frontmatter 字段解析：支持 YAML 列表、Python repr 字符串（如 "['read', 'grep']"）或 CSV 逗号分隔字符串。"""
    if not value:
        return None
    if isinstance(value, (list, tuple, set)):
        items = tuple(str(x).strip().strip("'\"") for x in value if str(x).strip())
        return items if items else None
    if isinstance(value, str):
        val = value.strip()
        if val.startswith("[") and val.endswith("]"):
            val = val[1:-1]
        parts = tuple(p.strip().strip("'\"") for p in val.split(",") if p.strip().strip("'\""))
        return parts if parts else None
    return None


def _parse_max_turns(value: object) -> int | None:
    """解析 maxTurns 字段为整数，布尔值或非法输入返回 None。"""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


class SubagentManager:
    """子代理仓储：发现 `*.md` 定义文件、按名称索引并生成 XML 清单。"""

    def __init__(
        self,
        dirs: Sequence[str | Path] | None = None,
        extra_dirs: Sequence[str | Path] | None = None,
    ):
        """初始化并扫描子代理目录。

        dirs 为 None 时自动探测 `<cwd>/.agents/agents`，为 `[]` 时显式禁用，非空序列仅扫描指定目录；
        extra_dirs 用于追加来自插件等外部来源的子代理目录。
        """
        self.subagents: dict[str, Subagent] = {}
        if dirs is None:
            dirs = [Path.cwd() / ".agents" / "agents"]
        for root in dirs:
            self._discover_dir(Path(root))
        if extra_dirs:
            for root in extra_dirs:
                self._discover_dir(Path(root))

    def _discover_dir(self, root: Path) -> None:
        """扫描单层 `*.md` 定义文件（不递归，忽略隐藏文件与 README.md），同名定义后者覆盖前者。"""
        if not root.is_dir():
            return
        for child in sorted(root.iterdir()):
            if child.is_dir() or child.suffix.lower() != ".md":
                continue
            if child.name.startswith(".") or child.name.lower() == "readme.md":
                continue
            sub = self._load_one(child)
            if sub is not None:
                self.subagents[sub.name] = sub

    def _load_one(self, path: Path) -> Subagent | None:
        """读取并解析单个子代理定义文件；I/O 失败、YAML 异常或缺少 description 时返回 None。"""
        try:
            text = path.read_text(encoding="utf-8-sig")
        except OSError:
            return None
        meta, body = parse_frontmatter(text)
        description = (meta.get("description") or "").strip()
        if not description:
            return None
        name = (meta.get("name") or "").strip() or path.stem
        return Subagent(
            name=name,
            description=description,
            content=body,
            file_path=path,
            model=meta.get("model") or None,
            effort=meta.get("effort") or None,
            max_turns=_parse_max_turns(meta.get("maxTurns")),
            tools=_split_csv(meta.get("tools")),
            disallowed_tools=_split_csv(meta.get("disallowedTools")) or (),
            skills=_split_csv(meta.get("skills")),
        )

    def get(self, name: str) -> Subagent | None:
        return self.subagents.get(name)

    def list(self) -> list[Subagent]:
        return list(self.subagents.values())

    def __len__(self) -> int:
        return len(self.subagents)

    def __contains__(self, name: str) -> bool:
        return name in self.subagents

    def format_prompt(self) -> str:
        """将已加载的全部子代理格式化为 `<available_agents>` XML 清单块，无定义时返回空字符串。"""
        if not self.subagents:
            return ""
        parts = ["<available_agents>"]
        for s in self.subagents.values():
            parts.append("  <agent>")
            parts.append(f"    <name>{s.name}</name>")
            parts.append(f"    <description>{s.description}</description>")
            parts.append("  </agent>")
        parts.append("</available_agents>")
        return "\n".join(parts)
