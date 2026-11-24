"""Skill（技能）管理：数据模型、文件系统发现与提示词格式化。

设计机制：
1. 静态清单注入：仅将技能名称与简要描述以 XML 块注入 System Prompt，控制上下文开销。
2. 按需显式激活：由宿主环境在调用时格式化并注入对应技能的完整正文。
3. 仓储自动发现：SkillManager 构造时扫描目标目录并建立内存索引。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Skill:
    """技能数据模型实体（不可变数据类）。

    Attributes:
        name: 技能唯一名称标识符（优先来自 frontmatter 中的 name，缺省回退为父目录名）。
        description: 技能的功能简述（来自 frontmatter，模型依赖此描述决定是否选用）。
        content: frontmatter 元数据区块之下的 Markdown 格式技能具体操作指令正文。
        file_path: 该技能对应的物理文件路径（Path 对象，主要供调试与追踪来源，不暴露给模型）。
    """

    name: str
    description: str
    content: str  # frontmatter 元数据之下的具体指令正文
    file_path: Path  # 物理路径（用于调试与调用来源标定，不作为模型工具参数）


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """解析 Markdown 文件首部的 YAML Frontmatter 元数据块。

    解析规范：
    - 首部必须以 `---\\n` 开始，并以换行闭合的 `\\n---\\n` 结束。
    - 中间区块由 PyYAML 安全加载为字典，解析出键值对。
    - 闭合标记之后的所有文本作为主体内容（Body）去除首尾空白后返回。

    容错策略：
    - 无 frontmatter、YAML 语法格式错误或解析后非字典结构时，
      均采取平滑降级策略：返回空字典与原始全文，绝不中断抛出异常或产生警告日志。

    Args:
        text: 待解析的 Markdown 文件完整文本内容。

    Returns:
        包含两个元素的元组：(元数据字典, 剩余的正文文本)。
    """
    # 统一换行为 LF 后检测前置 frontmatter 标记
    normalized = text.replace("\r\n", "\n")
    header = "---\n"
    if not normalized.startswith(header):
        return {}, text

    header_end = normalized.find("\n---\n", len(header))
    if header_end == -1:
        return {}, text

    yaml_text = normalized[len(header) : header_end]
    body = normalized[header_end + len("\n---\n") :].strip()

    try:
        fields = yaml.safe_load(yaml_text)
    except yaml.YAMLError:
        # YAML 语法错误时降级返回空元数据和原文
        return {}, text

    if not isinstance(fields, dict):
        return {}, text

    return {str(key): str(value) for key, value in fields.items()}, body


class SkillManager:
    """技能仓储（Repository）：负责技能的动态发现、内存索引以及提示词文本的生成。

    发现规则：
    1. 支持扫描每个来源目录的一层子目录，识别 `<name>/SKILL.md` 结构；
    2. 支持根级单技能简写，即直接识别 `<root>/SKILL.md`；
    3. 技能名称 `name` 优先取 frontmatter 中的 `name`，缺省时取其所在的目录名；
    4. 缺少 `description` 描述信息的技能将被自动静默跳过；
    5. 忽略所有以点（`.`）开头的隐藏目录；
    6. 目录不存在或路径读取失败时全流程静默跳过，保证系统健壮性。
    """

    def __init__(
        self,
        dirs: Sequence[str | Path] | None = None,
        extra_dirs: Sequence[str | Path] | None = None,
    ):
        """初始化技能仓储并即时执行扫描发现。

        Args:
            dirs: 主扫描来源目录列表。
                  - None: 默认自动探测当前工作目录下的 `<cwd>/.agents/skills`；
                  - 空列表 []: 显式禁用技能发现；
                  - 非空列表: 严格仅扫描列表内给定的目录。
            extra_dirs: 额外追加的技能来源目录列表（例如从插件 Plugin 解构出的技能目录）。
        """
        self.skills: dict[str, Skill] = {}

        if dirs is None:
            dirs = [Path.cwd() / ".agents" / "skills"]

        for root in dirs:
            self._discover_dir(Path(root))

        if extra_dirs:
            for root in extra_dirs:
                self._discover_dir(Path(root))

    def _discover_dir(self, root: Path) -> None:
        """扫描单一来源目录中的技能定义文件。

        识别两种文件组织方式：
        1. 根级单技能简写模式：`<root>/SKILL.md`；
        2. 子目录多技能组织模式：`<root>/<child_dir>/SKILL.md`。

        Args:
            root: 待扫描的根目录路径。
        """
        if not root.is_dir():
            return

        # 检查根级单 SKILL.md 简写结构
        root_skill = root / "SKILL.md"
        if root_skill.is_file():
            skill = self._load_one(root_skill)
            if skill is not None:
                self.skills[skill.name] = skill

        # 遍历一层子目录，识别 `<child>/SKILL.md` 并忽略隐藏目录
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name.startswith("."):
                continue
            skill_file = child / "SKILL.md"
            if not skill_file.is_file():
                continue
            skill = self._load_one(skill_file)
            if skill is not None:
                self.skills[skill.name] = skill

    def _load_one(self, path: Path) -> Skill | None:
        """从指定路径读取并解析单个 SKILL.md 文件为 Skill 实体对象。

        Args:
            path: SKILL.md 文件的绝对或相对物理路径。

        Returns:
            成功解析则返回 Skill 对象；若文件不存在、I/O 失败或缺少 description 则静默返回 None。
        """
        try:
            # 使用 utf-8-sig 兼容带 BOM 的 UTF-8 文件
            text = path.read_text(encoding="utf-8-sig")
        except OSError:
            return None

        metadata, body = parse_frontmatter(text)

        description = metadata.get("description")
        if not description:
            return None

        name = (metadata.get("name") or "").strip() or path.parent.name
        return Skill(name=name, description=description, content=body, file_path=path)

    def get(self, name: str) -> Skill | None:
        """按技能名称从仓储中检索指定的技能实体。

        Args:
            name: 技能唯一名称。

        Returns:
            若存在则返回对应 Skill 实体，否则返回 None。
        """
        return self.skills.get(name)

    def list(self) -> list[Skill]:
        """获取仓储中已加载的全部技能列表（保持发现先后顺序）。

        Returns:
            已注册 Skill 实体的列表。
        """
        return list(self.skills.values())

    def __len__(self) -> int:
        """返回当前仓储中已加载的技能总数。"""
        return len(self.skills)

    def __contains__(self, name: str) -> bool:
        """检查指定名称的技能是否存在于仓储中。"""
        return name in self.skills

    def format_prompt(self, names: Sequence[str] | None = None) -> str:
        """将全部（或指定名字子集）的技能元数据格式化为 XML 清单块。

        生成的 XML 结构将被注入到 Agent 的 System Prompt 中，
        仅向大模型提供每个技能的名称和简明功能描述，以便模型判断何时需要使用。

        Args:
            names: 指定只格式化的技能名称子集（常用于子代理按需挂载指定技能）。
                   若为 None 则格式化当前仓储中的所有可用技能；未知名项自动忽略。

        Returns:
            包裹在 `<available_skills>` 中的 XML 文本字符串；若无可用技能则返回空字符串。
        """
        skills = (self.skills[name] for name in names if name in self.skills) if names is not None else self.skills.values()
        parts = ["<available_skills>"]
        for skill in skills:
            parts.append("  <skill>")
            parts.append(f"    <name>{skill.name}</name>")
            parts.append(f"    <description>{skill.description}</description>")
            parts.append("  </skill>")

        if len(parts) == 1:
            return ""
        parts.append("</available_skills>")
        return "\n".join(parts)

    def format_invocation(self, name: str, instructions: str = "") -> str:
        """按技能名称获取其完整正文并包装为技能调用的注入提示词块。

        结构规范：
        `<skill name="..." location="...">\\n{content}\\n</skill>`
        如果提供了可选的附言或补充指令（instructions），则在尾部通过双换行进行衔接。

        Args:
            name: 需要调用的技能名称。
            instructions: 宿主或用户附加的具体上下文指令参数（可选）。

        Returns:
            格式化后可直接追加到上下文对话中的提示词字符串。

        Raises:
            ValueError: 当请求的技能名称未在仓储中注册时抛出，并在错误信息中列出所有可用的技能名称。
        """
        skill = self.get(name)
        if skill is None:
            available = ", ".join(sorted(self.skills)) or "(none)"
            raise ValueError(f"Unknown skill '{name}'. Available: {available}")

        block = f'<skill name="{skill.name}" location="{skill.file_path}">\n{skill.content}\n</skill>'
        return f"{block}\n\n{instructions}" if instructions else block
