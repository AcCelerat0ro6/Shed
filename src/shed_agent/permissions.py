"""业务权限门禁（PermissionGate）：面向编码与无人值守 SRE 排障的多层安全审批门禁。

架构设计概览
------------
本模块基于内核层 ``ToolCallHook`` 协议，在不侵入 ``shed_core`` 内核的前提下，
实现工具调用与 Shell 命令的四层静态威胁分析与权限管控：

1. **L1 原始文本注入与多行红线扫描**：
   前置识别多行脚本（``\\n`` / ``\\r``）、子 Shell 命令替换（``$()`` / 反引号）、
   参数展开绕过（``${``）等高危语法，一律剥夺自动免批资格。
2. **L2 引号感知词法分词与受控管道切分**：
   基于 ``shlex``（``posix=True, punctuation_chars=True``）精确区分引号内字面量与真实操作符。
   拦截分号链（``;``）、逻辑链（``&&`` / ``||``）、后台执行（``&``）、重定向（``>`` / ``>>`` / ``<``）
   及子 Shell 括号；仅对纯单竖线管道（``|``）拆分为子命令段逐段审查。
3. **L3 交互式“假死命令”与灾难性自锁命令前置硬阻断（Fail-Fast Block）**：
   识别裸 ``top``、``htop``、``vim``、``less``、``tail -f``、无限 ``ping`` 等会挂死远端通道
   直至超时的阻塞命令，以及 ``iptables -F``（清空防火墙踢断 SSH）、``rm -rf /``、``shutdown``
   等自锁/毁灭性命令，无需触发人工确认，直接返回 ``HookResult(block=True, reason=...)``
   并附带非交互式替代参数引导模型自动纠偏。
4. **L4 POSIX 短选项簇解构与声明式只读契约放行**：
   结合取值短选项感知解构粘连参数（区分 ``top -bn1`` 与裸 ``top``、``kubectl -owide`` 与 ``-w``），
   对 Linux 系统/网络/日志/容器/K8s 只读探针及安全过滤管道自动免批放行。

权限模式矩阵（PermissionMode）
------------------------------
- ``review``     : 默认人机协同模式。只读工具与安全 Shell 探针/管道自动放行；假死与自锁命令硬阻断；
                   其余潜在变更操作交由 ``confirm_callback`` 人工确认（无回调时默认放行以兼容本地脚本）。
- ``readonly``   : 纯只读定界模式（面向凌晨告警 Webhook 无头自动初筛）。只读工具与安全探针/管道自动放行；
                   假死命令、未知命令及一切写/变更操作（``write``、``edit``、未授权 ``bash``）无需回调一律硬阻断。
- ``strict``     : 零信任严格模式。除假死/自锁命令直接硬阻断外，包括只读工具在内的全部操作均须人工审批。
- ``autonomous`` : 全自主模式（``yolo`` 为其等价别名）。无条件放行全部工具调用（仅限受控沙箱环境使用）。
"""

from __future__ import annotations

import inspect
import shlex
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Literal

from shed_core.hooks import HookResult, ToolCallHook

# 权限运行模式枚举定义（向后兼容原有四种模式，新增 readonly 无头只读定界模式）
PermissionMode = Literal["review", "autonomous", "yolo", "strict", "readonly"]

# Shell 静态威胁分析决策三态：
# - "allow" : 确认为纯只读非阻塞操作，可在 review / readonly 模式下免批放行
# - "ask"   : 存在潜在状态变更、重定向、命令链或未知指令，review 下转人工审批，readonly 下直接拒绝
# - "block" : 确认为交互式挂起假死命令或灾难性自锁命令，前置硬阻断并返回自纠偏提示
BashDecision = Literal["allow", "ask", "block"]

# 免批只读工具集合（包含基础文件检索工具与故障知识库 RAG 检索工具 kb_search）
READONLY_TOOLS: frozenset[str] = frozenset({"read", "grep", "find", "ls", "kb_search"})

# 保留公开常量名以兼容外部模块的直接导入引用；实际判定由四层词法规则引擎驱动
SAFE_BASH_PREFIXES: tuple[str, ...] = (
    "git status",
    "git diff",
    "git log",
    "pytest",
    "python -m pytest",
    "uv run pytest",
)

# Shell 元标点字符集（配合 shlex punctuation_chars=True 识别未加引号的真实控制/重定向算子）
_SHELL_PUNCT_CHARS: frozenset[str] = frozenset("();<>|&")

# 原始命令字符串中禁止自动免批的高危展开与换行特征（防命令替换、变量注入与多行绕过）
_RAW_INJECTION_MARKERS: tuple[str, ...] = ("\n", "\r", "$(", "`", "${")

# 常见提权或执行包装前缀：剥离后用于穿透识别底层假死/自锁命令，但带包装前缀的命令绝不自动免批放行
_WRAPPER_COMMANDS: frozenset[str] = frozenset({"sudo", "nohup", "time", "nice", "ionice", "stdbuf", "timeout"})

# 高危系统根路径集合（用于前置硬阻断 rm -rf / 等灾难性误删，同时保留 rm -rf /tmp/test 走常规人工审批）
_CRITICAL_SYSTEM_PATHS: frozenset[str] = frozenset(
    {
        "/",
        "/*",
        "/bin",
        "/bin/*",
        "/boot",
        "/boot/*",
        "/dev",
        "/dev/*",
        "/etc",
        "/etc/*",
        "/lib",
        "/lib/*",
        "/lib64",
        "/lib64/*",
        "/proc",
        "/root",
        "/sbin",
        "/sbin/*",
        "/sys",
        "/usr",
        "/usr/*",
        "/var",
        "/var/*",
    }
)

# 纯交互式全屏/分页/终端复用黑名单：无论携带何种参数，在非交互子进程管道中均会导致挂死或异常，一律硬阻断
_INTERACTIVE_BLACKLIST: dict[str, str] = {
    "vim": "检测到交互式编辑器 [vim]，会挂死远程执行通道；请改用 read / edit / write 工具或 cat / head / tail。",
    "vi": "检测到交互式编辑器 [vi]，会挂死远程执行通道；请改用 read / edit / write 工具或 cat / head / tail。",
    "nvim": "检测到交互式编辑器 [nvim]，会挂死远程执行通道；请改用 read / edit / write 工具或 cat / head / tail。",
    "nano": "检测到交互式编辑器 [nano]，会挂死远程执行通道；请改用 read / edit / write 工具或 cat / head / tail。",
    "emacs": "检测到交互式编辑器 [emacs]，会挂死远程执行通道；请改用 read / edit / write 工具。",
    "less": "检测到交互式分页器 [less]，会等待键盘输入导致通道挂起；请改用 head -n 100、tail -n 100 或 cat。",
    "more": "检测到交互式分页器 [more]，会等待键盘输入导致通道挂起；请改用 head -n 100、tail -n 100 或 cat。",
    "most": "检测到交互式分页器 [most]，会等待键盘输入导致通道挂起；请改用 head -n 100 或 tail -n 100。",
    "htop": "检测到全屏交互式监控命令 [htop]，会挂死执行通道；请改用 top -bn1 或 ps aux --sort=-%cpu | head -n 20。",
    "btop": "检测到全屏交互式监控命令 [btop]，会挂死执行通道；请改用 top -bn1 或 ps aux | head -n 20。",
    "atop": "检测到全屏交互式监控命令 [atop]，会挂死执行通道；请改用 top -bn1 或 vmstat 1 3。",
    "iotop": "检测到交互式磁盘监控命令 [iotop]，默认持续刷新挂死通道；请改用 iostat -xz 1 2 或 pidstat -d 1 2。",
    "iftop": "检测到交互式网络监控命令 [iftop]，会挂死执行通道；请改用 ss -tulnp 或 ip -s link。",
    "nethogs": "检测到交互式流量监控命令 [nethogs]，会挂死执行通道；请改用 ss -tulnp 或 ip -s link。",
    "nmon": "检测到交互式监控命令 [nmon]，会挂死执行通道；请改用 top -bn1、vmstat 1 3 或 iostat -xz 1 2。",
    "glances": "检测到全屏交互式监控命令 [glances]，会挂死执行通道；请改用 top -bn1、df -h 或 free -m。",
    "watch": "检测到持续轮询命令 [watch]，会无限挂起通道；请直接单次执行目标诊断命令。",
    "tmux": "检测到终端复用器 [tmux]，禁止在自动化排障通道中开启交互式会话。",
    "screen": "检测到终端复用器 [screen]，禁止在自动化排障通道中开启交互式会话。",
    "telnet": "检测到交互式连接命令 [telnet]，易导致通道挂死；探测端口连通性请改用 ss -tulnp 或 curl --max-time 5。",
    "ftp": "检测到交互式文件传输命令 [ftp]，禁止在非交互通道中使用。",
    "sftp": "检测到交互式文件传输命令 [sftp]，禁止在非交互通道中使用。",
}

# 自定义子命令段额外校验函数签名：(flags, positionals, raw_segment) -> (BashDecision, reason)
SegmentValidator = Callable[[frozenset[str], tuple[str, ...], tuple[str, ...]], tuple[BashDecision, str | None]]


@dataclass(frozen=True)
class CommandRule:
    """单条可执行命令的声明式静态安全契约。

    Attributes:
        allowed_subcommands: 若非 None，则要求第一个非选项位置参数必须落入该只读子命令白名单，否则降级为 ``ask``。
        required_all_flags: 必须**同时**携带的全部选项集合（如 ``top`` 要求同时具备 ``-b`` 与 ``-n``）；缺失则触发 ``block``。
        required_any_flags: 必须**至少携带其一**的选项集合（如 ``ping`` 要求 ``-c`` 或 ``-w``）；全部缺失则触发 ``block``。
        missing_required_reason: 缺失必选非阻塞标志时返回给大模型的中文自纠偏提示。
        blocked_any_flags: 严禁携带的持续流式/交互挂死标志集合（如 ``-f`` / ``--follow``）；命中任意一个即触发 ``block``。
        blocked_flag_reason: 命中持续阻塞标志时返回给大模型的中文自纠偏提示。
        denied_any_flags: 涉及写盘/原地修改/清理的标志集合（如 ``sed -i``、``find -delete``）；命中后降级为 ``ask``。
        value_short_flags: 携带紧邻参数值的短选项集合（如 ``-o``、``-u``、``-n``），解构粘连短选项簇时遇到即停止拆分后续字符。
        split_short_clusters: 是否按 POSIX 规范拆分单横线粘连短选项簇（GNU ``find`` / ``ip`` 等单横线长词工具设为 ``False``）。
        max_positionals: 允许免批的最大非选项位置参数个数（如 ``mount`` 无参只读、带参挂载；``uniq`` 超 1 个位置参数会写输出文件）。
        allow_in_pipe: 是否允许作为多段管道（``cmd1 | cmd2``）的组成部分自动免批放行。
        custom_validator: 针对复杂多级子命令（如 ``docker stats``、``kubectl edit``、``ip addr add``）的精细化校验钩子。
    """

    allowed_subcommands: frozenset[str] | None = None
    required_all_flags: frozenset[str] = field(default_factory=frozenset)
    required_any_flags: frozenset[str] = field(default_factory=frozenset)
    missing_required_reason: str = ""
    blocked_any_flags: frozenset[str] = field(default_factory=frozenset)
    blocked_flag_reason: str = ""
    denied_any_flags: frozenset[str] = field(default_factory=frozenset)
    value_short_flags: frozenset[str] = field(default_factory=frozenset)
    split_short_clusters: bool = True
    max_positionals: int | None = None
    allow_in_pipe: bool = True
    custom_validator: SegmentValidator | None = None


@dataclass(frozen=True)
class PermissionRequest:
    """人工权限审批请求载荷。

    Attributes:
        action: 请求执行的工具名称（如 ``bash``、``write``、``edit``）。
        target: 操作核心目标摘要（文件路径或待执行的完整命令行文本）。
        details: 工具调用原始参数字典副本。
        preview: 针对 ``write`` / ``edit`` 工具提取的内容变更预览文本。
    """

    action: str
    target: str
    details: dict[str, Any] = field(default_factory=dict)
    preview: str | None = None


# 复杂工具精细化校验钩子（Docker / Kubectl / IP / 周期采样器 / Python 与 UV 测试）


def _validate_docker(
    flags: frozenset[str],
    positionals: tuple[str, ...],
    _raw: tuple[str, ...],
) -> tuple[BashDecision, str | None]:
    """校验 docker 子命令：区分无流快照、持续流阻塞与容器状态变更操作。"""
    if not positionals:
        return ("ask", None)

    subcmd = positionals[0]
    if subcmd == "stats":
        if "--no-stream" not in flags:
            return (
                "block",
                "检测到持续刷新命令 [docker stats]（缺少 --no-stream 会挂死通道），请改用 docker stats --no-stream。",
            )
        return ("allow", None)

    if subcmd == "logs":
        if flags & {"-f", "--follow"}:
            return (
                "block",
                "检测到持续流式命令 [docker logs -f]，会挂死远程通道；请改用 docker logs --tail 100 <容器名>。",
            )
        return ("allow", None)

    if subcmd == "events":
        if "--until" not in flags:
            return (
                "block",
                "检测到无限事件监听 [docker events]，请补充 --since 10m --until 0s 获取有限时间窗快照。",
            )
        return ("allow", None)

    if subcmd in {"ps", "inspect", "info", "version", "images", "top", "port"}:
        return ("allow", None)

    # 二级复合只读子命令支持（如 docker network ls / inspect，docker volume ls / inspect，docker compose ps / logs）
    if subcmd in {"network", "volume"} and len(positionals) >= 2 and positionals[1] in {"ls", "inspect"}:
        return ("allow", None)

    if subcmd == "compose" and len(positionals) >= 2:
        comp_sub = positionals[1]
        if comp_sub == "ps":
            return ("allow", None)
        if comp_sub == "logs":
            if flags & {"-f", "--follow"}:
                return (
                    "block",
                    "检测到持续流式命令 [docker compose logs -f]，请改用 docker compose logs --tail 100。",
                )
            return ("allow", None)

    return ("ask", None)


def _validate_kubectl(
    flags: frozenset[str],
    positionals: tuple[str, ...],
    _raw: tuple[str, ...],
) -> tuple[BashDecision, str | None]:
    """校验 kubectl 子命令：阻断 edit / watch / logs -f 等挂起操作，放行只读资源查询。"""
    if not positionals:
        return ("ask", None)

    subcmd = positionals[0]
    if subcmd == "edit":
        return (
            "block",
            "检测到交互式命令 [kubectl edit]，会拉起终端编辑器挂死通道；只读排查请用 kubectl get -o yaml。",
        )

    if subcmd in {"attach", "port-forward", "proxy"}:
        return (
            "block",
            f"检测到长连接阻塞命令 [kubectl {subcmd}]，会持续占用并挂死远程执行通道。",
        )

    if flags & {"-w", "--watch", "--watch-only"}:
        return (
            "block",
            f"检测到持续监听参数 [kubectl {subcmd} -w/--watch]，请移除 watch 参数以执行单次快照查询。",
        )

    if subcmd == "logs":
        if flags & {"-f", "--follow"}:
            return (
                "block",
                "检测到持续流式日志命令 [kubectl logs -f]，请改用 kubectl logs --tail=100 <Pod名称>。",
            )
        return ("allow", None)

    if subcmd in {"get", "describe", "top", "events", "cluster-info", "version", "api-resources", "explain"}:
        return ("allow", None)

    return ("ask", None)


def _validate_ip_command(
    _flags: frozenset[str],
    positionals: tuple[str, ...],
    _raw: tuple[str, ...],
) -> tuple[BashDecision, str | None]:
    """校验 Linux iproute2 (ip) 命令：放行 addr/route/link/neigh/rule 查看，拦截 add/del/set/flush 变更。"""
    if not positionals:
        return ("ask", None)

    obj = positionals[0]
    readonly_objects = {"addr", "address", "a", "route", "r", "link", "l", "neigh", "neighbour", "n", "rule"}
    if obj not in readonly_objects:
        return ("ask", None)

    # 无二级动作（默认等价于 show）或显式指定 show / list / get 时为纯只读操作
    if len(positionals) == 1 or positionals[1] in {"show", "list", "ls", "get"}:
        return ("allow", None)

    return ("ask", None)


def _validate_periodic_stat(
    _flags: frozenset[str],
    positionals: tuple[str, ...],
    raw: tuple[str, ...],
) -> tuple[BashDecision, str | None]:
    """校验 vmstat / iostat / mpstat / pidstat：拦截仅指定采样间隔未指定次数的无限输出用法（如 vmstat 1）。"""
    cmd_name = raw[0] if raw else "stat"
    numeric_pos = [p for p in positionals if p.isdigit()]
    if len(numeric_pos) == 1:
        interval = numeric_pos[0]
        return (
            "block",
            f"检测到无限采样命令 [{cmd_name} {interval}]（仅指定间隔未限制次数会持续挂起），"
            f"请改用 {cmd_name} {interval} 3 指定有限采样次数，或直接运行不带间隔参数的 {cmd_name}。",
        )
    return ("allow", None)


def _validate_python_pytest(
    _flags: frozenset[str],
    _positionals: tuple[str, ...],
    raw: tuple[str, ...],
) -> tuple[BashDecision, str | None]:
    """校验 python / python3 命令：仅在精确调用 `-m pytest` 时免批放行，防止 `python -c` 任意代码执行。"""
    if len(raw) >= 3 and raw[1] == "-m" and raw[2] == "pytest":
        return ("allow", None)
    return ("ask", None)


def _validate_uv_pytest(
    _flags: frozenset[str],
    _positionals: tuple[str, ...],
    raw: tuple[str, ...],
) -> tuple[BashDecision, str | None]:
    """校验 uv 命令：仅在精确调用 `uv run pytest` 时免批放行。"""
    if len(raw) >= 3 and raw[1] == "run" and raw[2] == "pytest":
        return ("allow", None)
    return ("ask", None)


# 内置 SRE 诊断探针与研发只读命令策略表
# 严禁将 xargs、tee、awk、sh、bash、eval、curl、wget 等具备任意执行或写文件能力的程序放入此表

DEFAULT_COMMAND_RULES: Mapping[str, CommandRule] = {
    # 1. 进程、CPU、内存与系统负载探针
    "ps": CommandRule(value_short_flags=frozenset({"-o", "-O", "-p", "-u", "-U", "-g", "-G", "-t", "-C"})),
    "pgrep": CommandRule(value_short_flags=frozenset({"-u", "-U", "-G", "-P", "-g", "-s", "-t", "-d"})),
    "pidof": CommandRule(),
    "pstree": CommandRule(),
    "top": CommandRule(
        required_all_flags=frozenset({"-b", "-n"}),
        missing_required_reason=(
            "检测到交互式挂起命令 [top]（缺少批处理或迭代次数标志会挂死通道），"
            "请改用 top -bn1（或 top -b -n 1）采集单帧快照。"
        ),
        value_short_flags=frozenset({"-n", "-d", "-p", "-u", "-U", "-w"}),
    ),
    "uptime": CommandRule(),
    "free": CommandRule(
        blocked_any_flags=frozenset({"-s", "--seconds"}),
        blocked_flag_reason="检测到持续轮询参数 [free -s]，请移除 -s 仅执行单次内存快照查询（如 free -m）。",
        value_short_flags=frozenset({"-s", "-c"}),
    ),
    "vmstat": CommandRule(
        value_short_flags=frozenset({"-S", "-d"}),
        custom_validator=_validate_periodic_stat,
    ),
    "iostat": CommandRule(custom_validator=_validate_periodic_stat),
    "mpstat": CommandRule(
        value_short_flags=frozenset({"-P", "-I", "-u"}),
        custom_validator=_validate_periodic_stat,
    ),
    "pidstat": CommandRule(
        value_short_flags=frozenset({"-p", "-u", "-C", "-U"}),
        custom_validator=_validate_periodic_stat,
    ),
    "lsof": CommandRule(
        blocked_any_flags=frozenset({"-r", "+r"}),
        blocked_flag_reason="检测到重复刷新参数 [lsof -r/+r]，请移除该参数以执行单次句柄查询。",
        value_short_flags=frozenset({"-p", "-u", "-c", "-i", "-d", "-s"}),
    ),
    # 2. 磁盘、挂载、硬件与主机元数据探针
    "df": CommandRule(value_short_flags=frozenset({"-t", "-x", "-B"})),
    "du": CommandRule(value_short_flags=frozenset({"-d", "-B", "-t"})),
    "lsblk": CommandRule(value_short_flags=frozenset({"-o", "-e", "-I"})),
    "mount": CommandRule(max_positionals=0, value_short_flags=frozenset({"-t"})),
    "uname": CommandRule(),
    "hostname": CommandRule(max_positionals=0),
    "hostnamectl": CommandRule(allowed_subcommands=frozenset({"status"})),
    "timedatectl": CommandRule(allowed_subcommands=frozenset({"status", "show"})),
    "lscpu": CommandRule(),
    "lsmem": CommandRule(),
    "lsmod": CommandRule(),
    "lspci": CommandRule(),
    "lsusb": CommandRule(),
    "nproc": CommandRule(),
    "arch": CommandRule(),
    "id": CommandRule(),
    "whoami": CommandRule(max_positionals=0),
    "w": CommandRule(),
    "who": CommandRule(),
    "last": CommandRule(value_short_flags=frozenset({"-n", "-f", "-t"})),
    "date": CommandRule(
        denied_any_flags=frozenset({"-s", "--set"}),
        value_short_flags=frozenset({"-d", "-r", "-f", "-I", "-s"}),
    ),
    "printenv": CommandRule(),
    # 3. 文件目录只读查看探针
    "ls": CommandRule(value_short_flags=frozenset({"-I", "-w", "-T"})),
    "stat": CommandRule(value_short_flags=frozenset({"-c", "--format"})),
    "file": CommandRule(),
    "which": CommandRule(),
    "whereis": CommandRule(),
    "realpath": CommandRule(),
    "readlink": CommandRule(),
    "pwd": CommandRule(max_positionals=0),
    "md5sum": CommandRule(),
    "sha256sum": CommandRule(),
    # 4. 网络套接字、路由与连通性探针
    "ss": CommandRule(value_short_flags=frozenset({"-f", "-A", "-D", "-F"})),
    "netstat": CommandRule(
        blocked_any_flags=frozenset({"-c", "--continuous"}),
        blocked_flag_reason="检测到持续刷新命令 [netstat -c]，请移除 -c 改用 ss -tulnp 或单次 netstat -tulnp。",
    ),
    "ip": CommandRule(
        split_short_clusters=False,
        custom_validator=_validate_ip_command,
    ),
    "ifconfig": CommandRule(max_positionals=1),
    "ping": CommandRule(
        required_any_flags=frozenset({"-c", "-w"}),
        missing_required_reason=(
            "检测到无限挂起命令 [ping]（Linux 下默认持续发包不退出），"
            "请改用 ping -c 4 <目标地址> 限制发包次数。"
        ),
        value_short_flags=frozenset({"-c", "-w", "-W", "-i", "-s", "-t", "-I", "-m"}),
    ),
    "ping6": CommandRule(
        required_any_flags=frozenset({"-c", "-w"}),
        missing_required_reason="检测到无限挂起命令 [ping6]，请改用 ping6 -c 4 <目标地址> 限制发包次数。",
        value_short_flags=frozenset({"-c", "-w", "-W", "-i", "-s", "-t", "-I"}),
    ),
    "dig": CommandRule(),
    "nslookup": CommandRule(),
    "host": CommandRule(),
    "traceroute": CommandRule(value_short_flags=frozenset({"-m", "-q", "-w", "-p", "-i", "-s"})),
    "tracepath": CommandRule(value_short_flags=frozenset({"-m", "-p", "-l"})),
    # 5. 系统服务、内核日志、容器与 Kubernetes 诊断探针
    "systemctl": CommandRule(
        allowed_subcommands=frozenset(
            {
                "status",
                "is-active",
                "is-enabled",
                "is-failed",
                "list-units",
                "list-unit-files",
                "list-timers",
                "list-sockets",
                "list-dependencies",
                "show",
                "cat",
            }
        ),
        value_short_flags=frozenset({"-t", "-p", "-n", "-o", "-H", "-M"}),
    ),
    "journalctl": CommandRule(
        blocked_any_flags=frozenset({"-f", "--follow"}),
        blocked_flag_reason=(
            "检测到持续流式命令 [journalctl -f]，会挂死远程执行通道；"
            "请改用 journalctl -u <服务名> -n 200 --no-pager 获取最近日志快照。"
        ),
        denied_any_flags=frozenset(
            {
                "--vacuum-size",
                "--vacuum-time",
                "--vacuum-files",
                "--rotate",
                "--flush",
                "--sync",
                "--relinquish-var",
            }
        ),
        value_short_flags=frozenset({"-u", "-n", "-p", "-S", "-U", "-t", "-o", "-D", "-F", "-M"}),
    ),
    "dmesg": CommandRule(
        blocked_any_flags=frozenset({"-w", "--follow"}),
        blocked_flag_reason="检测到持续监听命令 [dmesg -w]，请改用 dmesg -T | tail -n 100。",
        denied_any_flags=frozenset({"-c", "-C", "--clear", "--read-clear"}),
        value_short_flags=frozenset({"-l", "-f", "-F", "-s"}),
    ),
    "docker": CommandRule(
        value_short_flags=frozenset({"-H", "-c", "-l", "-f", "-n"}),
        custom_validator=_validate_docker,
    ),
    "kubectl": CommandRule(
        value_short_flags=frozenset({"-n", "-o", "-l", "-s", "-c"}),
        custom_validator=_validate_kubectl,
    ),
    # 6. 文本查看与安全只读过滤管道组件
    "cat": CommandRule(),
    "tac": CommandRule(),
    "nl": CommandRule(value_short_flags=frozenset({"-b", "-d", "-f", "-h", "-i", "-l", "-n", "-s", "-v", "-w"})),
    "head": CommandRule(value_short_flags=frozenset({"-n", "-c"})),
    "tail": CommandRule(
        blocked_any_flags=frozenset({"-f", "-F", "--follow"}),
        blocked_flag_reason=(
            "检测到持续流式追踪命令 [tail -f/-F]，会挂死远程执行通道；"
            "请改用 tail -n 100 <文件路径> 读取最近日志快照。"
        ),
        value_short_flags=frozenset({"-n", "-c", "-s"}),
    ),
    "grep": CommandRule(value_short_flags=frozenset({"-e", "-f", "-m", "-A", "-B", "-C", "-d", "-D"})),
    "egrep": CommandRule(value_short_flags=frozenset({"-e", "-f", "-m", "-A", "-B", "-C"})),
    "fgrep": CommandRule(value_short_flags=frozenset({"-e", "-f", "-m", "-A", "-B", "-C"})),
    "zgrep": CommandRule(value_short_flags=frozenset({"-e", "-f", "-m", "-A", "-B", "-C"})),
    "rg": CommandRule(value_short_flags=frozenset({"-e", "-f", "-m", "-A", "-B", "-C", "-g", "-t", "-T", "-M"})),
    "wc": CommandRule(),
    "cut": CommandRule(value_short_flags=frozenset({"-b", "-c", "-d", "-f"})),
    "tr": CommandRule(),
    "sort": CommandRule(
        denied_any_flags=frozenset({"-o", "--output"}),
        value_short_flags=frozenset({"-k", "-t", "-o", "-S", "-T"}),
    ),
    "uniq": CommandRule(
        max_positionals=1,
        value_short_flags=frozenset({"-f", "-s", "-w"}),
    ),
    "column": CommandRule(value_short_flags=frozenset({"-s", "-o", "-c", "-N", "-R", "-T", "-W"})),
    "sed": CommandRule(
        denied_any_flags=frozenset({"-i", "--in-place", "-f", "--file"}),
        value_short_flags=frozenset({"-e", "-f", "-l"}),
    ),
    "find": CommandRule(
        split_short_clusters=False,
        denied_any_flags=frozenset(
            {
                "-delete",
                "-exec",
                "-execdir",
                "-ok",
                "-okdir",
                "-fprint",
                "-fprint0",
                "-fprintf",
                "-fls",
            }
        ),
    ),
    # 7. 本地研发、版本控制与单元测试白名单命令（向后兼容）
    # 注：git 设 allow_in_pipe=False，单条 git status/diff/log 免批，拼装管道时走人工审批以防仓库敏感信息外泄
    "git": CommandRule(
        allowed_subcommands=frozenset({"status", "diff", "log"}),
        allow_in_pipe=False,
        value_short_flags=frozenset({"-C", "-c", "-n", "-U", "-S", "-G"}),
    ),
    "pytest": CommandRule(
        value_short_flags=frozenset({"-k", "-m", "-c", "-o", "-n", "-p", "-W"}),
    ),
    "python": CommandRule(
        split_short_clusters=False,
        custom_validator=_validate_python_pytest,
    ),
    "python3": CommandRule(
        split_short_clusters=False,
        custom_validator=_validate_python_pytest,
    ),
    "uv": CommandRule(
        split_short_clusters=False,
        custom_validator=_validate_uv_pytest,
    ),
}


# 四层词法解构与威胁判定核心算法


def extract_flags_and_positionals(
    segment: list[str] | tuple[str, ...],
    *,
    value_short_flags: frozenset[str] = frozenset(),
    split_short_clusters: bool = True,
) -> tuple[frozenset[str], tuple[str, ...]]:
    """从单个命令段的分词结果（不含 ``segment[0]`` 程序名）中提取规范化选项集合与位置参数序列。

    解析规则符合 POSIX ``getopt`` 与 GNU 长选项习惯：
    1. 遇到 ``--`` 终止符后，后续全部词元均归入 ``positionals``；
    2. ``--long=value`` 提取 ``--long`` 选项名；
    3. ``-abc`` 单横线短选项簇（当 ``split_short_clusters=True`` 时）逐字符拆解为 ``-a``、``-b``……
       一旦遇到数字（如 ``top -bn1`` 中的 ``1``）或命中 ``value_short_flags`` 中的带值短选项
       （如 ``kubectl -owide`` 中的 ``-o``、``journalctl -unginx`` 中的 ``-u``），立即终止当前词元拆分，
       防止将紧邻的参数值字母误判为其他短选项。
       若带值选项独立成词（如 ``-u nginx``），同时自动跳过紧随其后的参数值词元，确保子命令定位准确。
    """
    flags: set[str] = set()
    positionals: list[str] = []
    end_of_opts = False
    skip_next = False

    tokens = segment[1:]
    for tok in tokens:
        if skip_next:
            skip_next = False
            continue

        if end_of_opts:
            positionals.append(tok)
            continue

        if tok == "--":
            end_of_opts = True
            continue

        # 单个 "-" 代表标准输入流（stdin），归为位置参数
        if tok == "-" or not tok.startswith("-"):
            positionals.append(tok)
            continue

        # 双横线 GNU 长选项（如 --no-stream、--follow=true）
        if tok.startswith("--"):
            long_name = tok.split("=", 1)[0]
            flags.add(long_name)
            continue

        # 单横线长谓词模式（如 find -delete / ip -br addr）
        if not split_short_clusters:
            opt_name = tok.split("=", 1)[0]
            flags.add(opt_name)
            continue

        # POSIX 单横线短选项及粘连选项簇（如 -bn1、-tulnp、-u nginx、-owide）
        cluster = tok[1:]
        for idx, ch in enumerate(cluster):
            if ch.isdigit() or ch in ".,=:":
                # 遇到粘连数值参数（如 -100、-bn1 中的 1），停止继续拆分字符
                break
            short_flag = f"-{ch}"
            flags.add(short_flag)
            if short_flag in value_short_flags:
                # 若该带参短选项后无剩余粘连字符（如 "-u" 独立成词），则下一个词元为其参数值而非子命令
                if idx == len(cluster) - 1:
                    skip_next = True
                break

    return frozenset(flags), tuple(positionals)


def _unwrap_command_segment(segment: list[str]) -> tuple[list[str], bool]:
    """剥离常见命令包装前缀（如 ``sudo -n``、``nohup``、``env`` 变量赋值），定位底层真实被调程序。

    Returns:
        (底层命令段词元列表, 是否经过包装或环境变量修饰)
    """
    idx = 0
    n = len(segment)
    wrapped = False

    while idx < n:
        head = segment[idx]
        # 跳过形如 KEY=VALUE 的前置环境变量赋值，并标记为已修饰（禁止自动免批）
        if "=" in head and not head.startswith("-"):
            wrapped = True
            idx += 1
            continue

        base_name = PurePosixPath(head.replace("\\", "/")).name
        if base_name == "env":
            wrapped = True
            idx += 1
            while idx < n and (segment[idx].startswith("-") or "=" in segment[idx]):
                idx += 1
            continue

        if base_name in _WRAPPER_COMMANDS:
            wrapped = True
            idx += 1
            # 跳过 sudo / nice 等包装器自身的常见无参短选项（如 sudo -n / sudo -E）
            while idx < n and segment[idx].startswith("-"):
                opt = segment[idx]
                idx += 1
                if base_name == "sudo" and opt in {"-u", "-g", "-C", "-D", "-R", "-T"} and idx < n:
                    idx += 1
                elif base_name in {"nice", "timeout"} and opt in {"-n", "-k", "-s"} and idx < n:
                    idx += 1
            # timeout 首个非选项参数为时长（如 timeout 5 ping ...）
            if base_name == "timeout" and idx < n:
                idx += 1
            continue

        break

    return segment[idx:], wrapped


def _check_catastrophic_or_interactive_block(segment: list[str]) -> str | None:
    """检查单个子命令段是否命中 L3 交互式挂死黑名单或生产灾难/SSH 自锁红线。

    若命中硬阻断规则则返回详尽的中文拦截与纠偏原因；未命中则返回 ``None``。
    """
    core_seg, _ = _unwrap_command_segment(segment)
    if not core_seg:
        return None

    raw_prog = core_seg[0]
    prog_name = PurePosixPath(raw_prog.replace("\\", "/")).name

    # 1. 纯交互式全屏/分页/编辑器黑名单（任何路径调用如 /usr/bin/vim 一律拦截）
    if prog_name in _INTERACTIVE_BLACKLIST:
        return _INTERACTIVE_BLACKLIST[prog_name]

    # 2. 防火墙清空/重置自锁拦截（防止清空默认 DROP 链瞬间踢断 Agent 自身的 SSH 连接）
    if prog_name in {"iptables", "ip6tables"}:
        flags, positionals = extract_flags_and_positionals(
            core_seg,
            value_short_flags=frozenset({"-t", "-A", "-D", "-I", "-R", "-L", "-S", "-N", "-X", "-P"}),
        )
        if flags & {"-F", "--flush"}:
            return (
                f"检测到高危自锁命令 [{prog_name} -F/--flush]（清空防火墙规则极易瞬间阻断当前 SSH 会话），"
                f"已被门禁前置硬阻断；只读排查请使用 {prog_name} -L -n -v。"
            )
        if ("-P" in flags or "--policy" in flags) and any(p.upper() in {"DROP", "REJECT"} for p in core_seg[1:]):
            return (
                f"检测到高危防火墙默认策略变更 [{prog_name} -P DROP/REJECT]，易导致主机网络自锁，已被硬阻断。"
            )
        if any(p in {"-F", "--flush"} for p in positionals):
            return f"检测到高危自锁命令 [{prog_name} flush]，已被门禁前置硬阻断。"

    if prog_name == "nft" and len(core_seg) >= 2 and "flush" in core_seg[1:]:
        return "检测到高危防火墙清空命令 [nft flush]，极易踢断当前远程连接，已被前置硬阻断；查看规则请用 nft list ruleset。"

    if prog_name == "ufw" and len(core_seg) >= 2 and core_seg[1] in {"disable", "reset"}:
        return f"检测到高危防火墙变更命令 [ufw {core_seg[1]}]，已被前置硬阻断；查看状态请用 ufw status verbose。"

    # 3. 停机、重启与 PID 1 / 全进程树毁灭拦截
    if prog_name in {"shutdown", "reboot", "poweroff", "halt"}:
        return f"检测到主机停机/重启高危命令 [{prog_name}]，已被安全门禁前置硬阻断。"

    if prog_name in {"init", "telinit"} and len(core_seg) >= 2 and core_seg[1] in {"0", "6"}:
        return f"检测到系统运行级别切换停机/重启命令 [{prog_name} {core_seg[1]}]，已被前置硬阻断。"

    if prog_name == "kill":
        _, kill_pos = extract_flags_and_positionals(core_seg, value_short_flags=frozenset({"-s"}))
        # 处理形如 kill -9 -1 中 "-1" 被识别为短选项的情况
        has_minus_one = "-1" in core_seg[1:]
        if "1" in kill_pos or has_minus_one:
            return "检测到灾难性进程终止命令 [kill PID 1 / -1]（会导致系统崩溃或踢断 SSH 守护进程），已被前置硬阻断。"

    # 4. 磁盘格式化、裸块设备覆写与关键系统根目录递归删除拦截
    if prog_name == "mkfs" or prog_name.startswith("mkfs."):
        return f"检测到磁盘格式化高危命令 [{prog_name}]，已被安全门禁前置硬阻断。"

    if prog_name == "dd":
        for tok in core_seg[1:]:
            if tok.startswith("of=/dev/") and tok not in {"of=/dev/null", "of=/dev/stdout", "of=/dev/stderr"}:
                return f"检测到裸块设备覆写高危命令 [dd {tok}]，已被安全门禁前置硬阻断。"

    if prog_name == "rm":
        rm_flags, rm_pos = extract_flags_and_positionals(core_seg)
        is_recursive = bool(rm_flags & {"-r", "-R", "--recursive"})
        if is_recursive or "--no-preserve-root" in rm_flags:
            for target in rm_pos:
                normalized = target.rstrip("/") or "/"
                if target in _CRITICAL_SYSTEM_PATHS or normalized in _CRITICAL_SYSTEM_PATHS:
                    return f"检测到灾难性系统路径删除命令 [rm {target}]，已被安全门禁前置硬阻断。"

    return None


def _evaluate_single_segment(
    segment: list[str],
    *,
    in_pipe: bool,
    rules: Mapping[str, CommandRule],
) -> tuple[BashDecision, str | None]:
    """评估单段简单命令的安全等级（``allow`` / ``ask`` / ``block``）。"""
    if not segment:
        return ("ask", None)

    # 第一步：无论是否携带 sudo/路径前缀，优先执行交互式假死与灾难自锁硬阻断检查
    hard_block_reason = _check_catastrophic_or_interactive_block(segment)
    if hard_block_reason is not None:
        return ("block", hard_block_reason)

    # 同时穿透检查底层命令是否违反已知工具的挂死参数（如 sudo top、/usr/bin/tail -f）
    core_seg, is_wrapped = _unwrap_command_segment(segment)
    if core_seg:
        core_name = PurePosixPath(core_seg[0].replace("\\", "/")).name
        core_rule = rules.get(core_name)
        if core_rule is not None:
            c_flags, c_pos = extract_flags_and_positionals(
                core_seg,
                value_short_flags=core_rule.value_short_flags,
                split_short_clusters=core_rule.split_short_clusters,
            )
            if core_rule.blocked_any_flags and (c_flags & core_rule.blocked_any_flags):
                return ("block", core_rule.blocked_flag_reason or f"命令 [{core_name}] 包含持续阻塞参数。")
            if core_rule.required_all_flags and not core_rule.required_all_flags.issubset(c_flags):
                return ("block", core_rule.missing_required_reason or f"命令 [{core_name}] 缺少必要的非交互参数。")
            if core_rule.required_any_flags and not (c_flags & core_rule.required_any_flags):
                return ("block", core_rule.missing_required_reason or f"命令 [{core_name}] 缺少退出限制参数。")
            if core_rule.custom_validator is not None:
                c_decision, c_reason = core_rule.custom_validator(c_flags, c_pos, tuple(core_seg))
                if c_decision == "block":
                    return ("block", c_reason)

    # 第二步：免批白名单准入审查（要求未经过 sudo/env 包装，且禁止包含路径斜杠以防 ./ps 伪造二进制）
    prog = segment[0]
    if is_wrapped or "/" in prog or "\\" in prog or "=" in prog:
        return ("ask", None)

    rule = rules.get(prog)
    if rule is None:
        return ("ask", None)

    if in_pipe and not rule.allow_in_pipe:
        return ("ask", None)

    flags, positionals = extract_flags_and_positionals(
        segment,
        value_short_flags=rule.value_short_flags,
        split_short_clusters=rule.split_short_clusters,
    )

    # 检查写盘/原地修改参数（命中则降级为 ask）
    if rule.denied_any_flags and (flags & rule.denied_any_flags):
        return ("ask", None)

    # 检查最大位置参数数量约束
    if rule.max_positionals is not None and len(positionals) > rule.max_positionals:
        return ("ask", None)

    # 检查只读子命令白名单约束
    if rule.allowed_subcommands is not None:
        if not positionals or positionals[0] not in rule.allowed_subcommands:
            return ("ask", None)

    # 执行自定义精细校验器
    if rule.custom_validator is not None:
        return rule.custom_validator(flags, positionals, tuple(segment))

    return ("allow", None)


def analyze_bash_command(
    command: str,
    *,
    extra_rules: Mapping[str, CommandRule] | None = None,
) -> tuple[BashDecision, str | None]:
    """对输入 Shell 命令执行完整的四层静态威胁与只读契约分析。

    Args:
        command: 待执行的原始 Shell 命令文本。
        extra_rules: 可选的业务自定义命令安全规则映射（覆盖同名内置规则）。

    Returns:
        二元组 ``(decision, reason)``：
        - ``("allow", None)``：所有管道段均确认为纯只读且非阻塞的安全命令；
        - ``("ask", None)``：包含链式控制符、重定向、子 Shell 或未列入只读白名单的操作；
        - ``("block", reason)``：包含交互式假死命令或灾难性自锁命令，``reason`` 提供中文自纠偏指引。
    """
    cmd = command.strip()
    if not cmd:
        return ("ask", None)

    # L1：原始文本高危展开与多行注入快速红线扫描
    if any(marker in cmd for marker in _RAW_INJECTION_MARKERS):
        return ("ask", None)

    # L2：shlex 引号感知词法切分（精准分离引号内字面量与外层真实操作符）
    try:
        lexer = shlex.shlex(cmd, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        # 存在未闭合引号等畸形 Shell 语法，遵循 Fail-Closed 原则拒绝自动免批
        return ("ask", None)

    if not tokens:
        return ("ask", None)

    # 按单竖线 "|" 切分管道段；遇到任何分号、逻辑链、后台符、重定向或子 Shell 括号立即降级为 "ask"
    segments: list[list[str]] = []
    current_seg: list[str] = []

    for tok in tokens:
        if tok and set(tok) <= _SHELL_PUNCT_CHARS:
            if tok == "|":
                if not current_seg:
                    # 空管道段（如 "| grep a" 或 "ps || grep a" 畸形拆分），拒绝自动免批
                    return ("ask", None)
                segments.append(current_seg)
                current_seg = []
                continue
            # 命中 ";", "&&", "||", "&", ">", ">>", "<", "(", ")", "|&", "&>" 等非单管道算子
            return ("ask", None)

        current_seg.append(tok)

    if not current_seg:
        # 命令以悬空管道符结尾（如 "ps aux |"）
        return ("ask", None)
    segments.append(current_seg)

    # L3 & L4：合并规则表并对管道每一段逐一评估（最高威胁优先：任一段 block 则整体 block；全段 allow 才整体 allow）
    active_rules: Mapping[str, CommandRule] = (
        {**DEFAULT_COMMAND_RULES, **extra_rules} if extra_rules else DEFAULT_COMMAND_RULES
    )
    in_pipe = len(segments) > 1
    all_allowed = True

    for seg in segments:
        decision, reason = _evaluate_single_segment(seg, in_pipe=in_pipe, rules=active_rules)
        if decision == "block":
            return ("block", reason)
        if decision != "allow":
            all_allowed = False

    if all_allowed:
        return ("allow", None)
    return ("ask", None)


# 权限门禁核心实现（兼容 ToolCallHook 异步钩子协议）


class PermissionGate:
    """基于 ``ToolCallHook`` 实现的多模式业务安全审批门禁。

    支持在本地交互编码（``review`` / ``strict`` / ``autonomous``）与远程无人值守 SRE 排障
    （``readonly`` / ``review``）场景下复用同一套权限拦截契约，无需改动内核主循环。
    """

    def __init__(
        self,
        mode: PermissionMode = "review",
        confirm_callback: Callable[[PermissionRequest], Awaitable[bool] | bool] | None = None,
        *,
        extra_command_rules: Mapping[str, CommandRule] | None = None,
    ) -> None:
        """初始化权限门禁实例。

        Args:
            mode: 权限管控模式（``review`` / ``readonly`` / ``strict`` / ``autonomous`` / ``yolo``）。
                  其中 ``yolo`` 在内部标准化映射为 ``autonomous``。
            confirm_callback: 人工交互确认回调函数（支持同步 ``bool`` 或异步 ``Awaitable[bool]``）。
            extra_command_rules: 可选的自定义 Shell 命令只读/阻断规则表，用于无侵入扩展企业私有巡检工具。
        """
        self.mode: PermissionMode = "autonomous" if mode == "yolo" else mode
        self.confirm_callback = confirm_callback
        self._extra_command_rules = dict(extra_command_rules) if extra_command_rules else None

    async def __call__(self, hook: ToolCallHook) -> HookResult:
        """拦截并审查单次工具调用请求。"""
        # 0. 全自主模式（autonomous / yolo）：无条件放行全部操作
        if self.mode == "autonomous":
            return HookResult()

        tool_name = hook.tool_name
        args = hook.args or {}

        # 1. 只读工具快速免批放行（strict 零信任模式除外；涵盖 read/grep/find/ls/kb_search）
        if self.mode != "strict" and tool_name in READONLY_TOOLS:
            return HookResult()

        # 2. Shell 命令四层静态威胁分析与参数感知过滤
        if tool_name == "bash":
            cmd = str(args.get("command", "")).strip()
            decision, block_reason = analyze_bash_command(
                cmd,
                extra_rules=self._extra_command_rules,
            )
            # 2.1 交互式挂起“假死命令”与灾难性自锁命令：在所有受控模式下前置硬阻断，无需弹窗确认
            if decision == "block":
                return HookResult(
                    block=True,
                    reason=block_reason or f"高危或交互式阻塞命令已被安全门禁前置阻断: {cmd}",
                )
            # 2.2 纯只读安全探针或安全只读管道：在 review 与 readonly 模式下自动免批放行
            if decision == "allow" and self.mode in ("review", "readonly"):
                return HookResult()

        # 3. 纯只读定界模式（readonly）兜底硬阻断：
        #    解决无头排障（如凌晨告警 Webhook 自动初筛）无人工回调时默认放行写操作的安全隐患
        if self.mode == "readonly":
            return HookResult(
                block=True,
                reason=(
                    f"只读定界模式 (readonly) 下禁止执行潜在变更或未授权操作 [{tool_name}]；"
                    "当前阶段仅允许只读诊断工具与安全排障探针。"
                ),
            )

        # 4. 发起人工交互审批（review / strict 模式）
        if not self.confirm_callback:
            # 保持本地无回调脚本默认放行的向后兼容语义；无人值守只读场景请显式指定 mode="readonly"
            return HookResult()

        target = str(args.get("path") or args.get("command") or tool_name)
        preview = None
        if tool_name in ("write", "edit"):
            preview = args.get("content") or str(args.get("edits", ""))

        req = PermissionRequest(
            action=tool_name,
            target=target,
            details=args,
            preview=preview,
        )

        res = self.confirm_callback(req)
        approved = await res if inspect.isawaitable(res) else res
        if not approved:
            return HookResult(block=True, reason=f"用户拒绝了工具 [{tool_name}] 的执行请求。")

        return HookResult()


__all__ = [
    "BashDecision",
    "CommandRule",
    "DEFAULT_COMMAND_RULES",
    "PermissionGate",
    "PermissionMode",
    "PermissionRequest",
    "READONLY_TOOLS",
    "SAFE_BASH_PREFIXES",
    "analyze_bash_command",
    "extract_flags_and_positionals",
]
