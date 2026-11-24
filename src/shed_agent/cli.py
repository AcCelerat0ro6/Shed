"""shed Python CLI 入口模块。

提供通过 Python 启动 shed（如 `uv run shed`、`uv tool install .` 或 `python -m shed_agent.cli`）时的 `main()` 入口，
负责检测本机 Node.js 运行时并桥接至前端 TUI。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def find_tui_entry() -> Path | None:
    """按开发环境、包内目录与系统安装前缀顺序定位 TUI 启动入口脚本（shed.js）。"""
    repo_root = Path(__file__).resolve().parent.parent.parent

    # 1. 开发工作区路径：<repo_root>/shed-tui/bin/shed.js
    development_entry = repo_root / "shed-tui" / "bin" / "shed.js"
    if development_entry.exists():
        return development_entry

    # 2. 包内资源路径：<package_dir>/shed-tui/bin/shed.js
    bundled_entry = Path(__file__).resolve().parent / "shed-tui" / "bin" / "shed.js"
    if bundled_entry.exists():
        return bundled_entry

    # 3. 系统安装前缀共享路径：<sys.prefix>/share/shed/shed-tui/bin/shed.js
    prefix_entry = Path(sys.prefix) / "share" / "shed" / "shed-tui" / "bin" / "shed.js"
    if prefix_entry.exists():
        return prefix_entry

    return None


def main(argv: list[str] | None = None) -> int:
    """CLI 主入口，检测运行时并拉起 TUI 前端进程。"""
    if argv is None:
        argv = sys.argv[1:]

    node_executable = shutil.which("node")
    if not node_executable:
        print(
            "\n[shed] 错误: 未检测到 Node.js 运行时环境。\n"
            "shed 采用像素级 Pi-TUI 呈现层，需要 Node.js (>= 18.0) 运行交互界面。\n\n"
            "请安装 Node.js 后重试:\n"
            "  - 官方网站下载: https://nodejs.org\n"
            "  - 或使用极速版本管理器安装: fnm (https://github.com/Schniz/fnm) 或 nvm\n",
            file=sys.stderr,
        )
        return 1

    tui_entry = find_tui_entry()
    if not tui_entry or not tui_entry.exists():
        print(
            "\n[shed] 错误: 未找到 TUI 启动入口脚本 (shed.js)。\n"
            "请确保已正确构建前端: npm run build --prefix shed-tui\n",
            file=sys.stderr,
        )
        return 1

    command = [node_executable, str(tui_entry), *argv]
    if "--python-executable" not in argv:
        command.extend(["--python-executable", sys.executable])

    try:
        completed_process = subprocess.run(command)
        return completed_process.returncode
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"[shed] 启动异常: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
