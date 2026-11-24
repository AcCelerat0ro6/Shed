#!/usr/bin/env python3
"""shed 跨平台便携发行版打包脚本。

编译 TUI 前端、收集 Python 核心源码与元数据，并生成独立的 .zip 与 .tar.gz 归档包。
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


def get_version_from_pyproject(pyproject_path: Path) -> str:
    """从 pyproject.toml 解析项目版本号。"""
    content = pyproject_path.read_text(encoding="utf-8")
    match = re.search(r'version\s*=\s*"([^"]+)"', content)
    if not match:
        raise ValueError(f"Could not parse version from {pyproject_path}")
    return match.group(1)


def generate_launcher_scripts(dest_dir: Path) -> tuple[Path, Path]:
    """生成 Windows .cmd 与 POSIX sh 启动脚本。"""
    dest_dir.mkdir(parents=True, exist_ok=True)

    cmd_file = dest_dir / "shed.cmd"
    cmd_file.write_text(
        '@echo off\r\nsetlocal\r\nnode "%~dp0shed-tui\\bin\\shed.js" %*\r\n',
        encoding="utf-8",
    )

    sh_file = dest_dir / "shed"
    sh_file.write_text(
        '#!/usr/bin/env sh\nDIR="$(cd "$(dirname "$0")" && pwd)"\nexec node "$DIR/shed-tui/bin/shed.js" "$@"\n',
        encoding="utf-8",
    )
    try:
        sh_file.chmod(0o755)
    except Exception:
        pass

    return cmd_file, sh_file


def build_tui(repo_root: Path) -> None:
    """编译 TypeScript TUI 前端产物。"""
    print("[1/4] 构建 TypeScript TUI 前端...")
    npm_cmd = shutil.which("npm")
    if not npm_cmd:
        raise RuntimeError("未在当前系统 PATH 中找到 npm，无法编译 TUI。")
    subprocess.run([npm_cmd, "run", "build", "--prefix", "shed-tui"], cwd=repo_root, check=True)


def assemble_portable_package(repo_root: Path, dist_dir: Path, version: str) -> Path:
    """在暂存目录中组装完整的便携发行包内容。"""
    print("[2/4] 收集运行时资产与依赖...")
    pkg_name = f"shed-v{version}-portable"
    staging_dir = dist_dir / pkg_name
    if staging_dir.exists():
        try:
            shutil.rmtree(staging_dir)
        except OSError:
            pass
    staging_dir.mkdir(parents=True, exist_ok=True)

    # 1. 拷贝 Python 源码（排除字节码与本地缓存）
    shutil.copytree(
        repo_root / "src",
        staging_dir / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".ruff_cache", ".pytest_cache"),
    )

    # 2. 拷贝编译后的 TUI 产物
    (staging_dir / "shed-tui").mkdir(parents=True, exist_ok=True)
    shutil.copytree(repo_root / "shed-tui" / "dist", staging_dir / "shed-tui" / "dist")
    shutil.copytree(repo_root / "shed-tui" / "bin", staging_dir / "shed-tui" / "bin")
    shutil.copy2(repo_root / "shed-tui" / "package.json", staging_dir / "shed-tui" / "package.json")

    # 3. 拷贝元数据与说明文档
    shutil.copy2(repo_root / "pyproject.toml", staging_dir / "pyproject.toml")
    if (repo_root / "README.md").exists():
        shutil.copy2(repo_root / "README.md", staging_dir / "README.md")
    if (repo_root / "LICENSE").exists():
        shutil.copy2(repo_root / "LICENSE", staging_dir / "LICENSE")

    # 4. 生成跨平台启动脚本
    generate_launcher_scripts(staging_dir)

    return staging_dir


def create_archives(staging_dir: Path, dist_dir: Path, version: str) -> tuple[Path, Path]:
    """将暂存目录打包压缩为 .zip 与 .tar.gz 归档文件。"""
    print("[3/4] 压缩生成发布归档包...")
    base_name = f"shed-v{version}-portable"

    # 生成 .zip 压缩包
    zip_path = dist_dir / f"{base_name}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in staging_dir.rglob("*"):
            if file.is_file():
                arcname = file.relative_to(dist_dir)
                zf.write(file, arcname)

    # 生成 .tar.gz 压缩包
    tar_path = dist_dir / f"{base_name}.tar.gz"
    if tar_path.exists():
        tar_path.unlink()
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(staging_dir, arcname=base_name)

    return zip_path, tar_path


def main() -> int:
    """发行版打包命令行入口。"""
    parser = argparse.ArgumentParser(description="Build portable distributions for shed")
    parser.add_argument("--skip-build-tui", action="store_true", help="Skip npm run build step")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    dist_dir = repo_root / "dist"
    dist_dir.mkdir(parents=True, exist_ok=True)

    version = get_version_from_pyproject(repo_root / "pyproject.toml")
    print(f"=== 开始打包 shed v{version} ===")

    if not args.skip_build_tui:
        build_tui(repo_root)

    staging_dir = assemble_portable_package(repo_root, dist_dir, version)
    zip_path, tar_path = create_archives(staging_dir, dist_dir, version)

    print("[4/4] 打包完成！")
    print(f"  - 便携目录: {staging_dir}")
    print(f"  - Zip 归档: {zip_path} ({zip_path.stat().st_size / 1024:.1f} KB)")
    print(f"  - Tar 归档: {tar_path} ({tar_path.stat().st_size / 1024:.1f} KB)")
    print("========================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
