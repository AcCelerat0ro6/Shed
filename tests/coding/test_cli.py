"""Python CLI 入口（shed_agent.cli）测试。"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from shed_agent.cli import find_tui_entry, main


def test_find_tui_entry_in_dev_workspace():
    entry = find_tui_entry()
    assert entry is not None
    assert entry.exists()
    assert entry.name == "shed.js"


def test_cli_help_flag_displays_usage(capsys):
    with patch.object(sys, "argv", ["shed", "--help"]):
        code = main(["--help"])
        # 若环境存在 Node.js 则透传 --help，否则打印本地帮助信息
        out = capsys.readouterr().out
        assert code == 0 or "Usage" in out or "shed-cli" in out or "shed" in out


def test_cli_missing_node_outputs_friendly_guide(capsys):
    with patch("shutil.which", return_value=None):
        code = main(["-m", "gpt-4o"])
        assert code == 1
        captured = capsys.readouterr()
        combined = captured.out + captured.err
        assert "Node.js" in combined or "node" in combined
        assert "https://nodejs.org" in combined or "fnm" in combined or "nvm" in combined


def test_cli_launches_node_with_python_executable():
    # 使用真实存在的文件路径，使 exists() 检查返回 True
    fake_entry = Path(__file__).resolve()
    with (
        patch("shed_agent.cli.find_tui_entry", return_value=fake_entry),
        patch("shutil.which", return_value="/usr/bin/node"),
        patch("subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(returncode=0)
        code = main(["-c", "-m", "deepseek-chat"])
        assert code == 0
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        assert call_args[0] == "/usr/bin/node"
        assert str(fake_entry) in call_args[1]
        assert "-c" in call_args
        assert "-m" in call_args
        assert "deepseek-chat" in call_args
        assert "--python-executable" in call_args
        assert sys.executable in call_args
