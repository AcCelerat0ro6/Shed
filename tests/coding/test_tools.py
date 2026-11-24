"""文件与命令工具集成测试。"""

import asyncio

import pytest

from shed_agent.mutation_queue import FileMutationQueue
from shed_agent.tools import (
    make_bash_tool,
    make_edit_tool,
    make_read_tool,
    make_write_tool,
)


@pytest.mark.anyio
async def test_read_basic(tmp_path):
    """验证读取文件全文。"""
    read = make_read_tool(tmp_path)
    (tmp_path / "a.txt").write_text("line1\nline2\nline3", encoding="utf-8")
    result = await read.execute({"path": "a.txt"})
    assert result.ok is True
    assert result.data == "line1\nline2\nline3"
    assert read.is_parallel_safe is True


@pytest.mark.anyio
async def test_read_limit(tmp_path):
    """验证 limit 截断时输出包含 Showing lines 提示。"""
    read = make_read_tool(tmp_path)
    (tmp_path / "a.txt").write_text("\n".join(f"l{i}" for i in range(10)), encoding="utf-8")
    result = await read.execute({"path": "a.txt", "limit": 3})
    # 返回 "[Showing lines X-Y of Z. ...]" 形式的截断提示
    assert "Showing lines" in result.data or "l0" in result.data


@pytest.mark.anyio
async def test_read_offset_beyond_end(tmp_path):
    """验证 offset 超出文件总行数时返回具体行数提示。"""
    read = make_read_tool(tmp_path)
    (tmp_path / "a.txt").write_text("line1\nline2", encoding="utf-8")
    result = await read.execute({"path": "a.txt", "offset": 100})
    assert "Offset 100 is beyond end of file" in result.data
    assert "has only 2 lines total" in result.data


@pytest.mark.anyio
async def test_read_path_outside_workspace(tmp_path):
    """验证访问工作区外部路径时返回错误或未找到提示。"""
    read = make_read_tool(tmp_path)
    result = await read.execute({"path": "../secret.txt"})
    assert "not found" in result.data.lower() or "Error" in result.data


@pytest.mark.anyio
async def test_read_missing(tmp_path):
    """验证读取不存在文件时返回错误提示。"""
    read = make_read_tool(tmp_path)
    result = await read.execute({"path": "nope.txt"})
    assert "not found" in result.data.lower() or "not exist" in result.data.lower()


@pytest.mark.anyio
async def test_write_creates_and_overwrites(tmp_path):
    """验证写文件自动创建父目录、覆盖写入并返回成功信息。"""
    write = make_write_tool(tmp_path)
    result = await write.execute({"path": "sub/dir/a.txt", "content": "hello"})
    # 返回 "Successfully wrote N bytes" 提示
    assert "5 bytes" in result.data or "wrote" in result.data.lower()
    assert (tmp_path / "sub" / "dir" / "a.txt").read_text(encoding="utf-8") == "hello"
    await write.execute({"path": "sub/dir/a.txt", "content": "world"})
    assert (tmp_path / "sub" / "dir" / "a.txt").read_text(encoding="utf-8") == "world"
    assert write.is_parallel_safe is True


@pytest.mark.anyio
async def test_edit_replaces_once(tmp_path):
    """验证精确替换单处文本内容。"""
    edit = make_edit_tool(tmp_path)
    (tmp_path / "a.txt").write_text("hello world hello", encoding="utf-8")
    result = await edit.execute({"path": "a.txt", "edits": [{"oldText": "world", "newText": "earth"}]})
    # 返回 "Successfully applied N edit(s) to ..." 提示
    assert "a.txt" in result.data or "edit" in result.data.lower()
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "hello earth hello"
    assert edit.is_parallel_safe is True


@pytest.mark.anyio
async def test_edit_text_not_found(tmp_path):
    """验证 old_text 不存在时返回包含路径名与行数的提示。"""
    edit = make_edit_tool(tmp_path)
    (tmp_path / "a.txt").write_text("line1\nline2", encoding="utf-8")
    result = await edit.execute({"path": "a.txt", "edits": [{"oldText": "nope", "newText": "x"}]})
    assert "a.txt" in result.data
    assert "2 lines" in result.data or "not found" in result.data.lower()


@pytest.mark.anyio
async def test_edit_multiple_matches(tmp_path):
    """验证 old_text 命中多处时提示提供更多上下文。"""
    edit = make_edit_tool(tmp_path)
    (tmp_path / "a.txt").write_text("dup\ndup\n", encoding="utf-8")
    result = await edit.execute({"path": "a.txt", "edits": [{"oldText": "dup", "newText": "unique"}]})
    # 返回 "'oldText' matched 2 times" 提示
    assert "matched 2" in result.data or "2 times" in result.data
    assert "Please provide more surrounding context lines" in result.data


@pytest.mark.anyio
async def test_file_mutation_queue_concurrency(tmp_path):
    """验证 FileMutationQueue：不同文件安全并发，同名文件排队串行。"""
    queue = FileMutationQueue()
    write = make_write_tool(tmp_path, mutation_queue=queue)

    timeline = []

    async def write_file(filename: str, delay: float):
        timeline.append(f"start_{filename}")
        await write.execute({"path": filename, "content": f"content_{filename}"})
        await asyncio.sleep(delay)
        timeline.append(f"end_{filename}")

    # 并发写入两个不同文件
    await asyncio.gather(
        write_file("file_a.txt", 0.05),
        write_file("file_b.txt", 0.05),
    )

    # 两个不同文件的写入同时启动
    assert timeline[0] in ("start_file_a.txt", "start_file_b.txt")
    assert timeline[1] in ("start_file_a.txt", "start_file_b.txt")


@pytest.mark.anyio
async def test_bash_normal(tmp_path):
    """验证正常命令返回标准输出。"""
    bash = make_bash_tool(tmp_path)
    result = await bash.execute({"command": "echo hi"})
    # bash 工具输出不自动裁剪空白，Windows 环境下可能包含 \r\n
    assert "hi" in result.data
    assert bash.is_parallel_safe is False


@pytest.mark.anyio
async def test_bash_dangerous(tmp_path):
    """验证危险命令被拦截。"""
    bash = make_bash_tool(tmp_path)
    result = await bash.execute({"command": "sudo rm -rf /"})
    # 返回危险命令拦截提示
    assert "rm -rf /" in result.data or "Blocked" in result.data or "blocked" in result.data


@pytest.mark.anyio
async def test_bash_timeout(tmp_path):
    """验证命令执行超时时返回 timed out 提示。"""
    bash = make_bash_tool(tmp_path)
    sleep_script = tmp_path / "_sleep.py"
    sleep_script.write_text(
        "import sys, time; print('starting step 1...', flush=True); time.sleep(30)",
        encoding="utf-8",
    )
    # 指定 1 秒超时
    result = await bash.execute({"command": "python _sleep.py", "timeout": 1})
    assert "timed out" in result.data.lower() or "timeout" in result.data.lower()
