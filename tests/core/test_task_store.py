import asyncio
from pathlib import Path

import pytest  # pyright: ignore[reportMissingImports]

from shed_core.task_store import TaskStore  # pyright: ignore[reportMissingImports]


def test_task_store_create_and_get(tmp_path: Path):
    async def _test():
        store = TaskStore(tmp_path)
        t1 = await store.create(subject="Design schema", description="Create users table")
        assert t1.id == "task_1"
        assert t1.status == "pending"
        assert t1.subject == "Design schema"
        assert store.get("task_1").description == "Create users table"

    asyncio.run(_test())


def test_task_store_dag_dependencies_and_unblock(tmp_path: Path):
    async def _test():
        store = TaskStore(tmp_path)
        await store.create(subject="Task 1")
        await store.create(subject="Task 2")

        # task_2 依赖 task_1
        updated_t2, unblocked = await store.update("task_2", add_blocked_by=["task_1"])
        assert updated_t2.blocked_by == ["task_1"]
        assert unblocked == []

        # 完成 task_1 后应解除 task_2 的阻塞
        updated_t1, unblocked = await store.update("task_1", status="completed")
        assert updated_t1.status == "completed"
        assert "task_2" in unblocked

    asyncio.run(_test())


def test_task_store_cycle_detection(tmp_path: Path):
    async def _test():
        store = TaskStore(tmp_path)
        await store.create(subject="Task 1")
        await store.create(subject="Task 2")
        await store.update("task_2", add_blocked_by=["task_1"])

        with pytest.raises(ValueError, match="[Cc]ycle"):
            await store.update("task_1", add_blocked_by=["task_2"])

    asyncio.run(_test())


def test_task_store_single_in_progress_enforcement(tmp_path: Path):
    async def _test():
        store = TaskStore(tmp_path, enforce_single_in_progress=True)
        await store.create(subject="Task 1")
        await store.create(subject="Task 2")

        await store.update("task_1", status="in_progress")
        with pytest.raises(ValueError, match="already in progress"):
            await store.update("task_2", status="in_progress")

    asyncio.run(_test())


def test_task_store_render_board(tmp_path: Path):
    async def _test():
        store = TaskStore(tmp_path)
        await store.create(subject="Task 1")
        await store.create(subject="Task 2")
        await store.update("task_1", status="completed")
        await store.update("task_2", status="in_progress", active_form="writing code")

        board = store.render_board()
        assert "[x] task_1: Task 1" in board
        assert "[>] task_2: Task 2" in board
        assert "writing code" in board

    asyncio.run(_test())


def test_task_store_persistence(tmp_path: Path):
    async def _test():
        store1 = TaskStore(tmp_path)
        await store1.create(subject="Persisted task")

        store2 = TaskStore(tmp_path)
        assert len(store2.list()) == 1
        assert store2.get("task_1").subject == "Persisted task"

    asyncio.run(_test())


def test_task_store_batch_write(tmp_path: Path):
    async def _test():
        store = TaskStore(tmp_path)
        items = await store.batch_write(
            [
                {"subject": "Batch task 1", "status": "completed"},
                {"subject": "Batch task 2", "status": "in_progress"},
            ]
        )
        assert len(items) == 2
        assert store.get("task_1").status == "completed"
        assert store.get("task_2").status == "in_progress"

    asyncio.run(_test())


def test_task_store_invalid_operations(tmp_path: Path):
    async def _test():
        store = TaskStore(tmp_path)
        with pytest.raises(ValueError, match="cannot be empty"):
            await store.create(subject="   ")

        with pytest.raises(KeyError, match="not found"):
            store.get("task_999")

        with pytest.raises(KeyError, match="not found"):
            await store.update("task_999", status="completed")

        await store.create(subject="Task 1")
        with pytest.raises(ValueError, match="cannot depend on itself"):
            await store.update("task_1", add_blocked_by=["task_1"])

        with pytest.raises(KeyError, match="not found"):
            await store.update("task_1", add_blocked_by=["task_nonexistent"])

    asyncio.run(_test())


def test_task_store_multiple_creates_and_updates(tmp_path: Path):
    async def _test():
        store = TaskStore(tmp_path)

        # 1. 顺序创建 10 个任务
        tasks = [await store.create(subject=f"Task {i}") for i in range(10)]
        assert len(tasks) == 10
        unique_ids = {t.id for t in tasks}
        assert len(unique_ids) == 10
        assert len(store.list()) == 10

        # 2. 依次更新各任务的描述
        for i in range(10):
            await store.update(f"task_{i + 1}", description=f"Desc {i}")
            assert store.get(f"task_{i + 1}").description == f"Desc {i}"

        # 3. 从磁盘重新加载并校验一致性
        reloaded = TaskStore(tmp_path)
        assert len(reloaded.list()) == 10

    asyncio.run(_test())
