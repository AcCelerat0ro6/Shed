import asyncio
from pathlib import Path
import pytest
from shed_agent.mutation_queue import FileMutationQueue


@pytest.mark.anyio
async def test_file_mutation_queue_serializes_same_file(tmp_path: Path):
    queue = FileMutationQueue()
    target_file = tmp_path / "shared.txt"

    execution_order = []
    active_count = 0
    max_concurrent = 0

    async def worker(task_id: int):
        nonlocal active_count, max_concurrent
        async with queue.acquire(target_file):
            active_count += 1
            max_concurrent = max(max_concurrent, active_count)
            execution_order.append(f"start_{task_id}")
            await asyncio.sleep(0.02)
            execution_order.append(f"end_{task_id}")
            active_count -= 1

    # 启动 3 个操作同一文件的协程
    await asyncio.gather(worker(1), worker(2), worker(3))

    # 同一文件的并发数不得超过 1
    assert max_concurrent == 1
    # 每次 start 后必须紧跟对应的 end，不能交叉
    for i in range(0, len(execution_order), 2):
        start_id = execution_order[i].split("_")[1]
        end_id = execution_order[i + 1].split("_")[1]
        assert start_id == end_id


@pytest.mark.anyio
async def test_file_mutation_queue_parallel_for_different_files(tmp_path: Path):
    queue = FileMutationQueue()
    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "b.txt"

    active_count = 0
    max_concurrent = 0

    async def worker(target_file: Path):
        nonlocal active_count, max_concurrent
        async with queue.acquire(target_file):
            active_count += 1
            max_concurrent = max(max_concurrent, active_count)
            await asyncio.sleep(0.04)
            active_count -= 1

    # 启动 2 个操作不同文件的协程
    await asyncio.gather(worker(file_a), worker(file_b))

    # 操作不同文件的协程应支持并发执行
    assert max_concurrent == 2


@pytest.mark.anyio
async def test_file_mutation_queue_releases_lock_on_exception(tmp_path: Path):
    queue = FileMutationQueue()
    target_file = tmp_path / "error_test.txt"

    # 协程 1 抛出异常
    with pytest.raises(RuntimeError, match="disk error"):
        async with queue.acquire(target_file):
            raise RuntimeError("disk error")

    # 协程 2 应能立即获取锁，不发生死锁
    acquired = False
    async with queue.acquire(target_file):
        acquired = True

    assert acquired is True
