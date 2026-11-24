"""后台异步子进程执行器与进程树生命周期管理。"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from shed_core.message_queue import (  # pyright: ignore[reportMissingImports]
        MessageQueue,
    )


@dataclass
class BackgroundJob:
    """单个后台作业状态记录。"""

    id: str
    description: str
    status: Literal["running", "completed", "failed", "cancelled"] = "running"
    result: str | None = None
    exit_code: int | None = None
    process: asyncio.subprocess.Process | None = None
    task: asyncio.Task[None] | None = None
    started_at: float = field(default_factory=time.time)


def _kill_process_tree(proc: asyncio.subprocess.Process | None) -> None:
    """终止子进程及其完整子进程树，避免产生孤儿进程。"""
    if proc is None or proc.returncode is not None or not proc.pid:
        return
    with contextlib.suppress(Exception):
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)


class BackgroundRunner:
    """管理后台异步子进程生命周期，并在任务结束后向 MessageQueue 投递通知。"""

    def __init__(self, message_queue: MessageQueue) -> None:
        self.message_queue = message_queue
        self.jobs: dict[str, BackgroundJob] = {}
        self._counter = 0
        atexit.register(self._sync_cleanup)

    def _sync_cleanup(self) -> None:
        """进程退出时终止所有仍在运行的子进程树，避免遗留孤儿进程。"""
        for job in self.jobs.values():
            if job.status == "running" and job.process and job.process.returncode is None:
                _kill_process_tree(job.process)

    async def run_process(self, command: str, cwd: Path | str, description: str = "") -> str:
        """异步启动后台子进程并立即返回作业 ID。"""
        self._counter += 1
        job_id = f"bg_{self._counter:06x}"
        job = BackgroundJob(id=job_id, description=description or command, status="running")
        self.jobs[job_id] = job

        async def _worker() -> None:
            proc = None
            try:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    cwd=str(cwd),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=os.name != "nt",
                )
                job.process = proc
                stdout, stderr = await proc.communicate()
                output = (stdout.decode(errors="replace") + stderr.decode(errors="replace")).strip()
                job.exit_code = proc.returncode
                job.result = output[:20000] if output else "(no output)"
                job.status = "completed" if proc.returncode == 0 else "failed"
            except asyncio.CancelledError:
                job.status = "cancelled"
                job.result = "Background task cancelled"
                job.exit_code = -1
                if proc is not None:
                    _kill_process_tree(proc)
                return
            except Exception as e:
                job.status = "failed"
                job.result = str(e)
                job.exit_code = 1
            finally:
                if job.status == "cancelled" and proc is not None:
                    _kill_process_tree(proc)

            notification = (
                f'<task_notification id="{job.id}">\n'
                f"Background task {job.id} ({job.description}) {job.status} (exit code {job.exit_code}):\n"
                f"{job.result}\n"
                f"</task_notification>"
            )
            self.message_queue.add_followup(notification)

        job.task = asyncio.create_task(_worker())
        return job_id

    async def cancel_all(self) -> None:
        """取消所有运行中的后台作业并终止其子进程树。"""
        for job in self.jobs.values():
            if job.status == "running":
                job.status = "cancelled"
                _kill_process_tree(job.process)
                if job.task and not job.task.done():
                    job.task.cancel()
                if job.process:
                    with contextlib.suppress(Exception):
                        await job.process.wait()
