"""작업 대기열 — 넣은 순서대로 하나씩 처리한다 (그래픽카드 · 메모리를 한 작업이 다 쓰게)"""
from __future__ import annotations

import queue
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Job:
    name: str
    work: Callable[["Job"], dict]
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: str = "queued"          # queued | running | done | error
    progress: float = 0.0
    message: str = "대기 중"
    result: Optional[dict] = None

    def report(self, fraction: float, text: str = "") -> None:
        self.progress = max(0.0, min(1.0, fraction))
        self.message = f"{self.name} · {text}" if text else self.name

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "state": self.state, "progress": round(self.progress, 4),
                "msg": self.message, "result": self.result}


class JobQueue:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._q: "queue.Queue[Job]" = queue.Queue()
        self._worker = threading.Thread(target=self._loop, daemon=True, name="facegrid-worker")
        self._worker.start()

    def submit(self, job: Job) -> Job:
        self._jobs[job.id] = job
        self._q.put(job)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def waiting(self) -> int:
        return self._q.qsize()

    def _loop(self) -> None:
        while True:
            job = self._q.get()
            job.state = "running"
            job.message = f"{job.name} · 시작"
            try:
                job.result = job.work(job)
                job.progress, job.state = 1.0, "done"
                r = job.result or {}
                if "seconds" in r:   # 그리기 작업
                    ppl = r.get("people_used") or []
                    job.message = f"{job.name} 완료 · {round(r['seconds'])}초 · 사람 {len(ppl)}명"
                else:                # 사람 찾기 작업
                    job.message = f"{job.name} · 사람 {r.get('count', 0)}명 찾음"
            except Exception as e:  # 한 작업이 실패해도 다음 작업은 계속
                traceback.print_exc()
                job.state, job.message = "error", f"{job.name} — {e}"
            finally:
                self._q.task_done()
