"""사람 찾기 — 영상 전체를 한 번 훑어 머리를 추적하고 사람 번호(#1, #2 …)를 정한다

훑기와 그리기를 나눈 이유
  · 사람이 처음 나오는 프레임부터 가릴 수 있다 (번호가 정해진 뒤 앞 프레임까지 거슬러 적용)
  · 한두 프레임만 잡힌 오검출(확정되지 못한 추적)은 사람 목록에서 빠진다
  · 같은 설정이면 번호가 늘 같아서, 고른 사람만 가리기가 정확하다
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Callable, Optional

import cv2
import numpy as np

from .detector import Detector
from .geometry import Rect
from .media import MediaSource
from .settings import GridSettings
from .tracker import MIN_HITS, Track, Tracker

Progress = Callable[[float, str], None]
THUMB = 96


class CutDetector:
    """장면 전환(컷) 찾기 — 앞 프레임과 색 분포가 확 달라지면 컷.

    컷에서 추적을 새로 시작해야, 앞 장면 사람 박스가 '놓쳐도 유지' 동안 새 장면 위에 남지 않는다.
    """
    CORREL = 0.55    # 색 분포 상관이 이보다 낮고
    DIFF = 28.0      # 밝기 차 평균이 이보다 크면 컷

    def __init__(self):
        self._prev = None

    @staticmethod
    def _sig(frame: np.ndarray):
        small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist, cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def is_cut(self, frame: np.ndarray) -> bool:
        sig = self._sig(frame)
        prev, self._prev = self._prev, sig
        if prev is None:
            return False
        correl = cv2.compareHist(prev[0], sig[0], cv2.HISTCMP_CORREL)
        diff = float(np.abs(prev[1] - sig[1]).mean())
        return correl < self.CORREL and diff > self.DIFF


@dataclass
class Person:
    id: int                 # 1 부터
    first: int              # 처음 나온 프레임
    last: int
    frames: int             # 실제로 검출된 프레임 수 (HOLD 제외)
    best_frame: int = 0
    best_box: Optional[Rect] = None
    thumb: str = ""
    shot: int = 1           # 몇 번째 컷에 나왔나 (컷이 바뀌면 같은 사람도 새 번호)

    def describe(self, fps: Optional[float]) -> dict:
        t = (lambda f: round(f / fps, 2)) if fps else (lambda f: None)
        return {"id": self.id, "first": self.first, "last": self.last, "frames": self.frames, "shot": self.shot,
                "start_sec": t(self.first), "end_sec": t(self.last), "thumb": self.thumb}


@dataclass
class TrackScan:
    key: tuple
    frame_count: int
    fps: Optional[float]
    people: list[Person] = field(default_factory=list)
    cuts: list[int] = field(default_factory=list)          # 새 컷이 시작되는 프레임
    _frames: list[list[tuple[int, Rect, bool]]] = field(default_factory=list, repr=False)   # 프레임마다 (사람 번호, 박스, HOLD)

    @staticmethod
    def key_for(settings: GridSettings) -> tuple:
        """이 값들이 같으면 훑은 결과를 다시 쓸 수 있다"""
        return (settings.model, round(settings.threshold, 4), settings.hold, round(settings.smooth, 4))

    def tracks_at(self, index: int, people: tuple = ()) -> list[Track]:
        if index >= len(self._frames):
            return []
        wanted = set(people)
        return [Track(pid, box, missed=1 if lost else 0, hits=MIN_HITS) for pid, box, lost in self._frames[index]
                if not wanted or pid in wanted]

    def count_at(self, index: int) -> int:
        return len(self._frames[index]) if index < len(self._frames) else 0

    @property
    def max_together(self) -> int:
        return max((len(f) for f in self._frames), default=0)

    def describe(self) -> dict:
        return {"people": [p.describe(self.fps) for p in self.people], "count": len(self.people),
                "max_together": self.max_together, "frames": self.frame_count,
                "shots": len(self.cuts) + 1, "cuts": self.cuts}


class Scanner:
    def __init__(self, detector: Detector, settings: GridSettings):
        self._det = detector
        self._s = settings

    def run(self, source: MediaSource, progress: Progress = lambda f, t: None) -> TrackScan:
        s = self._s
        total = source.frame_count
        scan = TrackScan(TrackScan.key_for(s), total, source.fps)
        if not source.is_video:
            frame = next(source.frames())
            dets = self._det.detect_scored(frame, s.threshold)
            scan._frames = [[(i + 1, d.box, False) for i, d in enumerate(dets)]]
            scan.people = [Person(i + 1, 0, 0, 1, 0, d.box) for i, d in enumerate(dets)]
            self._thumbs(scan, source, {0: frame})
            progress(1.0, "사람 찾기")
            return scan

        tracker = Tracker(s.smooth, s.hold, s.threshold)
        cut = CutDetector()
        raw: list[list[tuple[int, Rect, bool, bool]]] = []
        size = source.size
        n = 0
        for frame in source.frames():
            if cut.is_cut(frame) and n > 0:   # 새 장면: 앞 장면 박스를 끌고 오지 않게 추적을 새로
                scan.cuts.append(n)
                tracker.reset()
            tracks = tracker.update(self._det.detect_scored(frame, tracker.low), size)
            raw.append([(t.id, t.box, t.lost, t.confirmed) for t in tracks])
            n += 1
            if n % 8 == 0:
                progress(min(1.0, n / max(total, 1)), f"사람 찾기 {n}/{total}")
        scan.frame_count = n

        # 확정된 적 있는 추적만 사람으로, 처음 나온 순서대로 #1, #2 …
        confirmed = {tid for f in raw for tid, _, _, ok in f if ok}
        order: dict[int, int] = {}
        for f in raw:
            for tid, *_ in f:
                if tid in confirmed and tid not in order:
                    order[tid] = len(order) + 1
        people = {pid: Person(pid, -1, -1, 0) for pid in order.values()}
        for i, f in enumerate(raw):
            row = []
            for tid, box, lost, _ in f:
                if tid not in order:
                    continue
                pid = order[tid]
                row.append((pid, box, lost))
                p = people[pid]
                p.first = i if p.first < 0 else p.first
                p.last = i
                if not lost:
                    p.frames += 1
                    if p.best_box is None or box.area > p.best_box.area:
                        p.best_frame, p.best_box = i, box
            scan._frames.append(row)
        for p in people.values():
            p.shot = 1 + sum(1 for c in scan.cuts if c <= p.first)
        scan.people = [people[k] for k in sorted(people)]
        self._thumbs(scan, source)
        progress(1.0, "사람 찾기 끝")
        return scan

    @staticmethod
    def _thumbs(scan: TrackScan, source: MediaSource, cache: dict | None = None) -> None:
        """사람마다 머리가 가장 크게 보인 프레임에서 얼굴 썸네일"""
        cache = cache or {}
        for p in scan.people:
            if p.best_box is None:
                continue
            try:
                frame = cache.get(p.best_frame)
                if frame is None:
                    frame = source.frame_at_index(p.best_frame)
                b = p.best_box.expanded(0.25, 0.25, 0.25).clipped(frame.shape[1], frame.shape[0])
                x0, y0, x1, y1 = b.as_int()
                crop = frame[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                side = max(crop.shape[:2])
                pad = np.zeros((side, side, 3), np.uint8)
                pad[(side - crop.shape[0]) // 2:(side - crop.shape[0]) // 2 + crop.shape[0],
                    (side - crop.shape[1]) // 2:(side - crop.shape[1]) // 2 + crop.shape[1]] = crop
                ok, buf = cv2.imencode(".jpg", cv2.resize(pad, (THUMB, THUMB), interpolation=cv2.INTER_AREA),
                                       [cv2.IMWRITE_JPEG_QUALITY, 85])
                p.thumb = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()
            except Exception:
                p.thumb = ""
