"""프레임 사이 머리 박스 잇기 (ByteTrack 방식을 줄인 것)

1. 점수가 높은 검출부터 모든 추적과 한꺼번에 짝을 맞춘다 (겹침이 큰 짝부터 → 사람이 엇갈려도 번호가 덜 바뀜)
2. 남은 추적은 점수가 낮은 검출과 한 번 더 맞춘다 (흐려지거나 가려져 점수가 떨어진 머리를 놓치지 않게)
3. 남은 높은 점수 검출은 새 추적이 된다. min_hits 프레임 이상 이어져야 '확정'
4. 짝이 없는 추적은 마지막 움직임대로 hold 프레임 동안 이어 간다 (화면 밖으로는 안 나가게 자름)

번호(id)는 내부 번호다. 사람 번호(#1, #2 …)는 tracks.TrackScan 이 확정된 추적에만 붙인다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from .detector import Detection
from .geometry import Rect

MATCH_IOU = 0.1        # 이보다 덜 겹치면 같은 머리로 보지 않음 (가까우면 중심 거리로 보조)
VELOCITY_KEEP = 0.5
VELOCITY_DECAY = 0.8
MIN_HITS = 3           # 이만큼 이어져야 사람으로 확정
LOW_RATIO = 0.5        # 낮은 점수 검출 기준 = 민감도 × 이 값


@dataclass
class Track:
    id: int
    box: Rect
    velocity: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    missed: int = 0
    hits: int = 1
    score: float = 1.0

    @property
    def lost(self) -> bool:
        return self.missed > 0

    @property
    def confirmed(self) -> bool:
        return self.hits >= MIN_HITS

    def predicted(self) -> Rect:
        return self.box.shifted(*self.velocity)

    def follow(self, det: Detection, smooth: float) -> None:
        new = self.box.lerp(det.box, 1 - smooth)
        step = [n - o for n, o in zip(new.as_tuple(), self.box.as_tuple())]
        self.velocity = tuple(s * (1 - VELOCITY_KEEP) + v * VELOCITY_KEEP for s, v in zip(step, self.velocity))
        self.box, self.missed, self.score = new, 0, det.score
        self.hits += 1

    def coast(self, size: Optional[tuple[int, int]]) -> None:
        box = self.predicted()
        if size is not None:   # 화면 밖으로 흘러가지 않게
            w, h = size
            dx = min(0.0, w - box.x1) + max(0.0, -box.x0)
            dy = min(0.0, h - box.y1) + max(0.0, -box.y0)
            box = box.shifted(dx, dy, dx, dy)
        if box.w < 2 or box.h < 2:   # 빨리 줄어들던 박스가 뒤집히지 않게
            box = self.box
        self.box = box
        self.velocity = tuple(v * VELOCITY_DECAY for v in self.velocity)
        self.missed += 1


def _affinity(a: Rect, b: Rect) -> float:
    """겹침(IoU). 안 겹쳐도 아주 가까우면(빨리 움직임) 작은 값"""
    iou = a.iou(b)
    if iou > 0:
        return iou
    (ax, ay), (bx, by) = a.center, b.center
    size = (a.w + a.h + b.w + b.h) / 4
    d = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    return 0.05 if d < size * 0.5 else 0.0


def _pair(tracks: Sequence[Track], dets: Sequence[Detection], min_aff: float) -> tuple[list, list, list]:
    """겹침이 큰 짝부터 전체에서 맞춘다. (짝들, 남은 추적 번호, 남은 검출 번호)"""
    cands = []
    for ti, t in enumerate(tracks):
        p = t.predicted()
        for di, d in enumerate(dets):
            a = _affinity(p, d.box)
            if a >= min_aff:
                cands.append((a, ti, di))
    cands.sort(reverse=True)
    used_t, used_d, pairs = set(), set(), []
    for a, ti, di in cands:
        if ti in used_t or di in used_d:
            continue
        used_t.add(ti)
        used_d.add(di)
        pairs.append((ti, di))
    return pairs, [i for i in range(len(tracks)) if i not in used_t], [i for i in range(len(dets)) if i not in used_d]


@dataclass
class Tracker:
    smooth: float = 0.5
    hold: int = 12
    high: float = 0.25          # 새 추적을 만들 수 있는 점수
    _tracks: list[Track] = field(default_factory=list, init=False, repr=False)
    _next_id: int = field(default=0, init=False, repr=False)

    @property
    def low(self) -> float:
        return self.high * LOW_RATIO

    def reset(self) -> None:
        """컷이 바뀌었을 때: 지금 추적을 모두 끝낸다 (번호는 이어서 새로)"""
        self._tracks = []

    def update(self, detections: Sequence[Detection | Rect], size: Optional[tuple[int, int]] = None) -> list[Track]:
        dets = [d if isinstance(d, Detection) else Detection(d, 1.0) for d in detections]
        hi = [d for d in dets if d.score >= self.high]
        lo = [d for d in dets if d.score < self.high]

        pairs, rest_t, rest_hi = _pair(self._tracks, hi, MATCH_IOU * 0.5)
        for ti, di in pairs:
            self._tracks[ti].follow(hi[di], self.smooth)
        left = [self._tracks[i] for i in rest_t]
        pairs2, rest2, _ = _pair(left, lo, MATCH_IOU)
        for ti, di in pairs2:
            left[ti].follow(lo[di], self.smooth)
        for i in rest2:
            left[i].coast(size)

        for di in rest_hi:
            self._tracks.append(Track(self._next_id, hi[di].box, score=hi[di].score))
            self._next_id += 1
        # 확정 전 추적은 한 번만 놓쳐도 버린다 (한두 프레임짜리 오검출 제거)
        self._tracks = [t for t in self._tracks if t.missed <= (self.hold if t.confirmed else 0)]
        return list(self._tracks)
