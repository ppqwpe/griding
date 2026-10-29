"""박스 계산"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self) -> float:
        return self.x1 - self.x0

    @property
    def h(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return max(0.0, self.w) * max(0.0, self.h)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2

    def expanded(self, x: float = 0.0, top: float = 0.0, bottom: float = 0.0) -> "Rect":
        return Rect(self.x0 - self.w * x, self.y0 - self.h * top, self.x1 + self.w * x, self.y1 + self.h * bottom)

    def clipped(self, width: int, height: int) -> "Rect":
        return Rect(max(0.0, self.x0), max(0.0, self.y0), min(float(width), self.x1), min(float(height), self.y1))

    def iou(self, other: "Rect") -> float:
        ix = max(0.0, min(self.x1, other.x1) - max(self.x0, other.x0))
        iy = max(0.0, min(self.y1, other.y1) - max(self.y0, other.y0))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def lerp(self, other: "Rect", t: float) -> "Rect":
        """t=0 이면 self, t=1 이면 other"""
        return Rect(*(a + (b - a) * t for a, b in zip(self.as_tuple(), other.as_tuple())))

    def shifted(self, dx0: float, dy0: float, dx1: float, dy1: float) -> "Rect":
        return Rect(self.x0 + dx0, self.y0 + dy0, self.x1 + dx1, self.y1 + dy1)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.x0, self.y0, self.x1, self.y1

    def as_int(self) -> tuple[int, int, int, int]:
        return tuple(int(round(v)) for v in self.as_tuple())  # type: ignore[return-value]

    def rounded(self, nd: int = 1) -> list[float]:
        return [round(float(v), nd) for v in self.as_tuple()]
