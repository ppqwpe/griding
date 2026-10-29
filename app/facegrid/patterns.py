"""격자 모양. 모양마다 Pattern 하위 클래스 하나가 자기 선을 그린다.

새 모양을 넣으려면 Pattern 을 상속해 draw() 를 만들고 REGISTRY 에 추가하면 된다.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np

from .geometry import Rect

ON = 255


class Pattern(ABC):
    key: str = ""
    label: str = ""
    description: str = ""

    @abstractmethod
    def draw(self, canvas: np.ndarray, box: Rect, cols: int, rows: int, thickness: int, overhang: float) -> None:
        """canvas(uint8 [H,W]) 위에 box 기준으로 이 모양을 255 로 그린다."""

    # 하위 클래스가 같이 쓰는 그리기 도구
    @staticmethod
    def _line(canvas, a, b, t):
        cv2.line(canvas, (int(round(a[0])), int(round(a[1]))), (int(round(b[0])), int(round(b[1]))), ON, t, cv2.LINE_AA)

    @classmethod
    def _vertical(cls, canvas, box, cols, t):
        for i in range(cols):
            x = box.x0 + box.w * (i + 0.5) / cols
            cls._line(canvas, (x, box.y0), (x, box.y1), t)

    @classmethod
    def _horizontal(cls, canvas, box, rows, t, overhang):
        oh = box.w * overhang
        for j in range(rows):
            y = box.y0 + box.h * (j + 0.5) / rows
            cls._line(canvas, (box.x0 - oh, y), (box.x1 + oh, y), t)


class GridPattern(Pattern):
    key, label, description = "grid", "격자", "세로줄 + 가로줄 (기본)"

    def draw(self, canvas, box, cols, rows, thickness, overhang):
        self._vertical(canvas, box, cols, thickness)
        self._horizontal(canvas, box, rows, thickness, overhang)


class VerticalLinesPattern(Pattern):
    key, label, description = "vlines", "세로줄", "세로줄만"

    def draw(self, canvas, box, cols, rows, thickness, overhang):
        self._vertical(canvas, box, cols, thickness)


class HorizontalLinesPattern(Pattern):
    key, label, description = "hlines", "가로줄", "가로줄만 (스캔라인 느낌)"

    def draw(self, canvas, box, cols, rows, thickness, overhang):
        self._horizontal(canvas, box, rows, thickness, overhang)


class DiagonalPattern(Pattern):
    key, label, description = "diagonal", "대각선 격자", "45도로 엇갈린 마름모 격자"

    def draw(self, canvas, box, cols, rows, thickness, overhang):
        # 이 머리 몫만 따로 그려 박스 밖으로 나간 선을 잘라 낸 뒤 합친다 (다른 머리 선은 건드리지 않음)
        layer = np.zeros_like(canvas)
        step = max(2.0, (box.w + box.h) / max(1, cols + rows))
        k = -box.h
        while k < box.w:
            self._line(layer, (box.x0 + k, box.y1), (box.x0 + k + box.h, box.y0), thickness)
            self._line(layer, (box.x0 + k, box.y0), (box.x0 + k + box.h, box.y1), thickness)
            k += step
        h, w = canvas.shape
        x0, y0, x1, y1 = box.clipped(w, h).as_int()
        keep = np.zeros_like(canvas)
        keep[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 1
        np.maximum(canvas, layer * keep, out=canvas)


class CrossPattern(Pattern):
    key, label, description = "cross", "X 십자", "대각선 X + 가운데 십자"

    def draw(self, canvas, box, cols, rows, thickness, overhang):
        cx, cy = box.center
        self._line(canvas, (box.x0, box.y0), (box.x1, box.y1), thickness)
        self._line(canvas, (box.x1, box.y0), (box.x0, box.y1), thickness)
        self._line(canvas, (cx, box.y0), (cx, box.y1), thickness)
        self._line(canvas, (box.x0, cy), (box.x1, cy), thickness)


class DotsPattern(Pattern):
    key, label, description = "dots", "점", "칸마다 동그라미"

    def draw(self, canvas, box, cols, rows, thickness, overhang):
        r = max(thickness, int(box.w / cols * 0.22))
        for i in range(cols):
            for j in range(rows):
                c = (int(box.x0 + box.w * (i + 0.5) / cols), int(box.y0 + box.h * (j + 0.5) / rows))
                cv2.circle(canvas, c, r, ON, -1, cv2.LINE_AA)


class CheckPattern(Pattern):
    key, label, description = "check", "체크", "바둑판처럼 칸을 번갈아 채움"

    def draw(self, canvas, box, cols, rows, thickness, overhang):
        cw, ch = box.w / cols, box.h / rows
        for i in range(cols):
            for j in range(rows):
                if (i + j) % 2 == 0:
                    a = (int(box.x0 + cw * i), int(box.y0 + ch * j))
                    b = (int(box.x0 + cw * (i + 1)), int(box.y0 + ch * (j + 1)))
                    cv2.rectangle(canvas, a, b, ON, -1)


class BoxPattern(Pattern):
    key, label, description = "box", "박스 채우기", "머리를 통째로 칠함"

    def draw(self, canvas, box, cols, rows, thickness, overhang):
        x0, y0, x1, y1 = box.as_int()
        cv2.rectangle(canvas, (x0, y0), (x1, y1), ON, -1)


class PatternRegistry:
    def __init__(self, patterns: list[Pattern]):
        self._items = {p.key: p for p in patterns}

    def get(self, key: str) -> Pattern:
        if key not in self._items:
            raise KeyError(f"모르는 모양: {key}")
        return self._items[key]

    def keys(self) -> list[str]:
        return list(self._items)

    def describe(self) -> list[dict]:
        return [{"key": p.key, "label": p.label, "description": p.description} for p in self._items.values()]


REGISTRY = PatternRegistry([
    GridPattern(), DiagonalPattern(), HorizontalLinesPattern(), VerticalLinesPattern(),
    DotsPattern(), CheckPattern(), CrossPattern(), BoxPattern(),
])
