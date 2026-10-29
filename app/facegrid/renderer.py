"""격자 불투명도 맵 만들기 · 원본에 합성"""
from __future__ import annotations

from typing import Iterable, Optional

import cv2
import numpy as np

from .blend import over_color
from .geometry import Rect
from .patterns import REGISTRY
from .settings import GridSettings


class GridRenderer:
    def __init__(self, settings: GridSettings):
        self._s = settings
        self._pattern = REGISTRY.get(settings.pattern)
        self._color = np.array(settings.bgr, np.float32)

    @property
    def settings(self) -> GridSettings:
        return self._s

    def head_area(self, box: Rect) -> Rect:
        s = self._s
        return box.expanded(s.expand_x, s.expand_top, s.expand_bottom)

    def alpha(self, size: tuple[int, int], boxes: Iterable[Rect]) -> Optional[np.ndarray]:
        """[H,W] float32 0..1. 머리가 하나도 없으면 None"""
        boxes = [b for b in boxes if b.w > 1 and b.h > 1]
        if not boxes:
            return None
        s = self._s
        h, w = size
        lines = np.zeros((h, w), np.uint8)
        fill = np.zeros((h, w), np.uint8) if s.fill_alpha > 0 else None
        shape = np.zeros((h, w), np.uint8) if s.clip == "mask" else None
        for raw in boxes:
            box = self.head_area(raw)
            t = max(s.min_px, int(round(box.w * s.thickness / 100)))
            self._pattern.draw(lines, box, s.cols, s.rows, t, s.overhang)
            x0, y0, x1, y1 = box.as_int()
            if fill is not None:
                cv2.rectangle(fill, (x0, y0), (x1, y1), 255, -1)
            if shape is not None:
                cx, cy = box.center
                cv2.ellipse(shape, (int(cx), int(cy)), (max(1, int(box.w / 2)), max(1, int(box.h / 2))), 0, 0, 360, 255, -1)
        a = lines.astype(np.float32) * (s.alpha / 255.0)
        if fill is not None:
            np.maximum(a, fill.astype(np.float32) * (s.fill_alpha / 255.0), out=a)
        if shape is not None:
            a *= cv2.GaussianBlur(shape, (0, 0), 2).astype(np.float32) / 255.0
        return a

    def composite(self, frame: np.ndarray, alpha: Optional[np.ndarray]) -> np.ndarray:
        if alpha is None:
            return frame
        return over_color(frame, alpha, self._color)

    def render(self, frame: np.ndarray, boxes: Iterable[Rect]) -> np.ndarray:
        return self.composite(frame, self.alpha(frame.shape[:2], boxes))

    def head_mask(self, size: tuple[int, int], boxes: Iterable[Rect]) -> np.ndarray:
        """머리 자리 흰색 / 나머지 검정 [H,W] uint8 (모양은 settings.mask_shape)"""
        h, w = size
        m = np.zeros((h, w), np.uint8)
        for raw in boxes:
            box = self.head_area(raw)
            x0, y0, x1, y1 = box.as_int()
            if self._s.mask_shape == "box":
                cv2.rectangle(m, (x0, y0), (x1, y1), 255, -1)
            else:
                cx, cy = box.center
                cv2.ellipse(m, (int(cx), int(cy)), (max(1, int(box.w / 2)), max(1, int(box.h / 2))), 0, 0, 360, 255, -1, cv2.LINE_AA)
        return m

    def cover(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """머리를 '가림 · 머리 전체' 색 · 불투명도로 채운 프레임 (_cover)"""
        return over_color(frame, mask, self._s.head_cover_bgr, self._s.head_cover_alpha)

    @staticmethod
    def matte(alpha: Optional[np.ndarray], size: tuple[int, int]) -> np.ndarray:
        """흰 격자 / 검은 배경 BGR 프레임"""
        m = np.zeros(size, np.uint8) if alpha is None else np.clip(alpha * 255, 0, 255).astype(np.uint8)
        return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
