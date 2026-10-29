"""확인용 그림 — 추적 overlay · 3칸 비교 · 밀착 시트

비교 그림 규칙은 2026-10-01 강아지 가이드 패키지와 같다:
칸 폭 640, 칸마다 위에 56px 검은 띠 + 영어 대문자 라벨 → 3칸이면 1920 폭.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .tracker import Track

PANEL_W = 640
LABEL_H = 56
LABEL_BG = (28, 22, 18)
LABEL_FG = (235, 235, 235)
SEEN = (80, 220, 80)
HOLD = (60, 160, 255)
COMPARE_LABELS = ("PLATE / ORIGINAL", "FACEGRID / SEEDANCE", "TRACK / HEAD BOXES")
COMBO_LABELS = ("PLATE / ORIGINAL", "RESULT / COMBO", "TRACK / HEAD BOXES")


class Composer:
    @staticmethod
    def overlay(frame: np.ndarray, tracks: Sequence[Track]) -> np.ndarray:
        out = frame.copy()
        t = max(2, frame.shape[1] // 640)
        for tr in tracks:
            x0, y0, x1, y1 = tr.box.as_int()
            col = HOLD if tr.lost else SEEN
            cv2.rectangle(out, (x0, y0), (x1, y1), col, t, cv2.LINE_AA)
            label = f"#{tr.id}" + (" HOLD" if tr.lost else "")
            cv2.putText(out, label, (x0, max(18, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * t, col, t, cv2.LINE_AA)
        return out

    @staticmethod
    def labels_for(recipe) -> tuple:
        """조합 결과면 가운데 칸 라벨을 RESULT 로"""
        return COMBO_LABELS if recipe is not None and not recipe.empty else COMPARE_LABELS

    @staticmethod
    def panel_height(width: int, height: int) -> int:
        return int(round(PANEL_W * height / width / 2)) * 2

    @classmethod
    def compare_size(cls, width: int, height: int, panels: int = 3) -> tuple[int, int]:
        return PANEL_W * panels, cls.panel_height(width, height) + LABEL_H

    @classmethod
    def compare(cls, panels: Sequence[np.ndarray], labels: Sequence[str] = COMPARE_LABELS) -> np.ndarray:
        h, w = panels[0].shape[:2]
        ph = cls.panel_height(w, h)
        cells = []
        for img, label in zip(panels, labels):
            bar = np.full((LABEL_H, PANEL_W, 3), LABEL_BG, np.uint8)
            cv2.putText(bar, label, (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.62, LABEL_FG, 1, cv2.LINE_AA)
            cells.append(np.vstack([bar, cv2.resize(img, (PANEL_W, ph), interpolation=cv2.INTER_AREA)]))
        return np.hstack(cells)

    @staticmethod
    def contact_sheet(frames: Sequence[np.ndarray], cols: int = 3, cell_w: int = 480) -> np.ndarray:
        h, w = frames[0].shape[:2]
        ch = int(round(cell_w * h / w))
        cells = [cv2.resize(f, (cell_w, ch), interpolation=cv2.INTER_AREA) for f in frames]
        while len(cells) % cols:
            cells.append(np.zeros_like(cells[0]))
        return np.vstack([np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)])
