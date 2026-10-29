"""색 섞기 — 그림이 있는 사각형 안만 계산한다

격자 · 가림은 머리 근처에만 있어서, 1920×1080 전체를 소수점으로 섞으면 대부분 헛일이다
(재 보니 한 번에 0.05초 — 카드 하나 · 결과물 하나마다 반복). 알파가 0 인 곳은 원본 그대로라
결과는 전체를 섞을 때와 픽셀 단위로 같다 (tests/test_blend.py).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def bounds(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    """[H,W] 에서 0 이 아닌 곳을 감싸는 (y0, y1, x0, x1) — 없으면 None"""
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def over_color(canvas: np.ndarray, alpha: np.ndarray, color, opacity: float = 1.0) -> np.ndarray:
    """canvas · (1 − a) + color · a 의 새 그림. alpha 는 [H,W] float 0~1 또는 uint8 0~255, opacity 는 전체 불투명도"""
    out = canvas.copy()
    box = bounds(alpha)
    if box is None or opacity <= 0:
        return out
    y0, y1, x0, x1 = box
    a = alpha[y0:y1, x0:x1].astype(np.float32)
    if alpha.dtype == np.uint8:
        a /= 255.0
    if opacity < 1:
        a *= opacity
    a = a[..., None]
    roi = out[y0:y1, x0:x1].astype(np.float32)
    out[y0:y1, x0:x1] = (roi * (1 - a) + np.asarray(color, np.float32) * a).astype(np.uint8)
    return out
