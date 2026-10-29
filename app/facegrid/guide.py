"""가린 표정 가이드 — 10/01 masked-face-guides 방식

1. 얼굴(또는 머리 전체)을 가림 색(기본 검정)으로 덮는다.
     face : MediaPipe 얼굴 점 478개의 바깥 테두리를 mask_grow 만큼 넓힌 다각형
     head : 머리 검출 박스를 둥근 네모로 (귀 · 머리카락 · 턱 아래까지)
2. 그 위에 표정 점 · 윤곽선을 guide_color(기본 빨강)로 그린다.
3. 옆모습 · 뒤통수처럼 표정을 못 찾은 머리도 머리 박스로 가려서, 얼굴이 드러나는 프레임이 없게 한다.

시댄스 Edit 에 넣으면 검은 부분은 새로 그릴 자리, 점 · 선은 눈 · 입 · 고개 움직임 참고가 된다.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .blend import over_color
from .facemesh import FaceLines, FacePoints, FaceShape
from .geometry import Rect
from .settings import GridSettings

HEAD_PAD = dict(x=0.08, top=0.06, bottom=0.18)   # head 모드 · 대신 가리기 여유 (머리 박스 비율)
REF_WIDTH = 1920                                  # 이 폭에서 선 2px · 점 지름 3px (10/01 과 같게)


class MaskedFaceGuide:
    def __init__(self, settings: GridSettings):
        self._s = settings

    # ── 가림 영역 ──
    def _face_polygon(self, face: FaceShape) -> np.ndarray:
        hull = cv2.convexHull(face.points.astype(np.float32)).reshape(-1, 2)
        center = hull.mean(0)
        return np.round(center + (hull - center) * (1 + 2 * self._s.mask_grow)).astype(np.int32)

    @staticmethod
    def _head_rect(mask: np.ndarray, box: Rect) -> None:
        r = box.expanded(HEAD_PAD["x"], HEAD_PAD["top"], HEAD_PAD["bottom"])
        x0, y0, x1, y1 = r.as_int()
        rad = max(2, int(min(r.w, r.h) * 0.22))
        cv2.rectangle(mask, (x0 + rad, y0), (x1 - rad, y1), 255, -1)
        cv2.rectangle(mask, (x0, y0 + rad), (x1, y1 - rad), 255, -1)
        for cx, cy in ((x0 + rad, y0 + rad), (x1 - rad, y0 + rad), (x0 + rad, y1 - rad), (x1 - rad, y1 - rad)):
            cv2.circle(mask, (cx, cy), rad, 255, -1, cv2.LINE_AA)

    @staticmethod
    def _has_face(box: Rect, faces: Sequence[FaceShape]) -> bool:
        wide = box.expanded(0.25, 0.25, 0.25)
        for f in faces:
            cx, cy = f.points.mean(0)
            if wide.x0 <= cx <= wide.x1 and wide.y0 <= cy <= wide.y1:
                return True
        return False

    def mask(self, size: tuple[int, int], faces: Sequence[FaceShape], heads: Sequence[Rect]) -> np.ndarray:
        h, w = size
        m = np.zeros((h, w), np.uint8)
        for f in faces:
            cv2.fillPoly(m, [self._face_polygon(f)], 255, cv2.LINE_AA)
        for box in heads:
            if self._s.mask_source == "head" or not self._has_face(box, faces):
                self._head_rect(m, box)
        return m

    # ── 그리기 ──
    def render(self, frame: np.ndarray, faces: Sequence[FaceShape], heads: Sequence[Rect],
               mask_faces: Sequence[FaceShape] | None = None) -> np.ndarray:
        """faces = 점 · 선을 그릴 얼굴, mask_faces = 가림 모양에 쓸 얼굴 (없으면 faces)"""
        m = self.mask(frame.shape[:2], faces if mask_faces is None else mask_faces, heads)
        out = over_color(frame, m, self._s.cover_bgr, self._s.cover_alpha)
        scale = frame.shape[1] / REF_WIDTH
        line = max(1, int(round(2 * scale)))
        dot = max(1, int(round(1.5 * scale)))
        col = self._s.guide_bgr
        FaceLines.draw(out, faces, col, line)
        return FacePoints.draw(out, faces, col, dot)
