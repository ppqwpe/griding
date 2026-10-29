"""표정 트래킹 — MediaPipe Face Landmarker (얼굴 점 478개)

10/01 강아지 가이드 패키지의 FACE 레이어와 같은 부위를 그린다:
얼굴 윤곽 · 눈 · 눈썹 · 입 · 코 (+ 홍채).
FaceTracker 를 상속하면 다른 표정 인식기로 바꿔 끼울 수 있다 (테스트는 가짜를 쓴다).
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from . import config

FACE_MODEL_PATH = config.APP_DIR / "model" / "face_landmarker.task"
PARTS = ("FACE_OVAL", "LEFT_EYE", "RIGHT_EYE", "LEFT_EYEBROW", "RIGHT_EYEBROW", "LIPS", "NOSE", "LEFT_IRIS", "RIGHT_IRIS")


def _connections() -> dict[str, list[tuple[int, int]]]:
    from mediapipe.tasks.python.vision import FaceLandmarksConnections as C
    out = {}
    for part in PARTS + ("TESSELATION",):
        out[part] = [(c.start, c.end) for c in getattr(C, f"FACE_LANDMARKS_{part}")]
    return out


@dataclass
class FaceShape:
    points: np.ndarray            # [N, 2] 픽셀 좌표

    def bounds(self) -> tuple[int, int, int, int]:
        x0, y0 = self.points.min(0)
        x1, y1 = self.points.max(0)
        return int(x0), int(y0), int(x1), int(y1)


class FaceLines:
    """FaceShape 들을 선으로 그리는 도구"""
    _CONN: dict | None = None

    @classmethod
    def connections(cls) -> dict:
        if cls._CONN is None:
            cls._CONN = _connections()
        return cls._CONN

    @classmethod
    def draw(cls, canvas: np.ndarray, faces: Iterable[FaceShape], color, thickness: int = 1,
             parts: Iterable[str] = PARTS) -> np.ndarray:
        conn = cls.connections()
        for f in faces:
            pts = np.round(f.points).astype(np.int32)
            n = len(pts)
            for part in parts:
                segs = [(pts[a], pts[b]) for a, b in conn.get(part, []) if a < n and b < n]
                for a, b in segs:
                    cv2.line(canvas, tuple(int(v) for v in a), tuple(int(v) for v in b), color, thickness, cv2.LINE_AA)
        return canvas


class FacePoints:
    """얼굴 점 478개를 점으로 찍는 도구 (10/01 face-points-lines 양식: 점 + 윤곽선)"""

    @staticmethod
    def draw(canvas: np.ndarray, faces: Iterable[FaceShape], color, radius: int = 1) -> np.ndarray:
        for f in faces:
            for x, y in np.round(f.points).astype(np.int32):
                cv2.circle(canvas, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
        return canvas

    @staticmethod
    def to_json(faces: Iterable[FaceShape]) -> list[list[list[float]]]:
        return [np.round(f.points, 1).tolist() for f in faces]


@dataclass
class _FaceMemory:
    template: Optional[np.ndarray] = None   # 마지막으로 제대로 찾은 얼굴 점 (머리 박스 기준 0~1)
    streak: int = 0                          # 연달아 찾은 프레임 수


class FaceContinuity:
    """프레임마다 얼굴을 찾았다 놓쳤다 해도 가림 · 점이 깜빡이지 않게 사람별로 이어 준다

      · 가짜 얼굴 거르기  : 머리 박스에 비해 너무 작은 얼굴(뒤통수 · 머리카락을 얼굴로 잘못 본 것)은 버림
      · 가림용 얼굴       : 놓친 프레임엔 그 사람의 마지막 얼굴 모양을 지금 머리 위치로 옮겨 씀
                            (가림이 얼굴 윤곽 ↔ 머리 네모로 갈아타지 않음). 한 번도 정면이 안 나온 사람만 머리 박스로.
      · 그릴 얼굴(점 · 선) : 두 프레임 이상 이어서 찾았을 때만 (한 프레임만 번쩍 나타나지 않게)
    """
    MIN_AREA = 0.18      # 얼굴 윤곽 넓이 / 머리 박스 넓이 — 실제 얼굴은 0.35~0.47, 가짜는 0.04~0.15 (실제 영상에서 잼)
    MIN_STREAK = 2

    def __init__(self):
        self._mem: dict[int, _FaceMemory] = {}

    @classmethod
    def valid(cls, face: FaceShape, box) -> bool:
        hull = cv2.convexHull(face.points.astype(np.float32))
        return cv2.contourArea(hull) >= cls.MIN_AREA * max(1.0, box.area)

    @staticmethod
    def _owner(face: FaceShape, tracks):
        cx, cy = face.points.mean(0)
        best, best_d = None, None
        for t in tracks:
            b = t.box
            if b.x0 <= cx <= b.x1 and b.y0 <= cy <= b.y1:
                hx, hy = b.center
                d = (cx - hx) ** 2 + (cy - hy) ** 2
                if best_d is None or d < best_d:
                    best, best_d = t, d
        return best

    def update(self, tracks, faces) -> tuple[list[FaceShape], list[FaceShape]]:
        """(그릴 얼굴들, 가림용 얼굴들)"""
        found: dict[int, FaceShape] = {}
        for f in faces:
            t = self._owner(f, tracks)
            if t is not None and t.id not in found and self.valid(f, t.box):
                found[t.id] = f
        draw, mask = [], []
        for t in tracks:
            mem = self._mem.setdefault(t.id, _FaceMemory())
            b = t.box
            f = found.get(t.id)
            if f is not None:
                mem.streak += 1
                mem.template = (f.points - np.array([b.x0, b.y0], np.float32)) / np.array([max(b.w, 1), max(b.h, 1)], np.float32)
                mask.append(f)
                if mem.streak >= self.MIN_STREAK:
                    draw.append(f)
            else:
                mem.streak = 0
                if mem.template is not None:   # 마지막 얼굴 모양을 지금 머리 위치로
                    mask.append(FaceShape(mem.template * np.array([b.w, b.h], np.float32) + np.array([b.x0, b.y0], np.float32)))
        return draw, mask


class FaceTracker(ABC):
    @abstractmethod
    def track(self, frame: np.ndarray, index: int, fps: float | None, heads: list = ()) -> list[FaceShape]:
        """heads: 머리 검출기가 찾은 박스들 (Rect). 쓰지 않는 구현도 있다."""

    def close(self) -> None:
        pass


class FaceModelMissingError(RuntimeError):
    pass


class MediaPipeFaceTracker(FaceTracker):
    """영상이면 VIDEO 모드(앞 프레임을 이어서 추적), 이미지면 IMAGE 모드"""

    # 0.4 는 뒤통수 · 머리카락을 얼굴로 잡았고, 0.9 는 진짜 정면 얼굴도 절반을 버렸다.
    # 0.75 에서 정면 얼굴은 거의 다 남고 뒤통수 가짜는 0 (10/27 두 영상으로 잼)
    CONFIDENCE = 0.75

    def __init__(self, video: bool, max_faces: int = 6, model_path: Path = FACE_MODEL_PATH, confidence: float = CONFIDENCE):
        os.environ.setdefault("GLOG_minloglevel", "2")
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        if not Path(model_path).is_file():
            raise FaceModelMissingError(f"표정 모델이 없습니다: {model_path}")
        self._mp = mp
        mode = vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=Path(model_path).read_bytes()),   # 한글 경로 문제 피하기
            running_mode=mode, num_faces=max_faces,
            min_face_detection_confidence=confidence, min_face_presence_confidence=confidence, min_tracking_confidence=confidence)
        self._lm = vision.FaceLandmarker.create_from_options(opts)
        self._video = video
        self._last_ms = -1

    def _detect(self, rgb: np.ndarray, index: int, fps):
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if self._video:
            ms = int(round(index * 1000 / (fps or 24)))
            ms = max(ms, self._last_ms + 1)   # 시간은 반드시 늘어나야 한다
            self._last_ms = ms
            return self._lm.detect_for_video(img, ms).face_landmarks
        return self._lm.detect(img).face_landmarks

    def track(self, frame, index, fps, heads=()):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return [FaceShape(np.array([(p.x * w, p.y * h) for p in face], np.float32)) for face in self._detect(rgb, index, fps)]

    def close(self):
        try:
            self._lm.close()
        except Exception:
            pass


class HeadCropFaceTracker(FaceTracker):
    """머리 박스마다 그 부분만 잘라 크게 키워서 표정을 찾는다.

    멀리 있어 작은 얼굴도 잡히고, 뒤통수처럼 얼굴이 안 보이는 머리는 자연스럽게 빠진다.
    두 사람이 겹쳐 한 조각에 얼굴이 둘 잡히면, 그 머리 가운데에 가장 가까운 얼굴만 그 머리의 것으로 본다.
    """
    CROP = 256       # 잘라서 키울 크기
    MARGIN = 0.35    # 머리 박스 둘레 여유

    def __init__(self, video: bool, model_path: Path = FACE_MODEL_PATH, whole: FaceTracker | None = None):
        self._crop = MediaPipeFaceTracker(video=False, max_faces=3, model_path=model_path)
        self._whole = whole   # 머리를 하나도 못 찾았을 때 화면 전체로 한 번 더 (없으면 생략)

    @staticmethod
    def _pick(found: list[FaceShape], head) -> FaceShape | None:
        """이 머리 박스 안에 중심이 있는 얼굴 중 머리 가운데와 가장 가까운 것"""
        hx, hy = head.center
        best, best_d = None, None
        for f in found:
            cx, cy = f.points.mean(0)
            if not (head.x0 <= cx <= head.x1 and head.y0 <= cy <= head.y1):
                continue
            d = (cx - hx) ** 2 + (cy - hy) ** 2
            if best_d is None or d < best_d:
                best, best_d = f, d
        return best

    def track(self, frame, index, fps, heads=()):
        H, W = frame.shape[:2]
        faces = []
        for b in heads:
            side = max(b.w, b.h) * (1 + 2 * self.MARGIN)
            cx, cy = b.center
            x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
            x1, y1 = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            crop = frame[y0:y1, x0:x1]
            s = self.CROP / max(crop.shape[:2])
            big = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
            found = [FaceShape(f.points / s + np.array([x0, y0], np.float32)) for f in self._crop.track(big, index, fps)]
            face = self._pick(found, b)
            if face is not None and not any(np.allclose(face.points.mean(0), g.points.mean(0), atol=2) for g in faces):
                faces.append(face)   # 겹친 두 머리가 같은 얼굴을 가져가지 않게
        if not faces and not heads and self._whole is not None:
            faces = self._whole.track(frame, index, fps)
        return faces

    def close(self):
        self._crop.close()
        if self._whole is not None:
            self._whole.close()
