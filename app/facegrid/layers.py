"""조합 — 카드(레이어)를 골라 순서대로 겹쳐 결과 한 장을 만든다

  Recipe(바탕, [레이어1, 레이어2, ...]).render(scene)
    바탕   original = 원본 위에 · black = 검은 바탕 위에
    레이어  아래 Layer 하위 클래스. 앞에 고른 것부터 칠하고, 뒤에 고른 것이 위에 덮인다.

레이어마다 isolate(scene) 로 투명 배경 그림만 따로 뽑을 수 있다 (_layer 결과물 — 압축 없이 편집기에서 겹치기용).

예) 10/01 가린 표정 가이드 = 원본 + [가림 · 얼굴 윤곽, 표정 선, 표정 점]
    시댄스 격자            = 원본 + [격자]
    가림 + 격자            = 원본 + [가림 · 얼굴 윤곽, 표정 점, 격자]

새 레이어 = Layer 상속 + LAYERS 에 추가 + GridSettings.LAYER_KEYS 에 키 추가.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np

from .blend import bounds, over_color
from .composer import Composer
from .facemesh import FaceLines, FacePoints, FaceShape
from .geometry import Rect
from .guide import MaskedFaceGuide, REF_WIDTH, draw_guide
from .renderer import GridRenderer
from .settings import GridSettings
from .tracker import Track


@dataclass
class Scene:
    """레이어가 그릴 때 쓰는 재료 (한 프레임)"""
    original: np.ndarray
    tracks: list[Track]
    renderer: GridRenderer
    faces: list[FaceShape] = field(default_factory=list)        # 점 · 선을 그릴 얼굴
    mask_faces: Optional[list[FaceShape]] = None                 # 가림 모양에 쓸 얼굴 (없으면 faces)
    _alpha: Optional[np.ndarray] = None
    _alpha_done: bool = False
    thick: float = 1.0                                           # 점 · 선 굵기 배율 (작은 예시 그림에서 잘 보이게)

    @classmethod
    def still(cls, frame: np.ndarray, heads: Sequence[Rect], settings: GridSettings, faces=(), thick: float = 1.0) -> "Scene":
        """한 장짜리 장면 (미리보기 · 예시 그림용, 추적 없이)"""
        return cls(frame, [Track(i, b) for i, b in enumerate(heads)], GridRenderer(settings), list(faces), thick=thick)

    @property
    def heads(self) -> list[Rect]:
        return [t.box for t in self.tracks]

    @property
    def alpha(self) -> Optional[np.ndarray]:
        if not self._alpha_done:
            self._alpha = self.renderer.alpha(self.original.shape[:2], self.heads)
            self._alpha_done = True
        return self._alpha

    @property
    def scale(self) -> float:
        return self.original.shape[1] / REF_WIDTH


def _fill(canvas: np.ndarray, mask: np.ndarray, bgr, alpha: float) -> np.ndarray:
    return over_color(canvas, mask, bgr, alpha)


class Layer(ABC):
    key: str = ""
    label: str = ""
    description: str = ""
    needs_faces: bool = False

    @abstractmethod
    def apply(self, canvas: np.ndarray, scene: Scene) -> np.ndarray:
        """canvas 위에 이 레이어를 그려 돌려준다 (canvas 를 바꿔도 됨)"""

    def isolate(self, scene: Scene) -> np.ndarray:
        """이 레이어만 투명 배경 BGRA 로 — 검은 바탕 · 흰 바탕에 각각 그려 둘의 차이로 알파를 구한다.
        원본 위에 이 그림을 알파로 얹으면 apply 와 같은 결과가 된다 (레이어 종류와 상관없이)."""
        black = self.apply(np.zeros_like(scene.original), scene)
        white = self.apply(np.full_like(scene.original, 255), scene)
        out = np.zeros((*black.shape[:2], 4), np.uint8)
        box = bounds(black.any(axis=2) | (white != 255).any(axis=2))   # 그린 곳만 계산 (나머지는 완전 투명)
        if box is None:
            return out
        y0, y1, x0, x1 = box
        b = black[y0:y1, x0:x1].astype(np.float32)
        w = white[y0:y1, x0:x1].astype(np.float32)
        a = np.clip(1.0 - (w - b).mean(axis=2) / 255.0, 0.0, 1.0)
        color = np.clip(b / np.maximum(a, 1e-3)[..., None], 0, 255)
        color[a < 1 / 255] = 0
        out[y0:y1, x0:x1] = np.dstack([color, a * 255]).round().astype(np.uint8)
        return out

    def describe(self) -> dict:
        return {"key": self.key, "label": self.label, "description": self.description, "needs_faces": self.needs_faces}


class FaceCoverLayer(Layer):
    key, label, needs_faces = "cover_face", "가림 · 얼굴 윤곽", True
    description = "얼굴 윤곽을 조금 넓혀 덮습니다. 표정을 못 찾은 옆모습과 뒤통수는 머리 박스로 덮습니다"

    def apply(self, canvas, scene):
        s = replace(scene.renderer.settings, mask_source="face")
        faces = scene.faces if scene.mask_faces is None else scene.mask_faces
        return _fill(canvas, MaskedFaceGuide(s).mask(canvas.shape[:2], faces, scene.heads), s.cover_bgr, s.cover_alpha)


class HeadCoverLayer(Layer):
    key, label = "cover_head", "가림 · 머리 전체"
    description = "귀와 머리카락까지 머리 전체를 덮습니다. 표정 인식 없이도 됩니다"

    def apply(self, canvas, scene):
        s = replace(scene.renderer.settings, mask_source="head")
        return _fill(canvas, MaskedFaceGuide(s).mask(canvas.shape[:2], [], scene.heads), s.head_cover_bgr, s.head_cover_alpha)


class GridLayer(Layer):
    key, label = "grid", "격자"
    description = "머리 위에 격자를 얹습니다"

    def apply(self, canvas, scene):
        return scene.renderer.composite(canvas, scene.alpha)


class FaceLinesLayer(Layer):
    key, label, needs_faces = "lines", "표정 선", True
    description = "눈, 눈썹, 입, 코, 얼굴 윤곽을 선으로 그립니다"

    def apply(self, canvas, scene):
        t = max(1, int(round(2 * scene.scale * scene.thick)))
        return draw_guide(canvas, scene.faces, scene.renderer.settings, line=t)


class FacePointsLayer(Layer):
    key, label, needs_faces = "points", "표정 점", True
    description = "얼굴 점 478개를 찍습니다"

    def apply(self, canvas, scene):
        r = max(1, int(round(1.5 * scene.scale * scene.thick)))
        return draw_guide(canvas, scene.faces, scene.renderer.settings, dot=r)


class HeadBoxLayer(Layer):
    key, label = "boxes", "추적 박스"
    description = "찾은 머리 박스와 번호. 초록은 찾은 것, 주황(HOLD)은 잠깐 놓쳐 유지한 것"

    def apply(self, canvas, scene):
        return Composer.overlay(canvas, scene.tracks)


LAYERS: dict[str, Layer] = {l.key: l for l in (
    FaceCoverLayer(), HeadCoverLayer(), GridLayer(), FaceLinesLayer(), FacePointsLayer(), HeadBoxLayer())}


class Recipe:
    """바탕 + 겹칠 레이어 순서"""

    def __init__(self, base: str, keys: Sequence[str]):
        self.base = base
        self.layers = [LAYERS[k] for k in keys if k in LAYERS]

    @classmethod
    def from_settings(cls, s: GridSettings) -> "Recipe":
        return cls(s.base, s.recipe)

    @property
    def empty(self) -> bool:
        return not self.layers

    @property
    def needs_faces(self) -> bool:
        return any(l.needs_faces for l in self.layers)

    def render(self, scene: Scene) -> np.ndarray:
        canvas = np.zeros_like(scene.original) if self.base == "black" else scene.original.copy()
        for layer in self.layers:
            canvas = layer.apply(canvas, scene)
        return canvas

    def title(self) -> str:
        return ("검은 바탕" if self.base == "black" else "원본") + " + " + " → ".join(l.label for l in self.layers)

    @staticmethod
    def describe() -> list[dict]:
        return [l.describe() for l in LAYERS.values()]
