"""예시 그림 — 지금 설정 예시 · 모양 카드 · 가리기 카드 · 조합 카드

플레이트를 넣기 전에는 예시 캐릭터 위에, 넣은 뒤에는 그 플레이트의 머리 부분 위에 그려 PNG(data URL)로 만든다.
  카드 (작은 그림)   model/sample_character.png      — 목 아래를 지운 머리 · 더듬이를 카드 가운데에
  조합 (큰 그림)     model/sample_character_body.png — 전신을 그림 가운데에
예시 캐릭터는 픽셀 그림이라 각지게 키우고, 바탕은 투명하게 뺀다 (화면 카드 색 위에 캐릭터만).
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import config
from .geometry import Rect
from .patterns import REGISTRY
from .renderer import GridRenderer
from .settings import GridSettings

THUMB = (240, 180)  # 카드 그림 크기 (w, h)
FACE_ASPECT = 0.8   # 사람 얼굴 가로 ÷ 세로 (sample_face.json 은 0~1 로 늘려 저장돼 있음)
SAMPLE_FACE = config.APP_DIR / "model" / "sample_face.json"
SAMPLE_IMAGE = config.APP_DIR / "model" / "sample_character.png"
SAMPLE_BODY = config.APP_DIR / "model" / "sample_character_body.png"
SAMPLE_CHARACTER_FACE = config.APP_DIR / "model" / "sample_character_face.json"   # 캐릭터 눈 · 입 · 윤곽에 맞춘 얼굴 점 (tools/make_character_face.py)
SAMPLE_HEAD = Rect(196, 100, 407, 276)      # 두 캐릭터 그림 안의 머리 (더듬이 빼고) — 같은 칸에서 잘라 위치가 같다
SAMPLE_FACE_BOX = (200, 96, 400, 300)      # 캐릭터 얼굴 점 파일이 없을 때 사람 얼굴 점을 얹을 자리
MATTE_BG = (35, 35, 35)                    # 캐릭터 바탕을 화면 카드 색(#232323)으로 바꿔 두고 투명하게 뺀다 (선 가장자리에 초록이 안 묻게)
CARD_THICK = 3.0                           # 예시 그림의 선 굵기(px) — 작은 카드에서도 보이게 (점은 이 비율로 따라감)
CARD_PAD = 0.14                            # 캐릭터 둘레 여백 (그림 크기 비율) — 가림 · 격자가 카드 밖으로 잘리지 않게


@dataclass
class Sample:
    """플레이트 없이 쓰는 예시 그림 하나"""
    frame: np.ndarray                     # BGR (바탕은 MATTE_BG 로 바꿔 둠)
    head: Rect                            # 그 안의 머리 박스
    keep: Optional[np.ndarray] = None     # 캐릭터 자리 [H,W] bool — 있으면 바탕을 투명하게 · 픽셀을 각지게
    region: Optional[Rect] = None         # 그림에 담을 곳 (가운데 정렬 기준). None 이면 머리 둘레
    offset: tuple[int, int] = (0, 0)      # 둘레에 여백을 붙여 원래 좌표가 밀린 만큼 (얼굴 점 파일 좌표에 더함)

    MARGIN = 256    # 캐릭터 둘레에 붙이는 빈 여백 — 키 큰 전신도 4:3 그림 안에 잘리지 않고 들어가게

    @property
    def pixel(self) -> bool:
        return self.keep is not None

    @classmethod
    def load(cls, path: Path, head: Rect, upto: Optional[float] = None) -> Optional["Sample"]:
        """캐릭터 그림을 읽고 바탕을 뺀다. upto 를 주면 그 높이까지(머리 · 더듬이)만 담을 곳으로"""
        try:
            img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        except OSError:
            return None
        if img is None:
            return None
        bg = np.median(img.reshape(-1, 3), 0)
        keep = np.abs(img.astype(np.int16) - bg.astype(np.int16)).sum(2) > 60
        img[~keep] = MATTE_BG
        rows = np.flatnonzero(keep.any(1))
        cols = np.flatnonzero(keep.any(0))
        if not rows.size:
            return None
        y1 = rows[-1] + 1 if upto is None else upto   # 머리 아래 가림이 들어갈 자리까지
        m = cls.MARGIN
        img = cv2.copyMakeBorder(img, m, m, m, m, cv2.BORDER_CONSTANT, value=MATTE_BG)
        keep = np.pad(keep, m)
        region = Rect(cols[0] + m, rows[0] + m, cols[-1] + 1 + m, y1 + m)
        head = Rect(head.x0 + m, head.y0 + m, head.x1 + m, head.y1 + m)
        return cls(img, head, keep, region, (m, m))

    @classmethod
    def drawn(cls) -> "Sample":
        """캐릭터 그림이 없을 때: 회색 배경 + 머리 · 어깨 실루엣"""
        w, h = THUMB
        img = np.zeros((h, w, 3), np.uint8)
        for y in range(h):  # 은은한 세로 그라데이션
            img[y] = (52 + y // 6, 50 + y // 6, 48 + y // 6)
        cv2.ellipse(img, (w // 2, h + 40), (95, 85), 0, 180, 360, (95, 92, 88), -1, cv2.LINE_AA)   # 어깨
        cv2.rectangle(img, (w // 2 - 18, 100), (w // 2 + 18, 140), (150, 160, 175), -1)          # 목
        cv2.ellipse(img, (w // 2, 78), (44, 56), 0, 0, 360, (160, 172, 190), -1, cv2.LINE_AA)     # 얼굴
        cv2.ellipse(img, (w // 2, 58), (48, 40), 0, 180, 360, (40, 42, 48), -1, cv2.LINE_AA)      # 머리카락
        for dx in (-16, 16):
            cv2.circle(img, (w // 2 + dx, 80), 4, (60, 60, 70), -1, cv2.LINE_AA)                  # 눈
        cv2.ellipse(img, (w // 2, 104), (12, 5), 0, 0, 180, (90, 90, 140), 2, cv2.LINE_AA)        # 입
        return cls(img, Rect(w / 2 - 50, 16, w / 2 + 50, 136))


@dataclass
class Cut:
    """잘라 키운 그림 한 장 + 원래 그림에서 옮겨 온 방법"""
    image: np.ndarray
    head: Rect
    x0: int
    y0: int
    scale: float
    keep: Optional[np.ndarray] = None     # 바탕을 투명하게 뺄 때 남길 곳 [H,W] uint8 0~255 (가장자리는 중간값)
    src: tuple[int, int] = (0, 0)         # 원래 그림에서 실제로 잘라 낸 크기 (h, w) — 투명 자리도 똑같이 자르려고

    def move(self, pts: np.ndarray) -> np.ndarray:
        return (pts - np.array([self.x0, self.y0], np.float32)) * self.scale


class Gallery:
    BIG = (THUMB[0] * 2, THUMB[1] * 2)

    def __init__(self):
        card = Sample.load(SAMPLE_IMAGE, SAMPLE_HEAD, upto=SAMPLE_HEAD.y1 + 0.25 * SAMPLE_HEAD.h)   # 머리 아래 가림 여유까지
        self._card = card or Sample.drawn()
        self._body = (Sample.load(SAMPLE_BODY, SAMPLE_HEAD) or self._card) if card else self._card
        self._face_box = SAMPLE_FACE_BOX if card else (THUMB[0] / 2 - 36, 30, THUMB[0] / 2 + 36, 126)
        self._character = card is not None

    # 테스트 · 예전 코드가 보는 이름
    @property
    def _frame(self) -> np.ndarray:
        return self._card.frame

    @property
    def _head(self) -> Rect:
        return self._card.head

    # ── 자르기 ──
    @staticmethod
    def crop_around(frame: np.ndarray, head: Rect, size: tuple[int, int] = THUMB) -> tuple[np.ndarray, Rect]:
        """머리가 카드 가운데 오도록 잘라 size 크기로 맞춘다"""
        c = Gallery._cut(frame, head, Gallery._around_head(head), size)
        return c.image, c.head

    @staticmethod
    def _around_head(head: Rect) -> Rect:
        side = max(head.h * 1.9, 40)
        cx, cy = head.center
        return Rect(cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)

    @staticmethod
    def _cut(frame: np.ndarray, head: Rect, region: Rect, size: tuple[int, int], pixel: bool = False,
             pad: float = 0.0) -> Cut:
        """region 을 가운데에 두고 size 비율로 넓혀 자른 뒤 size 로 키운다 (그림 밖으로 나가면 안쪽으로 밀어 넣음)"""
        tw, th = size
        H, W = frame.shape[:2]
        cx, cy = region.center
        side_w = max(region.w * (1 + 2 * pad), region.h * (1 + 2 * pad) * tw / th, 4)
        side_w = min(side_w, W, H * tw / th)
        side_h = side_w * th / tw
        x0 = int(round(min(max(0, cx - side_w / 2), max(0, W - side_w))))
        y0 = int(round(min(max(0, cy - side_h / 2), max(0, H - side_h))))
        crop = frame[y0:y0 + int(round(side_h)), x0:x0 + int(round(side_w))]
        s = tw / crop.shape[1]
        grow = cv2.INTER_NEAREST if pixel else cv2.INTER_CUBIC    # 키울 때만 해당 (줄일 때는 늘 고르게)
        img = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA if s < 1 else grow)
        box = Rect((head.x0 - x0) * s, (head.y0 - y0) * s, (head.x1 - x0) * s, (head.y1 - y0) * s)
        return Cut(img, box, x0, y0, s, src=crop.shape[:2])

    def _cut_sample(self, sample: Sample, size: tuple[int, int]) -> Cut:
        region = sample.region or self._around_head(sample.head)
        c = self._cut(sample.frame, sample.head, region, size, sample.pixel, pad=CARD_PAD if sample.region else 0.0)
        if sample.keep is not None:   # 그림과 똑같은 칸을 같은 방법으로 줄여서 가장자리 투명도까지 맞춘다
            sh, sw = c.src
            m = sample.keep[c.y0:c.y0 + sh, c.x0:c.x0 + sw].astype(np.uint8) * 255
            s = c.image.shape[1] / sw
            c.keep = cv2.resize(m, (c.image.shape[1], c.image.shape[0]),
                                interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST)
        return c

    def _base(self, frame, head, size: tuple[int, int] = THUMB) -> Cut:
        if frame is None or head is None:
            return self._cut_sample(self._card, size)
        return self._cut(frame, head, self._around_head(head), size)

    # ── 저장 ──
    @staticmethod
    def _encode(img: np.ndarray, cut: Optional[Cut] = None) -> str:
        """무손실 PNG (픽셀 그림 · 얇은 선이 뭉개지지 않게). 예시 캐릭터면 캐릭터 · 그린 것만 남기고 바탕은 투명"""
        if cut is not None and cut.keep is not None:
            drawn = (img != cut.image).any(axis=2)   # 그린 곳은 불투명 (바탕색이 카드 색과 같아 반투명 그림도 자연스럽게 보임)
            img = np.dstack([img, np.where(drawn, 255, cut.keep).astype(np.uint8)])
        ok, buf = cv2.imencode(".png", img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()

    @staticmethod
    def _thumb_settings(s: GridSettings) -> GridSettings:
        # 카드 그림은 작아서 선이 1px 아래로 사라지지 않게
        return s.with_overrides({"min_px": 2, "expand_x": 0, "expand_top": 0, "expand_bottom": 0})

    # ── 얼굴 점 ──
    def _sample_face(self):
        """예시 캐릭터 위 얼굴 점 — 캐릭터에 맞춰 둔 점이 있으면 그것, 없으면 사람 얼굴 점을 얼굴 자리에 같은 비율로"""
        from .facemesh import FaceShape
        if not hasattr(self, "_sample_face_cache"):
            self._sample_face_cache = None
            try:
                if self._character and SAMPLE_CHARACTER_FACE.is_file():
                    pts = np.array(json.loads(SAMPLE_CHARACTER_FACE.read_text(encoding="utf-8"))["points"], np.float32)
                    self._sample_face_cache = FaceShape(pts + np.array(self._card.offset, np.float32))
                else:
                    pts = np.array(json.loads(SAMPLE_FACE.read_text(encoding="utf-8"))["points"], np.float32)
                    x0, y0, x1, y1 = self._face_box
                    k = min((x1 - x0) / FACE_ASPECT, y1 - y0)
                    size = np.array([k * FACE_ASPECT, k], np.float32)
                    at = np.array([(x0 + x1) / 2, (y0 + y1) / 2], np.float32) - size / 2 + np.array(self._card.offset, np.float32)
                    self._sample_face_cache = FaceShape(pts * size + at)
            except (OSError, ValueError, KeyError):
                pass
        return self._sample_face_cache

    def _scene(self, frame, head, faces, body: bool = False):
        """(잘라 키운 그림, 얼굴들) — 플레이트면 머리 둘레, 예시면 캐릭터 (body=True 면 전신)"""
        from .facemesh import FaceShape
        if frame is None or head is None:
            f = self._sample_face()
            cut = self._cut_sample(self._body if body else self._card, self.BIG)
            faces = [f] if f is not None else []
        else:
            cut = self._cut(frame, head, self._around_head(head), self.BIG)
        return cut, [FaceShape(cut.move(f.points)) for f in (faces or [])]

    def _render(self, settings: GridSettings, keys, cut: Cut, fs) -> str:
        from .guide import REF_WIDTH
        from .layers import Recipe, Scene
        thick = CARD_THICK / 2 * REF_WIDTH / cut.image.shape[1]     # 1920 폭에서 선 2px 이 기준 → 카드에서 CARD_THICK px
        scene = Scene.still(cut.image, [cut.head], settings.with_overrides({"min_px": 2}), fs, thick=max(1.0, thick))
        out = Recipe(settings.base, keys).render(scene)
        return self._encode(out, cut if settings.base != "black" else None)   # 검은 바탕은 바탕이 곧 그림

    # ── 화면에 보내는 그림 ──
    def example(self, settings: GridSettings, frame: Optional[np.ndarray] = None, head: Optional[Rect] = None) -> str:
        """지금 격자 설정을 그대로 그린 예시 한 장"""
        cut = self._base(frame, head, self.BIG)
        s = settings.with_overrides({"min_px": 2})
        return self._encode(GridRenderer(s).render(cut.image, [cut.head]), cut)

    def layer_cards(self, settings: GridSettings, frame=None, head=None, faces=None) -> list[dict]:
        """조합 카드마다 그 레이어 하나만 그린 예시 (지금 바탕 · 색 설정 그대로)"""
        from .layers import Recipe
        cut, fs = self._scene(frame, head, faces)
        return [{**d, "image": self._render(settings, [d["key"]], cut, fs)} for d in Recipe.describe()]

    def combo(self, settings: GridSettings, frame=None, head=None, faces=None) -> str:
        """지금 고른 조합 전체를 그린 예시 — 예시 캐릭터면 전신 (조합이 비면 바탕만)"""
        cut, fs = self._scene(frame, head, faces, body=True)
        return self._render(settings, settings.recipe, cut, fs)

    COVERS = (("검정 · 타원", "#000000", "oval"), ("검정 · 네모", "#000000", "box"),
              ("초록 · 타원", "#00FF00", "oval"), ("초록 · 네모", "#00FF00", "box"))

    def covers(self, frame: Optional[np.ndarray] = None, head: Optional[Rect] = None) -> list[dict]:
        """_cover 레이어 예시 (가리기 색 × 모양)"""
        cut = self._base(frame, head)
        cards = []
        for name, color, shape in self.COVERS:
            r = GridRenderer(GridSettings.from_dict({"head_cover_color": color, "mask_shape": shape}))
            img = r.cover(cut.image, r.head_mask(cut.image.shape[:2], [cut.head]))
            cards.append({"name": name, "head_cover_color": color, "mask_shape": shape, "image": self._encode(img, cut)})
        return cards

    def patterns(self, frame: Optional[np.ndarray] = None, head: Optional[Rect] = None,
                 base: GridSettings | None = None) -> list[dict]:
        cut = self._base(frame, head)
        base = base or GridSettings()
        cards = []
        for p in REGISTRY.describe():
            s = self._thumb_settings(base.with_overrides({"pattern": p["key"]}))
            cards.append({**p, "image": self._encode(GridRenderer(s).render(cut.image, [cut.head]), cut)})
        return cards
