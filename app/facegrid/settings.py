"""격자 설정

GridSettings.from_dict() 는 화면에서 온 값을 검사한다:
모르는 키는 버리고, 숫자가 아니면 기본값, 범위를 벗어나면 끝값으로 맞춘다.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, fields, replace

from .patterns import REGISTRY

_HEX = re.compile(r"^#?[0-9a-fA-F]{6}$")
COLOR_KEYS = ("color", "cover_color", "head_cover_color", "guide_color")


@dataclass(frozen=True)
class GridSettings:
    pattern: str = "grid"
    cols: int = 5
    rows: int = 6
    color: str = "#DC281E"
    alpha: float = 1.0
    fill_alpha: float = 0.0
    thickness: float = 2.2      # 선 두께 = 머리 폭의 %
    min_px: int = 1
    clip: str = "box"           # box | mask(타원)
    expand_x: float = 0.0
    expand_top: float = 0.0
    expand_bottom: float = 0.0
    overhang: float = 0.06
    threshold: float = 0.25     # 머리 검출 민감도 (낮을수록 많이 잡음)
    smooth: float = 0.5
    hold: int = 12
    cover_color: str = "#000000"   # 가림 · 얼굴 윤곽 색 (_masked 의 가림도 같은 색)
    cover_alpha: float = 1.0       # 가림 · 얼굴 윤곽 불투명도 — 1 = 꽉 가림, 낮추면 아래 원본이 비침
    head_cover_color: str = "#000000"   # 가림 · 머리 전체 색 (따로 저장 _cover 도 이 색)
    head_cover_alpha: float = 1.0       # 가림 · 머리 전체 불투명도
    mask_shape: str = "oval"       # _cover · _headmask 모양: oval | box
    # _masked (가린 표정 가이드): 얼굴을 cover_color 로 가리고 그 위에 표정 점 · 선
    mask_source: str = "face"      # face = 얼굴 윤곽을 넓혀 가림 | head = 머리 전체(귀 · 머리카락까지)
    mask_grow: float = 0.12        # 얼굴 윤곽을 바깥으로 넓히는 정도 (얼굴 폭 비율)
    guide_color: str = "#DC281E"   # 점 · 선 색 (격자와 같은 빨강)
    guide_alpha: float = 1.0       # 점 · 선 불투명도
    # 조합 (_combo): 바탕 위에 recipe 순서대로 레이어를 겹친다 (layers.py 의 Layer.key)
    base: str = "original"         # original = 원본 위에 | black = 검은 바탕 위에
    recipe: tuple = ()             # 비어 있으면 아무것도 안 겹침 — 격자도 고를 때만 들어간다
    # 머리 찾기
    model: str = "fast"            # fast = YOLO11s | accurate = YOLO11m
    people: tuple = ()             # 가릴 사람 번호 (#1 = 1). 비어 있으면 모두
    # 저장 (mp4) — media.Bitrate
    bitrate: str = "fixed"         # fixed = 고정 CBR (2K 30Mbps · 4K 50Mbps) | source = 원본 영상 비트레이트로 고정

    LIMITS = {
        "cols": (1, 32), "rows": (1, 32), "alpha": (0.0, 1.0), "fill_alpha": (0.0, 1.0),
        "thickness": (0.1, 20.0), "min_px": (1, 32), "expand_x": (-0.5, 2.0), "expand_top": (-0.5, 2.0),
        "expand_bottom": (-0.5, 2.0), "overhang": (0.0, 1.0), "threshold": (0.01, 0.95),
        "smooth": (0.0, 0.95), "hold": (0, 240), "mask_grow": (0.0, 1.0), "cover_alpha": (0.0, 1.0), "head_cover_alpha": (0.0, 1.0),
        "guide_alpha": (0.0, 1.0),
    }
    CLIPS = ("box", "mask")
    MASK_SHAPES = ("oval", "box")
    MASK_SOURCES = ("face", "head")
    BASES = ("original", "black")
    MODELS = ("fast", "accurate")
    BITRATES = ("fixed", "source")
    LAYER_KEYS = ("cover_face", "cover_head", "grid", "lines", "points", "boxes")   # layers.LAYERS 와 같아야 함 (테스트로 확인)
    GRID_KEYS = ("pattern", "cols", "rows", "color", "alpha", "fill_alpha", "thickness", "clip")

    @classmethod
    def from_dict(cls, data: dict | None) -> "GridSettings":
        base = cls()
        values = {}
        if not isinstance(data, dict):
            data = None
        for f in fields(cls):
            if not data or f.name not in data:
                continue
            values[f.name] = cls._coerce(f.name, data[f.name], getattr(base, f.name))
        return replace(base, **values)

    @classmethod
    def _coerce(cls, name, raw, default):
        if name == "pattern":
            return raw if raw in REGISTRY.keys() else default
        if name == "clip":
            return raw if raw in cls.CLIPS else default
        if name == "mask_shape":
            return raw if raw in cls.MASK_SHAPES else default
        if name == "mask_source":
            return raw if raw in cls.MASK_SOURCES else default
        if name == "base":
            return raw if raw in cls.BASES else default
        if name == "model":
            return raw if raw in cls.MODELS else default
        if name == "bitrate":
            return raw if raw in cls.BITRATES else default
        if name == "people":
            if not isinstance(raw, (list, tuple)):
                return default
            ids = []
            for v in raw:
                try:
                    n = int(v)
                except (TypeError, ValueError):
                    continue
                if n >= 1 and n not in ids:
                    ids.append(n)
            return tuple(sorted(ids))
        if name == "recipe":
            if not isinstance(raw, (list, tuple)):
                return default
            seen = []
            for k in raw:   # 아는 레이어만, 순서 유지, 같은 레이어 두 번 금지
                if k in cls.LAYER_KEYS and k not in seen:
                    seen.append(k)
            return tuple(seen)
        if name in COLOR_KEYS:
            s = str(raw).strip()
            return ("#" + s.lstrip("#")).upper() if _HEX.match(s) else default
        try:
            f = float(raw)
            if f != f or f in (float("inf"), float("-inf")):   # NaN · 무한대
                return default
            v = int(f) if isinstance(default, int) else f
        except (TypeError, ValueError, OverflowError):
            return default
        lo, hi = cls.LIMITS.get(name, (v, v))
        return type(default)(min(max(v, lo), hi))

    def with_overrides(self, overrides: dict) -> "GridSettings":
        return GridSettings.from_dict({**self.to_dict(), **overrides})

    def to_dict(self) -> dict:
        d = asdict(self)
        d["recipe"] = list(d["recipe"])   # 화면 · json 과 같은 모양 (목록)
        d["people"] = list(d["people"])
        return d

    @staticmethod
    def _to_bgr(hex_color: str) -> tuple[int, int, int]:
        s = hex_color.lstrip("#")
        r, g, b = (int(s[i:i + 2], 16) for i in (0, 2, 4))
        return b, g, r

    @property
    def bgr(self) -> tuple[int, int, int]:
        return self._to_bgr(self.color)

    @property
    def cover_bgr(self) -> tuple[int, int, int]:
        return self._to_bgr(self.cover_color)

    @property
    def head_cover_bgr(self) -> tuple[int, int, int]:
        return self._to_bgr(self.head_cover_color)

    @property
    def guide_bgr(self) -> tuple[int, int, int]:
        return self._to_bgr(self.guide_color)
