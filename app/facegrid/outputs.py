"""결과물. 결과물마다 Output 하위 클래스 하나가 자기 파일을 책임진다.

처리기는 프레임마다 모든 Output 의 write(ctx) 를 부르고, 끝나면 close() 로 파일 이름을 받는다.
영상 입력이면 mp4, 이미지 입력이면 같은 그림을 이미지 파일로 저장한다.

  _combo      고른 카드를 순서대로 겹친 결과        ComboOutput
  _layerN_키  카드마다 투명 배경 무손실 MOV        LayerFilesOutput
  _facegrid   격자 · 시댄스용 (원본 소리)          FacegridOutput
  _compare    PLATE | FACEGRID | TRACK 3칸         CompareOutput
  _overlay    원본 위 머리 추적 확인                OverlayOutput
  _matte      검은 배경 + 흰 격자                  MatteOutput
  _preview    비교 정지 이미지 (가운데 프레임)      PreviewOutput
  _contact    결과 3×3 밀착 시트                   ContactOutput
  _masked     얼굴 검게 가림 + 초록 표정 점 · 선    MaskedGuideOutput
  _face       검은 배경 + 흰 표정 선               FaceLinesOutput
  _face_overlay  원본 위 초록 표정 선               FaceOverlayOutput
  _points     검은 배경 + 흰 얼굴 점 + _points.json FacePointsOutput
  _points_overlay  원본 위 초록 얼굴 점             FacePointsOverlayOutput
  _cover      머리를 한 색으로 꽉 채움              CoverOutput
  _headmask   흰 머리 / 검은 배경                  HeadMaskOutput
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .composer import Composer
from .facemesh import FaceLines, FacePoints, FaceShape
from .guide import MaskedFaceGuide
from .layers import Recipe, Scene
from .media import AlphaMovWriter, EncodeSpec, FFmpegWriter, MediaSource
from .probe import Bitrate
from .renderer import GridRenderer
from .tracker import Track


@dataclass
class FrameContext:
    index: int
    original: np.ndarray
    result: np.ndarray
    alpha: Optional[np.ndarray]
    tracks: list[Track]
    renderer: GridRenderer
    faces: list[FaceShape] = field(default_factory=list)   # 점 · 선을 그릴 얼굴 (표정 결과물이 있을 때만)
    mask_faces: Optional[list[FaceShape]] = None            # 가림 모양에 쓸 얼굴 (놓친 프레임은 마지막 얼굴을 옮긴 것)
    recipe: Optional[Recipe] = None                         # 조합 (비어 있으면 None)
    _overlay: Optional[np.ndarray] = None
    _mask: Optional[np.ndarray] = None
    _combo: Optional[np.ndarray] = None

    @property
    def scene(self) -> Scene:
        return Scene(self.original, self.tracks, self.renderer, self.faces, self.mask_faces, self.alpha, True)

    @property
    def combo(self) -> np.ndarray:
        """조합 결과 (조합이 없으면 격자 결과)"""
        if self.recipe is None or self.recipe.empty:
            return self.result
        if self._combo is None:
            self._combo = self.recipe.render(self.scene)
        return self._combo

    @property
    def overlay(self) -> np.ndarray:  # 여러 결과물이 같이 쓰므로 한 번만 그린다
        if self._overlay is None:
            self._overlay = Composer.overlay(self.original, self.tracks)
        return self._overlay

    @property
    def head_mask(self) -> np.ndarray:
        if self._mask is None:
            self._mask = self.renderer.head_mask(self.original.shape[:2], [t.box for t in self.tracks])
        return self._mask


@dataclass
class OutputTarget:
    folder: Path
    stem: str
    source: MediaSource
    frames: int = 0          # 실제 프레임 수 (사람 찾기에서 센 값). 0 이면 파일에 적힌 값
    bitrate: str = "fixed"   # GridSettings.bitrate

    @property
    def frame_count(self) -> int:
        return self.frames or self.source.frame_count

    def encode_spec(self, size: tuple[int, int]) -> EncodeSpec:
        """원본 크기 결과물은 고정 비트레이트(CBR), 크기가 다른 확인용(_compare 등)은 화질 기준"""
        spec = self.source.encode_spec
        if tuple(size) != tuple(self.source.size):
            return spec
        info = getattr(self.source, "info", None)
        w, h = size
        return replace(spec, kbps=Bitrate.choose(self.bitrate, w + w % 2, h + h % 2, info.video_kbps if info else 0))

    def path(self, suffix: str, ext: str) -> Path:
        return self.folder / f"{self.stem}_{suffix}{ext}"


def save_image(path: Path, img: np.ndarray, ext: str) -> None:
    params = [cv2.IMWRITE_JPEG_QUALITY, 92] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise RuntimeError(f"이미지 저장 실패: {path.name}")
    buf.tofile(str(path))


class Output(ABC):
    key: str = ""
    label: str = ""
    description: str = ""
    group: str = "grid"        # 화면에서 묶어 보여 줄 이름: grid | face | mask
    required: bool = False
    needs_faces: bool = False  # True 면 처리기가 표정 트래킹을 돌린다

    def __init__(self):
        self._target: Optional[OutputTarget] = None
        self._files: list[str] = []

    def wants_faces(self, settings) -> bool:
        """이 설정으로 만들 때 표정 트래킹이 필요한가 (조합은 고른 레이어에 따라 다름)"""
        return self.needs_faces

    def open(self, target: OutputTarget) -> None:
        self._target = target

    @abstractmethod
    def write(self, ctx: FrameContext) -> None: ...

    def close(self) -> list[str]:
        return self._files

    def abort(self) -> None:
        pass

    def _record(self, path: Path) -> None:
        self._files.append(path.name)


class FrameOutput(Output):
    """프레임마다 그림 하나 → 영상이면 mp4, 이미지면 그림 파일"""
    with_audio = True
    image_ext = ".png"

    def __init__(self):
        super().__init__()
        self._writer: Optional[FFmpegWriter] = None
        self._still: Optional[np.ndarray] = None

    @abstractmethod
    def make(self, ctx: FrameContext) -> np.ndarray: ...

    def frame_size(self, target: OutputTarget) -> tuple[int, int]:
        return target.source.size

    def open(self, target):
        super().open(target)
        src = target.source
        if src.is_video:
            audio = src.path if self.with_audio else None
            size = self.frame_size(target)
            self._writer = FFmpegWriter(target.path(self.key, ".mp4"), size, target.encode_spec(size), audio)

    def write(self, ctx):
        img = self.make(ctx)
        if self._writer is not None:
            self._writer.write(img)
        else:
            self._still = img

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._record(self._writer.path)
        elif self._still is not None:
            ext = self._image_ext()
            path = self._target.path(self.key, ext)
            save_image(path, self._still, ext)
            self._record(path)
        return self._files

    def _image_ext(self) -> str:
        return self.image_ext

    def abort(self):
        if self._writer is not None:
            self._writer.abort()


class ComboOutput(FrameOutput):
    key, label, group = "combo", "조합", "combo"
    description = "고른 카드를 순서대로 겹친 결과 한 개 (원본 해상도 · fps · 소리)"

    def wants_faces(self, settings):
        return Recipe.from_settings(settings).needs_faces

    def make(self, ctx):
        return ctx.combo


class LayerFilesOutput(Output):
    """고른 카드마다 투명 배경 무손실 파일 하나 — 원본 플레이트 위에 편집기에서 순서대로 쌓는다"""
    key, label, group = "layers", "카드별 투명 레이어", "combo"
    description = ("고른 카드마다 투명 배경 무손실 MOV 하나씩 (_layer1_grid.mov …, PNG 코덱 · 알파). "
                   "원본 위에 번호 순서대로 쌓으면 _combo 와 같고 H.264 압축이 없음. 이미지면 투명 PNG")

    def __init__(self):
        super().__init__()
        self._writers: list[AlphaMovWriter] = []
        self._layers: list = []
        self._stills: list[np.ndarray] = []

    def wants_faces(self, settings):
        return Recipe.from_settings(settings).needs_faces

    def _name(self, i: int) -> str:
        return f"layer{i + 1}_{self._layers[i].key}"

    def _start(self, recipe: Recipe) -> None:
        self._layers = list(recipe.layers)
        src = self._target.source
        if src.is_video:
            self._writers = [AlphaMovWriter(self._target.path(self._name(i), ".mov"), src.size, src.encode_spec)
                             for i in range(len(self._layers))]

    def write(self, ctx):
        if ctx.recipe is None or ctx.recipe.empty:
            return
        if not self._layers:
            self._start(ctx.recipe)
        scene = ctx.scene
        images = [layer.isolate(scene) for layer in self._layers]
        if self._writers:
            for w, img in zip(self._writers, images):
                w.write(img)
        else:
            self._stills = images

    def close(self):
        for w in self._writers:
            w.close()
            self._record(w.path)
        for i, img in enumerate(self._stills):
            path = self._target.path(self._name(i), ".png")
            save_image(path, img, ".png")
            self._record(path)
        return self._files

    def abort(self):
        for w in self._writers:
            w.abort()


class FacegridOutput(FrameOutput):
    key, label, description = "facegrid", "격자", "머리 위에 격자를 덮은 시댄스용 파일 (원본 해상도 · fps · 소리)"

    def make(self, ctx):
        return ctx.result

    def _image_ext(self):
        ext = self._target.source.path.suffix.lower()
        return ext if ext in (".png", ".jpg", ".jpeg") else ".png"


class OverlayOutput(FrameOutput):
    key, label, description = "overlay", "추적 확인", "원본 위 머리 박스 · 번호 (초록 = 검출, 주황 HOLD = 잠깐 놓쳐 유지)"

    def make(self, ctx):
        return ctx.overlay


class MatteOutput(FrameOutput):
    key, label, description = "matte", "격자 매트", "검은 배경 + 흰 격자 (AE 에서 따로 합성)"
    with_audio = False

    def make(self, ctx):
        return GridRenderer.matte(ctx.alpha, ctx.original.shape[:2])


class CompareOutput(FrameOutput):
    key, label, description = "compare", "비교 영상", "원본 | 결과(조합) | 추적 3칸, 칸마다 위 라벨"
    image_ext = ".jpg"

    def wants_faces(self, settings):   # 가운데 칸이 조합이라 표정 카드가 있으면 표정 인식 필요
        return Recipe.from_settings(settings).needs_faces

    def frame_size(self, target):
        return Composer.compare_size(*target.source.size)

    def make(self, ctx):
        return Composer.compare([ctx.original, ctx.combo, ctx.overlay], Composer.labels_for(ctx.recipe))


class FaceLinesOutput(FrameOutput):
    key, label, group = "face", "표정 가이드", "face"
    description = "검은 배경 + 흰 표정 선 (눈 · 눈썹 · 입 · 코 · 얼굴 윤곽). 10/01 강아지 _face 와 같은 양식"
    needs_faces = True
    with_audio = False

    def make(self, ctx):
        canvas = np.zeros_like(ctx.original)
        t = max(1, ctx.original.shape[1] // 960)
        return FaceLines.draw(canvas, ctx.faces, (255, 255, 255), t)


class FaceOverlayOutput(FrameOutput):
    key, label, group = "face_overlay", "표정 추적 확인", "face"
    description = "원본 위 초록 표정 선 + 얼굴 그물 (원본 소리)"
    needs_faces = True

    def make(self, ctx):
        out = ctx.original.copy()
        FaceLines.draw(out, ctx.faces, (40, 120, 40), 1, parts=("TESSELATION",))
        return FaceLines.draw(out, ctx.faces, (80, 230, 80), max(1, ctx.original.shape[1] // 960))


class FacePointsOutput(FrameOutput):
    key, label, group = "points", "표정 점", "face"
    description = "검은 배경 + 흰 얼굴 점 478개 + 윤곽선. 프레임별 점 좌표는 _points.json 으로 같이 저장"
    needs_faces = True
    with_audio = False

    def __init__(self):
        super().__init__()
        self._frames: list = []

    def make(self, ctx):
        self._frames.append(FacePoints.to_json(ctx.faces))
        canvas = np.zeros_like(ctx.original)
        r = max(1, ctx.original.shape[1] // 1280)
        FaceLines.draw(canvas, ctx.faces, (255, 255, 255), 1)
        return FacePoints.draw(canvas, ctx.faces, (255, 255, 255), r)

    def close(self):
        files = super().close()
        path = self._target.path(self.key, ".json")
        src = self._target.source
        data = {"source": src.path.name, "size": list(src.size), "fps": src.fps, "points_per_face": 478,
                "format": "frames[프레임][얼굴][점] = [x, y] 픽셀", "frames": self._frames}
        path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        self._record(path)
        return self._files


class FacePointsOverlayOutput(FrameOutput):
    key, label, group = "points_overlay", "표정 점 확인", "face"
    description = "원본 위 초록 얼굴 점 + 윤곽선 (원본 소리)"
    needs_faces = True

    def make(self, ctx):
        out = ctx.original.copy()
        r = max(1, ctx.original.shape[1] // 1280)
        FaceLines.draw(out, ctx.faces, (80, 230, 80), 1)
        return FacePoints.draw(out, ctx.faces, (80, 230, 80), r)


class MaskedGuideOutput(FrameOutput):
    key, label, group = "masked", "가린 표정 가이드", "face"
    description = ("얼굴을 검게 가리고 그 위에 초록 표정 점 · 선 (원본 소리). 10/01 masked-face-guides 방식. "
                   "표정을 못 찾은 옆모습 · 뒤통수도 머리 박스로 가림")
    needs_faces = True

    def open(self, target):
        super().open(target)
        self._guide = None

    def make(self, ctx):
        if self._guide is None:
            self._guide = MaskedFaceGuide(ctx.renderer.settings)
        return self._guide.render(ctx.original, ctx.faces, [t.box for t in ctx.tracks], ctx.mask_faces)


class HeadMaskOutput(FrameOutput):
    key, label, group = "headmask", "머리 마스크", "mask"
    description = "흰 머리 / 검은 배경 (AE 에서 가리기 · 교체할 때). 모양은 [가리기 모양]"
    with_audio = False

    def make(self, ctx):
        return cv2.cvtColor(ctx.head_mask, cv2.COLOR_GRAY2BGR)


class CoverOutput(FrameOutput):
    key, label, group = "cover", "가리기", "mask"
    description = "원본 위에 머리를 [가리기 색]으로 꽉 채움 (원본 소리). 10/01 _masked_green 과 같은 방식"

    def make(self, ctx):
        return ctx.renderer.cover(ctx.original, ctx.head_mask)


class PreviewOutput(Output):
    key, label, description = "preview", "미리보기", "비교 정지 이미지 (가운데 프레임)"

    def __init__(self):
        super().__init__()
        self._img = None

    def wants_faces(self, settings):
        return Recipe.from_settings(settings).needs_faces

    def write(self, ctx):
        if ctx.index == self._target.frame_count // 2 or self._img is None:
            self._img = Composer.compare([ctx.original, ctx.combo, ctx.overlay], Composer.labels_for(ctx.recipe))

    def close(self):
        if self._img is not None:
            path = self._target.path(self.key, ".jpg")
            save_image(path, self._img, ".jpg")
            self._record(path)
        return self._files


class ContactOutput(Output):
    key, label, description = "contact", "밀착 시트", "결과 프레임 9장을 3×3 으로 (영상만)"
    COUNT = 9

    def __init__(self):
        super().__init__()
        self._frames: list[np.ndarray] = []
        self._pick: set[int] = set()

    def wants_faces(self, settings):
        return Recipe.from_settings(settings).needs_faces

    def open(self, target):
        super().open(target)
        n = max(1, target.frame_count)
        self._pick = {int(round(i * (n - 1) / (self.COUNT - 1))) for i in range(self.COUNT)}

    def write(self, ctx):
        if self._target.source.is_video and ctx.index in self._pick:
            self._frames.append(ctx.combo)

    def close(self):
        if self._frames:
            path = self._target.path(self.key, ".jpg")
            save_image(path, Composer.contact_sheet(self._frames), ".jpg")
            self._record(path)
        return self._files


class OutputCatalog:
    _CLASSES = (ComboOutput, LayerFilesOutput, FacegridOutput, CompareOutput, OverlayOutput, MatteOutput, PreviewOutput, ContactOutput,
                MaskedGuideOutput, FaceLinesOutput, FaceOverlayOutput, FacePointsOutput, FacePointsOverlayOutput,
                CoverOutput, HeadMaskOutput)
    DEFAULT_ON = ("combo", "compare", "preview", "contact")

    def keys(self) -> list[str]:
        return [c.key for c in self._CLASSES]

    def build(self, keys) -> list[Output]:
        """고른 결과물들. keys=None 이면 기본값 (DEFAULT_ON)"""
        wanted = set(self.DEFAULT_ON if keys is None else keys)
        return [c() for c in self._CLASSES if c.required or c.key in wanted]

    def describe(self) -> list[dict]:
        return [{"key": c.key, "label": c.label, "description": c.description, "required": c.required,
                 "group": c.group, "default": c.required or c.key in self.DEFAULT_ON} for c in self._CLASSES]
