"""처리 흐름

  1. 사람 찾기 (scan.Scanner)   영상 전체를 훑어 머리를 추적하고 사람 번호를 정한다
  2. 그리기                      프레임마다 (고른 사람만) 조합 · 결과물을 그린다. 표정 카드가 있으면 표정 인식도.

같은 설정으로 이미 훑었으면(화면의 [사람 찾기]) 그 결과를 그대로 쓴다.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import cv2

from .detector import Detector
from .facemesh import FaceContinuity, FaceTracker, HeadCropFaceTracker
from .layers import Recipe, Scene
from .media import MediaSource, VideoSource
from .outputs import FrameContext, OutputCatalog, OutputTarget
from .probe import OutputCheck
from .renderer import GridRenderer
from .scan import Scanner, TrackScan
from .settings import GridSettings

Progress = Callable[[float, str], None]   # (0~1, 안내 문구)
SCAN_SHARE = 0.35                          # 전체 진행 막대 중 사람 찾기 몫


@dataclass
class JobSummary:
    source: str
    folder: str
    files: list[str] = field(default_factory=list)
    settings: dict = field(default_factory=dict)
    frames: int = 0
    fps: Optional[float] = None
    resolution: list[int] = field(default_factory=list)
    duration: Optional[float] = None
    seconds: float = 0.0
    device: str = ""
    people_count: int = 0                                          # 찾은 사람 수
    people_used: list[int] = field(default_factory=list)          # 실제로 그린 사람 번호
    max_together: int = 0                                          # 한 프레임에 동시에 나온 최대 인원
    people: list[dict] = field(default_factory=list)              # 사람별 나온 구간 (썸네일 제외)
    empty_grid_frames: list[int] = field(default_factory=list)    # 그릴 머리가 하나도 없던 프레임
    face_tracked: bool = False
    empty_face_frames: list[int] = field(default_factory=list)    # 표정을 못 찾은 프레임 (표정 카드를 골랐을 때)
    input: dict = field(default_factory=dict)                     # 넣은 파일 규격 (해상도 · fps · 색 · 소리 · 알아둘 점)
    checks: dict = field(default_factory=dict)                    # 만든 영상마다 원본과 규격 대조 (프레임 수 · 크기 · fps)
    heads: list[list[list[float]]] = field(default_factory=list)  # 프레임마다 [사람 번호, x0, y0, x1, y1]

    MEDIA = (".mp4", ".png", ".jpg", ".jpeg")

    @property
    def main_file(self) -> str:
        """받기 버튼에 걸 대표 파일: 조합 → 그 밖의 첫 영상 · 이미지 (json 은 제외)"""
        media = [f for f in self.files if f.lower().endswith(self.MEDIA)]
        combo = [f for f in media if "_combo." in f]
        return (combo or media or self.files or [""])[0]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=1), encoding="utf-8")

    def brief(self) -> dict:
        """화면에 보낼 요약 (프레임별 머리 위치는 빼고)"""
        d = asdict(self)
        d.pop("heads")
        d["empty"] = len(self.empty_grid_frames)
        d["empty_face"] = len(self.empty_face_frames)
        d["main"] = self.main_file
        return d


FaceTrackerFactory = Callable[[bool], FaceTracker]


class FaceModelError(RuntimeError):
    """표정 인식(MediaPipe)을 이 PC 에서 켜지 못함 — 아주 오래된 CPU · 깨진 설치 등"""


def default_face_tracker(video: bool) -> FaceTracker:
    try:
        return HeadCropFaceTracker(video)
    except Exception as e:
        raise FaceModelError(f"이 PC 에서 표정 인식을 켜지 못했습니다 ({e}). "
                             "표정 카드(가림 · 얼굴 윤곽, 표정 선, 표정 점)를 빼고 "
                             "가림 · 머리 전체나 격자로 다시 실행해 보세요.") from e


class FaceGridPipeline:
    def __init__(self, detector: Detector, catalog: OutputCatalog | None = None,
                 face_tracker: FaceTrackerFactory = default_face_tracker):
        self._detector = detector
        self._catalog = catalog or OutputCatalog()
        self._face_tracker = face_tracker

    @property
    def detector(self) -> Detector:
        return self._detector

    # ── 사람 찾기 ──
    def scan(self, source: MediaSource, settings: GridSettings, progress: Progress = lambda f, t: None) -> TrackScan:
        return Scanner(self._detector, settings).run(source, progress)

    # ── 한 프레임 미리보기 ──
    def preview(self, source: MediaSource, settings: GridSettings, seconds: float = 0.0,
                scan: Optional[TrackScan] = None) -> bytes:
        """한 프레임에 조합을 그대로 그려 본다. 훑어 둔 결과가 있으면 그 사람 번호 · 고른 사람만."""
        frame = source.frame_at(seconds)
        if scan is not None and scan.key == TrackScan.key_for(settings):
            idx = int(round(seconds * (source.fps or 24))) if source.is_video else 0
            tracks = scan.tracks_at(min(idx, scan.frame_count - 1), settings.people)
            heads = [t.box for t in tracks]
        else:
            heads = self._detector.detect(frame, settings.threshold)
            tracks = None
        recipe = Recipe.from_settings(settings)
        if recipe.empty:
            out = GridRenderer(settings).render(frame, heads)
        else:
            faces = []
            if recipe.needs_faces and heads:
                tracker = self._face_tracker(False)
                try:
                    faces = tracker.track(frame, 0, None, heads)
                finally:
                    tracker.close()
            sc = Scene.still(frame, heads, settings, faces)
            if tracks is not None:
                sc.tracks = tracks
            out = recipe.render(sc)
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buf.tobytes()

    @staticmethod
    def _check(source: VideoSource, frames: int, folder: Path, files: list[str]) -> dict:
        """만든 영상이 원본 규격을 지켰나 — 비교 영상(3칸)은 크기가 달라서 프레임 수 · fps 만"""
        out = {}
        for f in files:
            if f.lower().endswith((".mp4", ".mov")):
                r = OutputCheck.compare(source.info, frames, folder / f)
                if "_compare." in f:
                    r["problems"] = [p for p in r["problems"] if not p.startswith("크기")]
                    r["ok"] = not r["problems"]
                out[f] = r
        return out

    # ── 전체 ──
    def run(self, source: MediaSource, folder: Path, stem: str, settings: GridSettings,
            output_keys=None, progress: Progress = lambda f, t: None,
            scan: Optional[TrackScan] = None) -> JobSummary:
        folder.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        outputs = self._catalog.build(output_keys)
        if not outputs:
            raise ValueError("만들 결과물이 없습니다")

        if scan is None or scan.key != TrackScan.key_for(settings):
            scan = self.scan(source, settings, lambda f, t: progress(f * SCAN_SHARE, t))
        share = 0.0 if scan is None else SCAN_SHARE

        renderer = GridRenderer(settings)
        recipe = Recipe.from_settings(settings)
        target = OutputTarget(folder, stem, source, scan.frame_count, settings.bitrate)
        summary = JobSummary(source=source.path.name, folder=folder.name, settings=settings.to_dict(),
                             fps=source.fps, resolution=list(source.size), device=self._detector.device,
                             people_count=len(scan.people), max_together=scan.max_together,
                             people=[{k: v for k, v in p.describe(source.fps).items() if k != "thumb"} for p in scan.people])
        summary.input = source.describe()
        known = {p.id for p in scan.people}
        summary.people_used = sorted(known & set(settings.people)) if settings.people else sorted(known)
        total = max(scan.frame_count, 1)
        opened = []
        faces = self._face_tracker(source.is_video) if any(o.wants_faces(settings) for o in outputs) else None
        continuity = FaceContinuity() if source.is_video else None
        try:
            for o in outputs:
                o.open(target)
                opened.append(o)
            n = 0
            for frame in source.frames():
                tracks = scan.tracks_at(n, settings.people)
                boxes = [t.box for t in tracks]
                alpha = renderer.alpha(frame.shape[:2], boxes)
                ctx = FrameContext(n, frame, renderer.composite(frame, alpha), alpha, tracks, renderer, recipe=recipe)
                if faces is not None:
                    found = faces.track(frame, n, source.fps, [t.box for t in tracks if not t.lost])
                    if continuity is not None:
                        ctx.faces, ctx.mask_faces = continuity.update(tracks, found)
                    else:
                        ctx.faces = [f for f in found if any(FaceContinuity.valid(f, t.box) for t in tracks)] or found
                    if tracks and not ctx.faces:
                        summary.empty_face_frames.append(n)
                if not tracks:
                    summary.empty_grid_frames.append(n)
                summary.heads.append([[t.id, *t.box.rounded()] for t in tracks])
                for o in outputs:
                    o.write(ctx)
                n += 1
                if n % 6 == 0:
                    progress(share + (1 - share) * min(1.0, n / total), f"그리는 중 {n}/{total}")
            if n == 0:
                raise RuntimeError("프레임을 하나도 읽지 못했습니다")
            for o in outputs:
                summary.files += o.close()
        except BaseException:
            for o in opened:
                o.abort()
            raise
        finally:
            if faces is not None:
                faces.close()
        summary.face_tracked = faces is not None
        summary.frames = n
        summary.duration = round(n / source.fps, 3) if source.fps else None
        if isinstance(source, VideoSource):
            summary.checks = self._check(source, n, folder, summary.files)
        summary.seconds = round(time.time() - t0, 1)
        settings_file = folder / f"{stem}_settings.json"
        summary.files.append(settings_file.name)
        summary.save(settings_file)
        progress(1.0, "끝")
        return summary
