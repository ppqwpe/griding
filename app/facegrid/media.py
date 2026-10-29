"""입력(영상 · 이미지)과 영상 저장

MediaSource.open(path) 가 확장자를 보고 VideoSource 나 ImageSource 를 돌려준다.
둘 다 frames() 로 프레임을 내놓으므로 처리 코드는 입력 종류를 몰라도 된다.

영상은 ffmpeg 로 직접 푼다 (OpenCV 는 BT.709 영상을 BT.601 공식으로 풀어 색이 조금 틀어진다 — 재서 확인함).
  · 색공간 · 범위는 원본 표시대로 (없으면 HD 이상 709, SD 601)
  · 회전 정보가 있으면 돌려서 바로 선 화면으로
  · 가변 프레임이면 기준 fps 의 고정 프레임으로 맞춰서 (원본 타임라인 · 소리와 어긋나지 않게)
저장할 때도 같은 색 규격으로 바꾸고 표시까지 붙인다 (EncodeSpec).
"""
from __future__ import annotations

import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from . import config
from .probe import ColorSpec, MediaInfo, SpecTable

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_SWS = "accurate_rnd+full_chroma_int"     # 반올림 치우침 없이 (기본값은 전체가 2단계쯤 어두워짐 — 재서 확인함)


class UnsupportedMediaError(ValueError):
    pass


class MediaReadError(RuntimeError):
    pass


def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


@dataclass(frozen=True)
class EncodeSpec:
    """결과 영상을 원본에 맞춰 쓰는 규격"""
    rate: Fraction = Fraction(24)
    color: ColorSpec = field(default_factory=ColorSpec)
    sar: str = "1:1"
    audio_copy: bool = False      # 원본 소리를 다시 압축하지 않고 그대로
    kbps: int = 0                 # 고정 비트레이트(CBR). 0 이면 화질 기준(CRF) — 확인용 축소본만

    @property
    def rate_arg(self) -> str:
        return f"{self.rate.numerator}/{self.rate.denominator}"

    def x264_args(self) -> list[str]:
        if not self.kbps:
            return ["-crf", str(config.CRF)]
        k = f"{self.kbps}k"   # 평균 · 최소 · 최대를 같게 + nal-hrd=cbr (모자라면 채움 데이터로 맞춤)
        return ["-b:v", k, "-minrate", k, "-maxrate", k, "-bufsize", k, "-x264-params", "nal-hrd=cbr"]


class MediaSource(ABC):
    is_video: bool = False

    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise MediaReadError(f"파일이 없습니다: {self.path.name}")

    @staticmethod
    def open(path: Path) -> "MediaSource":
        ext = Path(path).suffix.lower()
        if ext in config.VIDEO_EXT:
            return VideoSource(path)
        if ext in config.IMAGE_EXT:
            return ImageSource(path)
        raise UnsupportedMediaError(f"지원하지 않는 파일입니다 ({ext or '확장자 없음'}). 영상: mp4·mov 등, 이미지: png·jpg 등")

    @property
    @abstractmethod
    def size(self) -> tuple[int, int]:
        """(width, height)"""

    @property
    @abstractmethod
    def fps(self) -> Optional[float]: ...

    @property
    @abstractmethod
    def frame_count(self) -> int: ...

    @abstractmethod
    def frames(self) -> Iterator[np.ndarray]: ...

    @abstractmethod
    def frame_at(self, seconds: float) -> np.ndarray: ...

    def frame_at_index(self, index: int) -> np.ndarray:
        return self.frame_at(index / (self.fps or 24))

    @property
    def encode_spec(self) -> EncodeSpec:
        return EncodeSpec()

    def describe(self) -> dict:
        """화면 · _settings.json 에 보여 줄 입력 규격"""
        w, h = self.size
        return {"width": w, "height": h, "summary": f"{w}×{h} 이미지", "notes": [],
                "table": SpecTable.for_image(w, h, self.path.suffix.lower())}


class VideoSource(MediaSource):
    is_video = True

    def __init__(self, path: Path):
        super().__init__(path)
        self.info = MediaInfo.read(self.path)
        if not self.info.ok:
            raise MediaReadError(f"영상을 읽지 못했습니다: {self.path.name}")
        self._fps = float(self.info.rate)
        if self.info.vfr and self.info.duration:
            self._count = int(round(self.info.duration * self._fps))
        else:
            cap = cv2.VideoCapture(str(self.path), cv2.CAP_FFMPEG)
            self._count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
            cap.release()
            if self._count <= 0 and self.info.duration:
                self._count = int(round(self.info.duration * self._fps))

    @property
    def size(self):
        return self.info.width, self.info.height

    @property
    def fps(self):
        return self._fps

    @property
    def frame_count(self):
        return max(1, self._count)

    @property
    def encode_spec(self) -> EncodeSpec:
        i = self.info
        return EncodeSpec(i.rate, i.color, i.sar if not i.square_pixels else "1:1", i.audio_copyable)

    def describe(self) -> dict:
        return self.info.describe()

    # ── 풀기 ──
    def _decode_args(self) -> list[str]:
        c = self.info.color
        args = ["-map", "0:v:0", "-an", "-sn", "-dn"]
        if self.info.vfr:
            args += ["-fps_mode", "cfr", "-r", self.encode_spec.rate_arg]
        args += ["-vf", f"scale=in_color_matrix={c.matrix}:in_range={c.range}:flags={_SWS},format=bgr24",
                 "-f", "rawvideo", "-"]
        return args

    def _pipe(self, before_input: list[str], after_input: list[str]) -> subprocess.Popen:
        cmd = [ffmpeg_exe(), "-v", "error", "-nostdin", *before_input, "-i", str(self.path), *after_input]
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=_NO_WINDOW)

    def _read(self, proc: subprocess.Popen) -> Iterator[np.ndarray]:
        w, h = self.size
        n = w * h * 3
        try:
            while True:
                buf = proc.stdout.read(n)
                if len(buf) < n:
                    return
                yield np.frombuffer(buf, np.uint8).reshape(h, w, 3).copy()
        finally:
            proc.stdout.close()
            if proc.poll() is None:
                proc.kill()
            proc.wait()

    def frames(self):
        yield from self._read(self._pipe([], self._decode_args()))

    def frame_at(self, seconds):
        t = min(max(0.0, seconds), max(0.0, self.info.duration - 1 / max(self._fps, 1)))
        frame = next(self._read(self._pipe(["-ss", f"{t:.4f}"], ["-frames:v", "1", *self._decode_args()])), None)
        if frame is None:   # 끝을 넘겼거나 seek 실패 → 첫 프레임
            frame = next(self._read(self._pipe([], ["-frames:v", "1", *self._decode_args()])), None)
        if frame is None:
            raise MediaReadError("프레임을 읽지 못했습니다")
        return frame

    def frame_at_index(self, index):
        # 프레임 시작 시각보다 조금 앞에서 찾으면 그 프레임이 나온다 (반올림으로 다음 프레임을 집지 않게)
        return self.frame_at(max(0.0, index / self._fps - 0.25 / self._fps))


class ImageSource(MediaSource):
    def __init__(self, path: Path):
        super().__init__(path)
        img = cv2.imdecode(np.fromfile(str(self.path), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise MediaReadError(f"이미지를 읽지 못했습니다: {self.path.name}")
        self._img = img

    @property
    def size(self):
        return self._img.shape[1], self._img.shape[0]

    @property
    def fps(self):
        return None

    @property
    def frame_count(self):
        return 1

    def frames(self):
        yield self._img.copy()

    def frame_at(self, seconds):
        return self._img.copy()


class FFmpegWriter:
    """BGR 프레임을 받아 H.264 mp4 로 저장. 색 · fps · 화면 비율은 spec(원본)대로, audio_from 을 주면 그 파일의 소리를 붙인다."""
    IN_PIX = "bgr24"
    CHANNELS = 3

    def __init__(self, path: Path, size: tuple[int, int], spec: EncodeSpec | float = EncodeSpec(),
                 audio_from: Optional[Path] = None):
        self.path = Path(path)
        self.spec = spec if isinstance(spec, EncodeSpec) else EncodeSpec(rate=Fraction(spec).limit_denominator(1001))
        w, h = size
        cmd = [ffmpeg_exe(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", self.IN_PIX,
               "-s", f"{w}x{h}", "-r", self.spec.rate_arg, "-i", "-"]
        if audio_from:
            audio = ["-c:a", "copy"] if self.spec.audio_copy else ["-c:a", "aac", "-b:a", config.AUDIO_BITRATE]
            cmd += ["-i", str(audio_from), "-map", "0:v", "-map", "1:a:0?", *audio]
        cmd += self._video_args() + ["-movflags", "+faststart", str(self.path)]
        self._size = (w, h)
        # 오류 메시지는 임시 파일로 (파이프로 받으면 경고가 많을 때 ffmpeg 가 멈출 수 있음)
        self._err = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=self._err, creationflags=_NO_WINDOW)

    def _video_args(self) -> list[str]:
        c = self.spec.color
        vf = (f"pad=ceil(iw/2)*2:ceil(ih/2)*2,scale=out_color_matrix={c.matrix}:out_range={c.range}:flags={_SWS},"
              f"format=yuv420p,setsar={self.spec.sar.replace(':', '/')}")
        return ["-vf", vf, "-c:v", "libx264", *self.spec.x264_args(), "-preset", "medium", *c.tags()]

    def write(self, frame: np.ndarray) -> None:
        if frame.ndim != 3 or frame.shape[2] != self.CHANNELS:
            raise ValueError(f"채널 수가 다릅니다: {frame.shape} (필요 {self.CHANNELS})")
        if (frame.shape[1], frame.shape[0]) != self._size:
            raise ValueError(f"프레임 크기가 다릅니다: {frame.shape[1]}x{frame.shape[0]} ≠ {self._size}")
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError):
            raise RuntimeError(f"영상 저장 중 멈춤: {self.path.name} — {self._error()}")

    def _error(self) -> str:
        try:
            self._err.seek(0)
            return self._err.read().decode("utf-8", "replace").strip()[-300:]
        except Exception:
            return ""

    def close(self) -> None:
        try:
            self._proc.stdin.close()
        except OSError:
            pass
        self._proc.wait()
        try:
            if self._proc.returncode != 0:
                raise RuntimeError(f"영상 저장 실패: {self.path.name} — {self._error()}")
        finally:
            self._err.close()

    def abort(self) -> None:
        try:
            self._proc.kill()
            self._proc.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            pass
        try:
            self._err.close()
        except OSError:
            pass


class AlphaMovWriter(FFmpegWriter):
    """BGRA 프레임을 투명 배경 무손실 MOV 로 저장 (PNG 코덱 · RGBA · 원본 크기 그대로).
    AE · 프리미어에서 원본 위에 그대로 겹친다. 색은 그림 RGB 그대로라 변환이 없다."""
    IN_PIX = "bgra"
    CHANNELS = 4

    def _video_args(self) -> list[str]:
        return ["-vf", f"setsar={self.spec.sar.replace(':', '/')}", "-c:v", "png", "-pix_fmt", "rgba"]
