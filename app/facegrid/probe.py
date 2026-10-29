"""영상 규격 읽기 — 들어온 파일이 어떤 규격인지, 나간 파일이 그 규격을 지켰는지

ffprobe 없이 앱에 들어 있는 ffmpeg 로 읽는다 (`ffmpeg -i` 의 스트림 줄).

  MediaInfo.read(path)   해상도 · 화면 비율(SAR) · 회전 · fps · 가변 프레임 · 코덱 · 비트 깊이 · 색공간 · 소리
  info.color             결과를 쓸 때 맞출 색 (행렬 · 범위 · 원색 · 감마)
  info.notes()           사람이 알아야 할 점 (10비트 → 8비트, HDR, 가변 프레임 …)
  OutputCheck            만든 영상의 프레임 수 · 길이 · 크기 · fps 를 원본과 대조
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Optional


def _ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _run(args: list[str]) -> str:
    return subprocess.run([_ffmpeg(), "-hide_banner", *args], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stderr


@dataclass(frozen=True)
class ColorSpec:
    """YUV ↔ RGB 를 바꿀 때 쓸 색 규격 (ffmpeg 이름)"""
    matrix: str = "bt709"        # scale 필터의 in/out_color_matrix: bt709 · bt601 · bt2020
    range: str = "tv"            # tv(16-235) · pc(0-255)
    space: str = "bt709"         # 태그 -colorspace
    primaries: str = "bt709"     # 태그 -color_primaries
    trc: str = "bt709"           # 태그 -color_trc

    @property
    def hdr(self) -> bool:
        return self.trc in ("smpte2084", "arib-std-b67")

    def tags(self) -> list[str]:
        return ["-colorspace", self.space, "-color_primaries", self.primaries, "-color_trc", self.trc,
                "-color_range", self.range]


_SPACE_TO_MATRIX = {"bt709": "bt709", "bt470bg": "bt601", "smpte170m": "bt601", "bt2020nc": "bt2020",
                    "bt2020c": "bt2020", "fcc": "fcc", "smpte240m": "smpte240m"}
_MATRIX_DEFAULTS = {"bt709": ("bt709", "bt709", "bt709"), "bt601": ("smpte170m", "smpte170m", "smpte170m"),
                    "bt2020": ("bt2020nc", "bt2020", "bt709")}


@dataclass(frozen=True)
class MediaInfo:
    path: str
    width: int = 0                 # 화면에 보이는 크기 (회전 반영)
    height: int = 0
    codec: str = ""
    pix_fmt: str = ""
    bit_depth: int = 8
    fps: float = 0.0               # 평균 fps
    tbr: float = 0.0               # 기준 fps (가변 프레임이면 fps 와 다름)
    duration: float = 0.0
    start: float = 0.0
    rotation: int = 0
    sar: str = "1:1"
    color: ColorSpec = field(default_factory=ColorSpec)
    color_tagged: bool = False     # 원본에 색공간 표시가 있었나 (없으면 크기로 짐작)
    audio_codec: str = ""
    audio_rate: int = 0
    audio_channels: str = ""
    video_kbps: int = 0            # 영상 비트레이트 (스트림 값, 없으면 전체 − 소리로 짐작)

    # ── 읽기 ──
    @classmethod
    def read(cls, path: Path) -> "MediaInfo":
        text = _run(["-i", str(path)])
        video = re.search(r"Stream #\d+:\d+[^:]*: Video: (.*)", text)
        if not video:
            return cls(str(path))
        line = video.group(1)
        codec = line.split()[0].strip(",")
        fmt = re.search(r", ((?:yuv|yuvj|rgb|bgr|gbr|gray|nv|p0|rgba|bgra|argb|abgr|ya)\w*)(?:\(([^)]*)\))?", line)
        pix_fmt = fmt.group(1) if fmt else ""
        extra = [t.strip() for t in (fmt.group(2) or "").split(",")] if fmt else []
        size = re.search(r", (\d{2,5})x(\d{2,5})", line)
        w, h = (int(size.group(1)), int(size.group(2))) if size else (0, 0)
        sar = re.search(r"\[SAR (\d+:\d+)", line)
        fps = re.search(r"([\d.]+) fps", line)
        tbr = re.search(r"([\d.]+k?) tbr", line)
        rot = re.search(r"rotation of (-?[\d.]+) degrees", text)
        rotation = int(round(float(rot.group(1)))) % 360 if rot else 0
        if rotation in (90, 270):
            w, h = h, w
        dur = re.search(r"Duration: (\d+):(\d+):([\d.]+)", text)
        start = re.search(r"start: (-?[\d.]+)", text)
        depth = re.search(r"p(\d{2})(?:le|be)", pix_fmt)
        audio = re.search(r"Stream #\d+:\d+[^:]*: Audio: (\w+)[^,]*, (\d+) Hz, ([^,]+)", text)
        color, tagged = cls._color(extra, h)
        kbps = re.search(r"(\d+) kb/s", line)
        if kbps:
            video_kbps = int(kbps.group(1))
        else:
            total = re.search(r"bitrate: (\d+) kb/s", text)
            sound = re.search(r"Stream #\d+:\d+[^:]*: Audio: .*?(\d+) kb/s", text)
            video_kbps = max(0, int(total.group(1)) - (int(sound.group(1)) if sound else 0)) if total else 0
        return cls(
            path=str(path), width=w, height=h, codec=codec, pix_fmt=pix_fmt,
            bit_depth=int(depth.group(1)) if depth else 8,
            fps=float(fps.group(1)) if fps else 0.0, tbr=cls._num(tbr.group(1)) if tbr else 0.0,
            duration=(int(dur.group(1)) * 3600 + int(dur.group(2)) * 60 + float(dur.group(3))) if dur else 0.0,
            start=float(start.group(1)) if start else 0.0, rotation=rotation,
            sar=sar.group(1) if sar else "1:1", color=color, color_tagged=tagged,
            audio_codec=audio.group(1) if audio else "", audio_rate=int(audio.group(2)) if audio else 0,
            audio_channels=audio.group(3).strip() if audio else "", video_kbps=video_kbps)

    @staticmethod
    def _num(s: str) -> float:
        return float(s[:-1]) * 1000 if s.endswith("k") else float(s)

    @staticmethod
    def _color(extra: list[str], height: int) -> tuple[ColorSpec, bool]:
        """yuv420p(tv, bt709, progressive) · yuv420p(tv, smpte170m/bt470bg/smpte170m) 에서 범위 · 색공간"""
        rng = "pc" if "pc" in extra else "tv"
        tagged = next((t for t in extra if t not in ("tv", "pc", "progressive", "top first", "bottom first")
                       and not t.startswith("top") and not t.startswith("bottom")), "")
        parts = tagged.split("/") if tagged else []
        space = parts[0] if parts else ""
        matrix = _SPACE_TO_MATRIX.get(space)
        if matrix is None:   # 표시가 없거나 모르는 값 → 크기로 짐작 (HD 이상 709, SD 601)
            matrix = "bt709" if height >= 720 else "bt601"
            return ColorSpec(matrix, rng, *_MATRIX_DEFAULTS[matrix]), False
        d_space, d_prim, d_trc = _MATRIX_DEFAULTS.get(matrix, ("bt709", "bt709", "bt709"))
        prim = parts[1] if len(parts) == 3 else d_prim
        trc = parts[2] if len(parts) == 3 else d_trc
        known = lambda v, d: v if v and v != "unknown" and v != "reserved" else d
        return ColorSpec(matrix, rng, known(space, d_space), known(prim, d_prim), known(trc, d_trc)), True

    # ── 판단 ──
    @property
    def ok(self) -> bool:
        return self.width > 0 and self.height > 0

    @property
    def vfr(self) -> bool:
        """평균 fps 와 기준 fps 가 다르면 가변 프레임 (휴대폰 · 화면 녹화에서 흔함)"""
        return self.fps > 0 and self.tbr > 0 and abs(self.fps - self.tbr) > 0.05

    @property
    def rate(self) -> Fraction:
        """결과에 쓸 정확한 fps — 23.98 · 29.97 · 59.94 는 /1001 로"""
        base = self.tbr if self.vfr else self.fps
        if base <= 0:
            return Fraction(24)
        for n in (24, 30, 60, 120, 48):
            if abs(base - n * 1000 / 1001) < 0.02:
                return Fraction(n * 1000, 1001)
        return Fraction(base).limit_denominator(1001)

    @property
    def square_pixels(self) -> bool:
        a, b = self.sar.split(":")
        return a == b or a == "0"

    @property
    def audio_copyable(self) -> bool:
        """mp4 에 그대로 넣을 수 있는 소리 (다시 압축하지 않음)"""
        return self.audio_codec in ("aac", "mp3", "ac3", "eac3", "alac")

    def notes(self) -> list[str]:
        n = []
        if self.bit_depth > 8:
            n.append(f"{self.bit_depth}비트 원본 → 결과 mp4 는 8비트. 화질을 지키려면 '카드별 투명 레이어'를 원본 위에 합성")
        if self.color.hdr:
            n.append("HDR 원본 — 결과 mp4 는 8비트라 띠가 생길 수 있음. 투명 레이어로 합성 권장")
        if self.vfr:
            n.append(f"가변 프레임 (평균 {self.fps:g} · 기준 {self.tbr:g} fps) → 결과는 고정 {float(self.rate):.3f} fps 로 맞춤")
        if not self.square_pixels:
            n.append(f"픽셀이 정사각형이 아님 (SAR {self.sar}) — 결과에도 같은 SAR 표시")
        if self.rotation:
            n.append(f"회전 정보 {self.rotation}° — 돌려서 바로 선 화면({self.width}×{self.height})으로 처리")
        if self.width % 2 or self.height % 2:
            n.append("가로 · 세로가 홀수 — mp4 는 1px 늘려 짝수로 저장 (투명 레이어는 원본 크기 그대로)")
        if not self.color_tagged:
            n.append(f"색공간 표시 없음 → {self.color.matrix} 로 가정")
        if self.audio_codec and not self.audio_copyable:
            n.append(f"소리 {self.audio_codec} 는 mp4 에 그대로 못 넣어 AAC 로 변환")
        for p in SeedanceLimits.problems(self.width, self.height, self.duration):
            n.append(f"시댄스 넣기: {p}")
        return n

    def describe(self) -> dict:
        d = asdict(self)
        d.pop("path")
        d["color"] = asdict(self.color)
        d.update(vfr=self.vfr, audio_copyable=self.audio_copyable, rate=f"{self.rate.numerator}/{self.rate.denominator}", notes=self.notes(),
                 summary=self.summary(), table=SpecTable.for_video(self))
        return d

    def summary(self) -> str:
        if not self.ok:
            return ""
        audio = f" · 소리 {self.audio_codec} {self.audio_rate // 1000}kHz" if self.audio_codec else " · 소리 없음"
        return (f"{self.width}×{self.height} · {self.fps:g}fps{' (가변)' if self.vfr else ''} · {self.duration:.2f}초 · "
                f"{self.codec} {self.bit_depth}bit · {self.color.matrix} {self.color.range}{audio}")


class Bitrate:
    """mp4 저장 비트레이트 — 고정(CBR): 가변(VBR · CRF)이 아니라 처음부터 끝까지 같은 값
      fixed  : 긴 변 2560px 까지(2K) 30Mbps · 그보다 크면(4K) 50Mbps
      source : 원본 영상 비트레이트로 고정 (원본 값을 모르면 fixed 값)
    """
    FIXED_2K = 30_000          # kbps
    FIXED_4K = 50_000
    SOURCE_RANGE = (1_000, 100_000)

    @classmethod
    def fixed(cls, width: int, height: int) -> int:
        return cls.FIXED_4K if max(width, height) > 2560 else cls.FIXED_2K

    @classmethod
    def choose(cls, mode: str, width: int, height: int, source_kbps: int = 0) -> int:
        if mode == "source" and source_kbps > 0:
            lo, hi = cls.SOURCE_RANGE
            return min(max(source_kbps, lo), hi)
        return cls.fixed(width, height)

    @staticmethod
    def label(kbps: int) -> str:
        return f"{kbps / 1000:.1f}".rstrip("0").rstrip(".") + "Mbps"


class SpecTable:
    """화면의 '입력 → 출력 규격표' — 원본 · 조합 mp4 · 투명 레이어가 어떤 규격으로 나오는지 (저장 규칙과 같은 계산)

    rows: [{"key", "label", "input", "combo", "layers", "same": {"combo": bool, "layers": bool}}]
    same 이 False 인 칸은 일부러 바뀌는 곳 (예: 10비트 → 8비트, 홀수 → 짝수) — why 에 이유
    """
    COLUMNS = {"input": "원본 (넣은 파일)", "combo": "조합 영상 _combo.mp4", "layers": "투명 레이어 _layerN.mov"}

    @staticmethod
    def _row(key, label, inp, combo, layers, why_combo="", why_layers=""):
        return {"key": key, "label": label, "input": inp, "combo": combo, "layers": layers,
                "same": {"combo": not why_combo, "layers": not why_layers},
                "why": {"combo": why_combo, "layers": why_layers}}

    @classmethod
    def for_video(cls, i: "MediaInfo") -> dict:
        w, h = i.width, i.height
        ew, eh = w + w % 2, h + h % 2
        fixed = Bitrate.fixed(ew, eh)
        rate = float(i.rate)
        fps = f"{rate:.3f}".rstrip("0").rstrip(".")
        fps_in = f"{i.fps:g}" + (f" (가변 · 기준 {i.tbr:g})" if i.vfr else "")
        c = i.color
        color_in = f"{c.matrix} · {c.range}" + ("" if i.color_tagged else " (표시 없음 → 짐작)")
        audio_in = f"{i.audio_codec} {i.audio_rate // 1000}kHz {i.audio_channels}" if i.audio_codec else "없음"
        if not i.audio_codec:
            audio_out, why_audio = "없음", ""
        elif i.audio_copyable:
            audio_out, why_audio = f"{i.audio_codec} 그대로 복사", ""
        else:
            audio_out, why_audio = "AAC 192k", f"{i.audio_codec} 는 mp4 에 못 넣어 변환"
        rows = [
            cls._row("size", "해상도", f"{w}×{h}", f"{ew}×{eh}", f"{w}×{h}",
                     "H.264 는 짝수만 → 1px 늘림" if (ew, eh) != (w, h) else ""),
            cls._row("fps", "fps", fps_in, fps, fps, "가변 → 고정으로 맞춤" if i.vfr else "", "가변 → 고정으로 맞춤" if i.vfr else ""),
            cls._row("length", "길이", f"{i.duration:.2f}초", f"{i.duration:.2f}초 (같은 프레임 수)", f"{i.duration:.2f}초 (같은 프레임 수)"),
            cls._row("codec", "코덱", i.codec, "H.264", "PNG 무손실", "시댄스 · 재생용 압축", ""),
            cls._row("bitrate", "비트레이트", Bitrate.label(i.video_kbps) if i.video_kbps else "모름",
                     f"{Bitrate.label(fixed)} 고정 (CBR)", "무손실 (용량은 그림에 따라)",
                     "" if i.video_kbps and abs(fixed - i.video_kbps) < 500 else "고정 비트레이트로 저장", ""),
            cls._row("bits", "비트 · 채널", f"{i.bit_depth}bit · {i.pix_fmt}", "8bit · yuv420p", "8bit · RGBA (투명)",
                     "8비트로 줄어듦 — 원본 화질은 투명 레이어로" if i.bit_depth > 8 else "", ""),
            cls._row("color", "색공간", color_in, f"{c.matrix} · {c.range} (표시 붙임)", "RGB 그대로 (변환 없음)"),
            cls._row("rotation", "회전", f"{i.rotation}°" if i.rotation else "없음", "바로 선 화면", "바로 선 화면",
                     "회전 정보를 적용해 저장" if i.rotation else "", "회전 정보를 적용해 저장" if i.rotation else ""),
            cls._row("sar", "픽셀 비율", i.sar, i.sar, i.sar),
            cls._row("audio", "소리", audio_in, audio_out, "없음 (그림만)", why_audio, "레이어는 그림만" if i.audio_codec else ""),
        ]
        # 화면에서 '원본 비트레이트 그대로' 를 고르면 bitrate 줄의 조합 칸을 이 값으로 바꿔 보여 준다
        options = {mode: Bitrate.choose(mode, ew, eh, i.video_kbps) for mode in ("fixed", "source")}
        return {"columns": cls.COLUMNS, "rows": rows, "bitrate": {"input": i.video_kbps, "options": options}}

    @classmethod
    def for_image(cls, w: int, h: int, ext: str) -> dict:
        rows = [cls._row("size", "해상도", f"{w}×{h}", f"{w}×{h}", f"{w}×{h}"),
                cls._row("codec", "형식", ext.lstrip(".").upper(), "PNG", "PNG (투명)"),
                cls._row("bits", "채널", "RGB", "RGB", "RGBA (투명)")]
        return {"columns": {"input": "원본 (넣은 파일)", "combo": "조합 이미지 _combo.png", "layers": "투명 레이어 _layerN.png"},
                "rows": rows}


class SeedanceLimits:
    """시댄스에 넣을 수 있는 영상인가 — 시댄스 앱 설명서(26.10.2803)의 파일 규칙"""
    MAX_MB = 200
    MIN_SEC = 2.0
    MAX_SEC = {"2.0": 15.2, "2.5": 30.2}
    SIDE = (300, 6000)
    ASPECT = (0.4, 2.5)

    @classmethod
    def problems(cls, width: int, height: int, seconds: float, megabytes: Optional[float] = None) -> list[str]:
        p = []
        if megabytes is not None and megabytes > cls.MAX_MB:
            p.append(f"용량 {megabytes:.0f}MB > {cls.MAX_MB}MB")
        if seconds and seconds < cls.MIN_SEC:
            p.append(f"길이 {seconds:.1f}초 < {cls.MIN_SEC:g}초")
        if seconds > cls.MAX_SEC["2.5"]:
            p.append(f"길이 {seconds:.1f}초 — 2.0 은 15초 · 2.5 는 30초까지 (잘라서 넣기)")
        elif seconds > cls.MAX_SEC["2.0"]:
            p.append(f"길이 {seconds:.1f}초 — 2.5 만 됨 (2.0 은 15초까지)")
        if width and height:
            if min(width, height) < cls.SIDE[0] or max(width, height) > cls.SIDE[1]:
                p.append(f"크기 {width}×{height} — 한 변 {cls.SIDE[0]}~{cls.SIDE[1]}px")
            if not cls.ASPECT[0] <= width / height <= cls.ASPECT[1]:
                p.append(f"가로÷세로 {width / height:.2f} — {cls.ASPECT[0]}~{cls.ASPECT[1]} 만")
        return p


class OutputCheck:
    """만든 영상이 원본과 같은 규격인지 — 프레임 수 · 길이 · 크기 · fps"""

    @staticmethod
    def frames(path: Path) -> int:
        """다시 풀지 않고 영상 패킷만 세어 프레임 수를 얻는다 (빠름)"""
        text = _run(["-i", str(path), "-map", "0:v:0", "-c", "copy", "-f", "null", "-"])
        found = re.findall(r"frame=\s*(\d+)", text)
        return int(found[-1]) if found else 0

    @classmethod
    def compare(cls, source: MediaInfo, source_frames: int, path: Path) -> dict:
        out = MediaInfo.read(path)
        frames = cls.frames(path)
        same_size = (out.width, out.height) == (source.width, source.height) or \
                    (out.width, out.height) == (source.width + source.width % 2, source.height + source.height % 2)
        rate = float(source.rate)
        result = {"frames": frames, "size": [out.width, out.height], "fps": out.fps, "duration": round(out.duration, 3),
                  "codec": out.codec, "pix_fmt": out.pix_fmt, "color": out.color.matrix if out.color_tagged else "",
                  "audio": out.audio_codec, "kbps": out.video_kbps}
        problems = []
        if frames != source_frames:
            problems.append(f"프레임 {frames} ≠ 원본 {source_frames}")
        if not same_size:
            problems.append(f"크기 {out.width}×{out.height} ≠ 원본 {source.width}×{source.height}")
        if rate and abs(out.fps - rate) > 0.02:
            problems.append(f"fps {out.fps:g} ≠ {rate:.3f}")
        result["ok"] = not problems
        result["problems"] = problems
        result["megabytes"] = round(Path(path).stat().st_size / 1e6, 1)
        result["seedance"] = SeedanceLimits.problems(out.width, out.height, out.duration, result["megabytes"])
        return result
