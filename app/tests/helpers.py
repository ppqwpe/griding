"""테스트 도구 — 사람 사진 없이 돌릴 수 있게 가짜 검출기와 합성 영상을 만든다."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from facegrid.detector import Detection, Detector  # noqa: E402
from facegrid.geometry import Rect  # noqa: E402
from facegrid.media import ffmpeg_exe  # noqa: E402

W, H, FPS, FRAMES = 320, 180, 24, 48
MISSING = set(range(20, 24))    # 이 프레임에선 일부러 못 찾음 (hold 로 메워져야 함)
GONE_AFTER = 40                 # 이후엔 머리가 사라짐 (hold 가 끝나면 격자도 사라져야 함)


def frame_index(frame) -> int:
    return int(round(frame[2, 2, 0] / 4))


def head_at(i: int) -> Rect:
    x = 60 + i * 3
    return Rect(x, 40, x + 60, 120)


class FakeDetector(Detector):
    """프레임 번호를 프레임 왼쪽 위 픽셀 밝기로 읽어서, 정해 둔 위치를 돌려준다."""
    device = "CPU"

    def detect_scored(self, frame, threshold):
        i = frame_index(frame)
        if i in MISSING or i >= GONE_AFTER:
            return []
        return [Detection(head_at(i), 0.9)]


# ── 여러 명 ─────────────────────────────────────────
# A: 왼쪽 → 오른쪽, B: 오른쪽 → 왼쪽 (24 프레임쯤 엇갈림)
# 10 프레임: 한 프레임짜리 오검출 (사람으로 세면 안 됨)
# 30~33 프레임: A 가 흐려져 점수가 낮음 (낮은 점수로 이어 붙여야 함 · 놓치면 안 됨)
FALSE_HIT = 10
A_BLUR = set(range(30, 34))


def person_a(i: int) -> Rect:
    x = 10 + i * 5
    return Rect(x, 40, x + 50, 110)


def person_b(i: int) -> Rect:
    x = 260 - i * 5
    return Rect(x, 50, x + 50, 120)


class CrowdDetector(Detector):
    device = "CPU"

    def detect_scored(self, frame, threshold):
        i = frame_index(frame)
        out = [Detection(person_a(i), 0.15 if i in A_BLUR else 0.9), Detection(person_b(i), 0.85)]
        if i == FALSE_HIT:
            out.append(Detection(Rect(140, 140, 170, 175), 0.6))
        return [d for d in out if d.score >= threshold]


def make_frame(i: int) -> np.ndarray:
    img = np.full((H, W, 3), 60, np.uint8)
    img[:6, :6] = i * 4                        # 프레임 번호 표식
    b = head_at(i)
    cv2.ellipse(img, (int(b.center[0]), int(b.center[1])), (30, 40), 0, 0, 360, (150, 170, 190), -1)
    return img


def make_video(path: Path, with_audio: bool = True) -> Path:
    cmd = [ffmpeg_exe(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-"]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={FRAMES / FPS}", "-c:a", "aac", "-shortest"]
    # 표식 픽셀이 압축으로 흐려지지 않게 무손실에 가깝게
    cmd += ["-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(FRAMES):
        p.stdin.write(make_frame(i).tobytes())
    p.stdin.close()
    assert p.wait() == 0
    return path


def make_image(path: Path) -> Path:
    ok, buf = cv2.imencode(path.suffix, make_frame(5))
    buf.tofile(str(path))
    return path


def probe(path: Path) -> dict:
    """ffprobe 없이 ffmpeg 로 스트림 정보를 읽는다"""
    out = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True, errors="replace").stderr
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.read()[0]:
        n += 1
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"frames": n, "size": (w, h), "audio": "Audio:" in out, "h264": "h264" in out}
