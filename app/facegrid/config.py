"""경로와 상수. 앱 폴더를 어디로 옮겨도 이 파일 기준으로 찾는다.

앱 폴더에 쓸 수 없는 곳(Program Files · 읽기 전용 USB · 네트워크 드라이브 등)에 두면
결과는 문서\\Griding results, 임시 파일은 운영체제의 앱 데이터 폴더로 자동으로 옮긴다
(Windows %LOCALAPPDATA%\\Griding · macOS ~/Library/Caches/Griding — platforms.py).
"""
import os
from pathlib import Path

from .platforms import current as _platform

APP_DIR = Path(__file__).resolve().parent.parent  # .../FaceGrid_Lite/app
ROOT_DIR = APP_DIR.parent                         # .../FaceGrid_Lite
MODEL_PATH = APP_DIR / "model" / "model.onnx"                     # YOLO11s head (빠름)
MODEL_ACCURATE_PATH = APP_DIR / "model" / "head_m" / "model.onnx"  # YOLO11m head (정확)
UI_PATH = APP_DIR / "ui.html"


def writable(folder: Path) -> bool:
    """이 폴더를 만들고 파일을 쓸 수 있나"""
    try:
        folder.mkdir(parents=True, exist_ok=True)
        probe = folder / ".write_test"
        probe.write_text("ok")
        probe.unlink()
        return True
    except OSError:
        return False


def pick_dir(*candidates: Path) -> Path:
    """쓸 수 있는 첫 폴더 (모두 안 되면 첫 폴더 — 시작할 때 server.main 이 알려 준다)"""
    return next((c for c in candidates if writable(c)), candidates[0])


_LOCAL = _platform().local_dir()
WORK_DIR = pick_dir(APP_DIR / "work", _LOCAL / "work")
RESULT_DIR = pick_dir(ROOT_DIR / "results", Path.home() / "Documents" / "Griding results", _LOCAL / "results")

# 그래픽카드 드라이버가 말썽이면 FACEGRID_CPU=1 로 켜면 처음부터 CPU 로만 돌린다
FORCE_CPU = os.environ.get("FACEGRID_CPU", "").strip() not in ("", "0")

HOST = "127.0.0.1"
FIRST_PORT = 7860
DETECT_SIZE = 640
NMS_IOU = 0.5
UPLOAD_MAX_BYTES = 8 * 1024 ** 3   # 8GB
UPLOAD_KEEP_HOURS = 24

VIDEO_EXT = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

CRF = 16
AUDIO_BITRATE = "192k"
