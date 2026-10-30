"""머리 검출 — deepghs/real_head_detection (Ultralytics YOLO11 head, AGPL-3.0)

얼굴 검출기와 달리 머리 전체(머리카락 · 옆얼굴 · 뒤통수)를 잡는다.
모델 두 가지 (settings.model):
  fast      head_detect_v0_s_yv11 (YOLO11s, 38MB)  — 빠름
  accurate  head_detect_v0_m_yv11 (YOLO11m, 80MB)  — 작은 머리 · 겹친 사람에 더 정확, 2~3배 느림

Detector 를 상속하면 다른 검출기로 바꿔 끼울 수 있다 (테스트는 FakeDetector 를 쓴다).
한 세션을 여러 스레드가 동시에 부르지 않도록 DetectorPool 이 용도별로 따로 만든다.
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from . import config
from .platforms import current as current_platform
from .geometry import Rect


@dataclass(frozen=True)
class Detection:
    box: Rect
    score: float


class Detector(ABC):
    device: str = "CPU"

    @abstractmethod
    def detect_scored(self, frame: np.ndarray, threshold: float) -> list[Detection]:
        """BGR 프레임에서 머리 박스 + 점수"""

    def detect(self, frame: np.ndarray, threshold: float) -> list[Rect]:
        return [d.box for d in self.detect_scored(frame, threshold)]


class ModelMissingError(RuntimeError):
    pass


# DirectML 세션 두 개가 서로 다른 스레드에서 동시에 돌면 멈춘다 (직접 확인함).
# 그래픽카드 실행은 프로세스 전체에서 한 번에 하나씩 — 한 번이 수십 ms 라 화면이 기다리는 시간은 짧다.
_GPU_LOCK = threading.Lock()


class HeadDetector(Detector):
    GPU = current_platform().gpu_provider   # Windows DirectML · macOS Core ML (없으면 CPU 만)

    def __init__(self, model_path: Path = config.MODEL_PATH, size: int = config.DETECT_SIZE):
        import onnxruntime as ort

        if not Path(model_path).is_file():
            raise ModelMissingError(f"머리 검출 모델이 없습니다: {model_path}")
        self._size = size
        self._path = str(model_path)
        self._opts = ort.SessionOptions()
        self._opts.log_severity_level = 3
        self._opts.enable_mem_pattern = False                              # DirectML 권장 설정
        self._opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if config.FORCE_CPU or not self.GPU or self.GPU not in ort.get_available_providers():
            self._use_cpu()
            return
        try:
            self._open([self.GPU, "CPUExecutionProvider"])
            self._warm_up()
            self.device = "GPU" if self._sess.get_providers()[0] == self.GPU else "CPU"
        except Exception:  # GPU 드라이버 문제 · 지원 안 되는 연산 → CPU 로
            self._use_cpu()

    def _open(self, providers) -> None:
        import onnxruntime as ort
        self._sess = ort.InferenceSession(self._path, self._opts, providers=list(providers))
        self._input = self._sess.get_inputs()[0].name

    def _use_cpu(self) -> None:
        self._open(["CPUExecutionProvider"])
        self.device = "CPU"

    def _run(self, blob):
        if self.device != "GPU":
            return self._sess.run(None, {self._input: blob})
        try:
            with _GPU_LOCK:
                return self._sess.run(None, {self._input: blob})
        except Exception:  # 작업 도중 그래픽카드 드라이버가 멈추거나 리셋되면 CPU 로 바꿔 이어서 한다
            self._use_cpu()
            return self._sess.run(None, {self._input: blob})

    def _warm_up(self) -> None:
        # 만들기는 됐는데 실제 실행에서 실패하는 그래픽카드가 있어서, 여기서 한 번 돌려 본다
        with _GPU_LOCK:
            self._sess.run(None, {self._input: np.zeros((1, 3, self._size, self._size), np.float32)})

    def _letterbox(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = frame.shape[:2]
        scale = self._size / max(h, w)
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        canvas = np.full((self._size, self._size, 3), 114, np.uint8)
        canvas[:nh, :nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return blob, scale

    def detect_scored(self, frame: np.ndarray, threshold: float) -> list[Detection]:
        if frame is None or frame.size == 0:
            return []
        h, w = frame.shape[:2]
        blob, scale = self._letterbox(frame)
        out = self._run(blob)[0][0]  # [5, N] = cx cy w h score
        keep = out[4] >= threshold
        if not keep.any():
            return []
        cx, cy, bw, bh, score = out[:, keep]
        xywh = np.stack([cx - bw / 2, cy - bh / 2, bw, bh], 1) / scale
        idx = np.array(cv2.dnn.NMSBoxes(xywh.tolist(), score.tolist(), threshold * 0.999, config.NMS_IOU)).reshape(-1).astype(int)
        out = []
        for i in idx:
            x, y, bw_, bh_ = xywh[i]
            box = Rect(x, y, x + bw_, y + bh_).clipped(w, h)
            if box.w > 2 and box.h > 2:
                out.append(Detection(box, float(score[i])))
        return out


MODELS = {"fast": config.MODEL_PATH, "accurate": config.MODEL_ACCURATE_PATH}


class DetectorPool:
    """모델 · 용도별 검출기. 화면(미리보기 · 예시)과 작업이 같은 세션을 동시에 쓰지 않게 따로 만든다."""

    def __init__(self, factory: Callable[[str], Detector] | None = None):
        self._factory = factory or (lambda model: HeadDetector(MODELS[model]))
        self._items: dict[tuple[str, str], Detector] = {}
        self._lock = threading.Lock()

    def get(self, model: str, purpose: str = "ui") -> Detector:
        model = model if model in MODELS else "fast"
        with self._lock:
            key = (model, purpose)
            if key not in self._items:
                self._items[key] = self._factory(model)
            return self._items[key]

    def available(self) -> dict[str, bool]:
        return {k: Path(p).is_file() for k, p in MODELS.items()}
