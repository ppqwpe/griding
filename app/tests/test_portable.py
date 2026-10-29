"""어느 PC 에서든 — 저장 폴더 대신 찾기, 그래픽카드 실패 시 CPU"""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from facegrid import config
from facegrid.detector import HeadDetector


class PickDirTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="fg 폴더_"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_first_writable(self):
        a, b = self.tmp / "가", self.tmp / "나"
        self.assertEqual(config.pick_dir(a, b), a)

    def test_falls_back_when_not_writable(self):
        blocked = self.tmp / "file_not_dir"
        blocked.write_text("x")                     # 파일이라 그 아래에 폴더를 못 만든다
        good = self.tmp / "결과"
        self.assertFalse(config.writable(blocked / "sub"))
        self.assertEqual(config.pick_dir(blocked / "sub", good), good)

    def test_all_blocked_returns_first(self):
        blocked = self.tmp / "f"
        blocked.write_text("x")
        self.assertEqual(config.pick_dir(blocked / "a", blocked / "b"), blocked / "a")


class _Sess:
    """가짜 onnxruntime 세션 — fail=True 면 run 에서 실패 (드라이버 리셋 흉내)"""
    def __init__(self, provider, fail=False):
        self._p, self.fail = provider, fail

    def get_providers(self):
        return [self._p]

    def get_inputs(self):
        return [mock.Mock(name="images")]

    def run(self, *_):
        if self.fail:
            raise RuntimeError("DXGI_ERROR_DEVICE_REMOVED")
        return [np.zeros((1, 5, 0), np.float32)]


class DeviceTest(unittest.TestCase):
    def _detector(self, gpu_fails_on_run=False, force_cpu=False):
        made = []

        def session(path, opts, providers):
            gpu = providers[0] == "DmlExecutionProvider"
            s = _Sess(providers[0], fail=gpu and gpu_fails_on_run and len(made) > 0)
            made.append(providers[0])
            return s

        ort = mock.Mock()
        ort.get_available_providers.return_value = ["DmlExecutionProvider", "CPUExecutionProvider"]
        ort.InferenceSession.side_effect = session
        with mock.patch.dict("sys.modules", {"onnxruntime": ort}), mock.patch.object(config, "FORCE_CPU", force_cpu), \
                mock.patch.object(HeadDetector, "_warm_up", lambda self: None):
            d = HeadDetector(config.MODEL_PATH)
            d._sess.fail = gpu_fails_on_run and d.device == "GPU"
            frame = np.zeros((120, 160, 3), np.uint8)
            return d, d.detect_scored(frame, 0.25), made

    def test_gpu_by_default(self):
        d, _, made = self._detector()
        self.assertEqual((d.device, made), ("GPU", ["DmlExecutionProvider"]))

    def test_force_cpu(self):
        d, _, made = self._detector(force_cpu=True)
        self.assertEqual((d.device, made), ("CPU", ["CPUExecutionProvider"]))

    def test_gpu_failure_mid_job_switches_to_cpu(self):
        d, dets, made = self._detector(gpu_fails_on_run=True)
        self.assertEqual(d.device, "CPU", "실패하면 CPU 로 바꿔 이어서")
        self.assertEqual(dets, [])
        self.assertEqual(made[-1], "CPUExecutionProvider")


if __name__ == "__main__":
    unittest.main()
