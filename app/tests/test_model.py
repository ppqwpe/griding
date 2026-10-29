"""실제 머리 검출 모델 테스트

FACEGRID_TEST_PLATE 에 사람이 나오는 영상 경로를 주면 실제 검출까지 확인한다 (없으면 그 부분은 건너뜀).
"""
import os
import unittest
from pathlib import Path

import numpy as np

import helpers  # noqa: F401
from facegrid.detector import HeadDetector, ModelMissingError
from facegrid.media import MediaSource


class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.det = HeadDetector()

    def test_loads_on_some_device(self):
        self.assertIn(self.det.device, ("GPU", "CPU"))

    def test_two_sessions_in_parallel_do_not_hang(self):
        """화면용 · 작업용 세션이 동시에 돌아도 멈추지 않는다 (DirectML 은 동시 실행하면 멈췄음)"""
        import threading
        from facegrid import config
        dets = [HeadDetector(), HeadDetector(config.MODEL_ACCURATE_PATH) if config.MODEL_ACCURATE_PATH.is_file() else HeadDetector()]
        frame = np.random.default_rng(0).integers(0, 255, (360, 640, 3), dtype=np.uint8)
        done = [0, 0]

        def loop(i):
            for _ in range(20):
                dets[i].detect_scored(frame, 0.1)
                done[i] += 1
        threads = [threading.Thread(target=loop, args=(i,), daemon=True) for i in (0, 1)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        self.assertEqual(done, [20, 20])

    def test_missing_model_is_clear(self):
        with self.assertRaises(ModelMissingError):
            HeadDetector(Path("없는모델.onnx"))

    def test_odd_inputs(self):
        self.assertEqual(self.det.detect(np.zeros((0, 0, 3), np.uint8), 0.25), [])
        self.assertEqual(self.det.detect(np.zeros((720, 1280, 3), np.uint8), 0.25), [])
        self.assertIsInstance(self.det.detect(np.zeros((1920, 1080, 3), np.uint8), 0.25), list)   # 세로 영상

    @unittest.skipUnless(os.environ.get("FACEGRID_TEST_PLATE"), "FACEGRID_TEST_PLATE 없음")
    def test_real_plate(self):
        src = MediaSource.open(Path(os.environ["FACEGRID_TEST_PLATE"]))
        hits = 0
        for t in (1, 5, 10, 15):
            boxes = self.det.detect(src.frame_at(t), 0.25)
            hits += bool(boxes)
            for b in boxes:
                self.assertGreater(b.area, 100)
        self.assertGreaterEqual(hits, 3)


if __name__ == "__main__":
    unittest.main()
