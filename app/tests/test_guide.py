"""가린 표정 가이드(_masked) 테스트"""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

import helpers
from facegrid.geometry import Rect
from facegrid.guide import MaskedFaceGuide
from facegrid.media import MediaSource
from facegrid.pipeline import FaceGridPipeline
from facegrid.settings import GridSettings
from test_face_layers import FakeFaceTracker, oval_face

SIZE = (360, 640)
HEAD = Rect(250, 60, 390, 240)
FACE = oval_face(320, 160, rx=45, ry=60)


class GuideTest(unittest.TestCase):
    def frame(self):
        return np.full((*SIZE, 3), 200, np.uint8)

    def test_face_mode_covers_face_and_draws_guide(self):
        out = MaskedFaceGuide(GridSettings()).render(self.frame(), [FACE], [HEAD])
        self.assertEqual(tuple(out[160, 300]), (0, 0, 0), "얼굴 안쪽은 가림 색")
        red = (out[..., 2] > 180) & (out[..., 1] < 90) & (out[..., 0] < 90)
        self.assertGreater(red.sum(), 50, "점 · 선이 기본 빨강으로 그려져야 함")
        self.assertEqual(tuple(out[10, 10]), (200, 200, 200), "얼굴 밖은 그대로")

    def test_grow_widens_mask(self):
        small = MaskedFaceGuide(GridSettings.from_dict({"mask_grow": 0})).mask(SIZE, [FACE], [HEAD])
        big = MaskedFaceGuide(GridSettings.from_dict({"mask_grow": 0.3})).mask(SIZE, [FACE], [HEAD])
        self.assertGreater(big.sum(), small.sum() * 1.3)

    def test_head_mode_covers_more(self):
        face = MaskedFaceGuide(GridSettings.from_dict({"mask_source": "face"})).mask(SIZE, [FACE], [HEAD])
        head = MaskedFaceGuide(GridSettings.from_dict({"mask_source": "head"})).mask(SIZE, [FACE], [HEAD])
        self.assertGreater(head.sum(), face.sum())
        self.assertEqual(head[HEAD.as_int()[1] + 2, 320], 255, "머리 윗부분(머리카락)까지")

    def test_no_face_still_covers_head(self):
        """옆모습 · 뒤통수: 표정을 못 찾아도 머리는 가려져야 시댄스에서 얼굴이 안 보인다"""
        m = MaskedFaceGuide(GridSettings()).mask(SIZE, [], [HEAD])
        cx, cy = map(int, HEAD.center)
        self.assertEqual(m[cy, cx], 255)
        self.assertEqual(MaskedFaceGuide(GridSettings()).mask(SIZE, [], []).max(), 0)

    def test_colors(self):
        s = GridSettings.from_dict({"cover_color": "#FFFFFF", "guide_color": "#FF0000"})
        out = MaskedFaceGuide(s).render(self.frame(), [FACE], [HEAD])
        self.assertEqual(tuple(out[160, 300]), (255, 255, 255))
        red = (out[..., 2] > 200) & (out[..., 1] < 80)
        self.assertGreater(red.sum(), 50)

    def test_settings_validation(self):
        s = GridSettings.from_dict({"mask_source": "x", "mask_grow": 5, "guide_color": "nope"})
        self.assertEqual((s.mask_source, s.mask_grow, s.guide_color), ("face", 1.0, "#DC281E"))


class GuidePipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="fg_가이드_"))
        cls.video = helpers.make_video(cls.tmp / "v.mp4")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_masked_output_never_leaves_face_open(self):
        pipe = FaceGridPipeline(helpers.FakeDetector(), face_tracker=FakeFaceTracker)
        out = self.tmp / "01_v"
        s = pipe.run(MediaSource.open(self.video), out, "v", GridSettings.from_dict({"hold": 6}), ["masked"])
        self.assertIn("v_masked.mp4", s.files)
        info = helpers.probe(out / "v_masked.mp4")
        self.assertEqual(info["frames"], helpers.FRAMES)
        self.assertTrue(info["audio"])
        import cv2
        cap = cv2.VideoCapture(str(out / "v_masked.mp4"))
        for i in range(helpers.GONE_AFTER):     # 머리가 있는 동안은 (표정을 못 찾는 20~23 포함) 머리 가운데가 가려져 있어야 함
            ok, f = cap.read()
            cx, cy = map(int, helpers.head_at(i).center)
            self.assertLess(int(f[cy, cx].max()), 40, f"{i} 프레임에서 얼굴이 드러남")
        cap.release()

    def test_combo_is_default_on(self):
        from facegrid.outputs import OutputCatalog
        self.assertIn("combo", [o.key for o in OutputCatalog().build(None)])


if __name__ == "__main__":
    unittest.main()
