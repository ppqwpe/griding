"""조합(카드 겹치기) 테스트"""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

import helpers
from facegrid.geometry import Rect
from facegrid.layers import LAYERS, Recipe, Scene
from facegrid.media import MediaSource
from facegrid.pipeline import FaceGridPipeline
from facegrid.settings import GridSettings
from test_face_layers import FakeFaceTracker, oval_face

SIZE = (360, 640)
HEAD = Rect(250, 60, 390, 240)
FACE = oval_face(320, 160, rx=45, ry=60)


def scene(settings, faces=(FACE,)):
    return Scene.still(np.full((*SIZE, 3), 200, np.uint8), [HEAD], settings, faces)


class RecipeTest(unittest.TestCase):
    def test_keys_match_settings(self):
        self.assertEqual(tuple(LAYERS), GridSettings.LAYER_KEYS)

    def test_settings_clean_recipe(self):
        s = GridSettings.from_dict({"recipe": ["points", "nope", "grid", "points"], "base": "x"})
        self.assertEqual((s.recipe, s.base), (("points", "grid"), "original"))
        self.assertEqual(GridSettings.from_dict({"recipe": "grid"}).recipe, (), "목록이 아니면 기본값 (비어 있음)")
        self.assertEqual(GridSettings().recipe, (), "격자도 고를 때만 들어간다")
        self.assertEqual(GridSettings.from_dict({"recipe": []}).recipe, ())

    def test_every_layer_draws_something(self):
        s = GridSettings()
        for key in LAYERS:
            with self.subTest(layer=key):
                base = scene(s).original
                out = Recipe("original", [key]).render(scene(s))
                self.assertGreater(int(np.abs(out.astype(int) - base).sum()), 0)

    def test_order_matters(self):
        s = GridSettings.from_dict({"cover_color": "#000000", "color": "#FF0000", "pattern": "box"})
        grid_on_top = Recipe("original", ["cover_head", "grid"]).render(scene(s))
        cover_on_top = Recipe("original", ["grid", "cover_head"]).render(scene(s))
        cx, cy = map(int, HEAD.center)
        self.assertEqual(tuple(grid_on_top[cy, cx]), (0, 0, 255), "격자(빨강 박스)가 위")
        self.assertEqual(tuple(cover_on_top[cy, cx]), (0, 0, 0), "가림(검정)이 위")

    def test_black_base(self):
        out = Recipe("black", ["points"]).render(scene(GridSettings()))
        self.assertEqual(tuple(out[5, 5]), (0, 0, 0))
        self.assertGreater(out.sum(), 0)

    def test_masked_guide_equals_cover_lines_points(self):
        """10/01 가린 표정 가이드 = 원본 + [가림 · 얼굴 윤곽, 표정 선, 표정 점]"""
        from facegrid.guide import MaskedFaceGuide
        s = GridSettings()
        a = Recipe("original", ["cover_face", "lines", "points"]).render(scene(s))
        b = MaskedFaceGuide(s).render(scene(s).original, [FACE], [HEAD])
        self.assertLess(np.abs(a.astype(int) - b).mean(), 1.0)

    def test_needs_faces(self):
        self.assertFalse(Recipe("original", ["grid", "cover_head", "boxes"]).needs_faces)
        self.assertTrue(Recipe("original", ["grid", "points"]).needs_faces)
        self.assertTrue(Recipe("original", []).empty)


class ComboPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="fg_조합_"))
        cls.video = helpers.make_video(cls.tmp / "v.mp4")
        cls.image = helpers.make_image(cls.tmp / "s.png")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_combo_video(self):
        FakeFaceTracker.instances.clear()
        pipe = FaceGridPipeline(helpers.FakeDetector(), face_tracker=FakeFaceTracker)
        s = GridSettings.from_dict({"recipe": ["cover_face", "points", "grid"], "hold": 6})
        out = self.tmp / "01_v"
        r = pipe.run(MediaSource.open(self.video), out, "v", s, ["combo", "compare"])
        self.assertEqual(r.files[0], "v_combo.mp4", "조합 결과가 대표 파일")
        self.assertEqual(r.settings["recipe"], ["cover_face", "points", "grid"])
        info = helpers.probe(out / "v_combo.mp4")
        self.assertEqual((info["frames"], info["audio"]), (helpers.FRAMES, True))
        self.assertEqual(len(FakeFaceTracker.instances), 1, "점 카드가 있으니 표정 인식을 돌림")

    def test_no_face_tracking_for_grid_only(self):
        FakeFaceTracker.instances.clear()
        pipe = FaceGridPipeline(helpers.FakeDetector(), face_tracker=FakeFaceTracker)
        pipe.run(MediaSource.open(self.video), self.tmp / "02_v", "v", GridSettings.from_dict({"recipe": ["grid", "boxes"]}), ["combo"])
        self.assertEqual(FakeFaceTracker.instances, [], "표정 카드가 없으면 표정 인식을 안 돌림")

    def test_combo_image_and_preview(self):
        pipe = FaceGridPipeline(helpers.FakeDetector(), face_tracker=FakeFaceTracker)
        s = GridSettings.from_dict({"recipe": ["cover_head", "grid"], "base": "black"})
        r = pipe.run(MediaSource.open(self.image), self.tmp / "03_s", "s", s, ["combo"])
        self.assertIn("s_combo.png", r.files)
        jpg = pipe.preview(MediaSource.open(self.video), GridSettings.from_dict({"recipe": ["cover_face", "lines"]}), 0.3)
        self.assertEqual(jpg[:2], b"\xff\xd8")


if __name__ == "__main__":
    unittest.main()
