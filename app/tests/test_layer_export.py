"""카드별 투명 레이어 (_layerN_키.mov) — 압축 없이 편집기에서 겹치기용"""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

import helpers
from facegrid.layers import LAYERS, Recipe
from facegrid.media import MediaSource, ffmpeg_exe
from facegrid.pipeline import FaceGridPipeline
from facegrid.settings import GridSettings
from test_combo import scene
from test_face_layers import FakeFaceTracker


def over(base: np.ndarray, rgba: np.ndarray) -> np.ndarray:
    a = rgba[..., 3:].astype(np.float32) / 255.0
    return base.astype(np.float32) * (1 - a) + rgba[..., :3].astype(np.float32) * a


def frames_rgba(path: Path, size) -> np.ndarray:
    """[프레임, H, W, 4] — OpenCV 는 PNG 코덱 MOV 를 못 읽어서 ffmpeg 로 푼다"""
    w, h = size
    raw = subprocess.run([ffmpeg_exe(), "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "bgra", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 4)


class IsolateTest(unittest.TestCase):
    def test_each_layer_over_original_equals_apply(self):
        """투명 레이어를 원본 위에 얹은 것 = 그 카드를 원본에 바로 그린 것"""
        for color in ("#000000", "#FFFFFF", "#22AA44"):
            s = GridSettings.from_dict({"cover_color": color, "fill_alpha": 0.3})
            for key, layer in LAYERS.items():
                with self.subTest(layer=key, cover=color):
                    sc = scene(s)
                    want = layer.apply(sc.original.copy(), sc).astype(np.float32)
                    diff = np.abs(over(sc.original, layer.isolate(sc)) - want)
                    # 안티앨리어싱 가장자리에서 8비트 반올림이 겹쳐 몇 단계 차이가 날 수 있다 (눈에 안 보임)
                    self.assertLess(diff.max(), 12.0)
                    self.assertLess(diff.mean(), 0.05)

    def test_stack_equals_combo(self):
        s = GridSettings()
        keys = ["cover_face", "lines", "points", "grid"]
        sc = scene(s)
        out = sc.original.astype(np.float32)
        for k in keys:
            out = over(out, LAYERS[k].isolate(sc))
        combo = Recipe("original", keys).render(scene(s)).astype(np.float32)
        self.assertLess(np.abs(out - combo).mean(), 0.5)

    def test_empty_where_nothing_drawn(self):
        rgba = LAYERS["grid"].isolate(scene(GridSettings()))
        self.assertEqual(int(rgba[2, 2, 3]), 0, "머리 밖은 완전히 투명")
        self.assertEqual(int(rgba[..., 3].max()), int(round(GridSettings().alpha * 255)))


class LayerFilesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="fg_레이어_"))
        cls.video = helpers.make_video(cls.tmp / "v.mp4")
        cls.image = helpers.make_image(cls.tmp / "s.png")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_video_layers(self):
        pipe = FaceGridPipeline(helpers.FakeDetector(), face_tracker=FakeFaceTracker)
        s = GridSettings.from_dict({"recipe": ["cover_face", "points", "grid"]})
        out = self.tmp / "01_v"
        r = pipe.run(MediaSource.open(self.video), out, "v", s, ["combo", "layers"])
        self.assertEqual([f for f in r.files if "_layer" in f],
                         ["v_layer1_cover_face.mov", "v_layer2_points.mov", "v_layer3_grid.mov"])
        self.assertEqual(r.main_file, "v_combo.mp4")
        info = helpers.probe(out / "v_layer3_grid.mov")
        self.assertEqual((info["audio"], info["h264"]), (False, False))
        frames = frames_rgba(out / "v_layer3_grid.mov", (helpers.W, helpers.H))
        self.assertEqual(len(frames), helpers.FRAMES)
        rgba = frames[0]
        self.assertEqual(int(rgba[2, 2, 3]), 0, "배경은 투명")
        self.assertGreater(int(rgba[..., 3].max()), 0, "격자는 보임")

    def test_image_layers_are_png(self):
        pipe = FaceGridPipeline(helpers.FakeDetector(), face_tracker=FakeFaceTracker)
        s = GridSettings.from_dict({"recipe": ["cover_head", "grid"]})
        r = pipe.run(MediaSource.open(self.image), self.tmp / "02_s", "s", s, ["layers"])
        self.assertEqual([f for f in r.files if "_layer" in f], ["s_layer1_cover_head.png", "s_layer2_grid.png"])

    def test_nothing_without_cards(self):
        pipe = FaceGridPipeline(helpers.FakeDetector())
        r = pipe.run(MediaSource.open(self.image), self.tmp / "03_s", "s", GridSettings(), ["layers", "overlay"])
        self.assertFalse([f for f in r.files if "_layer" in f])


if __name__ == "__main__":
    unittest.main()
