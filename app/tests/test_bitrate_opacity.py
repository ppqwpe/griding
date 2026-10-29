"""피드백 반영 — 가림 불투명도 · 고정 비트레이트(CBR)"""
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import helpers
from facegrid.blend import over_color
from facegrid.layers import LAYERS
from facegrid.media import MediaSource
from facegrid.pipeline import FaceGridPipeline
from facegrid.probe import Bitrate, MediaInfo
from facegrid.settings import GridSettings
from test_combo import scene


class OpacityTest(unittest.TestCase):
    def test_over_color_half(self):
        canvas = np.full((4, 4, 3), 200, np.uint8)
        mask = np.full((4, 4), 255, np.uint8)
        self.assertEqual(int(over_color(canvas, mask, (0, 0, 0), 0.5)[0, 0, 0]), 100)
        self.assertTrue(np.array_equal(over_color(canvas, mask, (0, 0, 0), 0.0), canvas))

    def test_cover_layers_follow_setting(self):
        for key in ("cover_face", "cover_head"):
            with self.subTest(layer=key):
                k = "cover_alpha" if key == "cover_face" else "head_cover_alpha"   # 가림마다 불투명도 따로
                full = scene(GridSettings.from_dict({k: 1.0}))
                none = scene(GridSettings.from_dict({k: 0.0}))
                half = scene(GridSettings.from_dict({k: 0.5}))
                orig = full.original
                a = LAYERS[key].apply(orig.copy(), full)
                self.assertTrue(np.array_equal(LAYERS[key].apply(orig.copy(), none), orig), "0% = 원본 그대로")
                b = LAYERS[key].apply(orig.copy(), half)
                hit = (a != orig).any(axis=2)
                self.assertTrue(hit.any())
                diff_full = np.abs(a[hit].astype(int) - orig[hit]).mean()
                diff_half = np.abs(b[hit].astype(int) - orig[hit]).mean()
                self.assertAlmostEqual(diff_half / diff_full, 0.5, delta=0.05)

    def test_layer_file_alpha_is_partial(self):
        sc = scene(GridSettings.from_dict({"head_cover_alpha": 0.4}))
        rgba = LAYERS["cover_head"].isolate(sc)
        inside = rgba[..., 3][rgba[..., 3] > 0]
        self.assertAlmostEqual(float(np.median(inside)), 0.4 * 255, delta=2)

    def test_setting_clamped(self):
        self.assertEqual(GridSettings.from_dict({"cover_alpha": 3}).cover_alpha, 1.0)
        self.assertEqual(GridSettings.from_dict({"cover_alpha": -1}).cover_alpha, 0.0)
        self.assertEqual(GridSettings().cover_alpha, 1.0)

    def test_each_mask_has_own_color_and_opacity(self):
        s = GridSettings.from_dict({"cover_color": "#FFFFFF", "cover_alpha": 0.3, "head_cover_color": "#00ff00", "head_cover_alpha": 0.8})
        self.assertEqual((s.cover_color, s.cover_alpha, s.head_cover_color, s.head_cover_alpha), ("#FFFFFF", 0.3, "#00FF00", 0.8))
        sc = scene(s)
        face = LAYERS["cover_face"].apply(sc.original.copy(), sc)
        head = LAYERS["cover_head"].apply(sc.original.copy(), sc)
        self.assertGreater(int(face.max()), 200, "얼굴 윤곽 가림은 흰색 쪽")
        g = head[(head != sc.original).any(axis=2)]
        self.assertTrue((g[:, 1] > g[:, 2]).all(), "머리 전체 가림은 초록 쪽")


class BitrateRuleTest(unittest.TestCase):
    def test_fixed_by_resolution(self):
        self.assertEqual(Bitrate.choose("fixed", 1920, 1080), 30_000)
        self.assertEqual(Bitrate.choose("fixed", 2560, 1440), 30_000)
        self.assertEqual(Bitrate.choose("fixed", 3840, 2160), 50_000)
        self.assertEqual(Bitrate.choose("fixed", 2160, 3840), 50_000, "세로 4K 도")

    def test_source(self):
        self.assertEqual(Bitrate.choose("source", 1920, 1080, 8_000), 8_000)
        self.assertEqual(Bitrate.choose("source", 1920, 1080, 0), 30_000, "모르면 고정값")
        self.assertEqual(Bitrate.choose("source", 1920, 1080, 180_000), 100_000)
        self.assertEqual(GridSettings.from_dict({"bitrate": "vbr"}).bitrate, "fixed")

    def test_parse_kbps(self):
        def read(text):
            with mock.patch("facegrid.probe._run", return_value=text):
                return MediaInfo.read(Path("x.mp4"))
        head = "  Duration: 00:00:09.05, start: 0.000000, bitrate: 9000 kb/s\n"
        audio = "  Stream #0:1(und): Audio: aac (LC), 48000 Hz, stereo, fltp, 192 kb/s\n"
        a = read(head + "  Stream #0:0(und): Video: h264 (High), yuv420p(tv, bt709, progressive), 1920x1080, 8012 kb/s, 24 fps, 24 tbr\n" + audio)
        b = read(head + "  Stream #0:0: Video: h264 (High), yuv420p(tv, bt709, progressive), 1920x1080, 24 fps, 24 tbr\n" + audio)
        self.assertEqual((a.video_kbps, b.video_kbps), (8012, 9000 - 192))


class CbrEncodeTest(unittest.TestCase):
    """실제로 저장해서 비트레이트를 잰다 — 원본 크기 결과는 고정값, 확인용 축소본은 화질 기준"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="fg cbr_"))
        cls.src = helpers.make_video(cls.tmp / "a.mp4")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _run(self, mode):
        out = self.tmp / mode
        pipe = FaceGridPipeline(helpers.FakeDetector())
        settings = GridSettings.from_dict({"recipe": ["grid"], "bitrate": mode})
        return pipe.run(MediaSource.open(self.src), out, "o", settings, ["combo", "compare"]), out

    def test_fixed(self):
        r, out = self._run("fixed")
        kbps = r.checks["o_combo.mp4"]["kbps"]
        self.assertAlmostEqual(kbps / 30_000, 1.0, delta=0.08, msg=f"재 보니 {kbps} kb/s")
        self.assertLess(r.checks["o_compare.mp4"]["kbps"], 15_000, "확인용은 고정값을 안 씀")
        self.assertTrue(r.checks["o_combo.mp4"]["ok"])

    def test_source(self):
        want = MediaInfo.read(self.src).video_kbps
        r, out = self._run("source")
        kbps = r.checks["o_combo.mp4"]["kbps"]
        target = Bitrate.choose("source", helpers.W, helpers.H, want)
        self.assertAlmostEqual(kbps / target, 1.0, delta=0.15, msg=f"원본 {want} → 목표 {target} · 재 보니 {kbps}")


if __name__ == "__main__":
    unittest.main()
