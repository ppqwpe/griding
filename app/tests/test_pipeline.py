"""처리 흐름 테스트 — 합성 영상 · 이미지로 결과 양식 전체를 만들어 본다 (한글 경로 포함)"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import helpers
from facegrid.composer import Composer
from facegrid.media import MediaSource, UnsupportedMediaError
from facegrid.pipeline import FaceGridPipeline
from facegrid.settings import GridSettings


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="fg_테스트_"))
        cls.video = helpers.make_video(cls.tmp / "플레이트 01.mp4")
        cls.image = helpers.make_image(cls.tmp / "스틸.png")
        cls.pipe = FaceGridPipeline(helpers.FakeDetector())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_source_factory(self):
        self.assertTrue(MediaSource.open(self.video).is_video)
        self.assertFalse(MediaSource.open(self.image).is_video)
        with self.assertRaises(UnsupportedMediaError):
            MediaSource.open(self.tmp / "a.txt")

    def test_video_package(self):
        out = self.tmp / "results" / "01_plate"
        progress = []
        s = self.pipe.run(MediaSource.open(self.video), out, "plate", GridSettings.from_dict({"hold": 6}),
                          ["facegrid", "compare", "overlay", "matte", "preview", "contact"], lambda d, t: progress.append(d))
        names = {f"plate_{k}" for k in ("facegrid.mp4", "compare.mp4", "overlay.mp4", "matte.mp4",
                                         "preview.jpg", "contact.jpg", "settings.json")}
        self.assertEqual(set(s.files), names)
        self.assertEqual(s.main_file, "plate_facegrid.mp4")
        for n in names:
            self.assertTrue((out / n).is_file(), n)

        main = helpers.probe(out / "plate_facegrid.mp4")
        self.assertEqual(main["frames"], helpers.FRAMES)
        self.assertEqual(main["size"], (helpers.W, helpers.H))
        self.assertTrue(main["audio"], "시댄스용 파일에 원본 소리가 있어야 함")
        self.assertTrue(main["h264"])
        self.assertFalse(helpers.probe(out / "plate_matte.mp4")["audio"], "매트에는 소리를 넣지 않음")
        cmp = helpers.probe(out / "plate_compare.mp4")
        self.assertEqual(cmp["size"], Composer.compare_size(helpers.W, helpers.H))
        self.assertEqual(cmp["frames"], helpers.FRAMES)

        # 20~23 프레임은 검출이 끊기지만 hold 로 메워져야 하고, 40 이후 hold(6) 가 끝나면 격자가 사라져야 한다
        empty = set(s.empty_grid_frames)
        self.assertFalse(empty & helpers.MISSING)
        self.assertEqual(empty, set(range(helpers.GONE_AFTER + 6, helpers.FRAMES)))
        self.assertEqual(progress[-1], 1.0)
        self.assertEqual(progress, sorted(progress), "진행 막대는 뒤로 가지 않음")

        data = json.loads((out / "plate_settings.json").read_text(encoding="utf-8"))
        self.assertEqual(data["frames"], helpers.FRAMES)
        self.assertEqual(len(data["heads"]), helpers.FRAMES)
        self.assertEqual(data["settings"]["hold"], 6)
        self.assertEqual({h[0] for f in data["heads"] for h in f}, {1}, "한 사람은 끝까지 같은 번호 (#1)")
        self.assertEqual((data["people_count"], data["people_used"], data["max_together"]), (1, [1], 1))

    def test_single_output(self):
        out = self.tmp / "results" / "02_min"
        s = self.pipe.run(MediaSource.open(self.video), out, "min", GridSettings(), ["facegrid"])
        self.assertEqual(s.files, ["min_facegrid.mp4", "min_settings.json"])

    def test_grid_can_be_off(self):
        out = self.tmp / "results" / "04_nogrid"
        s = self.pipe.run(MediaSource.open(self.video), out, "ng", GridSettings(), ["matte"])
        self.assertEqual(s.files, ["ng_matte.mp4", "ng_settings.json"])
        self.assertEqual(s.main_file, "ng_matte.mp4")
        with self.assertRaises(ValueError):
            self.pipe.run(MediaSource.open(self.video), self.tmp / "results" / "05_none", "x", GridSettings(), [])

    def test_image_package(self):
        out = self.tmp / "results" / "03_still"
        s = self.pipe.run(MediaSource.open(self.image), out, "still", GridSettings(),
                          ["facegrid", "compare", "overlay", "matte", "preview", "contact"])
        self.assertIn("still_facegrid.png", s.files)
        self.assertIn("still_compare.jpg", s.files)
        self.assertNotIn("still_contact.jpg", s.files, "밀착 시트는 영상만")
        self.assertEqual(s.frames, 1)
        self.assertEqual(s.empty_grid_frames, [])

    def test_every_pattern_runs(self):
        from facegrid.patterns import REGISTRY
        src = MediaSource.open(self.image)
        for key in REGISTRY.keys():
            with self.subTest(pattern=key):
                jpg = self.pipe.preview(src, GridSettings.from_dict({"pattern": key}))
                self.assertEqual(jpg[:2], b"\xff\xd8")

    def test_preview_past_end_falls_back(self):
        jpg = self.pipe.preview(MediaSource.open(self.video), GridSettings(), seconds=999)
        self.assertEqual(jpg[:2], b"\xff\xd8")


class GalleryTest(unittest.TestCase):
    def test_cards_without_plate(self):
        from facegrid.gallery import Gallery
        from facegrid.patterns import REGISTRY
        g = Gallery()
        self.assertTrue(g.example(GridSettings()).startswith("data:image/png;base64,"))
        cards = g.layer_cards(GridSettings())
        self.assertEqual(len(cards), 6)
        self.assertTrue(all(c["image"].startswith("data:image/png") for c in cards), "예시 캐릭터로 모두 그려짐")
        self.assertNotEqual(g.combo(GridSettings.from_dict({"recipe": ["grid"]})),
                            g.combo(GridSettings.from_dict({"recipe": ["cover_face", "points"]})))
        self.assertEqual([c["key"] for c in g.patterns()], REGISTRY.keys())
        self.assertEqual(len(g.covers()), 4)

    def test_example_follows_settings(self):
        from facegrid.gallery import Gallery
        g = Gallery()
        few, many = g.example(GridSettings.from_dict({"cols": 2})), g.example(GridSettings.from_dict({"cols": 12}))
        self.assertNotEqual(few, many)
        frame = helpers.make_frame(3)
        self.assertTrue(g.example(GridSettings(), frame, helpers.head_at(3)).startswith("data:image/png"))

    def test_crop_keeps_head_inside(self):
        from facegrid.gallery import THUMB, Gallery
        import numpy as np
        frame = np.zeros((1080, 1920, 3), np.uint8)
        for head in (helpers.Rect(0, 0, 100, 120), helpers.Rect(1800, 950, 1910, 1075), helpers.Rect(900, 400, 1000, 520)):
            img, box = Gallery.crop_around(frame, head)
            self.assertEqual(img.shape[1::-1], THUMB)
            self.assertGreaterEqual(box.x0, -1)
            self.assertLessEqual(box.x1, THUMB[0] + 1)


if __name__ == "__main__":
    unittest.main()
