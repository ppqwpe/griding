"""표정 · 가리기 레이어 테스트"""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

import helpers
from facegrid.facemesh import FaceLines, FaceShape, FaceTracker, MediaPipeFaceTracker
from facegrid.geometry import Rect
from facegrid.media import MediaSource
from facegrid.pipeline import FaceGridPipeline
from facegrid.renderer import GridRenderer
from facegrid.settings import GridSettings


def oval_face(cx, cy, rx=25, ry=35, n=478):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return FaceShape(np.stack([cx + rx * np.cos(t), cy + ry * np.sin(t)], 1).astype(np.float32))


class FakeFaceTracker(FaceTracker):
    """머리 위치에 타원 얼굴을 돌려준다. MISSING 프레임에선 못 찾음."""
    instances = []

    def __init__(self, video):
        self.video, self.closed = video, False
        FakeFaceTracker.instances.append(self)

    def track(self, frame, index, fps, heads=()):
        i = int(round(frame[2, 2, 0] / 4))
        if i in helpers.MISSING or i >= helpers.GONE_AFTER:
            return []
        return [oval_face(*helpers.head_at(i).center)]

    def close(self):
        self.closed = True


class MaskTest(unittest.TestCase):
    size = (200, 300)
    head = Rect(100, 50, 180, 150)

    def test_oval_vs_box(self):
        oval = GridRenderer(GridSettings.from_dict({"mask_shape": "oval"})).head_mask(self.size, [self.head])
        box = GridRenderer(GridSettings.from_dict({"mask_shape": "box"})).head_mask(self.size, [self.head])
        self.assertEqual(oval[100, 140], 255)
        self.assertEqual(box[52, 102], 255)
        self.assertEqual(oval[52, 102], 0)
        self.assertEqual(GridRenderer(GridSettings()).head_mask(self.size, []).max(), 0)

    def test_cover_colors(self):
        frame = np.full((*self.size, 3), 128, np.uint8)
        r = GridRenderer(GridSettings.from_dict({"head_cover_color": "#00FF00"}))   # _cover 는 머리 전체 가림 색
        out = r.cover(frame, r.head_mask(self.size, [self.head]))
        self.assertEqual(tuple(out[100, 140]), (0, 255, 0))
        self.assertEqual(tuple(out[5, 5]), (128, 128, 128))

    def test_cover_settings_validated(self):
        s = GridSettings.from_dict({"cover_color": "green", "mask_shape": "star"})
        self.assertEqual((s.cover_color, s.mask_shape), ("#000000", "oval"))

    def test_face_lines_draw_parts(self):
        canvas = np.zeros((200, 300, 3), np.uint8)
        FaceLines.draw(canvas, [oval_face(150, 100)], (255, 255, 255), 1)
        self.assertGreater(canvas.sum(), 0)
        self.assertEqual(canvas[:40].sum(), 0, "얼굴 밖엔 선이 없어야 함")


class FacePipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="fg_표정_"))
        cls.video = helpers.make_video(cls.tmp / "v.mp4")
        cls.pipe = FaceGridPipeline(helpers.FakeDetector(), face_tracker=FakeFaceTracker)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_face_and_mask_layers(self):
        FakeFaceTracker.instances.clear()
        out = self.tmp / "01_v"
        s = self.pipe.run(MediaSource.open(self.video), out, "v", GridSettings.from_dict({"hold": 6}),
                          ["facegrid", "face", "face_overlay", "points", "points_overlay", "cover", "headmask"])
        for k in ("facegrid", "face", "face_overlay", "points", "points_overlay", "cover", "headmask"):
            self.assertIn(f"v_{k}.mp4", s.files)
            self.assertEqual(helpers.probe(out / f"v_{k}.mp4")["frames"], helpers.FRAMES, k)
        self.assertFalse(helpers.probe(out / "v_face.mp4")["audio"])
        self.assertTrue(helpers.probe(out / "v_face_overlay.mp4")["audio"])
        self.assertTrue(helpers.probe(out / "v_cover.mp4")["audio"])
        self.assertTrue(s.face_tracked)
        self.assertFalse(helpers.probe(out / "v_points.mp4")["audio"])
        pts = json.loads((out / "v_points.json").read_text(encoding="utf-8"))
        self.assertEqual(len(pts["frames"]), helpers.FRAMES)
        self.assertEqual(pts["frames"][0], [], "첫 프레임은 아직 한 번만 봤으니 안 그림 (깜빡임 방지)")
        self.assertEqual(len(pts["frames"][1][0]), 478, "두 프레임째부터 얼굴 하나에 점 478개")
        self.assertEqual(pts["frames"][21], [], "못 찾은 프레임은 빈 목록")
        self.assertIn("v_points.json", s.files)
        # 표정이 비는 프레임 = 머리는 있는데(HOLD 포함) 얼굴을 못 찾은 프레임
        # 점 · 선이 없는 프레임 = 얼굴을 못 찾은 프레임 + 다시 찾은 첫 프레임(0, 24: 두 번째부터 그림)
        self.assertEqual(set(s.empty_face_frames), helpers.MISSING | set(range(helpers.GONE_AFTER, helpers.GONE_AFTER + 6)) | {0, 24})
        self.assertEqual(len(FakeFaceTracker.instances), 1)
        self.assertTrue(FakeFaceTracker.instances[0].video and FakeFaceTracker.instances[0].closed)

    def test_no_face_tracking_unless_asked(self):
        FakeFaceTracker.instances.clear()
        s = self.pipe.run(MediaSource.open(self.video), self.tmp / "02_v", "v", GridSettings(), ["compare", "cover"])
        self.assertFalse(s.face_tracked)
        self.assertEqual(FakeFaceTracker.instances, [], "표정 결과물을 안 골랐으면 표정 인식을 돌리지 않음")


class RealFaceTrackerTest(unittest.TestCase):
    def test_runs_on_blank(self):
        t = MediaPipeFaceTracker(video=True)
        try:
            self.assertEqual(t.track(np.zeros((360, 640, 3), np.uint8), 0, 24), [])
            self.assertEqual(t.track(np.zeros((360, 640, 3), np.uint8), 0, 24), [], "같은 시간이 와도 멈추지 않음")
        finally:
            t.close()

    @unittest.skipUnless(os.environ.get("FACEGRID_TEST_PLATE"), "FACEGRID_TEST_PLATE 없음")
    def test_real_plate_faces(self):
        src = MediaSource.open(Path(os.environ["FACEGRID_TEST_PLATE"]))
        t = MediaPipeFaceTracker(video=False)
        try:
            faces = t.track(src.frame_at(0.5), 0, None)
        finally:
            t.close()
        self.assertEqual(len(faces), 1)
        self.assertEqual(faces[0].points.shape, (478, 2))

    @unittest.skipUnless(os.environ.get("FACEGRID_TEST_PLATE"), "FACEGRID_TEST_PLATE 없음")
    def test_head_crop_finds_small_faces(self):
        """멀리 있는 작은 얼굴: 화면 전체로는 못 찾아도 머리 박스를 잘라 키우면 찾아야 한다"""
        from facegrid.detector import HeadDetector
        from facegrid.facemesh import HeadCropFaceTracker
        src = MediaSource.open(Path(os.environ["FACEGRID_TEST_PLATE"]))
        det, whole, crop = HeadDetector(), MediaPipeFaceTracker(video=False), HeadCropFaceTracker(video=False)
        try:
            frame = src.frame_at(8.0)            # 옆모습 · 전신 컷
            heads = det.detect(frame, 0.25)
            self.assertTrue(heads)
            got = crop.track(frame, 0, None, heads)
            self.assertGreaterEqual(len(got), len(whole.track(frame, 0, None)))
            for f in got:                         # 찾은 점은 머리 박스 근처에 있어야 함
                x0, y0, x1, y1 = f.bounds()
                self.assertTrue(any(h.expanded(0.5, 0.5, 0.5).iou(Rect(x0, y0, x1, y1)) > 0 for h in heads))
        finally:
            whole.close()
            crop.close()


if __name__ == "__main__":
    unittest.main()
