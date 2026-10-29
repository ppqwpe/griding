"""부품 단위 테스트 — 설정 · 박스 · 추적 · 모양 · 합성 · 보관소"""
import json
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import numpy as np

import helpers  # noqa: F401  (경로 설정)
from facegrid.geometry import Rect
from facegrid.patterns import REGISTRY
from facegrid.renderer import GridRenderer
from facegrid.settings import GridSettings
from facegrid.store import ResultStore, UploadError, UploadStore
from facegrid.tracker import Tracker


class SettingsTest(unittest.TestCase):
    def test_defaults(self):
        s = GridSettings()
        self.assertEqual((s.pattern, s.cols, s.rows, s.color, s.thickness), ("grid", 5, 6, "#DC281E", 2.2))

    def test_bad_values_fall_back_or_clamp(self):
        s = GridSettings.from_dict({"cols": "99", "rows": "abc", "color": "red", "pattern": "nope",
                                    "alpha": 3, "hold": -5, "clip": "x", "threshold": float("nan"), "junk": 1})
        self.assertEqual(s.cols, 32)
        self.assertEqual(s.rows, 6)
        self.assertEqual(s.color, "#DC281E")
        self.assertEqual(s.pattern, "grid")
        self.assertEqual(s.alpha, 1.0)
        self.assertEqual(s.hold, 0)
        self.assertEqual(s.clip, "box")
        self.assertEqual(s.threshold, 0.25)

    def test_color_normalized(self):
        self.assertEqual(GridSettings.from_dict({"color": "00ff7f"}).color, "#00FF7F")
        self.assertEqual(GridSettings.from_dict({"color": "#00ff7f"}).bgr, (127, 255, 0))

    def test_round_trip(self):
        s = GridSettings.from_dict({"cols": 7, "pattern": "dots"})
        self.assertEqual(GridSettings.from_dict(s.to_dict()), s)

    def test_limits_cover_every_number(self):
        numeric = [f for f, v in GridSettings().to_dict().items() if isinstance(v, (int, float))]
        self.assertEqual(sorted(numeric), sorted(GridSettings.LIMITS))


class RectTest(unittest.TestCase):
    def test_iou(self):
        a, b = Rect(0, 0, 10, 10), Rect(5, 0, 15, 10)
        self.assertAlmostEqual(a.iou(b), 50 / 150)
        self.assertEqual(a.iou(Rect(20, 20, 30, 30)), 0.0)

    def test_expand_and_clip(self):
        r = Rect(10, 10, 20, 30).expanded(x=0.5, top=0.5)
        self.assertEqual(r.as_tuple(), (5, 0, 25, 30))
        self.assertEqual(Rect(-5, -5, 50, 50).clipped(40, 30).as_tuple(), (0, 0, 40, 30))


class TrackerTest(unittest.TestCase):
    def test_same_head_keeps_id(self):
        t = Tracker(smooth=0.0, hold=3)
        ids = [t.update([Rect(10 + i, 10, 50 + i, 60)])[0].id for i in range(10)]
        self.assertEqual(set(ids), {0})

    def test_hold_then_drop(self):
        t = Tracker(smooth=0.0, hold=2)
        for _ in range(3):                       # 3 프레임 이어져야 확정
            t.update([Rect(0, 0, 40, 40)])
        self.assertTrue(t.update([])[0].lost)
        self.assertEqual(len(t.update([])), 1)
        self.assertEqual(t.update([]), [])

    def test_unconfirmed_dropped_at_first_miss(self):
        t = Tracker(hold=10)
        t.update([Rect(0, 0, 40, 40)])
        self.assertEqual(t.update([]), [], "확정 전 한 프레임짜리는 바로 버림")

    def test_coasting_stays_on_screen(self):
        t = Tracker(smooth=0.0, hold=30)
        for i in range(4):
            t.update([Rect(250 + i * 20, 0, 300 + i * 20, 40)], (320, 180))
        for _ in range(20):
            box = t.update([], (320, 180))[0].box
        self.assertLessEqual(box.x1, 320.01)
        self.assertGreater(box.w, 1)

    def test_two_heads_two_ids(self):
        t = Tracker()
        tracks = t.update([Rect(0, 0, 30, 30), Rect(100, 0, 130, 30)])
        self.assertEqual(sorted(x.id for x in tracks), [0, 1])

    def test_smoothing_moves_part_way(self):
        t = Tracker(smooth=0.5)
        t.update([Rect(0, 0, 40, 40)])
        box = t.update([Rect(10, 0, 50, 40)])[0].box
        self.assertAlmostEqual(box.x0, 5.0)


class RendererTest(unittest.TestCase):
    size = (200, 300)
    head = Rect(100, 50, 180, 150)

    def test_no_heads_gives_none(self):
        r = GridRenderer(GridSettings())
        self.assertIsNone(r.alpha(self.size, []))
        frame = np.zeros((*self.size, 3), np.uint8)
        self.assertIs(r.render(frame, []), frame)

    def test_every_pattern_draws_inside_only(self):
        for key in REGISTRY.keys():
            with self.subTest(pattern=key):
                a = GridRenderer(GridSettings.from_dict({"pattern": key})).alpha(self.size, [self.head])
                self.assertGreater(a[50:150, 100:180].sum(), 0)
                self.assertEqual(a[:, :80].sum(), 0, "왼쪽 빈 곳에 선이 생기면 안 됨")
                self.assertLessEqual(a.max(), 1.0)

    def test_diagonal_keeps_other_heads(self):
        two = [Rect(10, 10, 80, 90), Rect(150, 10, 220, 90)]
        a = GridRenderer(GridSettings.from_dict({"pattern": "diagonal"})).alpha(self.size, two)
        self.assertGreater(a[10:90, 10:80].sum(), 0)
        self.assertGreater(a[10:90, 150:220].sum(), 0)

    def test_alpha_and_fill(self):
        a = GridRenderer(GridSettings.from_dict({"alpha": 0.5})).alpha(self.size, [self.head])
        self.assertAlmostEqual(float(a.max()), 0.5, places=2)
        f = GridRenderer(GridSettings.from_dict({"alpha": 1, "fill_alpha": 0.2})).alpha(self.size, [self.head])
        self.assertAlmostEqual(float(f[52:148, 102:178].min()), 0.2, places=2)

    def test_mask_clip_trims_corners(self):
        box = GridRenderer(GridSettings.from_dict({"pattern": "box"})).alpha(self.size, [self.head])
        oval = GridRenderer(GridSettings.from_dict({"pattern": "box", "clip": "mask"})).alpha(self.size, [self.head])
        self.assertGreater(box[52, 102], 0.9)
        self.assertLess(oval[52, 102], 0.1)

    def test_composite_color(self):
        frame = np.zeros((*self.size, 3), np.uint8)
        out = GridRenderer(GridSettings.from_dict({"pattern": "box", "color": "#0000FF"})).render(frame, [self.head])
        self.assertEqual(tuple(out[100, 140]), (255, 0, 0))   # BGR

    def test_matte(self):
        a = GridRenderer(GridSettings()).alpha(self.size, [self.head])
        m = GridRenderer.matte(a, self.size)
        self.assertEqual(m.shape, (*self.size, 3))
        self.assertEqual(GridRenderer.matte(None, self.size).max(), 0)


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="fg_한글_"))

    def test_safe_stem(self):
        self.assertEqual(ResultStore.safe_stem("잔불 C031 (v2).mp4"), "잔불_C031__v2")
        self.assertEqual(ResultStore.safe_stem("...mp4"), "plate")

    def test_numbered_folders(self):
        rs = ResultStore(self.tmp / "결과")
        a, _ = rs.new_folder("a.mp4")
        b, _ = rs.new_folder("b.mp4")
        self.assertEqual((a.name, b.name), ("01_a", "02_b"))

    def test_rerun_versions_and_overwrite(self):
        rs = ResultStore(self.tmp / "결과")
        a, _ = rs.folder_for("a.mp4")                       # 처음 = 새 번호
        (a / "a_combo.mp4").write_bytes(b"1")
        v2, _ = rs.folder_for("a.mp4", a.name, "version")
        v3, _ = rs.folder_for("a.mp4", v2.name, "version")   # _v2 에서 다시 → _v3 (번호는 그대로)
        self.assertEqual((a.name, v2.name, v3.name), ("01_a", "01_a_v2", "01_a_v3"))
        self.assertTrue((a / "a_combo.mp4").exists(), "새 버전이면 예전 결과는 그대로")
        same, _ = rs.folder_for("a.mp4", a.name, "overwrite")
        self.assertEqual(same, a)
        self.assertEqual(list(a.iterdir()), [], "덮어쓰기면 그 폴더를 비우고 다시 씀")
        b, _ = rs.folder_for("b.mp4", "../../Windows", "overwrite")   # 결과 폴더 밖 · 없는 폴더는 새 번호로
        self.assertEqual(b.name, "02_b")
        self.assertEqual(rs.folder_for("b.mp4", "없는폴더", "version")[0].name, "03_b")

    def test_resolve_blocks_escape(self):
        rs = ResultStore(self.tmp / "결과")
        f, _ = rs.new_folder("x.mp4")
        (f / "x_facegrid.mp4").write_bytes(b"1")
        self.assertIsNotNone(rs.resolve(f"{f.name}/x_facegrid.mp4"))
        self.assertIsNone(rs.resolve("../../Windows/win.ini"))
        self.assertIsNone(rs.resolve(f"{f.name}/없음.mp4"))

    def test_verification(self):
        rs = ResultStore(self.tmp / "결과")
        f, stem = rs.new_folder("x.mp4")
        (f / f"{stem}_settings.json").write_text(json.dumps({"source": "x.mp4", "frames": 3, "empty_grid_frames": []}), encoding="utf-8")
        rows = json.loads(rs.write_verification().read_text(encoding="utf-8"))
        self.assertEqual(rows[0]["clip"], f.name)
        self.assertEqual(rows[0]["frames"], 3)

    def test_upload_rules(self):
        us = UploadStore(self.tmp / "work")
        with self.assertRaises(UploadError):
            us.save(BytesIO(b"abc"), 3, "virus.exe")
        with self.assertRaises(UploadError):
            us.save(BytesIO(b""), 0, "a.mp4")
        with self.assertRaises(UploadError):
            us.save(BytesIO(b"ab"), 10, "a.mp4")      # 덜 받음
        fid = us.save(BytesIO(b"abc"), 3, "판.mp4")
        self.assertEqual(us.path(fid).read_bytes(), b"abc")
        self.assertEqual(us.name(fid), "판.mp4")
        with self.assertRaises(UploadError):
            us.path("../../x.mp4")


if __name__ == "__main__":
    unittest.main()
