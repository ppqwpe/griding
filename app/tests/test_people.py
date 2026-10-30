"""여러 명 · 사람 찾기 · 리뷰에서 나온 오류 재발 방지"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

import helpers
from facegrid.detector import Detection
from facegrid.facemesh import FaceShape, HeadCropFaceTracker
from facegrid.geometry import Rect
from facegrid.media import MediaSource
from facegrid.pipeline import FaceGridPipeline, JobSummary
from facegrid.scan import Scanner
from facegrid.settings import GridSettings
from facegrid.store import ResultStore
from facegrid.tracker import Tracker


class CrowdScanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="fg_여럿_"))
        cls.video = helpers.make_video(cls.tmp / "crowd.mp4")
        cls.scan = Scanner(helpers.CrowdDetector(), GridSettings()).run(MediaSource.open(cls.video))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_counts_two_people_not_the_false_hit(self):
        self.assertEqual(len(self.scan.people), 2, "한 프레임짜리 오검출은 사람이 아님")
        self.assertEqual(self.scan.max_together, 2)
        self.assertTrue(all(p.thumb.startswith("data:image/jpeg") for p in self.scan.people))

    def test_first_frame_is_covered(self):
        """확정(3 프레임) 전 첫 프레임부터 사람으로 그려져야 한다"""
        self.assertEqual(self.scan.count_at(0), 2)
        self.assertEqual(self.scan.people[0].first, 0)

    def test_ids_do_not_swap_when_crossing(self):
        def who(i, pid):
            box = next(t.box for t in self.scan.tracks_at(i) if t.id == pid)
            return "A" if box.iou(helpers.person_a(i)) > box.iou(helpers.person_b(i)) else "B"
        first = {pid: who(0, pid) for pid in (1, 2)}
        for i in range(helpers.FRAMES):
            for pid in (1, 2):
                self.assertEqual(who(i, pid), first[pid], f"{i} 프레임에서 #{pid} 번호가 바뀜")

    def test_low_score_frames_stay_tracked(self):
        for i in helpers.A_BLUR:
            self.assertEqual(self.scan.count_at(i), 2, f"{i}: 흐려진 A 를 놓침")
            self.assertFalse(any(t.lost for t in self.scan.tracks_at(i)), "낮은 점수로라도 실제 위치를 따라가야 함")

    def test_filter_people(self):
        only = self.scan.tracks_at(5, (2,))
        self.assertEqual([t.id for t in only], [2])
        self.assertEqual(len(self.scan.tracks_at(5, ())), 2)
        self.assertEqual(self.scan.tracks_at(9999), [])

    def test_render_only_chosen_person(self):
        pipe = FaceGridPipeline(helpers.CrowdDetector())
        s = GridSettings.from_dict({"recipe": ["cover_head"], "people": [1]})
        out = self.tmp / "01"
        r = pipe.run(MediaSource.open(self.video), out, "c", s, ["combo"], scan=self.scan)
        self.assertEqual((r.people_count, r.people_used), (2, [1]))
        import cv2
        cap = cv2.VideoCapture(str(out / "c_combo.mp4"))
        ok, f = cap.read()
        cap.release()
        a_c = tuple(map(int, helpers.person_a(0).center))
        b_c = tuple(map(int, helpers.person_b(0).center))
        pid_of_a = 1 if self.scan.tracks_at(0, (1,))[0].box.iou(helpers.person_a(0)) > 0.3 else 2
        covered, open_ = (a_c, b_c) if pid_of_a == 1 else (b_c, a_c)
        self.assertLess(int(f[covered[1], covered[0]].max()), 30, "고른 사람은 가려짐")
        self.assertGreater(int(f[open_[1], open_[0]].max()), 45, "안 고른 사람은 그대로 (배경 밝기 60, 검게 안 칠해짐)")

    def test_scan_is_reused_and_deterministic(self):
        again = Scanner(helpers.CrowdDetector(), GridSettings()).run(MediaSource.open(self.video))
        self.assertEqual([p.describe(24) for p in again.people], [p.describe(24) for p in self.scan.people])
        self.assertEqual(again.key, self.scan.key)
        self.assertNotEqual(self.scan.key, Scanner(helpers.CrowdDetector(), GridSettings.from_dict({"hold": 3})).run(
            MediaSource.open(self.video)).key)


class CutTest(unittest.TestCase):
    """컷이 바뀌면 앞 장면 사람 박스가 '놓쳐도 유지' 동안 새 장면에 남으면 안 된다"""
    CUT = 24

    @classmethod
    def setUpClass(cls):
        import subprocess
        from facegrid.media import ffmpeg_exe
        cls.tmp = Path(tempfile.mkdtemp(prefix="fg_컷_"))
        cls.video = cls.tmp / "cut.mp4"
        p = subprocess.Popen([ffmpeg_exe(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s",
                              f"{helpers.W}x{helpers.H}", "-r", "24", "-i", "-", "-c:v", "libx264", "-qp", "0",
                              "-pix_fmt", "yuv444p", str(cls.video)], stdin=subprocess.PIPE)
        for i in range(helpers.FRAMES):
            f = helpers.make_frame(i)
            if i >= cls.CUT:                       # 새 장면: 색이 완전히 다름
                f[6:, :] = (40, 160, 230)
                f[:6, :6] = i * 4
            p.stdin.write(f.tobytes())
        p.stdin.close()
        p.wait()

        class BeforeCutOnly(helpers.FakeDetector):
            def detect_scored(self, frame, threshold):
                i = helpers.frame_index(frame)
                return [Detection(helpers.head_at(i), 0.9)] if i < CutTest.CUT else []
        cls.scan = Scanner(BeforeCutOnly(), GridSettings.from_dict({"hold": 12})).run(MediaSource.open(cls.video))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_cut_found(self):
        self.assertEqual(self.scan.cuts, [self.CUT])
        self.assertEqual(self.scan.describe()["shots"], 2)

    def test_no_box_carried_into_new_shot(self):
        for i in range(self.CUT, self.CUT + 12):
            self.assertEqual(self.scan.tracks_at(i), [], f"{i} 프레임에 앞 장면 박스가 남음")

    def test_cut_detector_ignores_small_changes(self):
        from facegrid.scan import CutDetector
        c = CutDetector()
        a = helpers.make_frame(0)
        self.assertFalse(c.is_cut(a))
        self.assertFalse(c.is_cut(helpers.make_frame(5)), "사람이 조금 움직인 건 컷이 아님")
        b = a.copy()
        b[:] = (40, 160, 230)
        self.assertTrue(c.is_cut(b))


class FaceContinuityTest(unittest.TestCase):
    """10/27 사용자 영상: 얼굴을 찾았다 놓쳤다 하며 가림이 얼굴 윤곽 ↔ 머리 네모로 88번 바뀌고, 뒤통수에 작은 가짜 얼굴이 생김"""

    def setUp(self):
        from facegrid.facemesh import FaceContinuity
        from facegrid.tracker import Track
        self.c = FaceContinuity()
        self.Track = Track
        self.box = Rect(100, 50, 200, 170)

    def face(self, box, scale=1.0):
        from test_face_layers import oval_face
        cx, cy = box.center
        return oval_face(cx, cy + 10, rx=40 * scale, ry=50 * scale)

    def test_tiny_false_face_rejected(self):
        t = self.Track(1, self.box)
        draw, mask = self.c.update([t], [self.face(self.box, 0.2)])    # 머리의 3% 정도
        self.assertEqual((draw, mask), ([], []), "뒤통수의 작은 가짜 얼굴은 버림")

    def test_mask_keeps_face_shape_when_lost(self):
        t = self.Track(1, self.box)
        f = self.face(self.box)
        self.c.update([t], [f])
        draw, mask = self.c.update([t], [f])
        self.assertEqual(len(draw), 1, "두 번째 프레임부터 그림")
        moved = self.Track(1, self.box.shifted(30, 0, 30, 0))
        draw, mask = self.c.update([moved], [])                           # 옆으로 돌아 얼굴을 놓침
        self.assertEqual(draw, [], "놓친 프레임엔 점 · 선을 안 그림")
        self.assertEqual(len(mask), 1, "가림은 마지막 얼굴 모양으로 계속 (네모로 갈아타지 않음)")
        self.assertAlmostEqual(float(mask[0].points.mean(0)[0]), float(f.points.mean(0)[0]) + 30, delta=1, msg="머리를 따라 옮겨짐")

    def test_single_frame_face_not_drawn(self):
        t = self.Track(1, self.box)
        draw, mask = self.c.update([t], [self.face(self.box)])
        self.assertEqual(draw, [])
        self.assertEqual(len(mask), 1)

    def test_never_seen_face_falls_back_to_head(self):
        draw, mask = self.c.update([self.Track(1, self.box)], [])
        self.assertEqual((draw, mask), ([], []), "정면을 한 번도 못 봤으면 머리 박스로 (가이드가 처리)")

    def test_two_people_keep_their_own_shapes(self):
        a, b = self.Track(1, self.box), self.Track(2, Rect(300, 50, 400, 170))
        fa, fb = self.face(a.box), self.face(b.box)
        self.c.update([a, b], [fa, fb])
        draw, mask = self.c.update([a, b], [fb])                          # a 만 놓침
        self.assertEqual(len(mask), 2)
        self.assertLess(float(mask[0].points.mean(0)[0]), 250, "a 자리엔 a 의 얼굴 모양")


class ReviewRegressionTest(unittest.TestCase):
    def test_settings_survive_bad_numbers(self):
        s = GridSettings.from_dict({"cols": 1e999, "alpha": float("inf"), "hold": "nan", "people": "x", "model": "big"})
        self.assertEqual((s.cols, s.alpha, s.hold, s.people, s.model), (5, 1.0, 12, (), "fast"))
        self.assertEqual(GridSettings.from_dict("not a dict"), GridSettings())

    def test_head_crop_picks_own_face(self):
        """겹친 두 머리: 각 머리는 자기 가운데 얼굴만 가져가고, 같은 얼굴을 두 번 쓰지 않는다"""
        left, right = Rect(0, 0, 100, 100), Rect(60, 0, 160, 100)
        f_left = FaceShape(np.array([[40, 50], [60, 50]], np.float32))
        f_right = FaceShape(np.array([[105, 50], [115, 50]], np.float32))
        self.assertIs(HeadCropFaceTracker._pick([f_right, f_left], left), f_left)
        self.assertIs(HeadCropFaceTracker._pick([f_left, f_right], right), f_right)
        self.assertIsNone(HeadCropFaceTracker._pick([f_right], Rect(0, 0, 50, 50)))

    def test_main_file_prefers_media(self):
        s = JobSummary("a", "f", files=["x_contact.jpg", "x_combo.mp4", "x_settings.json"])
        self.assertEqual(s.main_file, "x_combo.mp4")
        self.assertEqual(JobSummary("a", "f", files=["x_settings.json"]).main_file, "x_settings.json")
        self.assertEqual(JobSummary("a", "f", files=["x_matte.mp4", "x_settings.json"]).main_file, "x_matte.mp4")

    def test_open_folder_never_leaves_results(self):
        tmp = Path(tempfile.mkdtemp(prefix="fg_보관_"))
        rs = ResultStore(tmp / "results")
        self.assertEqual(rs.folder(".."), rs.root)
        self.assertEqual(rs.folder("../.."), rs.root)
        f, _ = rs.new_folder("a.mp4")
        (f / "x.mp4").write_bytes(b"1")
        rs.discard(f)
        self.assertFalse(f.exists(), "실패한 작업 폴더는 지움")
        rs.discard(tmp)            # 결과 폴더 밖은 건드리지 않음
        self.assertTrue(tmp.exists())
        shutil.rmtree(tmp, ignore_errors=True)

    def test_verification_numeric_order(self):
        tmp = Path(tempfile.mkdtemp(prefix="fg_순서_"))
        rs = ResultStore(tmp)
        for n in ("100_a", "11_b", "02_c"):
            (tmp / n).mkdir()
            (tmp / n / "x_settings.json").write_text(json.dumps({"frames": 1}), encoding="utf-8")
        (tmp / "02_c" / "y_settings.json").write_text("[1, 2]", encoding="utf-8")   # dict 가 아닌 파일도 견딤
        rows = json.loads(rs.write_verification().read_text(encoding="utf-8"))
        self.assertEqual([r["clip"] for r in rows], ["02_c", "11_b", "100_a"])
        shutil.rmtree(tmp, ignore_errors=True)

    def test_tracker_pairs_globally(self):
        """오래된 추적이 남의 박스를 먼저 가져가지 않게 (전체에서 겹침 큰 짝부터)"""
        t = Tracker(smooth=0.0)
        a, b = Rect(0, 0, 40, 40), Rect(30, 0, 70, 40)
        for _ in range(5):
            t.update([Detection(a, 0.9)])
        t.update([Detection(a, 0.9), Detection(b, 0.9)])
        tracks = t.update([Detection(a.shifted(2, 0, 2, 0), 0.9), Detection(b.shifted(2, 0, 2, 0), 0.9)])
        old = min(tracks, key=lambda x: x.id)
        self.assertGreater(old.box.iou(a), 0.8, "처음부터 있던 추적은 계속 a 를 따라감")


if __name__ == "__main__":
    unittest.main()
