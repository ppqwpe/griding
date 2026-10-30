"""원본 옆에 저장 · 파일 고르기 · 표정 선/점 불투명도"""
import http.client
import json
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.parse import quote

import numpy as np

import helpers
from facegrid import config
from facegrid.facemesh import FaceShape
from facegrid.guide import draw_guide
from facegrid.server import FaceGridApp, free_port, make_server
from facegrid.settings import GridSettings
from facegrid.store import BesideSource, ResultStore, UploadStore


class GuideAlphaTest(unittest.TestCase):
    def test_opacity_blends_lines_and_points(self):
        face = FaceShape(np.array([[20.0, 20.0], [40.0, 30.0]]))
        canvas = np.full((60, 60, 3), 100, np.uint8)
        full = draw_guide(canvas.copy(), [face], GridSettings(guide_color="#FF0000"), dot=3)
        half = draw_guide(canvas.copy(), [face], GridSettings(guide_color="#FF0000", guide_alpha=0.5), dot=3)
        none = draw_guide(canvas.copy(), [face], GridSettings(guide_color="#FF0000", guide_alpha=0.0), dot=3)
        self.assertEqual(tuple(full[20, 20]), (0, 0, 255))            # BGR 빨강
        self.assertTrue(abs(int(half[20, 20, 2]) - 178) <= 2)           # 100 과 255 의 중간
        self.assertTrue((none == canvas).all(), "0% 면 아무것도 안 그림")

    def test_setting_is_clamped(self):
        self.assertEqual(GridSettings.from_dict({"guide_alpha": 3}).guide_alpha, 1.0)
        self.assertEqual(GridSettings().guide_alpha, 1.0)


class BoxColorTest(unittest.TestCase):
    def test_box_color_and_opacity(self):
        from facegrid.geometry import Rect
        from facegrid.layers import LAYERS, Scene
        frame = np.full((120, 160, 3), 100, np.uint8)
        scene = Scene.still(frame, [Rect(40, 30, 100, 90)], GridSettings(box_color="#0000FF"))
        out = LAYERS["boxes"].apply(frame.copy(), scene)
        self.assertTrue(((out[..., 0] > 200) & (out[..., 2] < 60)).any(), "고른 파랑으로 박스를 그린다")
        faint = LAYERS["boxes"].apply(frame.copy(), Scene.still(frame, [Rect(40, 30, 100, 90)], GridSettings(box_color="#0000FF", box_alpha=0.0)))
        self.assertTrue((faint == frame).all(), "불투명도 0% 면 안 보임")


class BesideSourceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="fg_원본옆_"))
        self.src = self.tmp / "컷 01.mp4"
        self.src.write_bytes(b"x")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make(self, mode, stem_files=("combo.mp4", "settings.json")):
        b = BesideSource(self.src, mode)
        staging, stem = b.open()
        for f in stem_files:
            (staging / f"{stem}_{f}").write_text(mode)
        b.commit()
        return stem

    def test_version_then_overwrite(self):
        self.assertEqual(self.make("version"), "컷_01")
        self.assertEqual(self.make("version"), "컷_01_v2")
        self.assertEqual(self.make("version"), "컷_01_v3")
        self.assertEqual(self.make("overwrite"), "컷_01")
        self.assertEqual((self.tmp / "컷_01_combo.mp4").read_text(), "overwrite")
        self.assertFalse(list(self.tmp.glob(".griding_*")), "임시 폴더가 남지 않음")
        self.assertEqual(self.src.read_bytes(), b"x", "원본은 그대로")

    def test_failure_leaves_source_folder_clean(self):
        before = sorted(p.name for p in self.tmp.iterdir())
        b = BesideSource(self.src)
        staging, stem = b.open()
        (staging / f"{stem}_combo.mp4").write_text("half")
        b.discard()
        self.assertEqual(sorted(p.name for p in self.tmp.iterdir()), before)

    def test_link_never_deletes_original(self):
        up = UploadStore(self.tmp / "work")
        fid = up.link(self.src)
        self.assertEqual(up.path(fid), self.src.resolve())
        self.assertEqual(up.source_path(fid), self.src.resolve())
        self.assertEqual(up.name(fid), "컷 01.mp4")
        up.remove(fid)
        self.assertTrue(self.src.is_file(), "목록에서 빼도 원본은 남는다")

    def test_outside_files_only_from_that_run(self):
        rs = ResultStore(self.tmp / "results")
        (self.tmp / "other.txt").write_text("secret")
        key = rs.register(self.tmp, ["컷 01.mp4"])
        self.assertEqual(rs.resolve(f"{key}/컷 01.mp4"), self.src)
        self.assertIsNone(rs.resolve(f"{key}/other.txt"), "그 작업이 만든 파일만 내준다")
        self.assertIsNone(rs.resolve(f"{key}/../other.txt"))
        self.assertEqual(rs.folder(key), self.tmp.resolve())


class FakePicker:
    def __init__(self):
        self.paths = []

    def choose(self):
        return list(self.paths)


class PickAndSaveServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="fg_고르기_"))
        cls.shots = cls.tmp / "촬영 폴더"
        cls.shots.mkdir()
        cls.video = helpers.make_video(cls.shots / "plate A.mp4")
        cls.app = FaceGridApp(helpers.FakeDetector(), ResultStore(cls.tmp / "results"), UploadStore(cls.tmp / "work"), config.UI_PATH)
        cls.app.picker = FakePicker()
        cls.port = free_port(first=18200)
        cls.srv = make_server(cls.app, cls.port)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def call(self, method, path, body=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        c.request(method, path, body=json.dumps(body).encode() if isinstance(body, dict) else body, headers=headers or {})
        r = c.getresponse()
        raw = r.read()
        c.close()
        return r.status, dict(r.getheaders()), raw

    def run_job(self, fid, **extra):
        st, _, raw = self.call("POST", "/api/run", {"id": fid, "settings": {"recipe": ["grid"]}, "outputs": ["combo"], **extra})
        self.assertEqual(st, 200, raw)
        job_id = json.loads(raw)["id"]
        for _ in range(240):
            job = json.loads(self.call("GET", f"/api/job/{job_id}")[2])
            if job["state"] in ("done", "error"):
                self.assertEqual(job["state"], "done", job["msg"])
                return job["result"]
            time.sleep(0.25)
        self.fail("작업이 끝나지 않음")

    def test_pick_then_save_next_to_source(self):
        self.app.picker.paths = [self.video, self.shots / "notes.txt"]
        (self.shots / "notes.txt").write_text("not media")
        st, _, raw = self.call("POST", "/api/pick", {})
        j = json.loads(raw)
        self.assertEqual((st, len(j["plates"]), len(j["errors"])), (200, 1, 1))
        plate = j["plates"][0]
        self.assertEqual((plate["name"], plate["path"]), ("plate A.mp4", str(self.video)))
        self.assertFalse((self.tmp / "work" / plate["id"]).exists(), "고른 원본은 복사하지 않는다")

        st, h, raw = self.call("GET", "/source/" + plate["id"], headers={"Range": "bytes=0-9"})
        self.assertEqual((st, len(raw)), (206, 10))

        r1 = self.run_job(plate["id"], place="source")
        self.assertEqual((r1["place"], r1["where"]), ("source", str(self.shots.resolve())))
        self.assertTrue((self.shots / "plate_A_combo.mp4").is_file())
        self.assertTrue((self.shots / "plate_A_settings.json").is_file())
        self.assertFalse(list(self.shots.glob(".griding_*")))
        st, _, _ = self.call("GET", "/files/" + quote(r1["folder"]) + "/" + quote(r1["main"]))
        self.assertEqual(st, 200)

        r2 = self.run_job(plate["id"], place="source", save_mode="version")
        self.assertEqual(r2["main"], "plate_A_v2_combo.mp4")
        r3 = self.run_job(plate["id"], place="source", save_mode="overwrite")
        self.assertEqual(r3["main"], "plate_A_combo.mp4")
        self.assertTrue(self.video.is_file(), "원본은 그대로")

    def test_dragged_file_falls_back_to_results(self):
        st, _, raw = self.call("POST", "/api/upload", self.video.read_bytes(), {"X-Filename": quote("drag.mp4")})
        fid = json.loads(raw)["id"]
        r = self.run_job(fid, place="source")
        self.assertEqual(r["place"], "results")
        self.assertIn("위치를 몰라", r["note"])
        self.assertTrue(Path(r["where"]).is_dir())


if __name__ == "__main__":
    unittest.main()
