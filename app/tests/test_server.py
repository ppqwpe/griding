"""앱 서버 테스트 — 실제로 서버를 띄워 화면이 부르는 주소를 차례로 불러 본다"""
import http.client
import json
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.parse import quote

import helpers
from facegrid import config
from facegrid.server import FaceGridApp, free_port, make_server
from facegrid.store import ResultStore, UploadStore


class ServerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="fg_서버_"))
        cls.video = helpers.make_video(cls.tmp / "plate.mp4")
        cls.app = FaceGridApp(helpers.FakeDetector(), ResultStore(cls.tmp / "results"), UploadStore(cls.tmp / "work"), config.UI_PATH)
        from test_face_layers import FakeFaceTracker
        cls.app.face_probe_factory = lambda: FakeFaceTracker(False)
        cls.app.face_tracker_factory = FakeFaceTracker
        cls.port = free_port(first=18000)
        cls.srv = make_server(cls.app, cls.port)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def call(self, method, path, body=None, headers=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        data = json.dumps(body).encode() if isinstance(body, dict) else body
        c.request(method, path, body=data, headers=headers or {})
        r = c.getresponse()
        raw = r.read()
        c.close()
        return r.status, dict(r.getheaders()), raw

    def upload(self, path: Path, name: str = None):
        st, _, raw = self.call("POST", "/api/upload", path.read_bytes(), {"X-Filename": quote(name or path.name)})
        return st, json.loads(raw)

    def test_page_and_info(self):
        st, h, raw = self.call("GET", "/")
        self.assertEqual(st, 200)
        self.assertIn("Griding", raw.decode("utf-8"))
        st, _, raw = self.call("GET", "/api/info")
        info = json.loads(raw)
        self.assertNotIn("presets", info)
        self.assertEqual(info["limits"]["cols"], [1, 32])
        self.assertEqual(info["outputs"][0]["key"], "combo")
        self.assertFalse(any(o["required"] for o in info["outputs"]), "모든 파일을 끄고 켤 수 있음")
        self.assertIn("grid", [p["key"] for p in info["patterns"]])
        self.assertEqual(len(info["layers"]), 6)
        self.assertEqual([m["name"] for m in info["models"]], ["머리 검출", "추적", "표정 인식", "영상 읽기 · 저장"])
        self.assertEqual(info["defaults"]["cover_alpha"], 1.0)
        self.assertEqual(info["defaults"]["bitrate"], "fixed")
        for icon, kind in (("/favicon.ico", "image/x-icon"), ("/icon.png", "image/png"), ("/name.png", "image/png")):
            st, h, raw = self.call("GET", icon)
            self.assertEqual((st, h["Content-Type"]), (200, kind))
        st, _, raw = self.call("GET", "/i18n.json")   # 영어 화면 사전
        d = json.loads(raw)
        self.assertEqual(st, 200)
        self.assertEqual(d["exact"].get("격자"), "Grid")
        for t in d["templates"]:   # {} 개수와 {0}{1}… 번호가 맞아야 바꿔 끼울 수 있다
            n = t["ko"].count("{}")
            used = {int(i) for i in __import__("re").findall(r"\{(\d+)\}", t["en"])}
            self.assertTrue(used <= set(range(n)), t)

    def test_upload_rejects_bad_files(self):
        bad = self.tmp / "a.exe"
        bad.write_bytes(b"MZ")
        st, j = self.upload(bad)
        self.assertEqual(st, 400)
        self.assertIn("지원하지 않는", j["error"])
        fake = self.tmp / "broken.mp4"
        fake.write_bytes(b"not a video")
        st, j = self.upload(fake)
        self.assertEqual(st, 400)

    def test_full_flow(self):
        st, up = self.upload(self.video, "한글 플레이트.mp4")
        self.assertEqual(st, 200, up)
        self.assertTrue(up["video"])

        st, _, raw = self.call("POST", "/api/gallery", {"id": up["id"], "t": 0.2})
        g = json.loads(raw)
        self.assertEqual(st, 200)
        self.assertTrue(g["from_plate"] and g["found"])
        self.assertEqual(len(g["patterns"]), 8)
        self.assertEqual([c["key"] for c in g["cards"]], ["cover_face", "cover_head", "grid", "lines", "points", "boxes"])
        self.assertTrue(all(c["image"].startswith("data:image/png") for c in g["cards"]))
        self.assertTrue(g["combo"].startswith("data:image/png"))
        self.assertTrue(g["face_found"])
        st, _, raw = self.call("POST", "/api/gallery", {"settings": {"recipe": ["cover_face", "points"]}})
        empty = json.loads(raw)
        self.assertFalse(empty["from_plate"])
        self.assertTrue(empty["combo"].startswith("data:image"), "플레이트가 없어도 예시 얼굴로 보여 줌")
        self.assertNotIn("covers", g, "화면이 안 쓰는 가리기 예시는 안 보냄 (매번 그림 4장 아낌)")

        st, h, raw = self.call("POST", "/api/preview", {"id": up["id"], "t": 0.5, "settings": {"pattern": "dots"}})
        self.assertEqual((st, h["Content-Type"], raw[:2]), (200, "image/jpeg", b"\xff\xd8"))

        st, _, raw = self.call("POST", "/api/run", {"id": up["id"], "settings": {"cols": 7}, "outputs": ["facegrid", "compare", "preview"]})
        job = json.loads(raw)
        for _ in range(240):
            st, _, raw = self.call("GET", f"/api/job/{job['id']}")
            job = json.loads(raw)
            if job["state"] in ("done", "error"):
                break
            time.sleep(0.25)
        self.assertEqual(job["state"], "done", job["msg"])
        r = job["result"]
        self.assertEqual(r["folder"], "01_한글_플레이트")
        self.assertEqual(r["main"], "한글_플레이트_facegrid.mp4")
        self.assertEqual(r["settings"]["cols"], 7)
        self.assertNotIn("heads", r)

        url = "/files/" + quote(r["folder"]) + "/" + quote(r["main"])
        st, h, raw = self.call("GET", url, headers={"Range": "bytes=0-99"})
        self.assertEqual((st, len(raw), h["Content-Type"]), (206, 100, "video/mp4"))
        st, h, raw = self.call("GET", url)
        self.assertEqual(st, 200)
        self.assertEqual(int(h["Content-Length"]), len(raw))
        self.assertTrue((self.app.results.root / "verification.json").is_file())

    def wait(self, job_id):
        for _ in range(240):
            st, _, raw = self.call("GET", f"/api/job/{job_id}")
            job = json.loads(raw)
            if job["state"] in ("done", "error"):
                return job
            time.sleep(0.25)
        self.fail("작업이 끝나지 않음")

    def test_scan_then_run_uses_people(self):
        st, up = self.upload(self.video, "scan.mp4")
        st, _, raw = self.call("POST", "/api/scan", {"id": up["id"], "settings": {}})
        job = self.wait(json.loads(raw)["id"])
        self.assertEqual(job["state"], "done", job["msg"])
        res = job["result"]
        self.assertEqual((res["count"], res["max_together"]), (1, 1))
        self.assertTrue(res["people"][0]["thumb"].startswith("data:image/jpeg"))
        self.assertIn("사람 1명", job["msg"])
        st, _, raw = self.call("POST", "/api/run", {"id": up["id"], "settings": {"recipe": ["grid"], "people": [1]},
                                                    "outputs": ["combo"]})
        job = self.wait(json.loads(raw)["id"])
        self.assertEqual(job["state"], "done", job["msg"])
        self.assertEqual(job["result"]["people_used"], [1])
        self.assertEqual(job["result"]["main"], "scan_combo.mp4")

    def test_preview_not_blocked_by_running_job(self):
        """작업이 도는 동안에도 미리보기가 바로 온다 (작업과 화면이 검출기를 나눠 씀)"""
        st, up = self.upload(self.video, "busy.mp4")
        self.call("POST", "/api/run", {"id": up["id"], "settings": {"recipe": ["grid"]}, "outputs": ["combo", "compare"]})
        t0 = time.time()
        st, _, raw = self.call("POST", "/api/preview", {"id": up["id"], "settings": {"recipe": ["grid"]}})
        self.assertEqual(st, 200)
        self.assertLess(time.time() - t0, 5)

    def test_rejects_other_sites(self):
        st, _, _ = self.call("GET", "/api/info", headers={"Host": "evil.example:80"})
        self.assertEqual(st, 403, "DNS rebinding 막기")
        st, _, _ = self.call("POST", "/api/open", {"folder": ""}, {"Origin": "http://evil.example"})
        self.assertEqual(st, 403, "다른 웹페이지에서 보낸 요청 막기")
        st, _, _ = self.call("GET", "/api/info", headers={"Origin": f"http://127.0.0.1:{self.port}"})
        self.assertEqual(st, 200)

    def test_bad_inputs_are_400_not_500(self):
        st, up = self.upload(self.video, "bad.mp4")
        st, _, _ = self.call("POST", "/api/preview", {"id": up["id"], "t": "abc", "settings": "nope"})
        self.assertEqual(st, 200, "이상한 t · settings 는 기본값으로")
        st, _, _ = self.call("POST", "/api/preview", b"[1,2]", {"Content-Length": "5"})
        self.assertEqual(st, 400)

    def test_errors_are_friendly(self):
        st, _, raw = self.call("GET", "/files/..%2F..%2Fwin.ini")
        self.assertEqual(st, 404)
        st, _, raw = self.call("POST", "/api/run", {"id": "../../x.mp4"})
        self.assertEqual(st, 400)
        self.assertIn("다시 넣어", json.loads(raw)["error"])
        st, _, raw = self.call("POST", "/api/preview", b"{not json", {"Content-Length": "9"})
        self.assertEqual(st, 400)
        st, up = self.upload(self.video)
        st, _, raw = self.call("POST", "/api/run", {"id": up["id"], "outputs": []})
        self.assertEqual(st, 400)
        self.assertIn("하나 이상", json.loads(raw)["error"])
        st, _, raw = self.call("POST", "/api/run", {"id": up["id"], "outputs": ["combo"], "settings": {"recipe": []}})
        self.assertEqual(st, 400)
        self.assertIn("카드", json.loads(raw)["error"])
        st, _, _ = self.call("GET", "/api/job/nope")
        self.assertEqual(st, 404)
        st, _, _ = self.call("GET", "/nothing")
        self.assertEqual(st, 404)


if __name__ == "__main__":
    unittest.main()
