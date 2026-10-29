"""규격 — 넣은 파일을 제대로 읽고, 나간 파일이 원본 규격(색 · fps · 프레임 수 · 크기 · 소리)을 지키는지"""
import shutil
import subprocess
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

import helpers
from facegrid.detector import Detector
from facegrid.media import MediaSource, ffmpeg_exe
from facegrid.pipeline import FaceGridPipeline
from facegrid.probe import MediaInfo, OutputCheck
from facegrid.settings import GridSettings

FF = ffmpeg_exe()


class NoHeads(Detector):
    """머리를 하나도 안 찾음 → 결과 그림이 원본과 같아야 한다 (색 왕복만 재기)"""
    def detect_scored(self, frame, threshold):
        return []


def run(*args):
    subprocess.run([FF, "-v", "error", "-y", *args], check=True, capture_output=True)


def patches(w=320, h=180) -> np.ndarray:
    """진한 색 칸 (색공간이 틀리면 크게 달라지는 색들) — BGR"""
    img = np.zeros((h, w, 3), np.uint8)
    cols = [(0, 0, 230), (0, 200, 0), (220, 30, 30), (0, 220, 220), (220, 0, 220), (40, 160, 230), (128, 128, 128)]
    for i, c in enumerate(cols):
        img[:, i * w // len(cols):(i + 1) * w // len(cols)] = c
    return img


def make_colored(path: Path, matrix: str, space: str, size=(320, 180), frames=12, extra=(), pix="yuv420p"):
    """BGR 칸 그림 → 정확한 709 / 601 YUV 로 바꾸고 표시까지 붙인 영상"""
    w, h = size
    cmd = [FF, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", "24", "-i", "-",
           "-vf", f"scale=out_color_matrix={matrix}:out_range=tv:flags=accurate_rnd+full_chroma_int,format={pix}",
           "-c:v", "libx264", "-qp", "0", "-colorspace", space, "-color_primaries", space if space == "bt709" else "smpte170m",
           "-color_trc", space if space == "bt709" else "smpte170m", "-color_range", "tv", *extra, str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    img = patches(w, h)
    for _ in range(frames):
        p.stdin.write(img.tobytes())
    p.stdin.close()
    assert p.wait() == 0
    return img


def decode_first(path: Path, matrix: str, size) -> np.ndarray:
    """정답 풀기 — 표시된 색공간 공식으로 정확히"""
    w, h = size
    raw = subprocess.run([FF, "-v", "error", "-i", str(path), "-vf",
                          f"scale=in_color_matrix={matrix}:in_range=tv:flags=accurate_rnd+full_chroma_int,format=bgr24",
                          "-frames:v", "1", "-f", "rawvideo", "-"], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(h, w, 3).astype(np.float32)


class ParseTest(unittest.TestCase):
    def info(self, stream: str, extra: str = "") -> MediaInfo:
        text = (f"Input #0, mov,mp4, from 'x.mp4':\n  Duration: 00:00:09.05, start: 0.000000, bitrate: 1 kb/s\n"
                f"  Stream #0:0[0x1](und): Video: {stream}\n{extra}"
                "  Stream #0:1[0x2](und): Audio: aac (LC) (mp4a / 0x6134706D), 48000 Hz, stereo, fltp, 250 kb/s\n")
        from unittest import mock
        with mock.patch("facegrid.probe._run", return_value=text):
            return MediaInfo.read(Path("x.mp4"))

    def test_hevc_10bit_709(self):
        i = self.info("hevc (Main 10) (hvc1), yuv420p10le(tv, bt709, progressive), 1920x1080, 15134 kb/s, 24 fps, 24 tbr, 90k tbn")
        self.assertEqual((i.width, i.height, i.bit_depth, i.color.matrix, i.color.range, i.color_tagged),
                         (1920, 1080, 10, "bt709", "tv", True))
        self.assertEqual((i.rate, i.vfr, i.audio_codec, i.audio_copyable), (Fraction(24), False, "aac", True))
        self.assertTrue(any("10비트" in n for n in i.notes()))

    def test_601_triplet_and_ntsc_rate(self):
        i = self.info("vp9 (Profile 0), yuv420p(tv, smpte170m/bt470bg/smpte170m), 1080x1920, 3275 kb/s, 29.97 fps, 29.97 tbr, 15360 tbn")
        self.assertEqual((i.color.matrix, i.color.primaries, i.color.trc), ("bt601", "bt470bg", "smpte170m"))
        self.assertEqual(i.rate, Fraction(30000, 1001))

    def test_untagged_guess_by_size(self):
        hd = self.info("h264 (High), yuv420p(progressive), 1280x720, 1 kb/s, 25 fps, 25 tbr, 12800 tbn")
        sd = self.info("h264 (High), yuv420p(progressive), 720x480 [SAR 8:9 DAR 4:3], 1 kb/s, 23.98 fps, 23.98 tbr, 24k tbn")
        self.assertEqual((hd.color.matrix, hd.color_tagged, sd.color.matrix), ("bt709", False, "bt601"))
        self.assertEqual((sd.sar, sd.square_pixels, sd.rate), ("8:9", False, Fraction(24000, 1001)))

    def test_vfr_rotation_pc_range(self):
        i = self.info("h264 (High), yuvj420p(pc, bt709, progressive), 1920x1080, 1 kb/s, 29.83 fps, 30 tbr, 90k tbn",
                      "      Side data:\n        displaymatrix: rotation of -90.00 degrees\n")
        self.assertEqual((i.width, i.height, i.rotation, i.color.range), (1080, 1920, 270, "pc"))
        self.assertTrue(i.vfr)
        self.assertEqual(i.rate, Fraction(30))

    def test_png_alpha_mov(self):
        i = self.info("png (png  / 0x20676E70), rgba(pc, gbr/unknown/unknown, progressive), 1920x1080, 3635 kb/s, 24 fps, 24 tbr, 12288 tbn")
        self.assertEqual((i.codec, i.pix_fmt, i.width), ("png", "rgba", 1920))


class RoundTripTest(unittest.TestCase):
    """넣은 영상 → 결과 영상: 색 · 프레임 수 · fps · 크기 · 소리가 그대로인지 실제 파일로"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="fg 규격_"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _run(self, src: Path, keys=("combo",), recipe=("boxes",), detector=None):
        pipe = FaceGridPipeline(detector or helpers.FakeDetector())
        out = self.tmp / (src.stem + "_out")
        r = pipe.run(MediaSource.open(src), out, "o", GridSettings.from_dict({"recipe": list(recipe)}), list(keys))
        return r, out

    def _color_case(self, matrix, space, size):
        src = self.tmp / f"{space}_{size[0]}.mp4"
        make_colored(src, matrix, space, size)
        want = decode_first(src, matrix, size)
        got_in = MediaSource.open(src).frame_at(0).astype(np.float32)
        self.assertLess(np.abs(got_in - want).mean(), 0.6, "읽을 때 색이 원본과 같다")
        r, out = self._run(src, detector=NoHeads())
        res = decode_first(out / "o_combo.mp4", matrix, size)
        diff = res - want
        self.assertLess(np.abs(diff).mean(), 2.5, "결과 색이 원본과 같다 (압축 손실만)")
        self.assertLess(np.abs(diff.reshape(-1, 3).mean(0)).max(), 0.8, "전체가 밝거나 어둡게 치우치지 않는다")
        info = MediaInfo.read(out / "o_combo.mp4")
        self.assertEqual((info.color.matrix, info.color_tagged), (matrix, True), "결과에도 색공간 표시")
        self.assertTrue(all(c["ok"] for c in r.checks.values()), r.checks)

    def test_hd_709_color(self):
        self._color_case("bt709", "bt709", (320, 180))

    def test_sd_601_color(self):
        self._color_case("bt601", "smpte170m", (320, 180))

    def test_frames_fps_audio_and_layers_match(self):
        src = helpers.make_video(self.tmp / "a.mp4")   # 48프레임 · 24fps · AAC 소리
        r, out = self._run(src, ("combo", "layers", "compare"), ("grid",))
        self.assertEqual(r.input["width"], helpers.W)
        for f, c in r.checks.items():
            with self.subTest(file=f):
                self.assertTrue(c["ok"], c)
                self.assertEqual(c["frames"], helpers.FRAMES)
        combo = MediaInfo.read(out / "o_combo.mp4")
        self.assertEqual(combo.audio_codec, "aac", "소리는 그대로 복사")
        self.assertEqual(r.checks["o_layer1_grid.mov"]["codec"], "png")

    def test_odd_size(self):
        src = self.tmp / "odd.mp4"
        make_colored(src, "bt709", "bt709", (321, 181), pix="yuv444p")
        r, out = self._run(src, ("combo", "layers"))
        self.assertEqual(r.checks["o_combo.mp4"]["size"], [322, 182], "mp4 는 짝수로")
        self.assertEqual(r.checks["o_layer1_boxes.mov"]["size"], [321, 181], "레이어는 원본 크기 그대로")
        self.assertTrue(all(c["ok"] for c in r.checks.values()), r.checks)

    def test_rotation(self):
        flat = self.tmp / "flat.mp4"
        make_colored(flat, "bt709", "bt709", (320, 180))
        rot = self.tmp / "rot.mp4"
        run("-display_rotation", "90", "-i", str(flat), "-c", "copy", str(rot))
        src = MediaSource.open(rot)
        self.assertEqual(src.size, (180, 320), "돌려서 바로 선 크기")
        self.assertEqual(src.frame_at(0).shape[:2], (320, 180))
        r, _ = self._run(rot)
        self.assertTrue(all(c["ok"] for c in r.checks.values()), r.checks)

    def test_variable_frame_rate(self):
        base = self.tmp / "cfr.mp4"
        make_colored(base, "bt709", "bt709", (320, 180), frames=48)
        vfr = self.tmp / "vfr.mp4"   # 앞 24프레임은 24fps, 뒤 24프레임은 12fps 간격 → 3초
        run("-i", str(base), "-vf", "setpts='if(lt(N,24),N/24/TB,(1+(N-24)/12)/TB)'", "-fps_mode", "vfr",
            "-c:v", "libx264", "-qp", "0", str(vfr))
        info = MediaInfo.read(vfr)
        self.assertTrue(info.vfr, f"가변 프레임으로 읽혀야 함 ({info.fps} / {info.tbr})")
        src = MediaSource.open(vfr)
        n = sum(1 for _ in src.frames())
        self.assertAlmostEqual(n / src.fps, info.duration, delta=2 / src.fps, msg="고정 fps 로 맞춰 길이가 원본과 같다")
        r, _ = self._run(vfr)
        self.assertTrue(all(c["ok"] for c in r.checks.values()), r.checks)

    def test_10bit_source(self):
        src = self.tmp / "ten.mov"
        run("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=24:duration=1", "-c:v", "libx265", "-pix_fmt", "yuv420p10le",
            "-x265-params", "log-level=none", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", str(src))
        info = MediaInfo.read(src)
        self.assertEqual(info.bit_depth, 10)
        r, _ = self._run(src, ("combo", "layers"))
        self.assertTrue(all(c["ok"] for c in r.checks.values()), r.checks)
        self.assertTrue(any("10비트" in n for n in r.input["notes"]))


class CheckTest(unittest.TestCase):
    def test_counts_packets(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            v = helpers.make_video(tmp / "c.mp4", with_audio=False)
            self.assertEqual(OutputCheck.frames(v), helpers.FRAMES)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class SpecTableTest(unittest.TestCase):
    """화면 규격표 — 같게 나오는 칸 · 일부러 바뀌는 칸과 이유"""

    def table(self, stream):
        from unittest import mock
        text = ("  Duration: 00:00:09.05, start: 0.000000, bitrate: 1 kb/s\n"
                f"  Stream #0:0[0x1](und): Video: {stream}\n"
                "  Stream #0:1[0x2](und): Audio: pcm_s16le, 48000 Hz, stereo, s16, 1536 kb/s\n")
        with mock.patch("facegrid.probe._run", return_value=text):
            return {r["key"]: r for r in MediaInfo.read(Path("x.mov")).describe()["table"]["rows"]}

    def test_10bit_odd_pcm(self):
        t = self.table("prores (HQ), yuv422p10le(tv, bt709, progressive), 1921x1081, 1 kb/s, 23.98 fps, 23.98 tbr, 24k tbn")
        self.assertEqual((t["size"]["combo"], t["size"]["layers"]), ("1922×1082", "1921×1081"))
        self.assertFalse(t["size"]["same"]["combo"]); self.assertTrue(t["size"]["same"]["layers"])
        self.assertFalse(t["bits"]["same"]["combo"]); self.assertIn("투명 레이어", t["bits"]["why"]["combo"])
        self.assertEqual(t["fps"]["combo"], "23.976")
        self.assertEqual(t["audio"]["combo"], "AAC 192k"); self.assertIn("pcm_s16le", t["audio"]["why"]["combo"])
        self.assertTrue(t["color"]["same"]["combo"])

    def test_plain_h264_all_same(self):
        t = self.table("h264 (High), yuv420p(tv, bt709, progressive), 1920x1080, 1 kb/s, 24 fps, 24 tbr, 12288 tbn")
        diff = [k for k, r in t.items() if not r["same"]["combo"]]
        self.assertEqual(diff, ["codec", "bitrate", "audio"], "코덱은 늘 H.264 · 고정 비트레이트, 소리는 pcm 이라 변환 — 나머지는 같게")

    def test_image(self):
        from facegrid.probe import SpecTable
        rows = SpecTable.for_image(640, 480, ".jpg")["rows"]
        self.assertEqual(rows[0]["combo"], "640×480")


if __name__ == "__main__":
    unittest.main()
