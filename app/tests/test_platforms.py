"""운영체제별 처리 — 맥 쪽은 실제 명령 대신 가짜로 바꿔 윈도우에서도 돌린다"""
import io
import subprocess
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import helpers  # noqa: F401  (app 경로)
from facegrid import platforms, syscheck
from facegrid.picker import NativeFilePicker, PickerError
from facegrid.platforms import MacPlatform, Platform, WindowsPlatform, platform_for


def done(code=0, out=b"", err=b""):
    return subprocess.CompletedProcess([], code, out, err)


class PlatformChoiceTest(unittest.TestCase):
    def test_platform_for(self):
        self.assertIsInstance(platform_for("win32"), WindowsPlatform)
        self.assertIsInstance(platform_for("darwin"), MacPlatform)
        self.assertEqual(type(platform_for("linux")), Platform)

    def test_names_and_dirs(self):
        mac, win = MacPlatform(), WindowsPlatform()
        self.assertEqual((mac.launcher, mac.gpu_provider), ("Griding.command", "CoreMLExecutionProvider"))
        self.assertEqual((win.launcher, win.gpu_provider), ("Griding.bat", "DmlExecutionProvider"))
        self.assertEqual(mac.local_dir(), Path.home() / "Library" / "Caches" / "Griding")


class MacPickerTest(unittest.TestCase):
    def test_paths_from_osascript(self):
        mac = MacPlatform()
        out = "/Users/me/샷 01/plate a.mov\n/Users/me/b.mp4\n".encode()
        with mock.patch.object(platforms.subprocess, "run", return_value=done(0, out)) as run:
            paths = NativeFilePicker(mac).choose()
        self.assertEqual(paths, [Path("/Users/me/샷 01/plate a.mov"), Path("/Users/me/b.mp4")])
        cmd = run.call_args[0][0]
        self.assertEqual(cmd[0], "osascript")
        self.assertIn("with multiple selections allowed", " ".join(cmd))

    def test_cancel_is_empty_and_errors_raise(self):
        mac = MacPlatform()
        with mock.patch.object(platforms.subprocess, "run", return_value=done(1, b"", b"execution error: User canceled. (-128)")):
            self.assertEqual(mac.pick_files((), ()), [])
        with mock.patch.object(platforms.subprocess, "run", return_value=done(1, b"", b"something else")):
            self.assertRaises(PickerError, mac.pick_files, (), ())
        with mock.patch.object(platforms.subprocess, "run", side_effect=FileNotFoundError("osascript")):
            self.assertRaises(PickerError, mac.pick_files, (), ())

    def test_open_folder_uses_open(self):
        with mock.patch.object(platforms.subprocess, "Popen") as popen:
            MacPlatform().open_folder(Path("/Users/me/out"))
        self.assertEqual(popen.call_args[0][0][0], "open")

    def test_other_platform_has_no_dialog(self):
        self.assertRaises(PickerError, Platform().pick_files, (), ())


class WindowsPickerTest(unittest.TestCase):
    def test_json_from_powershell(self):
        out = '["C:\\\\샷\\\\a b.mp4","D:\\\\x.mov"]'.encode("utf-8-sig")
        with mock.patch.object(platforms.subprocess, "run", return_value=done(0, out)):
            self.assertEqual(WindowsPlatform().pick_files((".mp4",), (".png",)), [Path(r"C:\샷\a b.mp4"), Path(r"D:\x.mov")])
        self.assertIn("*.mp4;*.png", WindowsPlatform.picker_script((".mp4",), (".png",)))


class MacSystemCheckTest(unittest.TestCase):
    def test_mac_version_rule(self):
        self.assertEqual(syscheck.MacCheck("14.6", "arm64").run().status, "ok")
        old = syscheck.MacCheck("13.5", "arm64").run()
        self.assertEqual((old.status, old.fatal), ("fail", True))
        self.assertIn("Intel", syscheck.MacCheck("15.0", "x86_64").run().detail)

    def test_mac_check_list_skips_windows_only(self):
        with mock.patch.object(syscheck, "current_platform", return_value=MacPlatform()):
            names = [type(c).__name__ for c in syscheck.SystemCheck.checks()]
        self.assertIn("MacCheck", names)
        self.assertNotIn("RuntimeDllCheck", names)
        self.assertNotIn("WindowsCheck", names)

    def test_mac_memory_and_gpu_label(self):
        with mock.patch.object(syscheck, "current_platform", return_value=MacPlatform()), \
             mock.patch.object(syscheck, "_sysctl", side_effect=lambda k: {"hw.memsize": str(16 * 1024 ** 3), "machdep.cpu.brand_string": "Apple M2"}[k]):
            self.assertAlmostEqual(syscheck.CpuMemoryCheck.memory_gb(), 16.0)
            self.assertEqual(syscheck.GpuCheck.names(), ["Apple M2"])
            r = syscheck.GpuCheck("GPU").run()
        self.assertIn("Core ML", r.detail)


class MacLauncherTest(unittest.TestCase):
    ROOT = Path(__file__).resolve().parent.parent.parent

    def test_command_files(self):
        for name in ("Griding.command", "Griding_CPU.command"):
            raw = (self.ROOT / name).read_bytes()
            self.assertTrue(raw.startswith(b"#!/bin/bash\n"), name)
            self.assertNotIn(b"\r", raw, f"{name}: 줄 끝은 LF 여야 맥에서 돈다")
            self.assertIn(b"app/python-mac/bin/python3.12", raw)
            self.assertIn(b'PYTHONPATH="$PWD/app/lib-mac"', raw)
        self.assertIn(b"FACEGRID_CPU=1", (self.ROOT / "Griding_CPU.command").read_bytes())


class MacBundleTest(unittest.TestCase):
    """build_zip.MacBundle 이 실행 권한 · 링크를 zip 에 적는지 (작은 가짜 런타임으로)"""

    def test_exec_bits_and_links(self):
        import importlib.util
        import tarfile
        import tempfile
        tools = Path(__file__).resolve().parent.parent / "tools" / "build_zip.py"
        spec = importlib.util.spec_from_file_location("build_zip", tools)
        bz = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bz)
        tmp = Path(tempfile.mkdtemp(prefix="fg_mac_"))
        src = tmp / "rt"
        (src / "wheels").mkdir(parents=True)
        with tarfile.open(src / "python.tar.gz", "w:gz") as tar:
            data = b"\xcf\xfa\xed\xfe"
            ti = tarfile.TarInfo("python/bin/python3.12"); ti.size = len(data); ti.mode = 0o755
            tar.addfile(ti, io.BytesIO(data))
            li = tarfile.TarInfo("python/bin/python3"); li.type = tarfile.SYMTYPE; li.linkname = "python3.12"
            tar.addfile(li)
        with zipfile.ZipFile(src / "wheels" / "imageio_ffmpeg-0-py3-none-any.whl", "w") as w:
            w.writestr("imageio_ffmpeg/binaries/ffmpeg-macos-aarch64", b"x")
            w.writestr("imageio_ffmpeg/__init__.py", b"")
        bundle = bz.MacBundle(tmp / "out.zip", "arm64")
        bundle.src = src
        with mock.patch.object(bz.MacBundle, "app_files", return_value=[]):
            bundle.build()
        z = zipfile.ZipFile(tmp / "out.zip")
        mode = {i.filename: i.external_attr >> 16 for i in z.infolist()}
        self.assertEqual(mode["Griding/app/python-mac/bin/python3.12"] & 0o777, 0o755)
        self.assertEqual(mode["Griding/app/python-mac/bin/python3"] & 0o170000, 0o120000, "링크로 적힘")
        self.assertEqual(z.read("Griding/app/python-mac/bin/python3"), b"python3.12")
        self.assertEqual(mode["Griding/app/lib-mac/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64"] & 0o111, 0o111)
        self.assertEqual(mode["Griding/app/lib-mac/imageio_ffmpeg/__init__.py"] & 0o111, 0)


if __name__ == "__main__":
    unittest.main()
