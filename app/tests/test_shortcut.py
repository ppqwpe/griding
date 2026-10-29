"""바탕화면 바로가기 — 한 번만 만들고, 폴더를 옮기면 다시"""
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from facegrid.shortcut import DesktopShortcut


class ShortcutTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.mark = self.tmp / "shortcut.json"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _ensure(self, code=0):
        done = subprocess.CompletedProcess([], code, stdout="C:/Users/x/Desktop\n", stderr="")
        with mock.patch("facegrid.shortcut.os.name", "nt"), \
             mock.patch("facegrid.shortcut.subprocess.run", return_value=done) as run:
            note = DesktopShortcut(self.mark).ensure()
        return note, run

    def test_once(self):
        note, run = self._ensure()
        self.assertIn("바로가기", note)
        env = run.call_args.kwargs["env"]
        self.assertEqual(env["FG_BAT"], str(DesktopShortcut.BAT))
        self.assertTrue(env["FG_ICON"].endswith("griding.ico"))
        self.assertEqual(json.loads(self.mark.read_text("utf-8"))["bat"], str(DesktopShortcut.BAT))
        note, run = self._ensure()
        self.assertEqual(note, "")
        run.assert_not_called()

    def test_moved_folder_makes_again(self):
        self.mark.write_text(json.dumps({"bat": "D:\old\Griding.bat"}), "utf-8")
        note, run = self._ensure()
        run.assert_called_once()

    def test_failure_is_quiet(self):
        note, _ = self._ensure(code=1)
        self.assertEqual(note, "")
        self.assertFalse(self.mark.exists(), "실패하면 다음에 다시 시도")

    def test_icon_files_exist(self):
        self.assertTrue(DesktopShortcut.ICON.is_file())
        self.assertTrue(DesktopShortcut.ICON.with_suffix(".png").is_file())


if __name__ == "__main__":
    unittest.main()
