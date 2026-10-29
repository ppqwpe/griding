"""이 PC 점검 — 있는 것 · 없는 것을 보고, 못 쓰는 것만 막는다"""
import unittest
from pathlib import Path
from unittest import mock

from facegrid import config
from facegrid.syscheck import Check, ModelFilesCheck, StorageCheck, SystemCheck


class _Boom(Check):
    name = "터지는 점검"

    def run(self):
        raise RuntimeError("일부러")


class _Fatal(Check):
    name = "못 쓰는 것"

    def run(self):
        return self.result("fail", "없음", "다시 복사", fatal=True)


class SystemCheckTest(unittest.TestCase):
    def test_this_pc_is_usable(self):
        r = SystemCheck.run("GPU")
        names = [x.name for x in r.results]
        for n in ("운영체제", "프로그램 부품", "영상 저장 (ffmpeg)", "모델 파일", "그래픽카드", "표정 인식 (MediaPipe)", "저장 폴더"):
            self.assertIn(n, names)
        self.assertEqual(r.fatal, [], r.lines())
        self.assertTrue(r.to_dict()["ok"])

    def test_broken_check_does_not_stop_others(self):
        r = SystemCheck.run(checks=[_Boom(), StorageCheck()])
        self.assertEqual([x.status for x in r.results][0], "warn")
        self.assertEqual(len(r.results), 2)

    def test_fatal_is_reported(self):
        r = SystemCheck.run(checks=[_Fatal()])
        self.assertEqual(len(r.fatal), 1)
        self.assertFalse(r.to_dict()["ok"])
        self.assertIn("다시 복사", "\n".join(r.lines()))

    def test_missing_models(self):
        gone = Path(config.APP_DIR / "model" / "없는파일.onnx")
        files = list(ModelFilesCheck.FILES)
        with mock.patch.object(ModelFilesCheck, "FILES", ((gone, "머리 검출 · 빠름", "fail", "앱을 켤 수 없음"), files[2])):
            r = ModelFilesCheck().run()
        self.assertEqual((r.status, r.fatal), ("fail", True))
        with mock.patch.object(ModelFilesCheck, "FILES", (files[0], (gone, "표정 인식", "warn", "표정 카드 못 씀"))):
            r = ModelFilesCheck().run()
        self.assertEqual((r.status, r.fatal), ("warn", False))
        self.assertIn("표정 카드 못 씀", r.detail)


if __name__ == "__main__":
    unittest.main()
