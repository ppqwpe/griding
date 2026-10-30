"""이 PC 점검 — 켤 때마다 있는 것 · 없는 것을 보고 알려 준다

  SystemCheck.run(detector_device) → SystemReport
    report.print()        콘솔에 표로
    report.fatal          이게 있으면 앱을 켤 수 없음 (무엇을 하면 되는지 같이 알려 줌)
    report.to_dict()      화면 '이 PC 점검' 과 work\\pc_check.json

점검 하나 = Check 하위 클래스 하나. 새 점검은 Check 를 상속해 CHECKS 에 넣으면 된다.
상태: ok(됨) · warn(되지만 알아둘 것 — 일부 기능 꺼짐 · 느림) · fail(못 씀)
"""
from __future__ import annotations

import ctypes
import importlib
import json
import os
import platform
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import __version__, config

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
MARK = {"ok": "✓", "warn": "!", "fail": "✗"}


@dataclass
class CheckResult:
    name: str
    status: str            # ok · warn · fail
    detail: str
    fix: str = ""          # 안 되면 무엇을 하면 되는지
    fatal: bool = False    # fail 이면서 앱을 켤 수 없는 것


class Check(ABC):
    name = ""

    @abstractmethod
    def run(self) -> CheckResult: ...

    def result(self, status: str, detail: str, fix: str = "", fatal: bool = False) -> CheckResult:
        return CheckResult(self.name, status, detail, fix, fatal and status == "fail")


class WindowsCheck(Check):
    name = "운영체제"

    def run(self):
        rel, ver = platform.release(), platform.version()
        build = int(ver.split(".")[-1]) if ver.split(".")[-1].isdigit() else 0
        label = f"Windows {'11' if build >= 22000 else rel} ({ver}) · {platform.machine()}"
        if platform.system() != "Windows":
            return self.result("fail", label, "Griding 는 Windows 10/11 용입니다", fatal=True)
        if rel not in ("10", "11") and build < 10240:
            return self.result("warn", label, "Windows 10 이상을 권장합니다")
        return self.result("ok", label)


class _MemStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]


class CpuMemoryCheck(Check):
    name = "CPU · 메모리"

    def run(self):
        cpu = os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor() or "?"
        cores = os.cpu_count() or 1
        gb = 0.0
        try:
            m = _MemStatus()
            m.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                gb = m.ullTotalPhys / 1024 ** 3
        except (AttributeError, OSError):
            pass
        detail = f"{cpu} · {cores}스레드 · 메모리 {gb:.0f}GB" if gb else f"{cpu} · {cores}스레드"
        if gb and gb < 7.5:
            return self.result("warn", detail, "메모리 8GB 미만 — 4K · 긴 영상은 느리거나 멈출 수 있음")
        if cores < 4:
            return self.result("warn", detail, "코어가 적어 처리가 느릴 수 있음")
        return self.result("ok", detail)


class LibrariesCheck(Check):
    name = "프로그램 부품"
    NEEDED = ("numpy", "cv2", "onnxruntime", "mediapipe", "imageio_ffmpeg")

    def run(self):
        found, missing = [], []
        for mod in self.NEEDED:
            try:
                m = importlib.import_module(mod)
                found.append(f"{mod} {getattr(m, '__version__', '')}".strip())
            except Exception as e:   # DLL 이 없으면 ImportError 말고 OSError 도 난다
                missing.append(f"{mod} ({type(e).__name__})")
        if missing:
            fatal = any(m.split()[0] != "mediapipe" for m in missing)
            return self.result("fail" if fatal else "warn", "없음: " + ", ".join(missing),
                               "Griding 폴더를 통째로 다시 복사하세요 (일부만 옮기면 이렇게 됩니다). "
                               "그래도 안 되면 'Microsoft Visual C++ 재배포 패키지(x64)'를 설치하세요", fatal=fatal)
        return self.result("ok", " · ".join(found))


class RuntimeDllCheck(Check):
    name = "Visual C++ 런타임"
    DLLS = ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")

    def run(self):
        here = Path(config.APP_DIR / "python")
        system = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
        where = {d: ("앱" if (here / d).is_file() else "윈도우" if (system / d).is_file() else "") for d in self.DLLS}
        missing = [d for d, w in where.items() if not w]
        detail = " · ".join(f"{d} {w or '없음'}" for d, w in where.items())
        if missing:
            return self.result("warn", detail, "'Microsoft Visual C++ 재배포 패키지(x64)'를 설치하면 확실합니다")
        return self.result("ok", detail)


class FfmpegCheck(Check):
    name = "영상 저장 (ffmpeg)"
    ENCODERS = ("libx264", "png", "aac")

    def run(self):
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            ver = subprocess.run([exe, "-hide_banner", "-version"], capture_output=True, text=True, errors="replace",
                                 timeout=20, creationflags=_NO_WINDOW).stdout.split("\n")[0]
            enc = subprocess.run([exe, "-hide_banner", "-encoders"], capture_output=True, text=True, errors="replace",
                                 timeout=20, creationflags=_NO_WINDOW).stdout
        except Exception as e:
            return self.result("fail", f"ffmpeg 를 실행하지 못함 ({e})", "Griding 폴더를 통째로 다시 복사하세요", fatal=True)
        missing = [e for e in self.ENCODERS if f" {e} " not in enc]
        short = " ".join(ver.split()[:3])
        if missing:
            return self.result("fail", f"{short} · 없는 인코더: {', '.join(missing)}", "Griding 폴더를 다시 복사하세요", fatal=True)
        return self.result("ok", f"{short} · H.264 · PNG(투명 레이어) · AAC")


class ModelFilesCheck(Check):
    name = "모델 파일"

    FILES = (  # (경로, 이름, 없을 때 상태, 없을 때 영향)
        (config.MODEL_PATH, "머리 검출 · 빠름", "fail", "앱을 켤 수 없음"),
        (config.MODEL_ACCURATE_PATH, "머리 검출 · 정확", "warn", "'정확' 모델을 못 씀 (빠름만)"),
        (config.APP_DIR / "model" / "face_landmarker.task", "표정 인식", "warn", "표정 카드(가림·얼굴 윤곽 · 표정 선 · 점)를 못 씀"),
        (config.APP_DIR / "model" / "sample_face.json", "예시 얼굴", "warn", "카드 예시 그림의 얼굴 점이 안 보임"),
        (config.APP_DIR / "model" / "sample_character.png", "예시 캐릭터", "warn", "카드 예시가 그려 만든 머리로 바뀜"),
    )

    def run(self):
        have, problems, worst, fatal = [], [], "ok", False
        for path, label, bad, effect in self.FILES:
            p = Path(path)
            if p.is_file() and p.stat().st_size > 0:
                have.append(f"{label} {p.stat().st_size / 1e6:.0f}MB" if p.stat().st_size > 1e6 else label)
            else:
                problems.append(f"{label} 없음 → {effect}")
                worst = "fail" if bad == "fail" else ("warn" if worst == "ok" else worst)
                fatal = fatal or bad == "fail"
        detail = " · ".join(have) + (" | " + " · ".join(problems) if problems else "")
        return self.result(worst, detail, "Griding 폴더를 통째로 다시 복사하세요 (app\\model 폴더)" if problems else "", fatal=fatal)


class GpuCheck(Check):
    name = "그래픽카드"

    def __init__(self, device: str = ""):
        self._device = device

    @staticmethod
    def names() -> list[str]:
        # PATH 가 비어 있는 PC 도 있어서 윈도우 폴더의 PowerShell 을 직접 부른다
        ps = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
        try:
            out = subprocess.run([str(ps) if ps.is_file() else "powershell", "-NoProfile", "-NonInteractive", "-Command",
                                  "(Get-CimInstance Win32_VideoController).Name"], capture_output=True, text=True,
                                 errors="replace", timeout=15, creationflags=_NO_WINDOW).stdout
            return [l.strip() for l in out.splitlines() if l.strip()]
        except Exception:
            return []

    def run(self):
        try:
            import onnxruntime as ort
            dml = "DmlExecutionProvider" in ort.get_available_providers()
        except Exception:
            dml = False
        cards = ", ".join(self.names()) or "이름을 못 읽음"
        if config.FORCE_CPU:
            return self.result("warn", f"{cards} · CPU 로만 실행 (Griding_CPU.bat)", "그래픽카드를 쓰려면 Griding.bat 으로 켜세요")
        if self._device == "GPU":
            return self.result("ok", f"{cards} · DirectML 로 머리 찾기")
        why = "DirectML 없음" if not dml else "그래픽카드 실행 시험 실패 → CPU"
        return self.result("warn", f"{cards} · {why}",
                           "CPU 로도 결과는 같고 느리기만 합니다. 그래픽카드 드라이버를 최신으로 올리면 빨라질 수 있어요")


class FaceModelCheck(Check):
    name = "표정 인식 (MediaPipe)"

    def run(self):
        try:
            from .facemesh import HeadCropFaceTracker
            t = HeadCropFaceTracker(video=False)
            t.close()
            return self.result("ok", "켜짐 · CPU")
        except Exception as e:
            return self.result("warn", f"켜지 못함 ({type(e).__name__}: {str(e)[:120]})",
                               "표정 카드만 못 씁니다. 격자 · 가림·머리 전체는 됩니다. 아주 오래된 CPU(AVX 없음)면 이렇게 될 수 있어요")


class StorageCheck(Check):
    name = "저장 폴더"

    def run(self):
        moved = []
        if config.RESULT_DIR != config.ROOT_DIR / "results":
            moved.append("결과")
        if config.WORK_DIR != config.APP_DIR / "work":
            moved.append("임시")
        bad = [str(d) for d in (config.RESULT_DIR, config.WORK_DIR) if not config.writable(d)]
        try:
            free = shutil.disk_usage(config.RESULT_DIR).free / 1024 ** 3
        except OSError:
            free = 0
        detail = f"결과 → {config.RESULT_DIR} · 남은 공간 {free:.0f}GB"
        if bad:
            return self.result("fail", f"쓸 수 없음: {', '.join(bad)}", "Griding 폴더를 바탕화면이나 문서로 옮겨서 다시 실행하세요", fatal=True)
        if moved:
            return self.result("warn", detail, f"앱 폴더에 쓸 수 없어 {' · '.join(moved)} 폴더를 대신 위치로 옮겼습니다")
        if free < 5:
            return self.result("warn", detail, "공간이 5GB 미만 — 긴 영상 · 투명 레이어는 용량이 큽니다")
        return self.result("ok", detail)


@dataclass
class SystemReport:
    version: str
    results: list[CheckResult] = field(default_factory=list)

    @property
    def fatal(self) -> list[CheckResult]:
        return [r for r in self.results if r.fatal]

    @property
    def warnings(self) -> list[CheckResult]:
        return [r for r in self.results if r.status != "ok" and not r.fatal]

    def lines(self) -> list[str]:
        out = []
        for r in self.results:
            out.append(f"  [{MARK[r.status]}] {r.name:<16} {r.detail}")
            if r.fix and r.status != "ok":
                out.append(f"      → {r.fix}")
        return out

    def print(self) -> None:
        print("  이 PC 점검")
        for line in self.lines():
            print(line)
        n_warn, n_fail = len(self.warnings), len(self.fatal)
        print(f"  → {'켤 수 없음' if n_fail else '사용 가능'}" + (f" · 알아둘 것 {n_warn}개" if n_warn else "") + "\n", flush=True)

    def to_dict(self) -> dict:
        return {"version": self.version, "ok": not self.fatal, "results": [asdict(r) for r in self.results]}

    def save(self, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError:
            pass


class SystemCheck:
    """점검을 차례로 돌린다 — 하나가 터져도 나머지는 계속"""

    @staticmethod
    def checks(device: str = "") -> list[Check]:
        return [WindowsCheck(), CpuMemoryCheck(), RuntimeDllCheck(), LibrariesCheck(), FfmpegCheck(),
                ModelFilesCheck(), GpuCheck(device), FaceModelCheck(), StorageCheck()]

    @classmethod
    def run(cls, device: str = "", checks: list[Check] | None = None) -> SystemReport:
        report = SystemReport(__version__)
        for c in checks if checks is not None else cls.checks(device):
            try:
                report.results.append(c.run())
            except Exception as e:
                report.results.append(CheckResult(c.name, "warn", f"점검 중 오류: {type(e).__name__}: {e}"))
        return report
