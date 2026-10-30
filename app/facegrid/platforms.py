"""운영체제마다 다른 것만 모은 곳 — 나머지 코드는 current() 가 돌려주는 Platform 만 부른다

  Windows : PowerShell 파일 고르기 창 · 탐색기 · DirectML(그래픽카드) · %LOCALAPPDATA%\\Griding
  macOS   : osascript 파일 고르기 창 · Finder(open) · Core ML(Apple 칩 · 그래픽) · ~/Library/Caches/Griding
  그 밖   : 고르기 창 없음(끌어다 놓기만) · xdg-open · CPU · ~/.cache/griding
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


class PickerError(RuntimeError):
    pass


class Platform:
    name = "other"
    launcher = "Griding"                 # 화면 · 안내에 쓰는 켜는 파일 이름
    cpu_launcher = "Griding_CPU"
    gpu_provider = ""                    # onnxruntime 그래픽 가속 이름 (없으면 CPU 만)
    gpu_label = ""
    accel_name = "그래픽 가속"           # 화면 '준비됨 · …로 처리' 에 쓰는 말
    PICK_TIMEOUT = 30 * 60               # 고르기 창을 열어 둔 채 자리를 비워도 30분 뒤엔 풀린다

    def local_dir(self) -> Path:
        """앱 폴더에 쓸 수 없을 때 임시 파일을 둘 곳"""
        return Path.home() / ".cache" / "griding"

    def pick_files(self, video_ext, image_ext) -> list[Path]:
        raise PickerError("이 운영체제에서는 파일 고르기 창을 열 수 없습니다. 끌어다 놓아 주세요")

    def open_folder(self, path: Path) -> None:
        subprocess.Popen(["xdg-open", str(path)])

    @staticmethod
    def _run(cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(cmd, capture_output=True, timeout=timeout, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.TimeoutExpired) as e:
            raise PickerError(f"파일 고르기 창을 열지 못했습니다 ({e}). 끌어다 놓아 주세요")


class WindowsPlatform(Platform):
    name = "windows"
    launcher, cpu_launcher = "Griding.bat", "Griding_CPU.bat"
    gpu_provider, gpu_label = "DmlExecutionProvider", "DirectML"
    accel_name = "그래픽카드"

    def local_dir(self) -> Path:
        return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "Griding"

    @staticmethod
    def picker_script(video_ext, image_ext) -> str:
        """System.Windows.Forms.OpenFileDialog — 설치할 것 없음 (Windows 10/11 기본)"""
        exts = ";".join(f"*{e}" for e in tuple(video_ext) + tuple(image_ext))
        return "; ".join([
            "[Console]::OutputEncoding = [Text.Encoding]::UTF8",
            "Add-Type -AssemblyName System.Windows.Forms",
            "$owner = New-Object System.Windows.Forms.Form",
            "$owner.TopMost = $true",
            "$d = New-Object System.Windows.Forms.OpenFileDialog",
            "$d.Multiselect = $true",
            "$d.Title = 'Griding - plates'",
            f"$d.Filter = 'Video / image|{exts}|All files|*.*'",
            "if ($d.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) { ConvertTo-Json -Compress -InputObject @($d.FileNames) } else { '[]' }",
            "$owner.Dispose()",
        ])

    def pick_files(self, video_ext, image_ext) -> list[Path]:
        r = self._run(["powershell", "-NoProfile", "-STA", "-ExecutionPolicy", "Bypass", "-Command",
                       self.picker_script(video_ext, image_ext)], self.PICK_TIMEOUT)
        out = r.stdout.decode("utf-8-sig", "replace").strip()
        try:
            paths = json.loads(out or "[]")
        except ValueError:
            raise PickerError("파일 고르기 창을 열지 못했습니다. 끌어다 놓아 주세요")
        if isinstance(paths, str):
            paths = [paths]
        return [Path(p) for p in paths if isinstance(p, str) and p]

    def open_folder(self, path: Path) -> None:
        os.startfile(str(path))


class MacPlatform(Platform):
    name = "mac"
    launcher, cpu_launcher = "Griding.command", "Griding_CPU.command"
    gpu_provider, gpu_label = "CoreMLExecutionProvider", "Core ML"
    accel_name = "Core ML"
    CANCELLED = "-128"                   # osascript: 사용자가 취소

    def local_dir(self) -> Path:
        return Path.home() / "Library" / "Caches" / "Griding"

    @staticmethod
    def picker_script() -> list[str]:
        """AppleScript choose file — 여러 개 고르기, 줄마다 POSIX 경로 하나"""
        lines = ['activate',
                 'set picked to choose file with prompt "Griding - plates" with multiple selections allowed',
                 'set out to ""',
                 'repeat with f in picked',
                 'set out to out & POSIX path of f & linefeed',
                 'end repeat',
                 'return out']
        return [arg for line in lines for arg in ("-e", line)]

    def pick_files(self, video_ext, image_ext) -> list[Path]:
        r = self._run(["osascript", *self.picker_script()], self.PICK_TIMEOUT)
        if r.returncode != 0:
            if self.CANCELLED in r.stderr.decode("utf-8", "replace"):
                return []
            raise PickerError("파일 고르기 창을 열지 못했습니다. 끌어다 놓아 주세요")
        return [Path(l) for l in r.stdout.decode("utf-8", "replace").splitlines() if l.strip()]

    def open_folder(self, path: Path) -> None:
        subprocess.Popen(["open", str(path)])


def platform_for(system: str = sys.platform) -> Platform:
    if system.startswith("win"):
        return WindowsPlatform()
    if system == "darwin":
        return MacPlatform()
    return Platform()


_CURRENT = platform_for()


def current() -> Platform:
    return _CURRENT
