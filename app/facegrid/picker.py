"""Windows 파일 고르기 창 — 브라우저는 끌어다 놓은 파일의 위치를 알려 주지 않으므로,
원본 옆에 결과를 저장하려면 앱(서버)이 직접 고르기 창을 띄워 실제 경로를 받는다.

PowerShell 의 System.Windows.Forms.OpenFileDialog 를 쓴다 (설치할 것 없음, Windows 10/11 기본).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from . import config


class PickerError(RuntimeError):
    pass


class NativeFilePicker:
    TIMEOUT = 30 * 60   # 창을 열어 둔 채 자리를 비워도 30분 뒤엔 풀린다

    def _script(self) -> str:
        exts = ";".join(f"*{e}" for e in config.VIDEO_EXT + config.IMAGE_EXT)
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

    def choose(self) -> list[Path]:
        """고른 파일 경로들 (취소하면 빈 목록)"""
        try:
            r = subprocess.run(["powershell", "-NoProfile", "-STA", "-ExecutionPolicy", "Bypass", "-Command", self._script()],
                               capture_output=True, timeout=self.TIMEOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.TimeoutExpired) as e:
            raise PickerError(f"파일 고르기 창을 열지 못했습니다 ({e}). 끌어다 놓아 주세요")
        out = r.stdout.decode("utf-8-sig", "replace").strip()
        try:
            paths = json.loads(out or "[]")
        except ValueError:
            raise PickerError("파일 고르기 창을 열지 못했습니다. 끌어다 놓아 주세요")
        if isinstance(paths, str):
            paths = [paths]
        return [Path(p) for p in paths if isinstance(p, str) and p]
