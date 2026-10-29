"""바탕화면 바로가기 — 처음 켤 때 한 번, 아이콘 달린 Griding 바로가기를 만든다

bat 파일은 아이콘을 못 바꿔서, 아이콘(app/icon/griding.ico)이 달린 바로가기가 Griding.bat 을 실행한다.
만든 뒤에는 work/shortcut.json 에 적어 두고 다시 만들지 않는다 (지워도 되살아나지 않게).
폴더를 다른 곳으로 옮겨 켜면 새 위치로 한 번 다시 만든다.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from . import config

_SCRIPT = ("$d = [Environment]::GetFolderPath('Desktop'); $w = New-Object -ComObject WScript.Shell; "
           # 예전 이름(FaceGrid) 바로가기가 이 폴더를 가리키면 지운다 — 이름만 바뀐 같은 앱
           "$old = Join-Path $d 'FaceGrid.lnk'; "
           "if ((Test-Path $old) -and ($w.CreateShortcut($old).TargetPath -like ($env:FG_DIR + '\\*'))) { Remove-Item $old }; "
           "$s = $w.CreateShortcut((Join-Path $d 'Griding.lnk')); "
           "$s.TargetPath = $env:FG_BAT; $s.WorkingDirectory = $env:FG_DIR; "
           "$s.IconLocation = $env:FG_ICON + ',0'; $s.Description = 'Griding'; $s.Save(); $d")


class DesktopShortcut:
    ICON = config.APP_DIR / "icon" / "griding.ico"
    BAT = config.ROOT_DIR / "Griding.bat"

    def __init__(self, mark: Path | None = None):
        self._mark = mark or config.WORK_DIR / "shortcut.json"

    def _made_for_here(self) -> bool:
        try:
            return json.loads(self._mark.read_text("utf-8")).get("bat") == str(self.BAT)
        except (OSError, ValueError, AttributeError):
            return False

    def ensure(self) -> str:
        """만들었으면 안내 한 줄, 아니면 빈 문자열 (실패해도 앱은 그대로 켜짐)"""
        if os.name != "nt" or not self.BAT.is_file() or not self.ICON.is_file() or self._made_for_here():
            return ""
        ps = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
        env = {**os.environ, "FG_BAT": str(self.BAT), "FG_DIR": str(self.BAT.parent), "FG_ICON": str(self.ICON)}
        try:   # 경로는 환경 변수로 넘긴다 (한글 · 공백 · 따옴표가 있어도 안전)
            out = subprocess.run([str(ps) if ps.is_file() else "powershell", "-NoProfile", "-NonInteractive", "-Command", _SCRIPT],
                                 capture_output=True, text=True, encoding="mbcs" if os.name == "nt" else "utf-8", errors="replace", env=env, timeout=20,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except (OSError, subprocess.TimeoutExpired):
            return ""
        if out.returncode != 0:
            return ""
        try:
            self._mark.write_text(json.dumps({"bat": str(self.BAT)}, ensure_ascii=False), "utf-8")
        except OSError:
            pass
        return f"바탕화면에 Griding 바로가기를 만들었습니다 ({out.stdout.strip()}) — 다음부터 그걸 눌러 켜면 됩니다"
