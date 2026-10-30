"""파일 고르기 창 — 브라우저는 끌어다 놓은 파일의 위치를 알려 주지 않으므로,
원본 옆에 결과를 저장하려면 앱(서버)이 직접 고르기 창을 띄워 실제 경로를 받는다.

창을 띄우는 방법은 운영체제마다 다르다 (platforms.py: Windows PowerShell · macOS osascript).
"""
from __future__ import annotations

from pathlib import Path

from . import config
from .platforms import Platform, PickerError, current

__all__ = ["NativeFilePicker", "PickerError"]


class NativeFilePicker:
    def __init__(self, platform: Platform | None = None):
        self._platform = platform or current()

    def choose(self) -> list[Path]:
        """고른 파일 경로들 (취소하면 빈 목록)"""
        return self._platform.pick_files(config.VIDEO_EXT, config.IMAGE_EXT)
