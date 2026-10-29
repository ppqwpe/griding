"""Griding 실행 진입점 — Griding.bat 이 이 파일을 실행한다. 본체는 facegrid 패키지."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from facegrid.server import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(open_browser="--no-browser" not in sys.argv, make_shortcut="--no-shortcut" not in sys.argv))
