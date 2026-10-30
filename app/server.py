"""Griding 실행 진입점 — Griding.bat 이 이 파일을 실행한다. 본체는 facegrid 패키지."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from facegrid.server import main  # noqa: E402

if __name__ == "__main__":
    args = sys.argv[1:]
    port = int(args[args.index("--port") + 1]) if "--port" in args[:-1] else 7860   # --port N: 이 번호부터 빈 번호를 찾음
    sys.exit(main(open_browser="--no-browser" not in args, make_shortcut="--no-shortcut" not in args, first_port=port))
