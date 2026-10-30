"""배포 zip 만들기 — 받은 사람이 풀어서 켜는 파일만 누르면 되게

  app\\python\\python.exe app\\tools\\build_zip.py                 → 이 폴더 옆에 Griding_1.0_windows.zip
  app\\python\\python.exe app\\tools\\build_zip.py D:\\x.zip        → 원하는 곳에
  app\\python\\python.exe app\\tools\\build_zip.py --mac arm64     → Griding_1.0_mac_arm64.zip (Apple 칩)
  app\\python\\python.exe app\\tools\\build_zip.py --mac x86_64    → Griding_1.0_mac_intel.zip

zip 안의 맨 위 폴더 이름은 Griding. 쓰다가 생긴 것(app/work, results, __pycache__)과 깃 폴더는 넣지 않는다.
  윈도우: 앱 + app/python(임베디드 파이썬) + app/lib + *.bat
  맥    : 앱 + mac_runtime/<arch>/python.tar.gz → app/python-mac · 휠 → app/lib-mac + *.command
          (fetch_mac.py 가 받아 둔 것). 윈도우에서 만들어도 실행 권한(755) · 링크를 zip 에 그대로 적는다.
"""
from __future__ import annotations

import stat
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent.parent
TOP = PurePosixPath("Griding")
SKIP_DIRS = {"results", ".git", "mac_runtime", "app/work", "app/python", "app/lib"}
SKIP_NAMES = {"__pycache__", ".gitignore", ".gitattributes"}


class Bundle:
    """공통: 앱 파일을 모으고 zip 에 쓴다. 하위 클래스가 운영체제별 런타임 · 켜는 파일을 더한다."""
    launcher_suffix = ""

    def __init__(self, dest: Path):
        self.dest = dest

    def app_files(self) -> list[Path]:
        out = []
        for p in sorted(ROOT.rglob("*")):
            rel = p.relative_to(ROOT).as_posix()
            if any(rel == d or rel.startswith(d + "/") for d in SKIP_DIRS) or any(part in SKIP_NAMES for part in p.parts):
                continue
            if not p.is_file() or (p.parent == ROOT and p.suffix == ".zip"):   # 맨 위의 옛 배포 zip 은 뺀다
                continue
            if p.parent == ROOT and p.suffix in (".bat", ".command") and p.suffix != self.launcher_suffix:
                continue   # 다른 운영체제의 켜는 파일
            out.append(p)
        return out

    def check(self) -> None: ...

    def runtime(self, z: zipfile.ZipFile) -> int:
        return 0

    @staticmethod
    def put(z: zipfile.ZipFile, name: PurePosixPath, data: bytes, mode: int = 0o644, link: bool = False) -> None:
        info = zipfile.ZipInfo(str(name), date_time=(2026, 10, 30, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.create_system = 3                          # 유닉스 — 맥이 권한 비트를 읽는다
        info.external_attr = ((stat.S_IFLNK if link else stat.S_IFREG) | mode) << 16
        z.writestr(info, data, compresslevel=6)

    def app_mode(self, rel: str) -> int:
        return 0o644

    def build(self) -> tuple[int, float]:
        self.check()
        tmp = self.dest.with_suffix(".tmp")
        items = self.app_files()
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for p in items:
                rel = p.relative_to(ROOT).as_posix()
                self.put(z, TOP / rel, p.read_bytes(), self.app_mode(rel))
            n = len(items) + self.runtime(z)
        tmp.replace(self.dest)
        return n, self.dest.stat().st_size / 1e6


class WindowsBundle(Bundle):
    launcher_suffix = ".bat"
    NEEDED = (ROOT / "app" / "python" / "python.exe", ROOT / "app" / "lib")

    def check(self) -> None:
        missing = [str(p) for p in self.NEEDED if not p.exists()]
        if missing:
            raise SystemExit("파이썬 · 라이브러리가 없어 zip 을 만들 수 없습니다 (배포 zip 에서 app\\python, app\\lib 를 복사해 오세요): "
                             + ", ".join(missing))

    def runtime(self, z: zipfile.ZipFile) -> int:
        n = 0
        for base in ("app/python", "app/lib"):   # python312.zip 같은 안쪽 zip 도 그대로
            for p in sorted((ROOT / base).rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts:
                    self.put(z, TOP / p.relative_to(ROOT).as_posix(), p.read_bytes())
                    n += 1
        return n


class MacBundle(Bundle):
    launcher_suffix = ".command"
    SKIP_PY = ("python/lib/python3.12/test/", "python/lib/python3.12/idlelib/", "python/lib/python3.12/tkinter/",
               "python/lib/python3.12/turtledemo/", "python/share/", "python/include/")
    EXEC_IN_LIB = ("imageio_ffmpeg/binaries/",)        # 휠 안에서 실행해야 하는 파일 (ffmpeg)

    def __init__(self, dest: Path, arch: str):
        super().__init__(dest)
        self.arch = arch
        self.src = ROOT / "mac_runtime" / arch

    def check(self) -> None:
        if not (self.src / "python.tar.gz").is_file() or not list((self.src / "wheels").glob("*.whl")):
            raise SystemExit(f"{self.src} 에 맥용 파이썬 · 부품이 없습니다. 먼저 app/tools/fetch_mac.py {self.arch} 를 실행하세요")

    def app_mode(self, rel: str) -> int:
        return 0o755 if rel.endswith(".command") else 0o644

    def runtime(self, z: zipfile.ZipFile) -> int:
        n = 0
        with tarfile.open(self.src / "python.tar.gz") as tar:     # 권한 · 링크는 tar 에 적힌 그대로
            for m in tar.getmembers():
                if m.isdir() or any(m.name.startswith(s) for s in self.SKIP_PY) or "__pycache__" in m.name:
                    continue
                name = TOP / "app" / "python-mac" / m.name.split("/", 1)[1]
                if m.issym():
                    self.put(z, name, m.linkname.encode(), 0o777, link=True)
                elif m.isfile():
                    self.put(z, name, tar.extractfile(m).read(), m.mode & 0o777)
                else:
                    continue
                n += 1
        for whl in sorted((self.src / "wheels").glob("*.whl")):   # 휠 = zip — 풀어서 app/lib-mac 에
            with zipfile.ZipFile(whl) as w:
                for info in w.infolist():
                    if info.is_dir():
                        continue
                    mode = (info.external_attr >> 16) & 0o777 or 0o644
                    if any(info.filename.startswith(e) for e in self.EXEC_IN_LIB) or info.filename.endswith((".so", ".dylib")):
                        mode |= 0o755
                    self.put(z, TOP / "app" / "lib-mac" / info.filename, w.read(info), mode)
                    n += 1
        return n


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--mac" in args:
        arch = args[args.index("--mac") + 1] if len(args) > args.index("--mac") + 1 else "arm64"
        label = {"arm64": "arm64", "x86_64": "intel"}[arch]
        bundle = MacBundle(ROOT.parent / f"Griding_1.0_mac_{label}.zip", arch)
    else:
        bundle = WindowsBundle(Path(args[0]) if args else ROOT.parent / "Griding_1.0_windows.zip")
    n, mb = bundle.build()
    print(f"{bundle.dest} · 파일 {n}개 · {mb:.0f}MB")
