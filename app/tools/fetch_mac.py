"""맥 배포 zip 에 넣을 파이썬 · 부품을 받는다 (윈도우 PC 에서도 됨)

  python app/tools/fetch_mac.py            → mac_runtime/arm64/ · mac_runtime/x86_64/

  python.tar.gz   python-build-standalone 3.12 install_only (실행 권한 · 링크를 그대로 담은 원본 묶음)
  wheels/*.whl    app/lib 에 든 부품과 같은 버전의 맥용 휠 (pip download --platform macosx_*)

맥에 없는 판은 바꿔 받는다 (MAC_SUBSTITUTE):
  onnxruntime_directml → onnxruntime (DirectML 은 윈도우 전용)
  Intel 맥: onnxruntime 1.23.2 (Intel 판은 여기까지) · mediapipe 없음 (Intel 판은 0.10.21 까지인데
  numpy 1.x 용이라 같이 못 씀 → 표정 카드만 꺼지고 나머지는 된다)
mac_runtime/ 은 깃에 올리지 않는다 (.gitignore). build_zip.py --mac 이 여기서 zip 을 만든다.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
LIB = ROOT / "app" / "lib"
OUT = ROOT / "mac_runtime"
PYTHON = "3.12"
ARCHES = {
    "arm64": {"triple": "aarch64-apple-darwin", "platforms": [f"macosx_{v}_0_arm64" for v in (11, 12, 13, 14, 15)]},
    "x86_64": {"triple": "x86_64-apple-darwin", "platforms": [f"macosx_{v}_x86_64" for v in ("10_9", "10_15", "11_0", "12_0", "13_0", "14_0")]},
}
MAC_SUBSTITUTE = {
    "arm64": {"onnxruntime_directml": "onnxruntime"},
    "x86_64": {"onnxruntime_directml": "onnxruntime==1.23.2", "mediapipe": None},
}


class MacRuntimeFetcher:
    def __init__(self, arch: str):
        self.arch, self.spec = arch, ARCHES[arch]
        self.dir = OUT / arch

    def requirements(self) -> list[str]:
        """app/lib 의 부품 이름 · 버전 → 맥에서 받을 것"""
        out = []
        for d in sorted(LIB.glob("*.dist-info")):
            name, ver = d.name[:-len(".dist-info")].rsplit("-", 1)
            sub = MAC_SUBSTITUTE[self.arch].get(name, f"{name}=={ver}")
            if sub is None:
                continue
            out.append(sub if "==" in sub else f"{sub}=={ver}")
        return out

    def wheels(self) -> list[str]:
        dest = self.dir / "wheels"
        shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True)
        failed = []
        for req in self.requirements():
            cmd = [sys.executable, "-m", "pip", "download", "--quiet", "--no-deps", "--only-binary=:all:",
                   "--python-version", PYTHON, "--implementation", "cp", "--abi", "cp312", "--abi", "abi3", "--abi", "none",
                   *[a for p in self.spec["platforms"] + ["any"] for a in ("--platform", p)], "-d", str(dest), req]
            if subprocess.run(cmd, capture_output=True).returncode != 0:
                failed.append(req)
        return failed

    def python(self) -> str:
        rel = json.loads(urllib.request.urlopen(urllib.request.Request(
            "https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest", headers={"User-Agent": "griding"})).read())
        asset = next(a for a in rel["assets"]
                     if a["name"].startswith(f"cpython-{PYTHON}.") and a["name"].endswith(f"-{self.spec['triple']}-install_only.tar.gz"))
        self.dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(asset["browser_download_url"], self.dir / "python.tar.gz")
        (self.dir / "python.txt").write_text(asset["name"], encoding="utf-8")
        return asset["name"]


if __name__ == "__main__":
    for arch in sys.argv[1:] or ARCHES:
        f = MacRuntimeFetcher(arch)
        print(arch, f.python())
        bad = f.wheels()
        print(arch, "부품", len(list((f.dir / "wheels").glob("*.whl"))), "개" + (f" · 못 받음: {', '.join(bad)}" if bad else ""))
