"""파일 보관 — 올린 원본(UploadStore) · 결과 폴더(ResultStore)"""
from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import BinaryIO, Optional

from . import config

_ID = re.compile(r"^[0-9a-f]{12}\.[a-z0-9]{2,5}$")
_FOLDER = re.compile(r"^(\d{2,})_")
_VERSION = re.compile(r"_v(\d+)$")
SAVE_MODES = ("version", "overwrite")   # 같은 플레이트를 다시 실행할 때: 새 버전(_v2 …)으로 쌓기 | 같은 폴더에 덮어쓰기


class UploadError(ValueError):
    pass


class UploadStore:
    SPARE_BYTES = 2 * 1024 ** 3   # 올린 파일 말고도 결과를 쓸 자리 (넉넉히 2GB)

    def __init__(self, root: Path = config.WORK_DIR):
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def save(self, stream: BinaryIO, length: int, filename: str) -> str:
        name = Path(filename).name or "plate"
        ext = Path(name).suffix.lower()
        if ext not in config.VIDEO_EXT + config.IMAGE_EXT:
            raise UploadError(f"지원하지 않는 파일입니다: {name}")
        if length <= 0:
            raise UploadError("빈 파일입니다")
        if length > config.UPLOAD_MAX_BYTES:
            raise UploadError("파일이 너무 큽니다 (8GB 까지)")
        free = shutil.disk_usage(self._root).free
        if free < length + self.SPARE_BYTES:
            raise UploadError(f"디스크 공간이 모자랍니다 (남은 {free / 1e9:.1f}GB · 필요 {(length + self.SPARE_BYTES) / 1e9:.1f}GB 이상). "
                              "결과 폴더의 옛 결과를 지우거나 공간을 비운 뒤 다시 넣어 주세요")
        fid = uuid.uuid4().hex[:12] + ext
        path = self._root / fid
        left = length
        try:
            with open(path, "wb") as f:
                while left > 0:
                    chunk = stream.read(min(1 << 20, left))
                    if not chunk:
                        break
                    f.write(chunk)
                    left -= len(chunk)
        except OSError as e:   # 쓰는 중 디스크가 찼거나 끊김 — 반쯤 쓴 파일은 남기지 않는다
            path.unlink(missing_ok=True)
            raise UploadError(f"저장하지 못했습니다 ({e.strerror or e}). 디스크 공간을 확인해 주세요")
        if left:
            path.unlink(missing_ok=True)
            raise UploadError("올리는 중 끊겼습니다. 다시 넣어 주세요")
        self._meta(fid).write_text(json.dumps({"name": name}, ensure_ascii=False), encoding="utf-8")
        return fid

    def link(self, source: Path) -> str:
        """파일 고르기 창에서 고른 원본 — 복사하지 않고 위치만 기억한다 (원본 옆에 저장할 수 있게)"""
        source = Path(source)
        ext = source.suffix.lower()
        if ext not in config.VIDEO_EXT + config.IMAGE_EXT:
            raise UploadError(f"지원하지 않는 파일입니다: {source.name}")
        if not source.is_file():
            raise UploadError(f"파일이 없습니다: {source}")
        fid = uuid.uuid4().hex[:12] + ext
        self._meta(fid).write_text(json.dumps({"name": source.name, "path": str(source.resolve())}, ensure_ascii=False), encoding="utf-8")
        return fid

    def _info(self, fid: str) -> dict:
        try:
            d = json.loads(self._meta(fid).read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def path(self, fid: str) -> Path:
        if not _ID.match(fid or ""):
            raise UploadError("파일을 다시 넣어 주세요")
        p = self.source_path(fid) or self._root / fid
        if not p.is_file():
            raise UploadError("파일을 다시 넣어 주세요 (앱을 다시 켰다면 올린 파일이 지워졌습니다)")
        return p

    def source_path(self, fid: str) -> Optional[Path]:
        """고르기 창으로 넣은 원본의 실제 위치 (끌어다 놓은 파일은 None — 브라우저가 위치를 안 알려 줌)"""
        if not _ID.match(fid or ""):
            return None
        p = self._info(fid).get("path")
        return Path(p) if p else None

    def name(self, fid: str) -> str:
        return self._info(fid).get("name") or fid

    def _meta(self, fid: str) -> Path:
        return self._root / (fid + ".json")

    def remove(self, fid: str) -> None:
        """올린 사본과 기록만 지운다 — 고르기 창으로 넣은 원본은 절대 건드리지 않는다"""
        if _ID.match(fid or ""):
            for p in (self._root / fid, self._meta(fid)):
                p.unlink(missing_ok=True)

    def cleanup(self, keep_hours: float = config.UPLOAD_KEEP_HOURS) -> int:
        cutoff, n = time.time() - keep_hours * 3600, 0
        for p in self._root.iterdir():
            try:
                if p.is_file() and p.stat().st_mtime < cutoff:
                    p.unlink()
                    n += 1
            except OSError:
                pass
        return n


class BesideSource:
    """원본 폴더에 이름 뒤에 붙여 저장 — D:\\shots\\plate.mp4 → D:\\shots\\plate_combo.mp4 · plate_settings.json …

    만드는 동안은 원본 폴더 안 숨은 임시 폴더(.griding_xxxxxxxx)에 쓰고 다 되면 옮긴다.
    실패하면 임시 폴더만 지우므로 원본 폴더의 다른 파일은 건드리지 않는다.
      version   : plate → plate_v2 → plate_v3 … (plate_settings.json 이 있으면 쓰인 이름)
      overwrite : plate 그대로, 같은 이름 파일만 바꿔 씀
    """

    def __init__(self, source: Path, mode: str = "version"):
        self.dir = Path(source).parent
        self.base = ResultStore.safe_stem(Path(source).name)
        self.mode = mode
        self.staging: Optional[Path] = None

    def writable(self) -> bool:
        return self.dir.is_dir() and config.writable(self.dir)

    def _name(self, n: int) -> str:
        return self.base if n == 1 else f"{self.base}_v{n}"

    def stem(self) -> str:
        if self.mode == "overwrite":
            return self.base
        n = 1
        while (self.dir / f"{self._name(n)}_settings.json").exists():
            n += 1
        return self._name(n)

    def open(self) -> tuple[Path, str]:
        self.staging = self.dir / f".griding_{uuid.uuid4().hex[:8]}"
        self.staging.mkdir()
        return self.staging, self.stem()

    def commit(self) -> None:
        for f in self.staging.iterdir():
            os.replace(f, self.dir / f.name)
        self.staging.rmdir()
        self.staging = None

    def discard(self) -> None:
        if self.staging is not None:
            shutil.rmtree(self.staging, ignore_errors=True)
            self.staging = None


class ResultStore:
    def __init__(self, root: Path = config.RESULT_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._outside: dict[str, tuple[Path, set[str]]] = {}   # '@1' → (원본 옆 폴더, 그 작업이 만든 파일 이름)

    def register(self, folder: Path, files: list[str]) -> str:
        """결과 폴더 밖(원본 옆)에 쓴 결과에 '@번호' 이름을 붙여 기억 — 받기 · 폴더 열기용.
        그 작업이 만든 파일만 내준다 (원본 폴더의 다른 파일은 화면에서 못 읽음)."""
        folder = Path(folder).resolve()
        with self._lock:
            for k, (d, names) in self._outside.items():
                if d == folder:
                    names.update(files)
                    return k
            key = f"@{len(self._outside) + 1}"
            self._outside[key] = (folder, set(files))
        return key

    @staticmethod
    def safe_stem(filename: str) -> str:
        stem = Path(filename).stem
        s = "".join(c if (c.isalnum() or c in "-_") else "_" for c in stem).strip("_")
        return (s or "plate")[:80]

    def new_folder(self, filename: str) -> tuple[Path, str]:
        """결과/01_이름 → 02_이름 … (넣은 순서 번호)"""
        stem = self.safe_stem(filename)
        with self._lock:
            nums = [int(m.group(1)) for d in self.root.iterdir() if d.is_dir() and (m := _FOLDER.match(d.name))]
            folder = self.root / f"{(max(nums) + 1 if nums else 1):02d}_{stem}"
            folder.mkdir()
        return folder, stem

    def folder_for(self, filename: str, previous: str = "", mode: str = "version") -> tuple[Path, str]:
        """결과 폴더 고르기 — 처음이면 새 번호(new_folder), 다시 실행이면
             version   : 03_이름 → 03_이름_v2 → 03_이름_v3 … (예전 결과는 그대로)
             overwrite : 그 폴더를 비우고 다시 씀
        previous 는 그 플레이트가 지난번에 만든 폴더 이름 (결과 폴더 안에 실제로 있어야 씀)"""
        stem = self.safe_stem(filename)
        prev = self.root / Path(previous).name if previous else None
        if prev is None or not prev.is_dir() or not _FOLDER.match(prev.name) or prev.resolve().parent != self.root.resolve():
            return self.new_folder(filename)
        with self._lock:
            if mode == "overwrite":
                for f in prev.iterdir():
                    shutil.rmtree(f) if f.is_dir() else f.unlink()
                return prev, stem
            base = _VERSION.sub("", prev.name)
            used = [int(m.group(1)) for d in self.root.iterdir()
                    if d.is_dir() and d.name.startswith(base + "_v") and (m := _VERSION.search(d.name)) and d.name == f"{base}_v{m.group(1)}"]
            folder = self.root / f"{base}_v{max(used + [1]) + 1}"
            folder.mkdir()
        return folder, stem

    def resolve(self, relative: str) -> Optional[Path]:
        """/files/ 요청 경로 → 결과 폴더 안의 파일 (밖으로 못 나가게)"""
        parts = [p for p in relative.replace("\\", "/").split("/") if p not in ("", ".", "..")]
        if not parts:
            return None
        if parts[0] in self._outside:
            folder, names = self._outside[parts[0]]
            f = folder / parts[1] if len(parts) == 2 and parts[1] in names else None
            return f if f is not None and f.is_file() else None
        p = (self.root / Path(*parts)).resolve()
        try:
            p.relative_to(self.root.resolve())
        except ValueError:
            return None
        return p if p.is_file() else None

    def folder(self, name: str) -> Path:
        if name in self._outside:
            return self._outside[name][0]
        base = Path(name).name
        if not name or base in ("", ".", ".."):
            return self.root
        p = self.root / base
        return p if p.is_dir() else self.root

    def discard(self, folder: Path) -> None:
        """실패한 작업의 반쯤 만든 결과 폴더를 지운다 (결과 폴더 안에 있을 때만)"""
        try:
            folder = Path(folder).resolve()
            folder.relative_to(self.root.resolve())
        except (ValueError, OSError):
            return
        if folder != self.root.resolve():
            shutil.rmtree(folder, ignore_errors=True)

    @staticmethod
    def _order(d: Path) -> tuple:
        m = _FOLDER.match(d.name)
        return (int(m.group(1)) if m else 10 ** 9, d.name)

    FIELDS = ("source", "frames", "fps", "resolution", "duration", "people_count", "people_used", "max_together",
              "empty_grid_frames", "face_tracked", "empty_face_frames", "files", "device", "seconds")

    def write_verification(self) -> Path:
        rows = []
        for d in sorted((x for x in self.root.iterdir() if x.is_dir()), key=self._order):
            for sj in sorted(d.glob("*_settings.json")):
                try:
                    s = json.loads(sj.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if not isinstance(s, dict):
                    continue
                recipe = (s.get("settings") or {}).get("recipe") if isinstance(s.get("settings"), dict) else None
                rows.append({k: s.get(k) for k in self.FIELDS} | {"clip": d.name, "recipe": recipe})
        out = self.root / "verification.json"
        out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return out
