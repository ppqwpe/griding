"""Griding 앱 — 이 PC 안에서만 열리는 작은 웹 서버 + 브라우저 화면

FaceGridApp 이 부품(검출기 · 처리기 · 보관소 · 대기열 · 갤러리)을 갖고,
RequestHandler 는 주소(경로)마다 FaceGridApp 의 메서드 하나로 넘긴다.
"""
from __future__ import annotations

import json
import socket
import sys
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

from . import __version__, config
from .detector import Detector, DetectorPool
from .facemesh import HeadCropFaceTracker
from .layers import Recipe
from .gallery import Gallery
from .jobs import Job, JobQueue
from .media import MediaReadError, MediaSource, UnsupportedMediaError
from .outputs import OutputCatalog
from .patterns import REGISTRY
from .pipeline import FaceGridPipeline, default_face_tracker
from .scan import TrackScan
from .settings import GridSettings
from .shortcut import DesktopShortcut
from .picker import NativeFilePicker, PickerError
from .platforms import current as current_platform
from .store import SAVE_MODES, BesideSource, ResultStore, UploadError, UploadStore
from .syscheck import SystemCheck

CONTENT_TYPES = {".mp4": "video/mp4", ".mov": "video/quicktime", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                 ".json": "application/json; charset=utf-8", ".html": "text/html; charset=utf-8", ".ico": "image/x-icon"}


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class FaceGridApp:
    """부품을 가진 앱. 화면 요청(미리보기 · 예시)과 작업은 서로 다른 검출기를 써서 서로 기다리지 않는다."""
    SCAN_CACHE = 8

    def __init__(self, detectors: DetectorPool | Detector, results: ResultStore, uploads: UploadStore,
                 ui_path: Path = config.UI_PATH):
        if isinstance(detectors, Detector):   # 테스트: 검출기 하나를 모든 용도로
            one = detectors
            detectors = DetectorPool(lambda model: one)
        self.detectors = detectors
        self.results = results
        self.uploads = uploads
        self.catalog = OutputCatalog()
        self.gallery = Gallery()
        self.jobs = JobQueue()
        self.face_tracker_factory = default_face_tracker        # 작업 · 미리보기용 표정 인식기 만드는 법
        self.face_probe_factory = lambda: HeadCropFaceTracker(video=False)
        self.picker = NativeFilePicker()                        # 파일 고르기 창 (테스트에선 가짜로 바꿈)
        self._ui_path = ui_path
        self._ui_lock = threading.Lock()   # 화면 요청끼리는 차례로 (화면용 검출기 · 표정 예시기 공유)
        self._face_probe = None
        self._face_probe_failed = False
        self._scans: dict[tuple, TrackScan] = {}
        self.system: dict = {}             # 켤 때 한 '이 PC 점검' 결과 (화면에 보여 줌)

    @property
    def detector(self) -> Detector:
        return self.detectors.get("fast", "ui")

    def _pipeline(self, settings: GridSettings, purpose: str) -> FaceGridPipeline:
        return FaceGridPipeline(self.detectors.get(settings.model, purpose), self.catalog, self.face_tracker_factory)

    def _faces_for_example(self, frame, head):
        if self._face_probe_failed:
            return []
        try:
            if self._face_probe is None:
                self._face_probe = self.face_probe_factory()
            return self._face_probe.track(frame, 0, None, [head])
        except Exception:   # 표정 모델 문제로 예시가 안 나와도 앱은 계속 돈다 (한 번만 알림)
            traceback.print_exc()
            self._face_probe_failed = True
            return []

    # ── 정보 ──
    def ui(self) -> bytes:
        return self._ui_path.read_bytes()

    def info(self) -> dict:
        return {"version": __version__, "device": self.detector.device, "out": str(self.results.root),
                "defaults": GridSettings().to_dict(), "limits": GridSettings.LIMITS,
                "patterns": REGISTRY.describe(), "outputs": self.catalog.describe(), "layers": Recipe.describe(),
                "models": self.models(), "detect_models": self.detectors.available(), "waiting": self.jobs.waiting(), "launcher": current_platform().launcher, "accel": current_platform().accel_name,
                "system": self.system}

    def models(self) -> list[dict]:
        """화면에 보여 줄 '어떤 모델로 돌아가나'"""
        dev = f"그래픽 가속 ({current_platform().gpu_label})" if self.detector.device == "GPU" else "CPU"
        return [
            {"name": "머리 검출", "model": "YOLO11s head (빠름) · YOLO11m head (정확)",
             "source": "deepghs/real_head_detection · head_detect_v0_s_yv11 / _m_yv11 (Hugging Face)",
             "runs_on": f"ONNX Runtime · {dev}", "license": "AGPL-3.0 (Ultralytics 학습 · 저작자 표기 없음)", "when": "매 프레임 · 머리 위치 → 사람 번호 · 격자 · 가림 자리"},
            {"name": "추적", "model": "점수 2단계 짝맞추기 (ByteTrack 방식)", "source": "Griding 안에 직접 구현 (모델 파일 없음)",
             "runs_on": "CPU", "license": "-", "when": "프레임 사이 같은 사람 잇기 · 3프레임 이상 이어져야 사람으로 셈"},
            {"name": "표정 인식", "model": "MediaPipe Face Landmarker", "source": "Google MediaPipe · face_landmarker.task (얼굴 점 478개)",
             "runs_on": "MediaPipe · CPU", "license": "Apache-2.0", "when": "표정 선 · 표정 점 · 가림·얼굴 윤곽 카드를 골랐을 때만"},
            {"name": "영상 읽기 · 저장", "model": "FFmpeg · H.264 (조합) · PNG MOV (투명 레이어)", "source": "imageio-ffmpeg 에 들어 있는 ffmpeg",
             "runs_on": "CPU", "license": "LGPL/GPL",
             "when": "원본 색공간(709/601) · 회전 · 가변 프레임을 읽어 그대로 맞춰 저장 · 소리는 다시 압축 않고 복사 · 저장 뒤 규격 대조"},
        ]

    # ── 파일 ──
    def upload(self, stream, length: int, filename: str) -> dict:
        fid = None
        try:
            fid = self.uploads.save(stream, length, filename)
            src = MediaSource.open(self.uploads.path(fid))
        except (UploadError, UnsupportedMediaError, MediaReadError) as e:
            if fid:
                self.uploads.remove(fid)   # 열 수 없는 파일은 남기지 않음
            raise ApiError(str(e))
        return self._describe(fid, src)

    @staticmethod
    def _describe(fid: str, src: MediaSource) -> dict:
        return {"id": fid, "video": src.is_video, "size": list(src.size), "fps": src.fps, "frames": src.frame_count,
                "spec": src.describe()}

    def _source(self, fid) -> MediaSource:
        try:
            return MediaSource.open(self.uploads.path(str(fid or "")))
        except (UploadError, UnsupportedMediaError, MediaReadError) as e:
            raise ApiError(str(e))

    def _cached_scan(self, fid: str, settings: GridSettings) -> TrackScan | None:
        return self._scans.get((fid, TrackScan.key_for(settings)))

    def _remember_scan(self, fid: str, scan: TrackScan) -> None:
        self._scans[(fid, scan.key)] = scan
        while len(self._scans) > self.SCAN_CACHE:
            self._scans.pop(next(iter(self._scans)))

    @staticmethod
    def _seconds(body: dict) -> float:
        try:
            t = float(body.get("t") or 0)
        except (TypeError, ValueError):
            return 0.0
        return t if t == t and 0 <= t < 1e7 else 0.0

    # ── 미리보기 · 예시 ──
    def preview(self, body: dict) -> bytes:
        fid = body.get("id", "")
        src = self._source(fid)
        settings = GridSettings.from_dict(body.get("settings"))
        with self._ui_lock:
            return self._pipeline(settings, "ui").preview(src, settings, self._seconds(body), self._cached_scan(fid, settings))

    def gallery_cards(self, body: dict) -> dict:
        settings = GridSettings.from_dict(body.get("settings"))
        frame = head = None
        faces, heads = [], []
        if body.get("id"):
            src = self._source(body["id"])
            frame = src.frame_at(self._seconds(body))
            with self._ui_lock:
                heads = self.detectors.get(settings.model, "ui").detect(frame, settings.threshold)
                head = max(heads, key=lambda b: b.area) if heads else None
                faces = self._faces_for_example(frame, head) if head is not None else []
        use = frame if head is not None else None
        return {"found": head is not None, "from_plate": frame is not None, "face_found": bool(faces),
                "heads_here": len(heads),
                "cards": self.gallery.layer_cards(settings, use, head, faces),
                "combo": self.gallery.combo(settings, use, head, faces),
                "patterns": self.gallery.patterns(frame, head, settings)}

    # ── 작업 ──
    def scan(self, body: dict) -> dict:
        """사람 찾기 — 대기열에서 돌리고, 결과(사람 목록 · 썸네일)는 작업 결과로"""
        fid = body.get("id", "")
        src = self._source(fid)
        settings = GridSettings.from_dict(body.get("settings"))
        cached = self._cached_scan(fid, settings)

        def work(job: Job) -> dict:
            scan = cached or self._pipeline(settings, "job").scan(src, settings, job.report)
            self._remember_scan(fid, scan)
            return scan.describe()

        return self.jobs.submit(Job(f"{self.uploads.name(fid)} 사람 찾기", work)).to_dict()

    def run(self, body: dict) -> dict:
        fid = body.get("id", "")
        src = self._source(fid)
        name = self.uploads.name(fid)
        settings = GridSettings.from_dict(body.get("settings"))
        raw = body.get("outputs")
        keys = None if not isinstance(raw, list) else [k for k in raw if k in self.catalog.keys()]
        if keys is not None and not keys:
            raise ApiError("만들 파일을 하나 이상 골라 주세요")
        if (keys is None or "combo" in keys or "layers" in keys) and not settings.recipe:
            raise ApiError("조합할 카드를 하나 이상 골라 주세요")

        mode = body.get("save_mode") if body.get("save_mode") in SAVE_MODES else "version"
        previous = str(body.get("previous") or "")
        original = self.uploads.source_path(fid)
        beside = BesideSource(original, mode) if body.get("place") == "source" and original else None
        why = "" if body.get("place") != "source" or beside else "끌어 놓은 파일은 위치를 몰라 results 폴더에 저장했습니다"

        def compute(folder: Path, stem: str, job: Job, folder_name: str | None = None):
            scan = self._cached_scan(fid, settings)
            pipe = self._pipeline(settings, "job")
            if scan is None:
                scan = pipe.scan(src, settings, lambda f, t: job.report(f * 0.35, t))
                self._remember_scan(fid, scan)
            return pipe.run(src, folder, stem, settings, keys, job.report, scan, folder_name)

        def in_results(job: Job, note: str) -> dict:
            folder, stem = self.results.folder_for(name, previous, mode)
            try:
                summary = compute(folder, stem, job)
            except BaseException:
                self.results.discard(folder)   # 반쯤 만든 결과 폴더는 지운다
                raise
            finally:
                self.results.write_verification()
            return summary.brief() | {"place": "results", "where": str(folder), "note": note}

        def next_to_source(job: Job) -> dict:
            if not beside.writable():
                return in_results(job, f"원본 폴더에 쓸 수 없어 results 폴더에 저장했습니다 ({beside.dir})")
            staging, stem = beside.open()
            try:
                summary = compute(staging, stem, job, beside.dir.name)
                beside.commit()
            except BaseException:
                beside.discard()   # 원본 폴더엔 임시 폴더만 있었으니 그것만 지운다
                raise
            key = self.results.register(beside.dir, summary.files)
            return summary.brief() | {"folder": key, "place": "source", "where": str(beside.dir), "note": ""}

        return self.jobs.submit(Job(name, next_to_source if beside else lambda job: in_results(job, why))).to_dict()

    def pick(self) -> dict:
        """파일 고르기 창으로 고른 원본들 — 복사 없이 위치만 등록 (원본 옆에 저장할 수 있음)"""
        try:
            paths = self.picker.choose()
        except PickerError as e:
            raise ApiError(str(e))
        plates, errors = [], []
        for p in paths:
            try:
                fid = self.uploads.link(p)
                src = MediaSource.open(self.uploads.path(fid))
            except (UploadError, UnsupportedMediaError, MediaReadError) as e:
                errors.append(f"{p.name}: {e}")
                continue
            plates.append(self._describe(fid, src) | {"name": p.name, "bytes": p.stat().st_size, "path": str(p)})
        return {"plates": plates, "errors": errors}

    def job(self, job_id: str) -> dict:
        j = self.jobs.get(job_id)
        if not j:
            raise ApiError("작업을 찾지 못했습니다", 404)
        return j.to_dict()

    def open_folder(self, body: dict) -> dict:
        path = self.results.folder(str(body.get("folder") or ""))
        current_platform().open_folder(path)   # 탐색기 · Finder
        return {"ok": True, "path": str(path)}


class RequestHandler(BaseHTTPRequestHandler):
    app: FaceGridApp = None  # serve() 에서 채운다
    server_version = "Griding"

    def log_message(self, *args):
        pass

    # ── 응답 ──
    def _send(self, status: int, body, ctype: str = "application/json; charset=utf-8", headers: dict | None = None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json_body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            data = json.loads(self.rfile.read(n))
        except ValueError:
            raise ApiError("잘못된 요청입니다")
        return data if isinstance(data, dict) else {}

    def _guard(self, fn):
        try:
            fn()
        except ApiError as e:
            self._send(e.status, {"error": str(e)}, headers={"Connection": "close"} if self.command == "POST" else None)
            if self.command == "POST":
                self.close_connection = True
        except ConnectionError:   # 브라우저가 영상 요청을 끊는 건 흔한 일 (WinError 10053 포함)
            pass
        except Exception as e:
            traceback.print_exc()
            try:
                self._send(500, {"error": f"앱 오류: {e}"})
            except Exception:
                pass

    # ── 주소 ──
    def _allowed(self) -> bool:
        """다른 웹사이트가 몰래 이 앱에 요청하거나(DNS rebinding · CSRF) 파일을 읽지 못하게"""
        port = self.server.server_address[1]
        ok_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if (self.headers.get("Host") or "") not in ok_hosts:
            return False
        origin = self.headers.get("Origin")
        return origin is None or origin in {f"http://{h}" for h in ok_hosts}

    def do_GET(self):
        if not self._allowed():
            return self._send(403, {"error": "이 PC 의 Griding 화면에서만 쓸 수 있습니다"})
        self._guard(self._get)

    def do_POST(self):
        if not self._allowed():
            self.close_connection = True
            return self._send(403, {"error": "이 PC 의 Griding 화면에서만 쓸 수 있습니다"}, headers={"Connection": "close"})
        self._guard(self._post)

    def _get(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            return self._send(200, self.app.ui(), CONTENT_TYPES[".html"])
        if path == "/api/info":
            return self._send(200, self.app.info())
        if path == "/i18n.json":   # 영어 화면용 사전
            f = config.APP_DIR / "i18n_en.json"
            return self._send_file(f) if f.is_file() else self._send(200, {"exact": {}, "templates": []})
        if path in ("/favicon.ico", "/icon.png", "/name.png"):   # 아이콘 · 화면 왼쪽 위 픽셀 이름
            icon = DesktopShortcut.ICON
            return self._send_file(icon if path.endswith(".ico") else icon.with_name("griding_text.png") if path == "/name.png" else icon.with_suffix(".png"))
        if path.startswith("/api/job/"):
            return self._send(200, self.app.job(path.rsplit("/", 1)[-1]))
        if path.startswith("/source/"):   # 고르기 창으로 넣은 원본을 화면에서 틀기
            try:
                return self._send_file(self.app.uploads.path(unquote(path[len("/source/"):])))
            except UploadError as e:
                raise ApiError(str(e), 404)
        if path.startswith("/files/"):
            f = self.app.results.resolve(unquote(path[len("/files/"):]))
            if not f:
                raise ApiError("파일이 없습니다", 404)
            return self._send_file(f)
        raise ApiError("없는 주소입니다", 404)

    def _post(self):
        path = urlparse(self.path).path
        if path == "/api/upload":
            name = unquote(self.headers.get("X-Filename") or "plate.mp4")
            return self._send(200, self.app.upload(self.rfile, int(self.headers.get("Content-Length") or 0), name))
        body = self._json_body()
        routes = {"/api/preview": lambda: self._send(200, self.app.preview(body), "image/jpeg"),
                  "/api/gallery": lambda: self._send(200, self.app.gallery_cards(body)),
                  "/api/run": lambda: self._send(200, self.app.run(body)),
                  "/api/scan": lambda: self._send(200, self.app.scan(body)),
                  "/api/pick": lambda: self._send(200, self.app.pick()),
                  "/api/open": lambda: self._send(200, self.app.open_folder(body))}
        if path not in routes:
            raise ApiError("없는 주소입니다", 404)
        routes[path]()

    def _send_file(self, path: Path):
        size = path.stat().st_size
        start, end = 0, size - 1
        rng = self.headers.get("Range")
        if rng and "," in rng:
            rng = None   # 여러 구간은 지원하지 않음 → 전체
        if rng and rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            try:
                start = int(a) if a else max(0, size - int(b))
                end = int(b) if (a and b) else size - 1
            except ValueError:
                start, end = 0, size - 1
            end = min(end, size - 1)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream"))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Content-Disposition", f"inline; filename*=UTF-8''{quote(path.name)}")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)


def free_port(host: str = config.HOST, first: int = config.FIRST_PORT) -> int:
    for p in range(first, first + 100):
        with socket.socket() as s:
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    raise RuntimeError("빈 포트를 찾지 못했습니다")


def make_server(app: FaceGridApp, port: int) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (RequestHandler,), {"app": app})
    srv = ThreadingHTTPServer((config.HOST, port), handler)
    srv.daemon_threads = True
    return srv


def main(open_browser: bool = True, make_shortcut: bool = True, first_port: int = config.FIRST_PORT) -> int:
    print(f"\n  Griding {__version__} · 시댄스용 얼굴 격자\n  " + "-" * 34)
    for folder in (config.RESULT_DIR, config.WORK_DIR):
        if not config.writable(folder):
            print(f"\n  이 폴더에 저장할 수 없습니다: {folder}\n  Griding 폴더를 바탕화면이나 문서로 옮겨서 다시 실행하세요.")
            return 1
    print("  모델 불러오는 중 · 이 PC 점검 중...\n", flush=True)
    pool, detector, load_error = DetectorPool(), None, None
    try:
        detector = pool.get("fast", "ui")
    except Exception as e:
        load_error = e
    report = SystemCheck.run(detector.device if detector else "")
    report.print()
    report.save(config.WORK_DIR / "pc_check.json")
    if report.fatal or detector is None:
        if load_error is not None and not report.fatal:
            print(f"  모델을 불러오지 못했습니다: {load_error}\n  Griding 폴더를 통째로 다시 복사해 주세요.")
        print("  위 [✗] 항목을 해결한 뒤 다시 실행하세요.")
        return 1
    uploads = UploadStore()
    uploads.cleanup()
    app = FaceGridApp(pool, ResultStore(), uploads)
    app.system = report.to_dict()
    port = free_port(first=first_port)   # 쓰는 중이면 다음 번호
    srv = make_server(app, port)
    url = f"http://{config.HOST}:{port}/"
    print(f"  처리 장치: {current_platform().accel_name if detector.device == 'GPU' else 'CPU'}")
    print(f"\n  준비 완료 → {url}")
    print("  (브라우저가 안 열리면 위 주소를 크롬이나 엣지 주소창에 붙여넣으세요)")
    print(f"  결과 저장 폴더: {config.RESULT_DIR}")
    if make_shortcut:   # 바로가기는 뒤에서 (PowerShell 이 느린 PC 에서도 화면이 먼저 뜨게)
        threading.Thread(target=lambda: (lambda n: n and print(f"  {n}", flush=True))(DesktopShortcut().ensure()), daemon=True).start()
    print("\n  작업하는 동안 이 창을 열어 두세요. 다 끝나면 창을 닫으면 됩니다.\n", flush=True)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
