"""예시 캐릭터 얼굴 점 만들기 — 사람 얼굴 점 478개를 캐릭터 눈 · 입 · 머리 윤곽에 맞춰 옮긴다

  model/sample_face.json (사람 얼굴 0~1 배치) → model/sample_character_face.json (sample_character.png 의 픽셀 좌표)

  윤곽 · 눈 · 눈동자   부위 가운데에서 본 각도를 그대로 두고 캐릭터의 타원 위에 딱 맞게 (동그란 눈 · 동그란 머리)
  눈썹 · 입 · 코       부위 크기만 캐릭터 크기로 줄이고 늘림
  나머지 안쪽 점       위 부위 점들을 고정점으로 한 매끈한 휨(thin-plate spline)으로 같이 옮김

캐릭터를 바꾸면 아래 CHARACTER 숫자(그림 안 픽셀 위치)만 고치고 다시 실행:
  app\\python\\python.exe app\\tools\\make_character_face.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP))
from facegrid.facemesh import FaceLines  # noqa: E402

FACE_ASPECT = 0.8   # 사람 얼굴 가로 ÷ 세로 (0~1 배치를 펼 때)

# sample_character.png (512×512) 안의 위치 — (가운데 x, y, 가로 반지름, 세로 반지름)
CHARACTER = {
    "oval": (301, 190, 97, 76),         # 머리 윤곽 (더듬이 빼고, 조금 안쪽)
    "eye_l": (239, 209, 27, 25),        # 화면 왼쪽 눈 둘레 (검은 눈보다 살짝 크게)
    "eye_r": (353, 212, 28, 25),
    "iris_l": (239, 209, 13, 13),
    "iris_r": (353, 212, 13, 13),
    "brow_l": (239, 181, 14, 3),        # 눈 위 짧은 눈썹
    "brow_r": (353, 184, 14, 3),
    "lips": (294, 241, 12, 6),          # 작은 입
    "nose": (294, 228, 5, 4),
}
RING = ("FACE_OVAL", "LEFT_EYE", "RIGHT_EYE", "LEFT_IRIS", "RIGHT_IRIS")


def part_indices(conn: dict, part: str) -> np.ndarray:
    return np.unique(np.array(conn[part]).ravel())


def side_key(base: str, idx: np.ndarray, pts: np.ndarray, mid_x: float) -> str:
    """사람 점에서 이 부위가 화면 왼쪽인지 오른쪽인지 보고 캐릭터 부위 이름을 고른다"""
    return f"{base}_{'l' if pts[idx, 0].mean() < mid_x else 'r'}"


def to_ellipse(src: np.ndarray, target) -> np.ndarray:
    """부위 가운데에서 본 각도를 유지한 채 타원 위로 — 모든 점이 테두리에 (찌그러진 윤곽 없이)"""
    cx, cy, rx, ry = target
    c = src.mean(0)
    v = src - c
    r = np.hypot(v[:, 0], v[:, 1])
    rmax = r.max() or 1.0
    ang = np.arctan2(v[:, 1], v[:, 0])
    return np.stack([cx + np.cos(ang) * rx, cy + np.sin(ang) * ry], 1)


def to_box(src: np.ndarray, target) -> np.ndarray:
    """부위의 가로 · 세로 크기를 캐릭터 크기로"""
    cx, cy, rx, ry = target
    lo, hi = src.min(0), src.max(0)
    half = np.maximum((hi - lo) / 2, 1e-6)
    return (src - (lo + hi) / 2) / half * np.array([rx, ry]) + np.array([cx, cy])


def build(human01: np.ndarray) -> np.ndarray:
    human = human01 * np.array([FACE_ASPECT, 1.0])      # 사람 얼굴 비율로 편 좌표
    conn = FaceLines.connections()
    mid_x = human[[10, 1, 152], 0].mean()
    out = np.full_like(human, np.nan)
    for part in ("FACE_OVAL", "LEFT_EYE", "RIGHT_EYE", "LEFT_IRIS", "RIGHT_IRIS", "LEFT_EYEBROW", "RIGHT_EYEBROW", "LIPS", "NOSE"):
        idx = part_indices(conn, part)
        if part == "FACE_OVAL":
            key = "oval"
        elif part in ("LIPS", "NOSE"):
            key = part.lower()
        else:
            key = side_key({"EYE": "eye", "IRIS": "iris", "EYEBROW": "brow"}[part.split("_", 1)[1]], idx, human, mid_x)
        place = to_ellipse if part in RING else to_box
        out[idx] = place(human[idx], CHARACTER[key])
    for c in (468, 473):   # 눈동자 가운데 점 (연결선에는 없음) → 캐릭터 눈 가운데
        cx, cy, _, _ = CHARACTER["iris_l" if human[c, 0] < mid_x else "iris_r"]
        out[c] = (cx, cy)
    known = ~np.isnan(out[:, 0])
    rest = ~known
    out[rest] = ThinPlate(human[known], out[known]).apply(human[rest])
    cx, cy, rx, ry = CHARACTER["oval"]   # 머리 밖으로 나간 점은 윤곽 안으로
    v = (out - [cx, cy]) / [rx, ry]
    r = np.hypot(v[:, 0], v[:, 1])
    over = r > 0.98
    out[over] = np.array([cx, cy]) + v[over] / r[over, None] * 0.98 * np.array([rx, ry])
    return out


class ThinPlate:
    """고정점 src → dst 로 평면을 매끈하게 휘는 변환 (고정점 사이는 고무판처럼 따라감)"""

    def __init__(self, src: np.ndarray, dst: np.ndarray, smooth: float = 1e-3):
        self._src = src
        n = len(src)
        k = self._kernel(src, src) + np.eye(n) * smooth
        p = np.hstack([np.ones((n, 1)), src])
        a = np.zeros((n + 3, n + 3))
        a[:n, :n], a[:n, n:], a[n:, :n] = k, p, p.T
        b = np.vstack([dst, np.zeros((3, 2))])
        self._coef = np.linalg.solve(a, b)

    @staticmethod
    def _kernel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        r2 = ((a[:, None] - b[None]) ** 2).sum(2)
        return np.where(r2 > 0, r2 * np.log(np.maximum(r2, 1e-12)) / 2, 0.0)

    def apply(self, pts: np.ndarray) -> np.ndarray:
        n = len(self._src)
        k = self._kernel(pts, self._src)
        p = np.hstack([np.ones((len(pts), 1)), pts])
        return k @ self._coef[:n] + p @ self._coef[n:]


def main() -> None:
    human = np.array(json.loads((APP / "model" / "sample_face.json").read_text("utf-8"))["points"], np.float64)
    pts = build(human)
    note = "예시 캐릭터(sample_character.png) 위 얼굴 점 478개 — tools/make_character_face.py 로 만듦"
    (APP / "model" / "sample_character_face.json").write_text(
        json.dumps({"note": note, "points": np.round(pts, 2).tolist()}, ensure_ascii=False), "utf-8")
    print("저장:", APP / "model" / "sample_character_face.json")


if __name__ == "__main__":
    main()
