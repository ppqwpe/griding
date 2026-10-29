"""Griding 로고 만들기 — 앱 아이콘(ico · png)과 따로 쓸 로고 PNG 묶음

  app/icon/griding.ico · griding.png          앱 · 바로가기 · 브라우저 탭
  로고/griding_icon_{1024,512,256,128,64,32}.png   아이콘만 (투명 바탕)
  로고/griding_logo_dark.png · _light.png         아이콘 + 픽셀 글자 (어두운 · 밝은 바탕용, 투명)
  로고/griding_text_dark.png · _light.png         픽셀 글자만 — i 의 점 두 개는 격자와 같은 빨강
  app/icon/griding_text.png                       화면 왼쪽 위 이름

재료: app/model/sample_character_head.png (캐릭터 머리, 목 아래 없음 · 초록 바탕)
  app\\python\\python.exe app\\tools\\make_logo.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

APP = Path(__file__).resolve().parent.parent
ROOT = APP.parent
HEAD = APP / "model" / "sample_character_head.png"
NAME = "Griding"
TILE, RED = (21, 24, 29, 255), (220, 40, 30, 255)
S = 1024

# 픽셀 글자 — 한 줄에 한 행, '#' = 글자색 · 'o' = 빨강(i 의 점) · '.' = 빈칸. 높이 9칸 (7칸 + 아래로 내려가는 2칸)
GLYPHS = {
    "G": [".###.", "#...#", "#....", "#.###", "#...#", "#...#", ".###.", ".....", "....."],
    "r": [".....", ".....", "#.##.", "##..#", "#....", "#....", "#....", ".....", "....."],
    "i": ["o", ".", "#", "#", "#", "#", "#", ".", "."],
    "d": ["....#", "....#", ".####", "#...#", "#...#", "#...#", ".####", ".....", "....."],
    "n": [".....", ".....", "####.", "#...#", "#...#", "#...#", "#...#", ".....", "....."],
    "g": [".....", ".....", ".####", "#...#", "#...#", "#...#", ".####", "....#", "####."],
}


class PixelText:
    """GLYPHS 로 쓴 각진 글자 — 한 칸 = cell px 정사각형"""

    def __init__(self, text: str = NAME, gap: int = 1):
        self.text, self.gap = text, gap

    @property
    def grid(self) -> list[str]:
        rows = [""] * 9
        for i, ch in enumerate(self.text):
            g = GLYPHS[ch]
            for r in range(9):
                rows[r] += g[r] + ("." * self.gap if i < len(self.text) - 1 else "")
        return rows

    def render(self, cell: int, color, accent=RED) -> Image.Image:
        rows = self.grid
        img = Image.new("RGBA", (len(rows[0]) * cell, len(rows) * cell), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        for y, row in enumerate(rows):
            for x, c in enumerate(row):
                if c != ".":
                    d.rectangle((x * cell, y * cell, (x + 1) * cell - 1, (y + 1) * cell - 1), fill=accent if c == "o" else color)
        return img


class Logo:
    def __init__(self, head_path: Path = HEAD):
        self.head = self._cut_out(Image.open(head_path).convert("RGB"))

    @staticmethod
    def _cut_out(img: Image.Image) -> Image.Image:
        """바탕 초록을 투명으로 · 목 아래(폭이 머리의 35% 아래로 좁아지는 곳)를 자른다"""
        a = np.asarray(img).astype(int)
        bg = np.median(a.reshape(-1, 3), 0)
        alpha = np.clip((np.abs(a - bg).sum(2) - 40) * 6, 0, 255).astype(np.uint8)
        im = Image.fromarray(np.dstack([a.astype(np.uint8), alpha]), "RGBA")
        im = im.crop(im.getbbox())
        w = (np.asarray(im.getchannel("A")) > 128).sum(1)
        peak = int(w.argmax())
        cut = next((y for y in range(peak, len(w)) if w[y] < w.max() * 0.35), len(w))
        im = im.crop((0, 0, im.size[0], cut))
        return im.crop(im.getbbox())

    @staticmethod
    def _rounded_mask(size: int) -> Image.Image:
        m = Image.new("L", (size, size), 0)
        ImageDraw.Draw(m).rounded_rectangle((0, 0, size - 1, size - 1), radius=int(size * 0.22), fill=255)
        return m

    def icon(self) -> Image.Image:
        """어두운 둥근 네모 + 머리 + 빨간 4×4 격자 (머리 둥근 부분만)"""
        img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        ImageDraw.Draw(img).rounded_rectangle((0, 0, S - 1, S - 1), radius=int(S * 0.22), fill=TILE)
        k = S * 0.78 / max(self.head.size)
        head = self.head.resize((int(self.head.size[0] * k), int(self.head.size[1] * k)), Image.NEAREST)   # 픽셀 그림이라 각지게
        hx, hy = (S - head.size[0]) // 2, int(S * 0.5 - head.size[1] / 2)
        img.alpha_composite(head, (hx, hy))
        a = np.asarray(head.getchannel("A")) > 128
        rows = a.sum(1)
        wide = np.flatnonzero(rows > rows.max() * 0.55)      # 더듬이(가는 줄)는 빼고 머리만
        cols = np.flatnonzero(a[wide[0]:wide[-1]].any(0))
        x0, y0, x1, y1 = hx + cols[0], hy + wide[0], hx + cols[-1], hy + wide[-1]
        g = ImageDraw.Draw(img)
        for i in range(5):
            x = x0 + (x1 - x0) * i / 4
            y = y0 + (y1 - y0) * i / 4
            g.line((x, y0, x, y1), fill=RED, width=int(S * 0.03))
            g.line((x0, y, x1, y), fill=RED, width=int(S * 0.03))
        img.putalpha(Image.composite(img.getchannel("A"), self._rounded_mask(S), self._rounded_mask(S)))
        return img

    @staticmethod
    def wordmark(icon: Image.Image, text_color) -> Image.Image:
        """아이콘 + 픽셀 글자 가로형 (투명 바탕) — 글자 7칸 높이를 아이콘 높이의 절반쯤으로"""
        h = 512
        ic = icon.resize((h, h), Image.LANCZOS)
        cell = h // 14
        text = PixelText().render(cell, text_color)
        gap = int(h * 0.14)
        out = Image.new("RGBA", (h + gap + text.size[0] + cell, h), (0, 0, 0, 0))
        out.alpha_composite(ic, (0, 0))
        out.alpha_composite(text, (h + gap, (h - 7 * cell) // 2))   # 글자 몸통(7칸)을 세로 가운데에
        return out

    def save_all(self) -> list[Path]:
        icon = self.icon()
        out = []
        app_icon = APP / "icon"
        icon.resize((256, 256), Image.LANCZOS).save(app_icon / "griding.png")
        icon.save(app_icon / "griding.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
        out += [app_icon / "griding.png", app_icon / "griding.ico"]
        folder = ROOT / "로고"
        folder.mkdir(exist_ok=True)
        for size in (1024, 512, 256, 128, 64, 32):
            p = folder / f"griding_icon_{size}.png"
            (icon if size == S else icon.resize((size, size), Image.LANCZOS)).save(p)
            out.append(p)
        for name, color in (("dark", (237, 237, 237, 255)), ("light", (29, 29, 31, 255))):
            for kind, img in (("logo", self.wordmark(icon, color)), ("text", PixelText().render(24, color))):
                p = folder / f"griding_{kind}_{name}.png"
                img.save(p)
                out.append(p)
        p = app_icon / "griding_text.png"
        PixelText().render(3, (237, 237, 237, 255)).save(p)     # 화면 왼쪽 위 (어두운 화면 · 27px = 9칸 × 3)
        out.append(p)
        return out


if __name__ == "__main__":
    for p in Logo().save_all():
        print(p)
