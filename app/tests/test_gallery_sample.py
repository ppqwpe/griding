"""카드 예시 — 플레이트 없이도 예시 캐릭터 위에 그리고, 그림이 없으면 그려 만든 머리로"""
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from facegrid import gallery
from facegrid.gallery import SAMPLE_HEAD, Gallery, Sample
from facegrid.settings import GridSettings


class SampleTest(unittest.TestCase):
    def test_character_is_used(self):
        g = Gallery()
        m = Sample.MARGIN   # 캐릭터 둘레에 붙인 여백만큼 좌표가 밀림
        self.assertEqual(g._head.as_tuple(), (SAMPLE_HEAD.x0 + m, SAMPLE_HEAD.y0 + m, SAMPLE_HEAD.x1 + m, SAMPLE_HEAD.y1 + m))
        self.assertEqual(g._frame.shape[:2], (512 + 2 * m, 512 + 2 * m))
        f = g._sample_face()
        self.assertEqual(len(f.points), 478)
        x0, y0 = f.points.min(0)
        x1, y1 = f.points.max(0)
        h = g._head
        self.assertTrue(h.x0 <= x0 and x1 <= h.x1 and h.y0 <= y0 and y1 <= h.y1 + 5, "얼굴 점이 캐릭터 머리 안")
        for c, eye in ((468, (239 + m, 209 + m)), (473, (353 + m, 212 + m))):   # 눈동자 가운데 = 캐릭터 눈 가운데
            self.assertLess(np.hypot(*(f.points[c] - eye)), 3)

    def test_mask_cut_matches_picture(self):
        import cv2
        g = Gallery()
        for sample in (g._card, g._body):
            for size in ((240, 180), Gallery.BIG):
                c = g._cut_sample(sample, size)
                self.assertEqual(c.keep.shape, c.image.shape[:2])
                sh, sw = c.src   # 투명 자리를 그림과 같은 칸에서 같은 방법으로 줄였는지
                want = cv2.resize(sample.keep[c.y0:c.y0 + sh, c.x0:c.x0 + sw].astype(np.uint8) * 255, size,
                                  interpolation=cv2.INTER_AREA)
                self.assertTrue(np.array_equal(c.keep, want))
                bg = c.keep == 0   # 완전히 투명한 곳은 그림도 바탕색 (가장자리에 다른 색이 안 묻게)
                self.assertTrue((c.image[bg] == gallery.MATTE_BG).all())

    def test_combo_shows_whole_body_cards_show_head(self):
        g = Gallery()
        body = self._rgba(g.combo(GridSettings()))[..., 3] > 0
        head = self._rgba(g.layer_cards(GridSettings())[5]["image"])[..., 3] > 0
        rows_b, rows_h = np.flatnonzero(body.any(1)), np.flatnonzero(head.any(1))
        self.assertGreater(rows_b[-1] - rows_b[0], body.shape[0] * 0.75, "조합 예시는 전신 (더듬이 ~ 발)")
        self.assertLess(rows_b[-1], body.shape[0] - 2, "발이 잘리지 않음")
        for m in (body, head):   # 캐릭터가 가로 가운데
            cols = np.flatnonzero(m.any(0))
            self.assertLess(abs((cols[0] + cols[-1]) / 2 - m.shape[1] / 2), m.shape[1] * 0.06)

    def test_png_and_crisp(self):
        g = Gallery()
        url = g.combo(GridSettings())      # 카드 없이 바탕만 = 잘라 키운 캐릭터 그대로
        self.assertTrue(url.startswith("data:image/png;base64,"), "무손실 PNG")
        import base64, cv2
        img = cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8), 1)
        new = len(np.unique(img.reshape(-1, 3), axis=0))
        src = len(np.unique(g._body.frame.reshape(-1, 3), axis=0))
        self.assertLessEqual(new, src, "키울 때 픽셀을 각지게 — 원래 없던 섞인 색이 생기지 않음")

    @staticmethod
    def _rgba(url):
        import base64, cv2
        return cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",", 1)[1]), np.uint8), cv2.IMREAD_UNCHANGED)

    def test_no_background(self):
        g = Gallery()
        grid = self._rgba(g.layer_cards(GridSettings())[2]["image"])
        self.assertEqual(grid.shape[2], 4, "투명 PNG")
        a = grid[..., 3]
        self.assertEqual(a[0, 0], 0, "구석(바탕)은 투명")
        self.assertTrue((a == 255).mean() > 0.15, "캐릭터 · 격자는 남음")
        black = self._rgba(g.combo(GridSettings.from_dict({"recipe": ["points"], "base": "black"})))
        self.assertTrue(black.shape[2] == 3 or (black[..., 3] == 255).all(), "'검은 바탕' 은 바탕이 곧 그림이라 그대로")

    def test_lines_thick_enough(self):
        g = Gallery()
        lines = self._rgba(g.layer_cards(GridSettings())[3]["image"])[..., :3].astype(int)
        base = self._rgba(g.combo(GridSettings()))[..., :3].astype(int)
        drawn = (np.abs(lines - base).sum(2) > 0).astype(np.uint8)
        import cv2
        thin = cv2.erode(drawn, np.ones((2, 2), np.uint8))
        self.assertGreater(thin.sum(), drawn.sum() * 0.3, "선이 1px 가 아니라 두껍게 (깎아도 남음)")

    def test_cards_render(self):
        g = Gallery()
        s = GridSettings.from_dict({"recipe": ["cover_face", "points", "grid"]})
        cards = g.layer_cards(s)
        self.assertEqual(len(cards), 6)
        self.assertTrue(all(c["image"].startswith("data:image/png;base64,") for c in cards))
        self.assertTrue(g.combo(s).startswith("data:image/png"))

    def test_missing_image_falls_back(self):
        with mock.patch.object(gallery, "SAMPLE_IMAGE", Path("없는그림.png")):
            g = Gallery()
        self.assertEqual(g._frame.shape[:2], (gallery.THUMB[1], gallery.THUMB[0]))
        self.assertTrue(g.layer_cards(GridSettings())[0]["image"].startswith("data:image/png"))


if __name__ == "__main__":
    unittest.main()
