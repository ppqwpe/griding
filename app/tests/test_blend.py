"""그림이 있는 곳만 섞어도 전체를 섞을 때와 픽셀 하나까지 같은지"""
import time
import unittest

import numpy as np

from facegrid.blend import bounds, over_color
from facegrid.layers import LAYERS
from facegrid.settings import GridSettings
from test_combo import scene


def full_blend(canvas, alpha, color):
    a = alpha.astype(np.float32)
    if alpha.dtype == np.uint8:
        a /= 255.0
    a = a[..., None]
    return (canvas.astype(np.float32) * (1 - a) + np.asarray(color, np.float32) * a).astype(np.uint8)


def full_isolate(layer, sc):
    """고치기 전 방식 (화면 전체 계산)"""
    black = layer.apply(np.zeros_like(sc.original), sc).astype(np.float32)
    white = layer.apply(np.full_like(sc.original, 255), sc).astype(np.float32)
    a = np.clip(1.0 - (white - black).mean(axis=2) / 255.0, 0.0, 1.0)
    color = np.clip(black / np.maximum(a, 1e-3)[..., None], 0, 255)
    color[a < 1 / 255] = 0
    return np.dstack([color, a * 255]).round().astype(np.uint8)


class BlendTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.img = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
        self.a8 = np.zeros((1080, 1920), np.uint8)
        self.a8[300:500, 700:900] = rng.integers(0, 256, (200, 200), dtype=np.uint8)
        self.a8[100:120, 1500:1700] = 255

    def test_same_as_full_frame(self):
        for alpha in (self.a8, self.a8.astype(np.float32) / 255 * 0.7):
            with self.subTest(dtype=alpha.dtype):
                np.testing.assert_array_equal(over_color(self.img, alpha, (30, 40, 220)), full_blend(self.img, alpha, (30, 40, 220)))

    def test_empty_alpha_and_no_mutation(self):
        before = self.img.copy()
        out = over_color(self.img, np.zeros((1080, 1920), np.float32), (0, 0, 0))
        np.testing.assert_array_equal(out, before)
        over_color(self.img, self.a8, (0, 0, 0))
        np.testing.assert_array_equal(self.img, before, "입력 그림은 바꾸지 않는다")
        self.assertIsNone(bounds(np.zeros((4, 4), bool)))
        self.assertEqual(bounds(self.a8), (100, 500, 700, 1700))

    def test_faster(self):
        t = time.perf_counter(); [full_blend(self.img, self.a8, (1, 2, 3)) for _ in range(5)]; slow = time.perf_counter() - t
        t = time.perf_counter(); [over_color(self.img, self.a8, (1, 2, 3)) for _ in range(5)]; fast = time.perf_counter() - t
        self.assertLess(fast, slow)

    def test_isolate_same_as_before(self):
        for color in ("#000000", "#FFFFFF"):
            s = GridSettings.from_dict({"cover_color": color, "fill_alpha": 0.3})
            for key, layer in LAYERS.items():
                with self.subTest(layer=key, cover=color):
                    np.testing.assert_array_equal(layer.isolate(scene(s)), full_isolate(layer, scene(s)))


if __name__ == "__main__":
    unittest.main()
