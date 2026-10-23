"""CPU Torch tests of own synthetic preview tensors; no ComfyUI/model imports."""
import struct
import unittest

import torch
from sampling_preview import FACTORS, BIAS, render, _resize_axis


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


class TensorPreviewTests(unittest.TestCase):
    def test_native_scalar_order_pixels_input_and_rng_are_unchanged(self):
        x = ((torch.arange(64 * 3 * 5, dtype=torch.float32) % 37) - 18).reshape(1, 64, 3, 5) / 7
        before, rng = x.clone(), torch.get_rng_state().clone()
        width, height, pixels = render(torch, x, 64)
        expected = bytearray()
        for y in range(3):
            for z in range(5):
                for rgb in range(3):
                    value = 0.
                    for channel in range(64):
                        value = f32(value + f32(x[0, channel, y, z].item() * f32(FACTORS[channel][rgb])))
                    value = f32(value + f32(BIAS[rgb]))
                    value = f32(f32(value * .5) + .5)
                    expected.append(int(f32(min(1., max(0., value)) * 255.)))
        self.assertEqual((width, height), (5, 3))
        self.assertEqual(pixels, bytes(expected))
        self.assertTrue(torch.equal(x, before))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_triangle_separable_rounding_and_bounded_output(self):
        image = torch.tensor([[[0., 1., 2.], [100., 101., 102.]],
                              [[200., 201., 202.], [255., 254., 253.]]])
        result = _resize_axis(torch, _resize_axis(torch, image, 0, 1), 1, 1)
        self.assertEqual(result.flatten().tolist(), [138.75, 139.25, 139.75])
        x = torch.arange(64 * 3 * 5, dtype=torch.float32).reshape(1, 64, 3, 5)
        self.assertEqual(render(torch, x, 2)[:2], (2, 1))
        self.assertEqual(len(render(torch, x, 2)[2]), 6)
        with self.assertRaises(ValueError):
            render(torch, torch.zeros(1, 24, 2, 2), 2)


if __name__ == "__main__":
    unittest.main()
