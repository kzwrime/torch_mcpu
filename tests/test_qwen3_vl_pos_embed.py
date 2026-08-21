# Owner(s): ["module: PrivateUse1"]

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

import torch_mcpu  # noqa: F401


def _reference(embed, t, h, w, num_grid, merge_size):
    dtype = embed.dtype
    h_idxs = torch.linspace(0, num_grid - 1, h, dtype=torch.float32)
    w_idxs = torch.linspace(0, num_grid - 1, w, dtype=torch.float32)
    h_floor = h_idxs.to(torch.long)
    w_floor = w_idxs.to(torch.long)
    h_ceil = torch.clamp(h_floor + 1, max=num_grid - 1)
    w_ceil = torch.clamp(w_floor + 1, max=num_grid - 1)
    dh = h_idxs - h_floor
    dw = w_idxs - w_floor
    dh_grid, dw_grid = torch.meshgrid(dh, dw, indexing="ij")
    h_floor_grid, w_floor_grid = torch.meshgrid(h_floor, w_floor, indexing="ij")
    h_ceil_grid, w_ceil_grid = torch.meshgrid(h_ceil, w_ceil, indexing="ij")

    w11 = dh_grid * dw_grid
    w10 = dh_grid - w11
    w01 = dw_grid - w11
    w00 = torch.ones_like(dh_grid) - dh_grid - w01
    h_grid = torch.stack(
        [h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid]
    )
    w_grid = torch.stack(
        [w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid]
    )
    indices = (h_grid * num_grid + w_grid).reshape(4, -1)
    weights = torch.stack([w00, w01, w10, w11], dim=0).reshape(4, -1, 1)
    combined = (embed[indices] * weights.to(dtype=dtype)).sum(dim=0)
    combined = combined.reshape(
        h // merge_size,
        merge_size,
        w // merge_size,
        merge_size,
        embed.shape[1],
    )
    combined = combined.permute(0, 2, 1, 3, 4).reshape(1, -1, embed.shape[1])
    return combined.expand(t, -1, -1).reshape(-1, embed.shape[1]).to(dtype=dtype)


class TestQwen3VLPosEmbed(TestCase):
    def test_matches_reference(self):
        for dtype in (torch.float32, torch.bfloat16):
            for t, h, w in ((1, 2, 2), (1, 8, 12), (3, 6, 10)):
                torch.manual_seed(42)
                embed_cpu = torch.randn((16 * 16, 32), dtype=dtype) * 0.25
                embed = embed_cpu.to("mcpu")
                output = torch.empty((t * h * w, 32), dtype=dtype, device="mcpu")
                h_scale = (16 - 1) / (h - 1) if h > 1 else 0.0
                w_scale = (16 - 1) / (w - 1) if w > 1 else 0.0

                torch.ops.mcpu.vllm_qwen3_vl_bilinear_pos_embed(
                    embed,
                    output,
                    h,
                    w,
                    h_scale,
                    w_scale,
                    16,
                    2,
                    32,
                    32,
                )
                torch.mcpu.synchronize()

                expected = _reference(embed_cpu, t, h, w, 16, 2)
                self.assertEqual(output.shape, (t * h * w, 32))
                self.assertEqual(output.dtype, dtype)
                torch.testing.assert_close(
                    output.cpu(),
                    expected,
                    atol=5e-5 if dtype is torch.float32 else 1e-2,
                    rtol=1e-5 if dtype is torch.float32 else 1e-2,
                )

    def test_rejects_invalid_output_shape(self):
        embed = torch.randn((16 * 16, 8), dtype=torch.bfloat16, device="mcpu")
        output = torch.empty((7, 8), dtype=torch.bfloat16, device="mcpu")
        with self.assertRaisesRegex(RuntimeError, "multiple"):
            torch.ops.mcpu.vllm_qwen3_vl_bilinear_pos_embed(
                embed,
                output,
                2,
                2,
                15.0,
                15.0,
                16,
                2,
                8,
                8,
            )


if __name__ == "__main__":
    run_tests()
