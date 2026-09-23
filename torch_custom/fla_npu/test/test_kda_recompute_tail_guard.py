"""Host checks for KDA recompute tail padding.

Zero-padded gate rows are not a zero safe-gate step, so kg on the last
partial chunk has to be rebuilt from the last real gk token.
"""

import unittest
import warnings

import torch

from fla_npu.ops.ascendc._kda_policy import run_kda_recompute_with_tail_guard


class KdaRecomputeTailGuardTest(unittest.TestCase):
    def test_repair_kg_uses_last_real_token(self):
        seqlen = 70
        heads = 2
        dim = 4
        k = torch.ones(1, heads, seqlen, dim)
        seen = {}

        def launch(q, k_in, v, g, beta, a, cu_arg, indices_arg):
            padded = q.shape[2]
            seen["cu"] = cu_arg
            seen["indices"] = indices_arg
            row = torch.arange(padded, dtype=torch.float32).view(1, 1, padded, 1)
            gk = row.expand(1, heads, padded, dim).contiguous()
            wrong = torch.full((1, heads, padded, dim), -1.0)
            return gk, wrong, wrong, wrong, wrong.clone()

        q = torch.zeros(1, heads, seqlen, dim)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            gk, _w, _u, _qg, kg = run_kda_recompute_with_tail_guard(
                q, k, q, q, torch.zeros(1, heads, seqlen),
                torch.zeros(1, heads, seqlen, 64),
                launch, cu_seqlens=(0, seqlen), chunk_size=64, repair_kg=True,
            )

        self.assertEqual(seen["cu"], (0, 128))
        self.assertEqual(seen["indices"], (0, 0, 0, 1))
        self.assertEqual(gk.shape[2], seqlen)
        start = 64
        gk_last = gk[:, :, seqlen - 1:seqlen, :]
        expected = torch.exp2((gk_last - gk[:, :, start:seqlen, :]).clamp(-80, 80))
        self.assertTrue(torch.allclose(kg[:, :, start:seqlen, :].float(), expected))
        self.assertTrue(torch.equal(kg[:, :, :start, :], torch.full((1, heads, start, dim), -1.0)))


if __name__ == "__main__":
    unittest.main()
