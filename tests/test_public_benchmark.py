import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trajectory_prediction.public_benchmark import history_inputs, marginal_errors, collate_official
from trajectory_prediction.models import integrate_acceleration


class PublicBenchmarkTests(unittest.TestCase):
    def test_actual_seconds_and_coordinate_roundtrip(self):
        times = torch.arange(4) * 5.
        p0 = torch.tensor([1000., -100., 30.])
        velocity = torch.tensor([3., 4., 2.])
        p = (p0 + times[:, None] * velocity)[None, None]
        features, context, edges = history_inputs(p, torch.ones(1, 1, dtype=torch.bool), 5)
        torch.testing.assert_close(features[0, 0, :, 12], torch.tensor([-1.5, -1., -.5, 0.]))
        recovered_v = torch.einsum("bnij,bnj->bni", context["R_local_to_shared"], context["v0_local"])
        torch.testing.assert_close(recovered_v[0, 0], velocity)
        torch.testing.assert_close(features[0, 0, :, 9], torch.ones(4))
        torch.testing.assert_close(features[0, 0, :, 8], torch.zeros(4))
        torch.testing.assert_close(features[0, 0, :, 10], torch.zeros(4))
        torch.testing.assert_close(features[0, 0, :, 11], torch.full((4,), .5))
        future_times = torch.arange(1, 25) * 5.
        local, _, _ = integrate_acceleration(torch.zeros(1, 1, 1, 24, 3), context["v0_local"], future_times)
        recovered = torch.einsum("bnij,bknhj->bknhi", context["R_local_to_shared"], local) + context["p0_shared"][:, None, :, None]
        torch.testing.assert_close(recovered[0, 0, 0], p[0, 0, -1] + future_times[:, None] * velocity)
        self.assertTrue(torch.isfinite(edges).all())

    def test_permutation_mask_and_relative_geometry(self):
        p = torch.zeros(1, 3, 4, 3)
        p[0, 0, :, 0] = torch.arange(4) * 5.
        p[0, 1] = p[0, 0] + torch.tensor([100., 20., 3.])
        p[0, 2] = float("nan")
        mask = torch.tensor([[True, True, False]])
        f, ctx, edge = history_inputs(p, mask, 5)
        self.assertTrue(torch.isfinite(f).all() and torch.isfinite(edge).all())
        torch.testing.assert_close(edge[0, 0, 1, :, :3], torch.tensor([.1, .02, .003]).expand(3, 3))
        perm = torch.tensor([2, 0, 1])
        fp, cp, ep = history_inputs(p[:, perm], mask[:, perm], 5)
        torch.testing.assert_close(fp, f[:, perm])
        torch.testing.assert_close(ep, edge[:, perm][:, :, perm])
        for key in ctx: torch.testing.assert_close(cp[key], ctx[key][:, perm])

    def test_km_to_metres_and_scene_membership(self):
        x = torch.ones(2, 4, 3); y = torch.ones(2, 24, 3) * 2
        obs, target, mask, ids = collate_official([(x, y, 17), (x[:1], y[:1], 19)])
        self.assertEqual(obs[0, 0, 0, 0].item(), 1000.)
        self.assertEqual(target[0, 0, 0, 0].item(), 2000.)
        self.assertEqual(mask.tolist(), [[True, True], [True, False]])
        self.assertEqual(ids.tolist(), [17, 19])

    def test_official_marginal_minima_are_independent(self):
        y = torch.zeros(1, 1, 2, 3)
        pred = torch.zeros(1, 2, 1, 2, 3)
        pred[0, 0, 0, :, 0] = torch.tensor([0., 4.])
        pred[0, 1, 0, :, 0] = torch.tensor([10., 0.])
        a, f = marginal_errors(pred, y, torch.ones(1, 1, dtype=torch.bool))
        self.assertEqual(a.item(), 2.)
        self.assertEqual(f.item(), 0.)


if __name__ == "__main__": unittest.main()
