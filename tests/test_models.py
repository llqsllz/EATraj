"""Forecast-model invariants, supervised losses and numerical integration."""

import itertools
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trajectory_prediction.models import (  # noqa: E402
    TrajectoryModel,
    assemble_independent_scenes,
    assemble_factorized_scenes,
    integrate_acceleration,
    trajectory_loss,
)


def inputs(device="cpu", batch=2, actors=3, steps=6):
    torch.manual_seed(17)
    x = torch.randn(batch, actors, steps, 13, device=device)
    x[..., 12] = torch.linspace(-1, 0, steps, device=device)
    mask = torch.ones(batch, actors, dtype=torch.bool, device=device)
    times = torch.tensor([0.2, 0.5, 1.0, 2.0], device=device)
    context = {
        "p0_shared": torch.randn(batch, actors, 3, device=device) * 10,
        "v0_local": torch.randn(batch, actors, 3, device=device) * 2,
        "R_local_to_shared": torch.eye(3, device=device).expand(batch, actors, 3, 3).clone(),
    }
    edges = torch.randn(batch, actors, actors, 3, 13, device=device)
    return x, mask, times, context, edges


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Tiny invariant checks should not spawn large CPU thread pools.
        torch.set_num_threads(1)

    def test_shapes_k1_independent_and_joint(self):
        x, mask, times, context, edges = inputs()
        for joint, modes, interaction in [(False, 1, False), (False, 5, True), (True, 5, True)]:
            model = TrajectoryModel(width=16, layers=1, modes=modes, joint=joint, interaction=interaction).eval()
            output = model(x, mask, times, context, edges)
            self.assertEqual(output["trajectories"].shape, (2, modes, 3, 4, 3))
            self.assertTrue(torch.isfinite(output["trajectories"]).all())
            if joint:
                self.assertEqual(output["scene_logits"].shape, (2, modes))
                self.assertNotIn("actor_logits", output)
            else:
                self.assertEqual(output["actor_logits"].shape, (2, 3, modes))
                self.assertNotIn("scene_logits", output)

    def test_ascent_inspired_decoders_and_pose_context(self):
        x, mask, times, context, edges = inputs()
        for decoder in ("ascent", "flight", "hybrid"):
            model = TrajectoryModel(
                width=16, layers=1, modes=3, interaction=True, joint=True,
                motion_decoder=decoder, ascent_pose=True,
            ).eval()
            output = model(x, mask, times, context, edges)
            self.assertEqual(output["trajectories"].shape, (2, 3, 3, 4, 3))
            self.assertTrue(torch.isfinite(output["trajectories"]).all())
            self.assertEqual(output["constraint_diagnostics"]["decoder"], decoder)

    def test_ascent_inspired_modules_preserve_actor_equivariance(self):
        x, mask, times, context, edges = inputs()
        model = TrajectoryModel(
            width=16, layers=1, modes=3, interaction=True, joint=True,
            motion_decoder="flight", ascent_pose=True,
        ).eval()
        order = [2, 0, 1]
        original = model(x, mask, times, context, edges)
        permuted = model(x[:, order], mask[:, order], times,
                         {key: value[:, order] for key, value in context.items()},
                         edges[:, order][:, :, order])
        torch.testing.assert_close(
            permuted["trajectories"], original["trajectories"][:, :, order],
            atol=2e-5, rtol=2e-5,
        )
        torch.testing.assert_close(permuted["scene_logits"], original["scene_logits"], atol=2e-5, rtol=2e-5)

    def test_ascent_decoder_rejects_physical_projection(self):
        with self.assertRaisesRegex(ValueError, "do not support constraints"):
            TrajectoryModel(motion_decoder="flight", constraints={"Vmax": 10, "Wmax": 3, "Amax": 2})

    def test_actor_permutation_equivariance_with_relations_and_joint_modes(self):
        x, mask, times, context, edges = inputs()
        model = TrajectoryModel(width=16, layers=1, modes=3, interaction=True, joint=True).eval()
        order = [2, 0, 1]
        original = model(x, mask, times, context, edges)
        permuted = model(x[:, order], mask[:, order], times,
                         {key: value[:, order] for key, value in context.items()},
                         edges[:, order][:, :, order])
        torch.testing.assert_close(permuted["trajectories"], original["trajectories"][:, :, order], atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(permuted["scene_logits"], original["scene_logits"], atol=2e-5, rtol=2e-5)

    def test_strict_independent_actor_is_invariant_to_other_actor(self):
        x, mask, times, context, edges = inputs()
        model = TrajectoryModel(width=16, layers=1, modes=3, interaction=False, joint=False).eval()
        original = model(x, mask, times, context, edges)
        changed_x = x.clone()
        changed_x[:, 1:] += 100
        changed_context = {key: value.clone() for key, value in context.items()}
        changed_context["p0_shared"][:, 1:] += 1000
        changed_context["v0_local"][:, 1:] += 50
        changed = model(changed_x, mask, times, changed_context, edges * 100)
        torch.testing.assert_close(changed["trajectories"][:, :, 0], original["trajectories"][:, :, 0], atol=0, rtol=0)
        torch.testing.assert_close(changed["actor_logits"][:, 0], original["actor_logits"][:, 0], atol=0, rtol=0)
        fewer_mask = mask.clone()
        fewer_mask[:, 1:] = False
        fewer = model(changed_x, fewer_mask, times, changed_context)
        torch.testing.assert_close(fewer["trajectories"][:, :, 0], original["trajectories"][:, :, 0], atol=1e-5, rtol=1e-5)

    def test_relation_edges_have_an_effect(self):
        x, mask, times, context, edges = inputs()
        model = TrajectoryModel(width=16, layers=1, modes=2, interaction=True).eval()
        original = model(x, mask, times, context, edges)
        changed_edges = edges.clone()
        changed_edges[:, 0, 1] += 5
        changed = model(x, mask, times, context, changed_edges)
        self.assertGreater(float((original["trajectories"][:, :, 0] - changed["trajectories"][:, :, 0]).abs().max()), 1e-5)

    def test_padding_nan_is_ignored_without_changing_real_outputs(self):
        x, mask, times, context, edges = inputs()
        model = TrajectoryModel(width=16, layers=1, modes=2, interaction=True, joint=True).eval()
        mask[:, -1] = False
        original = model(x, mask, times, context, edges)
        x[:, -1] = torch.nan
        for value in context.values():
            value[:, -1] = torch.nan
        edges[:, -1] = torch.nan
        edges[:, :, -1] = torch.nan
        changed = model(x, mask, times, context, edges)
        torch.testing.assert_close(changed["trajectories"], original["trajectories"])
        self.assertTrue(torch.isfinite(changed["trajectories"]).all())
        self.assertEqual(float(changed["trajectories"][:, :, -1].abs().max()), 0)

    def test_zero_acceleration_recovers_shared_coordinate_cv(self):
        x, mask, times, context, _ = inputs(batch=1, actors=2)
        model = TrajectoryModel(width=16, layers=1).eval()
        torch.nn.init.zeros_(model.acceleration_head[-1].weight)
        torch.nn.init.zeros_(model.acceleration_head[-1].bias)
        context["R_local_to_shared"][:] = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        context["v0_local"][:] = torch.tensor([1.0, 0.0, 2.0])
        output = model(x, mask, times, context)
        expected = context["p0_shared"][:, :, None] + times[None, None, :, None] * torch.tensor([0.0, 1.0, 2.0])
        torch.testing.assert_close(output["trajectories"][:, 0], expected)

    def test_invalid_forward_protocol_is_rejected(self):
        x, mask, times, context, edges = inputs()
        model = TrajectoryModel(width=16, layers=1, interaction=True)
        with self.assertRaisesRegex(ValueError, "history_edges"):
            model(x, mask, times, context)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            model(x, mask, times.flip(0), context, edges)
        bad_context = {key: value.clone() for key, value in context.items()}
        bad_context["R_local_to_shared"][0, 0] = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        with self.assertRaisesRegex(ValueError, "yaw-only"):
            model(x, mask, times, bad_context, edges)
        with self.assertRaisesRegex(ValueError, "at least one"):
            model(x, torch.zeros_like(mask), times, context, edges)

    def test_missing_official_mamba_dependency_has_explicit_error(self):
        with mock.patch.dict(sys.modules, {"mamba_ssm": None}):
            with self.assertRaisesRegex(ImportError, "official mamba_ssm.Mamba2"):
                TrajectoryModel(width=16, layers=1, temporal="mamba2")

    def _gradient_smoke(self, device, joint):
        x, mask, times, context, edges = inputs(device=device)
        mask[1, -1] = False
        model = TrajectoryModel(width=16, layers=1, modes=3, interaction=joint, joint=joint,
                                constraints={"Vmax": 10, "Wmax": 3, "Amax": 2}).to(device)
        output = model(x, mask, times, context, edges)
        target = torch.randn(2, 3, 4, 3, device=device)
        valid = mask[..., None].expand(2, 3, 4).clone()
        valid[0, 0, 1] = False
        target[~valid] = torch.nan
        loss, diagnostics = trajectory_loss(output, target, valid, mask, joint=joint)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)
        self.assertEqual(diagnostics["n_supervised_scenes"], 2)
        self.assertEqual(diagnostics["n_supervised_actors"], 5)

    def test_cpu_forward_backward_independent_and_joint(self):
        for joint in (False, True):
            with self.subTest(joint=joint):
                self._gradient_smoke("cpu", joint)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_forward_backward_independent_and_joint(self):
        for joint in (False, True):
            with self.subTest(joint=joint):
                self._gradient_smoke("cuda", joint)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_autocast_preserves_float32_physics(self):
        x, mask, times, context, edges = inputs(device="cuda")
        angle = torch.tensor(0.723, device="cuda")
        c, s = torch.cos(angle), torch.sin(angle)
        context["R_local_to_shared"][..., 0, 0] = c
        context["R_local_to_shared"][..., 0, 1] = -s
        context["R_local_to_shared"][..., 1, 0] = s
        context["R_local_to_shared"][..., 1, 1] = c
        model = TrajectoryModel(width=16, layers=1, modes=2, interaction=True, joint=True,
                                constraints={"Vmax": 10, "Wmax": 3, "Amax": 2}).cuda()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(x, mask, times, context, edges)
            loss, _ = trajectory_loss(output, torch.zeros(2, 3, 4, 3, device="cuda"),
                                      torch.ones(2, 3, 4, dtype=torch.bool, device="cuda"), mask)
        loss.backward()
        self.assertEqual(output["trajectories"].dtype, torch.float32)
        self.assertEqual(output["local_velocities"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


class IntegrationTests(unittest.TestCase):
    def test_vectorized_unconstrained_matches_float64_loop_and_gradients(self):
        torch.manual_seed(37)
        proposal = torch.randn(2, 3, 4, 100, 3, dtype=torch.float64, requires_grad=True)
        v0 = torch.randn(2, 4, 3, dtype=torch.float64) * 10
        times = (torch.rand(100, dtype=torch.float64) + 0.1).cumsum(0)
        actual_p, actual_v, _ = integrate_acceleration(proposal, v0, times)
        steps = torch.diff(times, prepend=times.new_zeros(1))
        velocity = v0[:, None].expand(2, 3, 4, 3)
        position = torch.zeros_like(velocity)
        positions, velocities = [], []
        for h, dt in enumerate(steps):
            next_velocity = velocity + 10.0 * proposal[..., h, :] * dt
            position = position + 0.5 * (velocity + next_velocity) * dt
            velocity = next_velocity
            positions.append(position)
            velocities.append(velocity)
        expected_p, expected_v = torch.stack(positions, dim=-2), torch.stack(velocities, dim=-2)
        torch.testing.assert_close(actual_p, expected_p, atol=1e-10, rtol=1e-12)
        torch.testing.assert_close(actual_v, expected_v, atol=1e-12, rtol=1e-12)
        actual_gradient = torch.autograd.grad(actual_p.square().mean() + actual_v.square().mean(), proposal, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected_p.square().mean() + expected_v.square().mean(), proposal)[0]
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-10, rtol=1e-12)

    def test_variable_step_zero_acceleration_is_cv(self):
        times = torch.tensor([0.1, 0.7, 2.0], dtype=torch.float64)
        v0 = torch.tensor([[[3.0, 4.0, 1.0]]], dtype=torch.float64)
        proposal = torch.zeros((1, 2, 1, 3, 3), dtype=torch.float64)
        position, velocity, _ = integrate_acceleration(proposal, v0, times)
        expected = v0[:, None, :, None] * times[None, None, None, :, None]
        torch.testing.assert_close(position, expected.expand_as(position), atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(velocity, v0[:, None, :, None].expand_as(velocity))

    def test_constant_acceleration_matches_analytic_position(self):
        times = torch.tensor([0.1, 0.7, 2.0], dtype=torch.float64)
        v0 = torch.tensor([[[3.0, 4.0, 1.0]]], dtype=torch.float64)
        proposal = torch.ones((1, 1, 1, 3, 3), dtype=torch.float64) * 0.2
        position, _, _ = integrate_acceleration(proposal, v0, times)
        expected = v0[:, None, :, None] * times[None, None, None, :, None] + times[None, None, None, :, None].square()
        torch.testing.assert_close(position, expected, atol=1e-14, rtol=1e-14)

    def test_constraints_bound_speed_vertical_and_effective_acceleration(self):
        torch.manual_seed(11)
        proposal = torch.randn(2, 3, 4, 40, 3, dtype=torch.float64) * 10
        v0 = torch.randn(2, 4, 3, dtype=torch.float64) * 20
        times = torch.tensor([0.1, 0.3, 0.2, 0.5] * 10, dtype=torch.float64).cumsum(0)
        bounds = {"Vmax": 20.0, "Wmax": 3.0, "Amax": 2.0}
        position, velocity, diag = integrate_acceleration(proposal, v0, times, bounds)
        vmax = diag["effective_speed_bound"][:, None, :, None]
        wmax = diag["effective_vertical_bound"][:, None, :, None]
        self.assertTrue((torch.linalg.vector_norm(velocity, dim=-1) <= vmax + 1e-10).all())
        self.assertTrue((velocity[..., 2].abs() <= wmax + 1e-10).all())
        previous = torch.cat((v0[:, None, :, None].expand(2, 3, 4, 1, 3), velocity[..., :-1, :]), dim=-2)
        dt = torch.diff(times, prepend=times.new_zeros(1))
        effective_acceleration = (velocity - previous) / dt[None, None, None, :, None]
        self.assertLessEqual(float(torch.linalg.vector_norm(effective_acceleration, dim=-1).max()), 2.0 + 1e-10)
        displacement = torch.diff(position, dim=-2, prepend=torch.zeros_like(position[..., :1, :]))
        torch.testing.assert_close(displacement, 0.5 * (velocity + previous) * dt[None, None, None, :, None], atol=1e-12, rtol=1e-12)
        self.assertTrue(diag["speed_bound_expanded"].any())
        self.assertTrue(diag["projection_active"].any())

    def test_constrained_small_signal_matches_unconstrained_physical_gain(self):
        """Constraint activation should add saturation, not change local gain."""
        times = torch.tensor([0.1, 0.2], dtype=torch.float64)
        proposal = torch.tensor(
            [[[[[1.0e-5, -2.0e-5, 3.0e-5], [0.0, 0.0, 0.0]]]]],
            dtype=torch.float64,
        )
        v0 = torch.zeros((1, 1, 3), dtype=torch.float64)
        _, velocity, _ = integrate_acceleration(
            proposal,
            v0,
            times,
            {"Vmax": 1_000.0, "Wmax": 1_000.0, "Amax": 2.0},
        )
        inferred_acceleration = velocity[..., 0, :] / times[0]
        expected_acceleration = 10.0 * proposal[..., 0, :]
        torch.testing.assert_close(
            inferred_acceleration,
            expected_acceleration,
            atol=1.0e-10,
            rtol=1.0e-4,
        )

    def test_tiny_acceleration_bound_remains_finite_and_respected(self):
        proposal = torch.tensor(
            [[[[[1.0, -2.0, 3.0], [0.1, 0.2, -0.3]]]]], dtype=torch.float64
        )
        v0 = torch.zeros((1, 1, 3), dtype=torch.float64)
        times = torch.tensor([0.1, 0.2], dtype=torch.float64)
        _, velocity, _ = integrate_acceleration(
            proposal,
            v0,
            times,
            {"Vmax": 1.0e-9, "Wmax": 1.0e-9, "Amax": 1.0e-12},
        )
        self.assertTrue(torch.isfinite(velocity).all())
        previous = torch.cat((v0[:, None, :, None], velocity[..., :-1, :]), dim=-2)
        dt = torch.diff(times, prepend=times.new_zeros(1))
        effective_acceleration = (velocity - previous) / dt[None, None, None, :, None]
        self.assertLessEqual(
            float(torch.linalg.vector_norm(effective_acceleration, dim=-1).max()),
            1.0e-12 + 1.0e-13,
        )

    def test_zero_bounds_and_zero_state_have_finite_gradients(self):
        proposal = torch.zeros((1, 1, 1, 2, 3), requires_grad=True)
        position, velocity, _ = integrate_acceleration(proposal, torch.zeros((1, 1, 3)), torch.tensor([0.2, 0.4]),
                                                      {"Vmax": 0, "Wmax": 0, "Amax": 0})
        (position.sum() + velocity.sum()).backward()
        self.assertEqual(float(position.abs().max()), 0)
        self.assertEqual(float(velocity.abs().max()), 0)
        self.assertTrue(torch.isfinite(proposal.grad).all())


class LossTests(unittest.TestCase):
    def test_joint_matching_differs_from_independent_actor_matching(self):
        prediction = torch.zeros(1, 2, 2, 2, 3, requires_grad=True)
        with torch.no_grad():
            prediction[0, 0, 1, :, 0] = 10
            prediction[0, 1, 0, :, 0] = 10
        output = {"trajectories": prediction, "scene_logits": torch.zeros(1, 2), "actor_logits": torch.zeros(1, 2, 2)}
        target, valid, actors = torch.zeros(1, 2, 2, 3), torch.ones(1, 2, 2, dtype=torch.bool), torch.ones(1, 2, dtype=torch.bool)
        joint_loss, joint_diag = trajectory_loss(output, target, valid, actors, score_weight=0, joint=True)
        independent_loss, independent_diag = trajectory_loss(output, target, valid, actors, score_weight=0, joint=False)
        self.assertAlmostEqual(float(joint_loss), 0.000833333333, places=9)
        self.assertEqual(float(independent_loss), 0)
        self.assertEqual(joint_diag["matched_ADE_m"], 5)
        self.assertEqual(independent_diag["matched_ADE_m"], 0)
        self.assertEqual(joint_diag["mode_counts"], [1, 0])
        self.assertEqual(independent_diag["mode_counts"], [1, 1])

    def test_equal_valid_time_actor_and_scene_weighting(self):
        prediction = torch.zeros(2, 1, 2, 2, 3, requires_grad=True)
        with torch.no_grad():
            prediction[0, 0, 0, :, 0] = 1
            prediction[0, 0, 1, :, 0] = 3
            prediction[1, 0, 0, :, 0] = 5
        valid = torch.tensor([[[True, True], [False, True]], [[True, True], [False, False]]])
        target = torch.zeros(2, 2, 2, 3)
        target[~valid] = torch.nan
        output = {"trajectories": prediction, "scene_logits": torch.zeros(2, 1)}
        loss, diag = trajectory_loss(output, target, valid, torch.ones(2, 2, dtype=torch.bool), joint=True)
        self.assertAlmostEqual(float(loss), 0.00025, places=9)
        self.assertEqual(diag["matched_ADE_m"], 3.5)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_all_missing_is_explicit_zero_with_safe_backward(self):
        prediction = torch.full((1, 1, 2, 2, 3), torch.nan, requires_grad=True)
        output = {"trajectories": prediction}
        loss, diag = trajectory_loss(output, torch.full((1, 2, 2, 3), torch.nan),
                                     torch.zeros(1, 2, 2, dtype=torch.bool), torch.ones(1, 2, dtype=torch.bool))
        loss.backward()
        self.assertEqual(float(loss), 0)
        self.assertEqual(diag["n_supervised_scenes"], 0)
        self.assertIsNone(diag["matched_ADE_m"])
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_nonfinite_supervised_truth_is_rejected(self):
        output = {"trajectories": torch.zeros(1, 1, 2, 2, 3), "scene_logits": torch.zeros(1, 1)}
        target = torch.zeros(1, 2, 2, 3)
        target[0, 0, 0] = torch.nan
        with self.assertRaisesRegex(ValueError, "target must be finite"):
            trajectory_loss(output, target, torch.ones(1, 2, 2, dtype=torch.bool), torch.ones(1, 2, dtype=torch.bool))


class AssemblyTests(unittest.TestCase):
    def test_factorized_top_product_combinations_match_exhaustive_reference(self):
        trajectory = np.zeros((2, 2, 2, 1, 3))
        for g in range(2):
            for l in range(2):
                trajectory[g, l, :, :, 0] = 10 * g + l
        scene_logits = np.array([2.0, 0.5])
        local_logits = np.array([
            [[2.0, 0.0], [0.0, 1.0]],
            [[0.5, 1.5], [1.0, 0.2]],
        ])
        assembled, scores = assemble_factorized_scenes(trajectory, scene_logits, local_logits, scene_modes=5)
        scene_lp = scene_logits - np.logaddexp.reduce(scene_logits)
        local_lp = local_logits - np.logaddexp.reduce(local_logits, axis=-1, keepdims=True)
        reference = []
        for g in range(2):
            for l0 in range(2):
                for l1 in range(2):
                    reference.append((scene_lp[g] + local_lp[g, 0, l0] + local_lp[g, 1, l1], g, l0, l1))
        reference.sort(reverse=True)
        self.assertEqual(assembled.shape, (5, 2, 1, 3))
        np.testing.assert_allclose(scores, [x[0] for x in reference[:5]])
        for actual, (_, g, l0, l1) in zip(assembled, reference[:5]):
            self.assertEqual(tuple(actual[:, 0, 0].astype(int)), (10 * g + l0, 10 * g + l1))

    def test_heap_top_product_combinations_match_exhaustive_reference(self):
        trajectory = np.zeros((3, 2, 1, 3))
        for mode in range(3):
            trajectory[mode, :, :, 0] = mode
        logits = np.array([[2., 1., -1.], [0.5, 1.3, -0.2]])
        assembled, scores = assemble_independent_scenes(trajectory, logits, scene_modes=5)
        log_probs = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
        reference = sorted([(sum(log_probs[i, m] for i, m in enumerate(combo)), combo)
                            for combo in itertools.product(range(3), repeat=2)], reverse=True)[:5]
        self.assertEqual(assembled.shape, (5, 2, 1, 3))
        np.testing.assert_allclose(scores, [pair[0] for pair in reference])
        for actual, (_, expected) in zip(assembled, reference):
            self.assertEqual(tuple(actual[:, 0, 0].astype(int)), expected)
        self.assertEqual(len({tuple(scene[:, 0, 0]) for scene in assembled}), 5)

    def test_k1_is_not_duplicated(self):
        trajectory = np.arange(18).reshape(1, 2, 3, 3)
        assembled, scores = assemble_independent_scenes(trajectory, np.zeros((2, 1)), scene_modes=5)
        np.testing.assert_array_equal(assembled, trajectory)
        np.testing.assert_array_equal(scores, [0])


if __name__ == "__main__":
    unittest.main()
