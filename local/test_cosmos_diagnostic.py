"""Download-free checks for the Cosmos numerical comparison harness."""
import json
import unittest
from unittest.mock import patch
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
import numpy as np
from diffusers.models.transformers import transformer_cosmos
from diffusers import FlowMatchEulerDiscreteScheduler

from diagnose_cosmos_video import (capture_native_attention, differences, frame_differences,
                                   denoised_prediction_adapter, diagnostic_geometry, parse_arguments,
                                   contract_metadata, create_attention_bridges, attention_evidence,
                                   AuditedAttention, AuditedBlasAttention, main)
from cosmos_cumetal import CuMetalAttention


class DiagnosticTest(unittest.TestCase):
    def test_exact_comparison(self):
        x = torch.tensor([1.0, -2.0])
        result = differences(x, x.clone())
        self.assertTrue(result["finite"])
        self.assertEqual(result["max_abs"], 0)
        self.assertEqual(result["relative_rms"], 0)

    def test_known_error_and_zero_reference(self):
        result = differences(torch.ones(4), torch.zeros(4))
        self.assertEqual(result["max_abs"], 1)
        self.assertEqual(result["rms"], 1)
        self.assertTrue(result["relative_rms"] > 1e10)

    def test_nonfinite_detection(self):
        self.assertFalse(differences(torch.tensor([float("nan")]), torch.ones(1))["finite"])

    def test_uint8_frame_subtraction_does_not_wrap(self):
        result = frame_differences(np.array([0, 255], dtype=np.uint8),
                                   np.array([255, 0], dtype=np.uint8))
        self.assertEqual(result, {"max_abs": 255.0, "mean_abs": 255.0})

    def test_dispatch_restored_after_exception(self):
        original = transformer_cosmos.dispatch_attention_fn
        x = torch.ones(1, 2, 1, 4)
        records = []
        with patch.object(transformer_cosmos, "dispatch_attention_fn", return_value=x) as native:
            with self.assertRaisesRegex(RuntimeError, "test"), capture_native_attention(records):
                transformer_cosmos.dispatch_attention_fn(x, x, x)
                raise RuntimeError("test")
            self.assertIs(transformer_cosmos.dispatch_attention_fn, native)
        self.assertIs(transformer_cosmos.dispatch_attention_fn, original)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["device"], "cpu")

    def test_denoised_euler_analytic_step_and_restoration(self):
        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(sigmas=[2.0, 1.0])
        original = scheduler.step
        with denoised_prediction_adapter(scheduler):
            result = scheduler.step(torch.tensor([1.0]), scheduler.timesteps[0],
                                    torch.tensor([4.0]), return_dict=False)[0]
        torch.testing.assert_close(result, torch.tensor([2.5]))
        self.assertEqual(scheduler.step, original)

    def test_adapter_refuses_zero_sigma_and_restores(self):
        scheduler = FlowMatchEulerDiscreteScheduler()
        scheduler.set_timesteps(sigmas=[0.0])
        original = scheduler.step
        with self.assertRaisesRegex(ValueError, "positive sigma"), denoised_prediction_adapter(scheduler):
            scheduler.step(torch.ones(1), scheduler.timesteps[0], torch.ones(1))
        self.assertEqual(scheduler.step, original)

    def test_native_temporal_rope_matches_unscaled_positions(self):
        rope = transformer_cosmos.CosmosRotaryPosEmbed(
            hidden_size=128, max_size=(128, 240, 240), rope_scale=(1.0, 3.0, 3.0))
        states = torch.zeros(1, 16, 3, 8, 8)
        native = rope(states, fps=None)
        at_24 = rope(states, fps=24)
        at_16 = rope(states, fps=16)
        for expected, unscaled, modulated in zip(native, at_24, at_16, strict=True):
            torch.testing.assert_close(expected, unscaled, rtol=0, atol=0)
            self.assertGreater(float((expected - modulated).abs().max()), 0.1)
            # Temporal frame zero is unchanged; only later positions differ.
            torch.testing.assert_close(expected[:16], modulated[:16], rtol=0, atol=0)

    def test_geometry_records_actual_dense_worker_budget(self):
        small = diagnostic_geometry(9, 192, 320)
        self.assertEqual(small["self_attention_tokens"], 720)
        self.assertEqual(small["dense_score_bytes"], 16 * 720 * 720 * 4)
        self.assertTrue(small["cumetal_worker_supported"])
        full = diagnostic_geometry(93, 704, 1280)
        self.assertEqual(full["latent_frames"], 24)
        self.assertEqual(full["self_attention_tokens"], 84480)
        self.assertFalse(full["cumetal_worker_supported"])
        self.assertFalse(diagnostic_geometry(33, 192, 320)["cumetal_worker_supported"])
        tiled = diagnostic_geometry(9, 704, 1280)
        self.assertFalse(tiled["cumetal_worker_supported"])
        self.assertTrue(tiled["cumetal_blas_worker_supported"])
        self.assertEqual(tiled["cumetal_blas_plans"]["self_attention"]["tiles"], 42)
        self.assertFalse(full["cumetal_blas_worker_supported"])

    def test_mps_only_has_no_binary_dependency_and_zero_cuda_counters(self):
        with patch("diagnose_cosmos_video.AuditedAttention") as legacy, \
                patch("diagnose_cosmos_video.AuditedBlasAttention") as blas:
            self.assertEqual(create_attention_bridges(["mps"]), {})
            legacy.assert_not_called()
            blas.assert_not_called()
            bridges = create_attention_bridges(["mps", "cumetal-blas"])
            self.assertEqual(bridges, {"cumetal-blas": blas.return_value})
            legacy.assert_not_called()
            blas.assert_called_once_with()
        evidence = attention_evidence({})
        self.assertEqual(evidence["cumetal_calls"], 0)
        self.assertEqual(evidence["apple_gpu_launches"], 0)
        self.assertEqual(evidence["sampled_attention"], [])

    def test_audit_rejects_nondefault_options_before_worker(self):
        q = torch.ones(1, 1, 2, 4)
        with TemporaryDirectory() as directory:
            binary = Path(directory) / "fake-worker"
            binary.touch()
            for bridge_class in (AuditedAttention, AuditedBlasAttention):
                bridge = bridge_class(binary)
                for options in ({"is_causal": True}, {"scale": 0.25}, {"dropout_p": 0.1},
                                {"attn_mask": torch.ones(2, 2)}, {"enable_gqa": True}):
                    with self.subTest(bridge=bridge_class, options=options), \
                            patch.object(CuMetalAttention, "__call__") as worker, \
                            self.assertRaisesRegex(NotImplementedError, "audit only"):
                        bridge(q, q, q, **options)
                    worker.assert_not_called()
                self.assertEqual(bridge.calls, 0)
                self.assertEqual(bridge.audit, [])

    def test_audit_partial_positional_defaults_are_safe(self):
        q = torch.ones(1, 1, 2, 4)
        with TemporaryDirectory() as directory:
            binary = Path(directory) / "fake-worker"
            binary.touch()
            for bridge_class in (AuditedAttention, AuditedBlasAttention):
                for options in ((None,), (None, 0.0)):
                    bridge = bridge_class(binary)
                    with patch.object(CuMetalAttention, "__call__", return_value=q) as worker, patch("builtins.print"):
                        actual = bridge(q, q, q, *options)
                    self.assertIs(actual, q)
                    worker.assert_called_once_with(q, q, q, None, 0.0, False, None, False)
                    self.assertEqual(len(bridge.audit), 1)
                    self.assertEqual(bridge.audit[0]["cumetal_vs_cpu_fp32"]["max_abs"], 0)

    def test_cli_rejects_duplicate_backends(self):
        for backend in ("mps", "cumetal", "cumetal-blas"):
            with self.subTest(backend=backend), patch("sys.stderr"), self.assertRaises(SystemExit) as error:
                parse_arguments(["--summarize-only", "--backends", backend, backend])
            self.assertEqual(error.exception.code, 2)

    def test_blas_is_explicit_and_geometry_gate_is_independent(self):
        with TemporaryDirectory() as directory:
            image = Path(directory) / "input.png"
            image.touch()
            common = ["--image", str(image), "--output-dir", str(Path(directory) / "new")]
            self.assertEqual(parse_arguments(common).backends, ["mps", "cumetal"])
            native = parse_arguments([*common, "--backends", "cumetal-blas", "--height", "704", "--width", "1280"])
            self.assertEqual(native.backends, ["cumetal-blas"])
            for backend, frames in (("cumetal", "9"), ("cumetal-blas", "13")):
                with patch("sys.stderr"), self.assertRaises(SystemExit):
                    parse_arguments([*common, "--backends", backend, "--height", "704", "--width", "1280",
                                     "--frames", frames])

    def test_invalid_geometry_is_rejected(self):
        for geometry in ((4, 192, 320), (10, 192, 320), (97, 192, 320),
                         (9, 48, 320), (9, 720, 320), (9, 193, 320),
                         (9, 192, 48), (9, 192, 1296), (9, 192, 321)):
            with self.subTest(geometry=geometry), self.assertRaises(ValueError):
                diagnostic_geometry(*geometry)

    def test_cli_preserves_baseline_and_accepts_native_geometry(self):
        with TemporaryDirectory() as directory:
            image = Path(directory) / "input.png"
            image.touch()
            baseline = parse_arguments(["--image", str(image), "--output-dir", str(Path(directory) / "new")])
            self.assertEqual((baseline.frames, baseline.height, baseline.width, baseline.steps),
                             (9, 192, 320, 8))
            self.assertFalse(baseline.native_rope)
            self.assertFalse(baseline.native_ab2)
            self.assertFalse(baseline.posterior_mode)
            self.assertEqual(baseline.export_fps, 8)
            native = parse_arguments(["--image", str(image), "--output-dir", str(Path(directory) / "new"),
                                      "--backends", "mps", "--native-rope",
                                      "--frames", "93", "--height", "704", "--width", "1280",
                                      "--steps", "35", "--export-fps", "16"])
            self.assertTrue(native.native_rope)
            self.assertEqual((native.frames, native.height, native.width, native.export_fps),
                             (93, 704, 1280, 16))

    def test_cli_refuses_invalid_controls_and_large_cumetal_geometry(self):
        with TemporaryDirectory() as directory:
            image = Path(directory) / "input.png"
            image.touch()
            for options in (["--frames", "10"], ["--width", "1296"], ["--height", "720"],
                            ["--steps", "0"], ["--steps", "36"], ["--export-fps", "0"],
                            ["--export-fps", "61"], ["--frames", "33"],
                            ["--visual-quality", "good"], ["--image", str(image) + ".missing"]):
                with self.subTest(options=options), patch("sys.stderr"), self.assertRaises(SystemExit) as error:
                    parse_arguments(["--image", str(image), "--output-dir", str(Path(directory) / "new"), *options])
                self.assertEqual(error.exception.code, 2)

    def test_summary_does_not_require_source_image_or_worker_geometry(self):
        args = parse_arguments(["--summarize-only", "--image", "/not/a/real/input.png",
                                "--frames", "93", "--height", "704", "--width", "1280",
                                "--visual-quality", "requires inspection"])
        self.assertTrue(args.summarize_only)

    def test_summary_preserves_existing_latent_comparison(self):
        with TemporaryDirectory() as directory:
            path = Path(directory)
            report = {"runs": {"mps": {}, "cumetal-blas": {}},
                      "comparisons_to_mps": {"cumetal-blas": {"latents": {"max_abs": 0.01}}}}
            (path / "comparison.json").write_text(json.dumps(report))
            frames = np.zeros((1, 2, 2, 3), dtype=np.uint8)
            np.save(path / "mps-frames.npy", frames)
            np.save(path / "cumetal-blas-frames.npy", frames)
            args = parse_arguments(["--summarize-only", "--output-dir", directory])
            with patch("diagnose_cosmos_video.parse_arguments", return_value=args), \
                    patch("diagnose_cosmos_video.contact_sheet"), patch("builtins.print"):
                main()
            comparison = json.loads((path / "comparison.json").read_text())["comparisons_to_mps"]["cumetal-blas"]
            self.assertEqual(comparison["latents"], {"max_abs": 0.01})
            self.assertEqual(comparison["frames_difference"], {"max_abs": 0.0, "mean_abs": 0.0})

    def test_cli_contract_flags_and_evaluation_counts(self):
        with TemporaryDirectory() as directory:
            image = Path(directory) / "input.png"
            image.touch()
            common = ["--image", str(image), "--output-dir", str(Path(directory) / "new")]
            baseline = contract_metadata(parse_arguments(common))
            self.assertEqual(baseline["model_evaluations"], 8)
            self.assertEqual(baseline["transition_steps"], 7)
            self.assertEqual(baseline["conditioning_latents"], "posterior_sample")
            self.assertFalse(baseline["final_clean_prediction"])
            self.assertEqual(baseline["contract_experiment_differences"], [])
            mode_only = contract_metadata(parse_arguments([*common, "--posterior-mode"]))
            self.assertEqual(mode_only["sampler_adapter"], "none")
            self.assertEqual(mode_only["conditioning_latents"], "posterior_mode")
            native = parse_arguments([*common, "--native-ab2", "--posterior-mode", "--steps", "36"])
            metadata = contract_metadata(native)
            self.assertEqual(metadata["model_evaluations"], 36)
            self.assertEqual(metadata["transition_steps"], 35)
            self.assertTrue(metadata["final_clean_prediction"])
            self.assertEqual(metadata["sampler_adapter"], "experimental_native_ab2_final_clean")
            self.assertIn("FP32", " ".join(metadata["contract_experiment_differences"]))
            self.assertEqual(metadata["conditioning_latents"], "posterior_mode")
            self.assertEqual(parse_arguments([*common, "--native-ab2", "--steps", "2"]).steps, 2)
            for options in (["--native-ab2", "--steps", "1"], ["--native-ab2", "--steps", "37"],
                            ["--posterior-mode", "--steps", "36"], ["--native-ab2", "--denoised-euler"]):
                with self.subTest(options=options), patch("sys.stderr"), self.assertRaises(SystemExit) as error:
                    parse_arguments([*common, *options])
                self.assertEqual(error.exception.code, 2)

    def test_existing_evidence_refused_without_affecting_summary(self):
        with TemporaryDirectory() as directory:
            (Path(directory) / "comparison.json").write_text("{}")
            with patch("sys.stderr"), self.assertRaises(SystemExit) as error:
                parse_arguments(["--output-dir", directory])
            self.assertEqual(error.exception.code, 2)
            self.assertTrue(parse_arguments(["--output-dir", directory, "--summarize-only"]).summarize_only)


if __name__ == "__main__":
    unittest.main()
