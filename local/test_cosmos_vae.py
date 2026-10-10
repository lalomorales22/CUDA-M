"""CPU-only, download-free tests for the Wan VAE diagnostic."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from diagnose_cosmos_vae import denormalize_latents, frame_metrics, load_final_latents, tensor_frames


class VAEDiagnosticTest(unittest.TestCase):
    def test_normalization_inverse_float32(self):
        config = SimpleNamespace(latents_mean=[-0.5, 1.0], latents_std=[2.0, 0.25])
        raw = torch.tensor([2.5, 0.75]).reshape(1, 2, 1, 1, 1)
        normalized = (raw - torch.tensor(config.latents_mean).reshape_as(raw))
        normalized = normalized / torch.tensor(config.latents_std).reshape_as(raw) * 2
        torch.testing.assert_close(denormalize_latents(normalized.bfloat16(), config, 2), raw)

    def test_bad_normalization_refused(self):
        x = torch.ones(1, 2, 1, 1, 1)
        for std in ([1, 0], [1, float("nan")], [1]):
            with self.assertRaises(ValueError):
                denormalize_latents(x, SimpleNamespace(latents_mean=[0, 0], latents_std=std))
        with self.assertRaises(ValueError):
            denormalize_latents(x, SimpleNamespace(latents_mean=[0, 0], latents_std=[1, 1]), 0)

    def test_metrics_exact_and_known_error(self):
        a = np.zeros((2, 2, 2, 3), dtype=np.float32)
        exact = frame_metrics(a, a)
        self.assertEqual(exact[0]["rmse"], 0)
        self.assertTrue(exact[0]["psnr_exact"])
        self.assertIsNone(exact[0]["psnr_db"])
        result = frame_metrics(a + 1, a)[1]
        self.assertEqual(result["rmse"], 1)
        self.assertAlmostEqual(result["psnr_db"], 6.020599913)
        self.assertEqual(result["clipped_fraction"], 1)

    def test_nonfinite_or_wrong_shape_refused(self):
        a = np.zeros((2, 2, 2, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            frame_metrics(a, a[0])
        a[0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            frame_metrics(a, a)

    def test_tensor_layout(self):
        x = torch.arange(24).reshape(1, 3, 2, 2, 2)
        result = tensor_frames(x, 2, 2, 2)
        self.assertEqual(result.shape, (2, 2, 2, 3))
        np.testing.assert_array_equal(result[1, 0, 0], [4, 12, 20])
        with self.assertRaises(ValueError):
            tensor_frames(x, 3, 2, 2)

    def test_final_archive_uses_numeric_step_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latents.npz"
            shape = (1, 2, 1, 1, 1)
            values = {f"step_{i}": np.full(shape, i, dtype=np.float32) for i in reversed(range(12))}
            np.savez(path, **values)
            tensor, step = load_final_latents(path, shape)
            self.assertEqual(step, 11)
            self.assertTrue((tensor == 11).all())

    def test_archive_invalid_data_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latents.npz"
            shape = (1, 2, 1, 1, 1)
            a = np.ones(shape, dtype=np.float32)
            cases = [{"other": a}, {"step_1": a}, {"step_0": a.astype(np.int32)},
                     {"step_0": np.ones((2,), dtype=np.float32)}, {"step_0": a * np.nan}]
            for values in cases:
                np.savez(path, **values)
                with self.assertRaises(ValueError):
                    load_final_latents(path, shape)


if __name__ == "__main__":
    unittest.main()
