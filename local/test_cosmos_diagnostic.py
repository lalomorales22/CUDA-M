"""Download-free checks for the Cosmos numerical comparison harness."""
import unittest
from unittest.mock import patch

import torch
import numpy as np
from diffusers.models.transformers import transformer_cosmos
from diffusers import FlowMatchEulerDiscreteScheduler

from diagnose_cosmos_video import (capture_native_attention, differences, frame_differences,
                                   denoised_prediction_adapter)


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


if __name__ == "__main__":
    unittest.main()
