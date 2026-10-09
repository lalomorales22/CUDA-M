"""Download-free regression tests for Cosmos guardrail portability."""
import unittest
from unittest.mock import patch, PropertyMock

import torch
import cosmos_guardrail.cosmos_guardrail as guardrail

from run_cosmos_video import guardrail_weight_staging, make_safety_checker


class GuardrailPortabilityTest(unittest.TestCase):
    def test_weight_staging_and_restoration(self):
        original = guardrail.load_model
        model = torch.nn.Linear(2, 1)
        state = {"module." + key: value.clone() for key, value in model.state_dict().items()}
        with patch("torch.load", return_value={"state_dict": state}) as load:
            with guardrail_weight_staging():
                self.assertIs(guardrail.load_model(model, "test.pth"), model)
                load.assert_called_once_with("test.pth", map_location="cpu", weights_only=True)
            self.assertIs(guardrail.load_model, original)

    def test_bad_weights_refused_and_patch_restored(self):
        original = guardrail.load_model
        with patch("torch.load", return_value={"weight": torch.zeros(1, 2)}):
            with self.assertRaises(RuntimeError), guardrail_weight_staging():
                guardrail.load_model(torch.nn.Linear(2, 1), "test.pth")
        self.assertIs(guardrail.load_model, original)

    def test_device_dtype_and_fail_closed(self):
        qwen = guardrail.Qwen3Guard.__new__(guardrail.Qwen3Guard)
        torch.nn.Module.__init__(qwen)
        qwen.model = torch.nn.Linear(2, 1)
        with patch.object(guardrail.CosmosSafetyChecker, "__init__", torch.nn.Module.__init__), \
             patch.object(guardrail.CosmosSafetyChecker, "nn_models", new_callable=PropertyMock,
                          return_value=[qwen]), \
             patch.object(qwen, "extract_label_and_categories", side_effect=RuntimeError("GPU error")):
            checker = make_safety_checker()
            self.assertEqual(checker.device, torch.device("cpu"))
            self.assertEqual(checker.dtype, torch.float32)
            self.assertIs(checker.to(dtype=torch.float64), checker)
            self.assertEqual(checker.dtype, torch.float64)
            with self.assertRaisesRegex(RuntimeError, "GPU error"):
                qwen.is_safe("test")


if __name__ == "__main__":
    unittest.main()
