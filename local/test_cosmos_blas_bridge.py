"""CPU-only protocol, bounds, and fail-closed evidence tests; no worker executes."""
import os
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cosmos_cumetal import CuMetalAttention, CuMetalBlasAttention, blas_attention_plan


def evidence(dimensions):
    batch, heads, queries, keys, width, value_width = dimensions
    plan = blas_attention_plan(*dimensions)
    lines = []
    for index in range(plan["tiles"]):
        count = min(plan["tile_queries"], queries - index * plan["tile_queries"])
        lines.append('CUMETAL_PROVENANCE event=kernel_launch kernel="_Z19cosmos_blas_softmaxPfiiii" '
            'source=generic_nvvm semantic_quality=exact device=apple_gpu device_name="Apple M4" '
            f'launch_success=true grid=({batch * heads * count},1,1) block=(128,1,1)')
    lines.append('COSMOS_BLAS_COMPLETION status=passed gemm_backend=cumetal_cublas_mps '
        f'softmax_backend=cumetal_cuda_metal tiles={plan["tiles"]} gemm_calls={2 * plan["tiles"]} '
        f'tile_queries={plan["tile_queries"]} workspace_elements={plan["workspace_elements"]} '
        'device_name="Apple M4"')
    return "\n".join(lines)


class BridgeTest(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.binary = Path(self.directory.name) / "nonexecuting-worker"
        self.binary.touch()
        self.bridge = CuMetalBlasAttention(self.binary)
        self.dimensions = (1, 2, 257, 7, 3, 5)
        self.trace = evidence(self.dimensions)
        self.q = torch.zeros(1, 2, 257, 3)
        self.k = torch.zeros(1, 2, 7, 3)
        self.v = torch.zeros(1, 2, 7, 5)
        self.output = np.zeros((1, 2, 257, 5), dtype="<f4").tobytes()

    def worker_result(self, trace=None, output=None, code=0):
        return SimpleNamespace(returncode=code, stderr=(self.trace if trace is None else trace).encode(),
                               stdout=self.output if output is None else output)

    def test_protocol_and_separate_softmax_and_application_evidence(self):
        with patch.dict(os.environ, {"CUMETAL_FAKE_SUCCESS": "1"}), patch("cosmos_cumetal.subprocess.run",
                return_value=self.worker_result()) as worker:
            actual = self.bridge(self.q, self.k, self.v)
        self.assertEqual(actual.shape, (1, 2, 257, 5))
        self.assertEqual(actual.device.type, "cpu")
        self.assertEqual(self.bridge.calls, 1)
        self.assertEqual(len(self.bridge.launches), 2)
        self.assertEqual(len(self.bridge.completion_receipts), 1)
        self.assertEqual(self.bridge.completion_receipts[0]["gemm_calls"], "4")
        self.assertTrue(all("softmax" in launch["kernel"] for launch in self.bridge.launches))
        self.assertEqual(worker.call_args.args, ([str(self.binary)],))
        kwargs = worker.call_args.kwargs
        header = struct.unpack("<8If", kwargs["input"][:36])
        self.assertEqual(header[:8], (0x434d4131, *self.dimensions, 0))
        self.assertAlmostEqual(header[-1], 3 ** -0.5, places=7)
        self.assertEqual(len(kwargs["input"]), 36 + 4 * (self.q.numel() + self.k.numel() + self.v.numel()))
        self.assertNotIn("CUMETAL_FAKE_SUCCESS", kwargs["env"])
        self.assertEqual(kwargs["env"]["CUMETAL_ENABLE_LLMC_CPU_EMULATION"], "0")
        self.assertEqual(kwargs["env"]["CUMETAL_DISABLE_LLMC_EMULATION"], "1")

    def test_fp16_bf16_roundtrip_keeps_dtype_and_uses_float32_payload(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype), patch("cosmos_cumetal.subprocess.run",
                    return_value=self.worker_result()) as worker:
                result = self.bridge(self.q.to(dtype), self.k.to(dtype), self.v.to(dtype), is_causal=True, scale=0.25)
                header = struct.unpack("<8If", worker.call_args.kwargs["input"][:36])
                self.assertEqual(header[-2:], (1, 0.25))
                self.assertEqual(result.dtype, dtype)

    def test_receipt_failures_are_rejected(self):
        lines = self.trace.splitlines()
        replacements = {
            "missing receipt": "\n".join(lines[:-1]),
            "duplicate receipt": self.trace + "\n" + lines[-1],
            "failed status": self.trace.replace("status=passed", "status=failed"),
            "wrong gemm backend": self.trace.replace("gemm_backend=cumetal_cublas_mps", "gemm_backend=cpu"),
            "wrong softmax backend": self.trace.replace("softmax_backend=cumetal_cuda_metal", "softmax_backend=cpu"),
            "wrong tile count": self.trace.replace("tiles=2", "tiles=3"),
            "wrong gemm count": self.trace.replace("gemm_calls=4", "gemm_calls=2"),
            "wrong tile width": self.trace.replace("tile_queries=256", "tile_queries=257"),
            "wrong workspace": self.trace.replace("workspace_elements=3584", "workspace_elements=3585"),
            "wrong receipt device": self.trace.rsplit('device_name="Apple M4"', 1)[0] + 'device_name="Apple M1"',
            "non Apple device": self.trace.replace("Apple M4", "CPU"),
            "unknown receipt field": self.trace + " unverified=true",
            "duplicate receipt key": self.trace + " status=passed",
            "malformed quote": self.trace + ' invalid="',
            "malformed token": self.trace + " invalid",
            "validation-only": self.trace + "\nCOSMOS_BLAS_VALIDATION status=passed gpu_executed=false",
            "receipt before launches": "\n".join([lines[-1], *lines[:-1]]),
        }
        for name, trace in replacements.items():
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                self.bridge._validate_trace(trace, self.dimensions)

    def test_softmax_provenance_failures_are_rejected(self):
        lines = self.trace.splitlines()
        replacements = {
            "missing tile": "\n".join(lines[1:]),
            "extra tile": "\n".join([lines[0], *lines]),
            "wrong kernel": self.trace.replace("cosmos_blas_softmax", "fake_blas_softmax"),
            "wrong device": self.trace.replace("device=apple_gpu", "device=cpu"),
            "failed launch": self.trace.replace("launch_success=true", "launch_success=false"),
            "approximate semantics": self.trace.replace("semantic_quality=exact", "semantic_quality=approximate"),
            "wrong first tile shape": self.trace.replace("grid=(512,1,1)", "grid=(2,1,1)"),
            "wrong last tile shape": self.trace.replace("grid=(2,1,1)", "grid=(512,1,1)"),
            "wrong block": self.trace.replace("block=(128,1,1)", "block=(64,1,1)"),
            "duplicate launch key": self.trace.replace("launch_success=true", "launch_success=true launch_success=true"),
        }
        for name, trace in replacements.items():
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                self.bridge._validate_trace(trace, self.dimensions)

    def test_failures_never_increment_success_counters(self):
        results = [self.worker_result(output=b""), self.worker_result(output=self.output + b"x"),
                   self.worker_result(output=np.full((1, 2, 257, 5), np.nan, dtype="<f4").tobytes()),
                   self.worker_result(code=1), self.worker_result(trace="")]
        for result in results:
            with self.subTest(size=len(result.stdout), code=result.returncode), patch("cosmos_cumetal.subprocess.run",
                    return_value=result), self.assertRaises(RuntimeError):
                self.bridge(self.q, self.k, self.v)
            self.assertEqual(self.bridge.calls, 0)
            self.assertEqual(self.bridge.launches, [])
            self.assertEqual(self.bridge.completion_receipts, [])

    def test_unsupported_inputs_never_start_worker(self):
        with patch("cosmos_cumetal.subprocess.run") as worker:
            for options in ({"dropout_p": 0.2}, {"enable_gqa": True}, {"attn_mask": torch.ones(257, 7)},
                            {"scale": float("nan")}):
                with self.subTest(options=options), self.assertRaises((NotImplementedError, ValueError)):
                    self.bridge(self.q, self.k, self.v, **options)
            with self.assertRaises(ValueError):
                self.bridge(torch.full_like(self.q, float("nan")), self.k, self.v)
            with self.assertRaises(NotImplementedError):
                self.bridge(self.q.clone().requires_grad_(), self.k, self.v)
            worker.assert_not_called()

    def test_legacy_bounds_and_three_kernel_provenance_unchanged(self):
        legacy = CuMetalAttention(self.binary)
        with self.assertRaises(NotImplementedError):
            legacy._check_dimensions(1, 1, 4097, 1, 1, 1)
        self.bridge._check_dimensions(1, 1, 4097, 1, 1, 1)
        with self.assertRaises(RuntimeError):
            legacy._validate_trace(self.trace, self.dimensions)
        lines = ['CUMETAL_PROVENANCE event=kernel_launch kernel="' + name + '" '
                 'semantic_quality=exact device=apple_gpu device_name="Apple M4" launch_success=true'
                 for name in ("cosmos_scores", "cosmos_softmax", "cosmos_values")]
        launches, receipts = legacy._validate_trace("\n".join(lines), self.dimensions)
        self.assertEqual(len(launches), 3)
        self.assertEqual(receipts, [])
        for invalid in (lines[:2], lines + lines[:1], [line.replace("apple_gpu", "cpu") for line in lines]):
            with self.assertRaises(RuntimeError):
                legacy._validate_trace("\n".join(invalid), self.dimensions)


class BoundTest(unittest.TestCase):
    def test_supported_training_resolution_geometry_and_adaptive_tile(self):
        plan = blas_attention_plan(1, 16, 10560, 10560, 128, 128)
        self.assertEqual(plan["tiles"], 42)
        self.assertEqual(plan["tile_queries"], 256)
        self.assertEqual(plan["workspace_elements"], 43253760)
        self.assertEqual(plan["multiply_accumulates"], 456759705600)
        adaptive = blas_attention_plan(1, 128, 65, 16384, 1, 1)
        self.assertEqual(adaptive["tile_queries"], 32)
        self.assertEqual(adaptive["tiles"], 3)
        self.assertEqual(adaptive["workspace_elements"], 67108864)
        boundary = blas_attention_plan(1, 32, 8192, 8192, 128, 128)
        self.assertEqual(boundary["tensor_elements"], 134217728)
        self.assertEqual(boundary["multiply_accumulates"], 549755813888)

    def test_invalid_or_excessive_bounds(self):
        for dimensions in ((0, 1, 1, 1, 1, 1), (True, 1, 1, 1, 1, 1), (1, 1, 1.0, 1, 1, 1),
                           (1, 1, 16385, 1, 1, 1), (2, 65, 1, 1, 1, 1), (1, 1, 1, 1, 129, 1),
                           (1, 128, 16384, 1, 128, 128), (1, 16, 16384, 16384, 128, 128)):
            with self.subTest(dimensions=dimensions), self.assertRaises(NotImplementedError):
                blas_attention_plan(*dimensions)


if __name__ == "__main__":
    unittest.main()
