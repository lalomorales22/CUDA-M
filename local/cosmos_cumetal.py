#!/usr/bin/env python3
"""Explicit, bounded PyTorch SDPA bridge to CUDA source compiled by CuMetal."""
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import struct
import subprocess

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "local/out"


class CuMetalAttention:
    """Inference only. FP16/BF16 inputs explicitly use FP32 attention accumulation."""

    def __init__(self, binary=None):
        self.binary = Path(binary or OUT / "cosmos-attention")
        if not self.binary.is_file():
            raise RuntimeError("build local/cosmos_attention.cu with ./cuda-mac compile first")
        self.launches = []
        self.completion_receipts = []
        self.calls = 0

    def _check_dimensions(self, batch, heads, queries, keys, width, value_width):
        if (batch * heads > 128 or max(queries, keys) > 4096 or
                max(width, value_width) > 128 or batch * heads * queries * keys > 67108864):
            raise NotImplementedError("attention exceeds the worker's bounded dimensions")

    def _validate_trace(self, trace, dimensions):
        launches = []
        for line in trace.splitlines():
            if line.startswith("CUMETAL_PROVENANCE "):
                event = dict(token.split("=", 1) for token in shlex.split(line)[1:] if "=" in token)
                if event.get("event") == "kernel_launch":
                    launches.append(event)
        for name in ("cosmos_scores", "cosmos_softmax", "cosmos_values"):
            matching = [e for e in launches if name in e.get("kernel", "")]
            if len(matching) != 1:
                raise RuntimeError("missing or duplicate Apple-GPU launch: " + name)
        if len(launches) != 3 or any(e.get("device") != "apple_gpu" or
                e.get("launch_success") != "true" or e.get("semantic_quality") != "exact" or
                not e.get("device_name", "").startswith("Apple ") for e in launches):
            raise RuntimeError("worker did not provide exact, completed Apple-GPU provenance")
        return launches, []

    def __call__(self, query, key, value, attn_mask=None, dropout_p=0.0,
                 is_causal=False, scale=None, enable_gqa=False):
        if attn_mask is not None or dropout_p != 0.0 or enable_gqa:
            raise NotImplementedError("masks, dropout, and grouped-query attention are unsupported")
        tensors = (query, key, value)
        if any(t.ndim != 4 for t in tensors):
            raise ValueError("expected tensors in [batch, heads, sequence, channels] layout")
        if any(t.requires_grad for t in tensors):
            raise NotImplementedError("CuMetal attention has no backward implementation")
        if any(t.dtype != query.dtype or t.device != query.device for t in tensors):
            raise ValueError("query, key, and value must share dtype and device")
        if query.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise NotImplementedError("only FP32, FP16, and BF16 inputs are supported")
        if query.device.type not in ("cpu", "mps"):
            raise NotImplementedError("only CPU staging tensors and MPS tensors are supported")
        batch, heads, queries, width = query.shape
        keys, value_width = key.shape[2], value.shape[3]
        if key.shape[:2] != (batch, heads) or value.shape[:3] != (batch, heads, keys) or key.shape[3] != width:
            raise ValueError("incompatible query, key, and value shapes")
        if min(batch, heads, queries, keys, width, value_width) <= 0:
            raise ValueError("empty attention dimensions are unsupported")
        dimensions = (batch, heads, queries, keys, width, value_width)
        self._check_dimensions(*dimensions)
        factor = float(scale) if scale is not None else width ** -0.5
        if not math.isfinite(factor):
            raise ValueError("scale must be finite")
        staged = [t.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy() for t in tensors]
        if any(not np.isfinite(t).all() for t in staged):
            raise ValueError("non-finite input is unsupported")
        payload = struct.pack("<8If", 0x434d4131, batch, heads, queries, keys, width,
                              value_width, int(is_causal), factor)
        payload += b"".join(t.tobytes() for t in staged)
        env = {k: v for k, v in os.environ.items() if not k.startswith("CUMETAL_")}
        env.update(CUMETAL_TRACE_GPU="1", CUMETAL_ENABLE_LLMC_CPU_EMULATION="0",
                   CUMETAL_DISABLE_LLMC_EMULATION="1", CUMETAL_CACHE_DIR=str(OUT / "cache"))
        result = subprocess.run([str(self.binary)], input=payload, capture_output=True,
                                env=env, timeout=120)
        trace = result.stderr.decode(errors="replace")
        if result.returncode:
            raise RuntimeError("CuMetal worker failed: " + trace)
        launches, receipts = self._validate_trace(trace, dimensions)
        expected_bytes = batch * heads * queries * value_width * 4
        if len(result.stdout) != expected_bytes:
            raise RuntimeError("worker output size mismatch")
        array = np.frombuffer(result.stdout, dtype="<f4").copy().reshape(batch, heads, queries, value_width)
        if not np.isfinite(array).all():
            raise RuntimeError("worker returned non-finite output")
        self.calls += 1
        self.launches.extend(launches)
        self.completion_receipts.extend(receipts)
        return torch.from_numpy(array).to(device=query.device, dtype=query.dtype)

    @contextmanager
    def diffusers(self):
        """Replace only Cosmos's attention dispatch for the duration of one inference."""
        from diffusers.models.transformers import transformer_cosmos
        original = transformer_cosmos.dispatch_attention_fn

        def dispatch(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
            if kwargs:
                raise NotImplementedError("unsupported attention options: " + str(sorted(kwargs)))
            return self(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                        attn_mask, dropout_p, is_causal).transpose(1, 2)

        transformer_cosmos.dispatch_attention_fn = dispatch
        try:
            yield
        finally:
            transformer_cosmos.dispatch_attention_fn = original


def blas_attention_plan(batch, heads, queries, keys, width, value_width):
    """Mirror the experimental tiled worker's allocation and operation limits."""
    dimensions = (batch, heads, queries, keys, width, value_width)
    if any(not isinstance(d, int) or isinstance(d, bool) or not 1 <= d <= 16384 for d in dimensions):
        raise NotImplementedError("BLAS attention dimensions must be integers in 1..16384")
    head_count = batch * heads
    if head_count > 128 or max(width, value_width) > 128:
        raise NotImplementedError("BLAS attention exceeds head/channel bounds")
    tensor_elements = head_count * (queries + keys) * (width + value_width)
    multiply_accumulates = head_count * queries * keys * (width + value_width)
    if tensor_elements > 134217728 or multiply_accumulates > 549755813888:
        raise NotImplementedError("BLAS attention exceeds bounded memory or operation budget")
    tile = min(queries, 256, 67108864 // (head_count * keys))
    if tile < 1:
        raise NotImplementedError("BLAS attention exceeds score workspace")
    return {"tile_queries": tile, "tiles": (queries + tile - 1) // tile,
            "workspace_elements": head_count * tile * keys, "tensor_elements": tensor_elements,
            "multiply_accumulates": multiply_accumulates}


def _strict_trace_record(line):
    try:
        fields = {}
        for token in shlex.split(line)[1:]:
            key, value = token.split("=", 1)
            if not key or key in fields:
                raise ValueError("empty or duplicate field")
            fields[key] = value
        return fields
    except ValueError as error:
        raise RuntimeError("malformed worker evidence: " + str(error)) from error


class CuMetalBlasAttention(CuMetalAttention):
    """Experimental tiled cuBLAS/MPS GEMMs plus CUDA-M softmax.

    ``launches`` contains actual softmax runtime kernel records only.
    ``completion_receipts`` contains separate application-level assertions that
    synchronous cuBLAS calls completed. These are NOT GEMM kernel provenance.
    """

    def __init__(self, binary=None):
        path = Path(binary or OUT / "cosmos-attention-blas")
        if not path.is_file():
            raise RuntimeError("build local/cosmos_attention_blas.cu with ./cuda-mac compile first")
        super().__init__(path)

    def _check_dimensions(self, *dimensions):
        blas_attention_plan(*dimensions)

    def _validate_trace(self, trace, dimensions):
        plan = blas_attention_plan(*dimensions)
        launches, receipts = [], []
        for line in trace.splitlines():
            if line.startswith("COSMOS_BLAS_VALIDATION"):
                raise RuntimeError("validation-only execution is not GPU completion evidence")
            if line.startswith("CUMETAL_PROVENANCE "):
                event = _strict_trace_record(line)
                if event.get("event") == "kernel_launch":
                    if receipts:
                        raise RuntimeError("softmax launch appeared after its completion receipt")
                    launches.append(event)
            elif line.startswith("COSMOS_BLAS_COMPLETION"):
                if not line.startswith("COSMOS_BLAS_COMPLETION "):
                    raise RuntimeError("malformed BLAS completion receipt")
                receipts.append(_strict_trace_record(line))
        if len(receipts) != 1:
            raise RuntimeError("missing or duplicate BLAS application completion receipt")
        receipt = receipts[0]
        expected = {"status": "passed", "gemm_backend": "cumetal_cublas_mps",
                    "softmax_backend": "cumetal_cuda_metal", "tiles": str(plan["tiles"]),
                    "gemm_calls": str(2 * plan["tiles"]), "tile_queries": str(plan["tile_queries"]),
                    "workspace_elements": str(plan["workspace_elements"]),
                    "device_name": receipt.get("device_name", "")}
        if receipt != expected or not receipt.get("device_name", "").startswith("Apple "):
            raise RuntimeError("incorrect BLAS application completion receipt")
        if len(launches) != plan["tiles"]:
            raise RuntimeError("missing or duplicate tiled softmax GPU launch")
        batch, heads, queries, _, _, _ = dimensions
        for index, event in enumerate(launches):
            count = min(plan["tile_queries"], queries - index * plan["tile_queries"])
            if (event.get("kernel") not in ("cosmos_blas_softmax", "_Z19cosmos_blas_softmaxPfiiii")
                    or event.get("device") != "apple_gpu" or event.get("launch_success") != "true"
                    or event.get("semantic_quality") != "exact"
                    or event.get("device_name") != receipt["device_name"]
                    or event.get("grid") != f"({batch * heads * count},1,1)"
                    or event.get("block") != "(128,1,1)"):
                raise RuntimeError("incorrect tiled softmax GPU provenance")
        return launches, receipts


def self_test(report_path=None):
    torch.manual_seed(20261009)
    bridge = CuMetalAttention()
    report = {"status": "running", "scope": "attention and reduced random-weight Cosmos transformer; no pretrained inference",
              "torch": torch.__version__, "cases": []}
    report["sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (ROOT / "local/cosmos_attention.cu", Path(__file__),
                                  bridge.binary, ROOT / "build/libcumetal.dylib")}
    path = Path(report_path) if report_path is not None else OUT / "cosmos-attention-results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        for batch, heads, queries, keys, width, value_width, causal, scale in (
                (1, 1, 1, 1, 1, 1, False, None),
                (1, 2, 17, 23, 32, 16, False, None),
                (2, 3, 33, 19, 64, 32, True, 0.25),
                (1, 2, 19, 33, 128, 128, True, None),
                (1, 16, 64, 64, 128, 128, False, None)):
            q = torch.randn(batch, heads, queries, width)
            k = torch.randn(batch, heads, keys, width)
            v = torch.randn(batch, heads, keys, value_width)
            reference = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal, scale=scale)
            actual = bridge(q, k, v, is_causal=causal, scale=scale)
            torch.testing.assert_close(actual, reference, rtol=2e-4, atol=2e-5)
            error = float((actual - reference).abs().max())
            report["cases"].append({"shape": [batch, heads, queries, keys, width, value_width],
                                    "causal": causal, "max_abs_error": error})
            print(f"PASS attention {report['cases'][-1]}", flush=True)

        q = torch.randn(1, 2, 7, 32)
        for dtype in (torch.float16, torch.bfloat16):
            for device in ("cpu", "mps"):
                if device == "mps" and not torch.backends.mps.is_available():
                    continue
                x = q.to(dtype=dtype, device=device)
                reference = torch.nn.functional.scaled_dot_product_attention(x.cpu(), x.cpu(), x.cpu())
                actual = bridge(x, x, x).cpu()
                torch.testing.assert_close(actual, reference, rtol=0.01, atol=0.005)
                report["cases"].append({"dtype": str(dtype), "device": device,
                                        "max_abs_error": float((actual.float() - reference.float()).abs().max())})
        for kwargs in ({"dropout_p": 0.1}, {"attn_mask": torch.ones(7, 7)}, {"enable_gqa": True}):
            try:
                bridge(q, q, q, **kwargs)
            except NotImplementedError:
                pass
            else:
                raise AssertionError("unsupported mode was accepted")
        for invalid in (q[..., :0], torch.full_like(q, float("nan")), q.requires_grad_()):
            try:
                bridge(invalid, invalid, invalid)
            except (ValueError, NotImplementedError):
                pass
            else:
                raise AssertionError("invalid input was accepted")
        for payload in (b"", struct.pack("<8If", 0, 1, 1, 1, 1, 1, 1, 0, 1.0),
                        struct.pack("<8If", 0x434d4131, 1, 1, 1, 1, 1, 1, 0, 1.0)):
            result = subprocess.run([str(bridge.binary)], input=payload, capture_output=True, timeout=10)
            if result.returncode == 0 or result.stdout:
                raise AssertionError("malformed worker input was accepted")
        report["negative_checks"] = 9

        from diffusers.models.transformers.transformer_cosmos import CosmosTransformer3DModel
        model = CosmosTransformer3DModel(in_channels=4, out_channels=4, num_attention_heads=2,
            attention_head_dim=32, num_layers=1, text_embed_dim=32, adaln_lora_dim=16,
            max_size=(2, 8, 8), concat_padding_mask=False, extra_pos_embed_type=None).eval()
        args = {"hidden_states": torch.randn(1, 4, 2, 8, 8), "timestep": torch.tensor([1.0]),
                "encoder_hidden_states": torch.randn(1, 7, 32)}
        with torch.no_grad():
            reference = model(**args).sample
            before = bridge.calls
            with bridge.diffusers():
                actual = model(**args).sample
        if bridge.calls - before != 2 or not actual.abs().max().item():
            raise AssertionError("Cosmos self/cross attention did not execute or output was trivial")
        torch.testing.assert_close(actual, reference, rtol=2e-4, atol=2e-5)
        report["cosmos_transformer"] = {"random_weights": True, "attention_calls": 2,
                                        "max_abs_error": float((actual - reference).abs().max())}
        if torch.backends.mps.is_available():
            model.to("mps")
            mps_args = {k: v.to("mps") for k, v in args.items()}
            with torch.no_grad(), bridge.diffusers():
                mps_actual = model(**mps_args).sample.cpu()
            torch.testing.assert_close(mps_actual, reference, rtol=4e-4, atol=5e-5)
            report["cosmos_transformer_mps"] = {
                "random_weights": True, "other_tensor_operations": "PyTorch MPS",
                "attention": "CuMetal CUDA source -> Metal",
                "max_abs_error": float((mps_actual - reference).abs().max())}
        report.update(status="passed", calls=bridge.calls, launches=bridge.launches)
        print("PASS reduced Cosmos transformer: self and cross attention ran through CUDA-M", flush=True)
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"VERIFIED {bridge.calls} attention calls, {len(bridge.launches)} Apple-GPU launches; {path}")


if __name__ == "__main__":
    self_test()
