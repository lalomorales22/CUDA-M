#!/usr/bin/env python3
"""Bounded real-GPU bridge qualification; no model downloads or pretrained claims."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import torch
import torch.nn.functional as F

from cosmos_cumetal import CuMetalBlasAttention, ROOT, OUT


def error_metrics(actual, reference):
    actual, reference = actual.detach().float().cpu(), reference.detach().float().cpu()
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise AssertionError("comparison contains non-finite output")
    delta = actual - reference
    return {"max_abs": float(delta.abs().max()),
            "relative_rms": float(delta.square().mean().sqrt() /
                                  reference.square().mean().sqrt().clamp_min(1e-12))}


def verify(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError("refusing to overwrite existing evidence: " + str(path))
    if not torch.backends.mps.is_available():
        raise RuntimeError("this qualification requires an available Apple GPU")
    torch.manual_seed(20261009)
    bridge = CuMetalBlasAttention()
    report = {"status": "running", "seed": 20261009, "torch": torch.__version__,
              "scope": "experimental tiled BLAS bridge; reduced random-weight Cosmos transformer, not pretrained inference",
              "cpu_fallback_enabled": False, "dtype_cases": [], "transformer_cases": [],
              "evidence_boundary": "softmax counts are actual exact Apple-GPU runtime launches; GEMM counts are application completion receipts, not GEMM kernel provenance"}
    report["sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (Path(__file__), ROOT / "local/cosmos_cumetal.py",
                  ROOT / "local/cosmos_attention_blas.cu", bridge.binary,
                  ROOT / "build/libcumetal.dylib")}
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        q, k, v = torch.randn(1, 2, 257, 32), torch.randn(1, 2, 23, 32), torch.randn(1, 2, 23, 16)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for device in ("cpu", "mps"):
                tensors = [t.to(device=device, dtype=dtype) for t in (q, k, v)]
                reference = F.scaled_dot_product_attention(*(t.float().cpu() for t in tensors)).to(dtype)
                before = len(bridge.launches)
                actual = bridge(*tensors)
                if actual.dtype != dtype or actual.device.type != device:
                    raise AssertionError("bridge changed tensor dtype or device")
                tolerance = {"rtol": 2e-4, "atol": 2e-5} if dtype == torch.float32 else {"rtol": 0.01, "atol": 0.005}
                torch.testing.assert_close(actual.cpu(), reference, **tolerance)
                record = {"device": device, "dtype": str(dtype), "dimensions": [1, 2, 257, 23, 32, 16],
                          "softmax_kernel_launches": len(bridge.launches) - before,
                          "vs_cpu_fp32_rounded": error_metrics(actual, reference), "tolerance": tolerance}
                report["dtype_cases"].append(record)
                print("PASS dtype " + json.dumps(record), flush=True)
        actual = bridge(q, k, v, is_causal=True, scale=0.25)
        reference = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=0.25)
        torch.testing.assert_close(actual, reference, rtol=2e-4, atol=2e-5)
        report["causal_custom_scale"] = error_metrics(actual, reference)

        from diffusers.models.transformers import transformer_cosmos
        model = transformer_cosmos.CosmosTransformer3DModel(
            in_channels=4, out_channels=4, num_attention_heads=2, attention_head_dim=32,
            num_layers=1, text_embed_dim=32, adaln_lora_dim=16, max_size=(2, 8, 8),
            concat_padding_mask=False, extra_pos_embed_type=None).eval()
        args = {"hidden_states": torch.randn(1, 4, 2, 8, 8), "timestep": torch.tensor([1.0]),
                "encoder_hidden_states": torch.randn(1, 7, 32)}
        with torch.no_grad():
            cpu_reference = model(**args).sample
            for device in ("cpu", "mps"):
                model.to(device)
                device_args = {key: value.to(device) for key, value in args.items()}
                native = model(**device_args).sample
                original_dispatch = transformer_cosmos.dispatch_attention_fn
                before = bridge.calls
                with bridge.diffusers():
                    actual = model(**device_args).sample
                if transformer_cosmos.dispatch_attention_fn is not original_dispatch:
                    raise AssertionError("Cosmos attention dispatch was not restored")
                if bridge.calls - before != 2 or not actual.abs().max().item():
                    raise AssertionError("self/cross attention did not execute or output was trivial")
                torch.testing.assert_close(actual.cpu(), cpu_reference, rtol=4e-4, atol=5e-5)
                torch.testing.assert_close(actual, native, rtol=4e-4, atol=5e-5)
                record = {"device": device, "random_weights": True, "attention_calls": 2,
                          "vs_cpu_native": error_metrics(actual, cpu_reference),
                          "vs_device_native": error_metrics(actual, native)}
                report["transformer_cases"].append(record)
                print("PASS reduced transformer " + json.dumps(record), flush=True)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        report.update(elapsed_seconds=time.monotonic() - started, attention_calls=bridge.calls,
                      softmax_kernel_launches=len(bridge.launches),
                      application_completion_receipts=len(bridge.completion_receipts),
                      application_reported_gemm_calls=sum(int(r["gemm_calls"]) for r in bridge.completion_receipts),
                      device_names=sorted({r["device_name"] for r in bridge.completion_receipts}))
        path.write_text(json.dumps(report, indent=2) + "\n")
    print("VERIFIED " + str(path), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=OUT / "cosmos-blas-bridge-results.json")
    verify(parser.parse_args().report)
