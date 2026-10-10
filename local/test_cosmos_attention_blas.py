#!/usr/bin/env python3
"""CPU protocol gates by default; --gpu opts into CUDA-M/MPS numerical tests."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shlex
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
MAGIC = 0x434D4131


def header(shape=(1, 1, 1, 1, 1, 1), causal=0, scale=1.0, magic=MAGIC):
    return struct.pack("<8If", magic, *shape, causal, scale)


def environment():
    env = {k: v for k, v in os.environ.items() if not k.startswith("CUMETAL_")}
    env.update(CUMETAL_TRACE_GPU="1", CUMETAL_ENABLE_LLMC_CPU_EMULATION="0",
               CUMETAL_DISABLE_LLMC_EMULATION="1",
               CUMETAL_CACHE_DIR=str(ROOT / "local/out/cache"),
               CUMETAL_TOOL_XCRUN=str(ROOT / "local/toolchain/xcrun"))
    return env


def execute(binary, payload, validate=False, extra=()):
    return subprocess.run([str(binary), *(["--validate-only"] if validate else []), *extra],
                          input=payload, capture_output=True, env=environment(), timeout=120)


def fields(line):
    return dict(token.split("=", 1) for token in shlex.split(line)[1:] if "=" in token)


def protocol_tests(binary):
    cases = []
    good = header() + struct.pack("<3f", 1, 2, 3)
    for name, payload, error in (
        ("empty", b"", "truncated header"),
        ("short-header", header()[:-1], "truncated header"),
        ("magic", header(magic=0), "invalid header"),
        ("causal", header(causal=2), "invalid header"),
        ("nan-scale", header(scale=math.nan), "invalid header"),
        ("inf-scale", header(scale=math.inf), "invalid header"),
        ("empty-dimension", header((1, 1, 0, 1, 1, 1)), "dimension out of range"),
        ("large-dimension", header((1, 1, 1, 16385, 1, 1)), "dimension out of range"),
        ("heads", header((1, 129, 1, 1, 1, 1)), "unsupported attention dimensions"),
        ("width", header((1, 1, 1, 1, 129, 1)), "unsupported attention dimensions"),
        ("value-width", header((1, 1, 1, 1, 1, 129)), "unsupported attention dimensions"),
        ("memory-budget", header((1, 128, 16384, 1, 128, 128)), "bounded memory or operation budget"),
        ("operation-budget", header((1, 16, 16384, 16384, 128, 128)), "bounded memory or operation budget"),
        ("truncated-q", header(), "truncated input"),
        ("truncated-k", header() + struct.pack("<f", 1), "truncated input"),
        ("truncated-v", header() + struct.pack("<2f", 1, 2), "truncated input"),
        ("nonfinite-q", header() + struct.pack("<3f", math.nan, 2, 3), "non-finite input"),
        ("nonfinite-k", header() + struct.pack("<3f", 1, math.inf, 3), "non-finite input"),
        ("nonfinite-v", header() + struct.pack("<3f", 1, 2, -math.inf), "non-finite input"),
        ("trailing-data", good + b"x", "unexpected trailing input"),
        # This reaches input reading, proving the named target passes shape/budget guards.
        ("720p-target-admitted", header((1, 16, 10560, 10560, 128, 128)), "truncated input"),
    ):
        result = execute(binary, payload, validate=True)
        trace = result.stderr.decode(errors="replace")
        assert result.returncode != 0 and not result.stdout and error in trace, (name, trace)
        assert "CUMETAL_PROVENANCE" not in trace and "COSMOS_BLAS_COMPLETION" not in trace, name
        cases.append(name)
    result = execute(binary, good, validate=True)
    trace = result.stderr.decode(errors="replace")
    assert result.returncode == 0 and not result.stdout, trace
    receipt = fields(trace.strip())
    assert receipt["gpu_executed"] == "false" and receipt["workspace_elements"] == "1", trace
    assert "CUMETAL_PROVENANCE" not in trace, trace
    cases.append("valid-cpu-only-input")
    bounded_shape = (1, 32, 257, 16384, 1, 1)
    bounded_count = 32 * (257 + 2 * 16384)
    result = execute(binary, header(bounded_shape) + struct.pack("<f", 0) * bounded_count,
                     validate=True)
    trace = result.stderr.decode(errors="replace")
    assert result.returncode == 0 and not result.stdout, trace
    receipt = fields(trace.strip())
    assert receipt["gpu_executed"] == "false" and receipt["tile_queries"] == "128", trace
    assert int(receipt["workspace_elements"]) == 64 * 1024 * 1024, trace
    assert "CUMETAL_PROVENANCE" not in trace, trace
    cases.append("workspace-bound-reduces-tile")
    result = execute(binary, good, extra=("--unknown",))
    assert result.returncode != 0 and not result.stdout and b"unsupported argument" in result.stderr
    cases.append("unsupported-argument")
    return cases


def gpu_tests(binary):
    import numpy as np
    import torch
    torch.manual_seed(20261010)
    cases = []
    shapes = (
        (1, 1, 1, 1, 1, 1, False, 1.0),
        (1, 2, 17, 23, 32, 16, False, None),
        (2, 3, 33, 19, 64, 32, True, 0.25),
        (1, 2, 19, 33, 128, 128, True, None),
        (1, 2, 513, 257, 32, 17, False, 0.37),
        (2, 3, 513, 301, 32, 17, True, 0.25),
        (1, 1, 257, 513, 16, 31, True, None),
        (1, 2, 257, 23, 7, 9, False, 0.0),
        (1, 16, 64, 64, 128, 128, False, -0.1),
        (1, 32, 129, 16384, 1, 1, False, None),
    )
    for batch, heads, queries, keys, width, value_width, causal, scale in shapes:
        shape = (batch, heads, queries, keys, width, value_width)
        q = torch.randn(batch, heads, queries, width)
        k = torch.randn(batch, heads, keys, width)
        v = torch.randn(batch, heads, keys, value_width)
        factor = width ** -0.5 if scale is None else scale
        payload = header(shape, int(causal), factor)
        payload += b"".join(t.numpy().tobytes() for t in (q, k, v))
        started = time.monotonic()
        result = execute(binary, payload)
        elapsed = time.monotonic() - started
        trace = result.stderr.decode(errors="replace")
        assert result.returncode == 0, trace
        assert len(result.stdout) == batch * heads * queries * value_width * 4
        output = torch.from_numpy(np.frombuffer(result.stdout, dtype="<f4").copy()).reshape(
            batch, heads, queries, value_width)
        reference = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=causal, scale=scale)
        torch.testing.assert_close(output, reference, rtol=3e-5, atol=3e-5)
        tile_queries = min(queries, 256, (64 * 1024 * 1024) // (batch * heads * keys))
        expected_tiles = (queries + tile_queries - 1) // tile_queries
        launches = [fields(line) for line in trace.splitlines()
                    if line.startswith("CUMETAL_PROVENANCE ")]
        launches = [event for event in launches if event.get("event") == "kernel_launch"]
        assert len(launches) == expected_tiles, trace
        assert all("cosmos_blas_softmax" in event.get("kernel", "") and
                   event.get("device") == "apple_gpu" and event.get("launch_success") == "true" and
                   event.get("semantic_quality") == "exact" and
                   event.get("device_name", "").startswith("Apple ") for event in launches), trace
        receipts = [fields(line) for line in trace.splitlines() if line.startswith("COSMOS_BLAS_COMPLETION ")]
        assert len(receipts) == 1, trace
        receipt = receipts[0]
        assert receipt["status"] == "passed" and receipt["gemm_backend"] == "cumetal_cublas_mps", receipt
        assert int(receipt["tiles"]) == expected_tiles and int(receipt["gemm_calls"]) == 2 * expected_tiles
        assert int(receipt["tile_queries"]) == tile_queries
        assert int(receipt["workspace_elements"]) == batch * heads * tile_queries * keys
        cases.append({"shape": shape, "causal": causal, "scale": scale,
                      "max_absolute_error": (output - reference).abs().max().item(),
                      "elapsed_seconds": elapsed, "softmax_launches": launches, "receipt": receipt})
        print("PASS GPU", shape, "causal", causal, "tiles", expected_tiles, flush=True)
    result = execute(binary, header() + struct.pack("<3f", 1e30, 1e30, 1.0))
    assert result.returncode != 0 and not result.stdout and b"non-finite output" in result.stderr
    return cases


def cosmos_shape_test(binary):
    """Every 720p/9-frame attention element, without a full dense CPU score matrix."""
    import numpy as np
    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError("native MPS reference requires an Apple GPU")
    torch.manual_seed(20261011)
    shape = (1, 16, 10560, 10560, 128, 128)
    q, k, v = [torch.randn(1, 16, 10560, 128) for _ in range(3)]
    reference = torch.empty_like(q)
    torch.mps.synchronize()
    reference_started = time.monotonic()
    mq, mk, mv = [tensor.to("mps") for tensor in (q, k, v)]
    peak_mps_driver_bytes = torch.mps.driver_allocated_memory()
    # SDPA is independent across query rows. Preserve all keys for each tile.
    for offset in range(0, 10560, 256):
        part = torch.nn.functional.scaled_dot_product_attention(mq[:, :, offset:offset + 256], mk, mv)
        reference[:, :, offset:offset + 256] = part.cpu()
        peak_mps_driver_bytes = max(peak_mps_driver_bytes, torch.mps.driver_allocated_memory())
    torch.mps.synchronize()
    reference_seconds = time.monotonic() - reference_started
    del mq, mk, mv, part
    torch.mps.empty_cache()
    payload = header(shape, 0, 128 ** -0.5)
    payload += b"".join(tensor.numpy().tobytes() for tensor in (q, k, v))
    started = time.monotonic()
    result = execute(binary, payload)
    worker_seconds = time.monotonic() - started
    trace = result.stderr.decode(errors="replace")
    assert result.returncode == 0, trace
    assert len(result.stdout) == reference.numel() * 4
    output = torch.from_numpy(np.frombuffer(result.stdout, dtype="<f4").copy()).reshape_as(reference)
    torch.testing.assert_close(output, reference, rtol=3e-5, atol=3e-5)
    error = output - reference
    launches = [fields(line) for line in trace.splitlines() if line.startswith("CUMETAL_PROVENANCE ")]
    launches = [event for event in launches if event.get("event") == "kernel_launch"]
    assert len(launches) == 42 and all("cosmos_blas_softmax" in event.get("kernel", "") and
        event.get("device") == "apple_gpu" and event.get("launch_success") == "true" and
        event.get("semantic_quality") == "exact" and
        event.get("device_name", "").startswith("Apple ") for event in launches), trace
    receipts = [fields(line) for line in trace.splitlines() if line.startswith("COSMOS_BLAS_COMPLETION ")]
    assert len(receipts) == 1, trace
    receipt = receipts[0]
    assert receipt["status"] == "passed" and receipt["gemm_backend"] == "cumetal_cublas_mps", receipt
    assert receipt["softmax_backend"] == "cumetal_cuda_metal", receipt
    assert int(receipt["tiles"]) == 42 and int(receipt["gemm_calls"]) == 84, receipt
    assert int(receipt["workspace_elements"]) == 43253760 and int(receipt["tile_queries"]) == 256, receipt
    summary = {"shape": shape, "dtype": "float32", "causal": False,
        "scope": "standalone target attention geometry, not full-model inference",
        "reference": "native MPS SDPA, 256-query tiles preserving all 10560 keys",
        "max_absolute_error": error.abs().max().item(),
        "relative_rms_error": (error.square().mean().sqrt() / reference.square().mean().sqrt()).item(),
        "worker_seconds": worker_seconds, "reference_seconds": reference_seconds,
        "score_workspace_bytes": 43253760 * 4,
        "reference_peak_observed_mps_driver_bytes": peak_mps_driver_bytes,
        "child_process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        "receipt": receipt, "softmax_launches": launches}
    print("PASS Cosmos target geometry", {key: value for key, value in summary.items()
          if key not in ("softmax_launches", "receipt")}, flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true", help="run actual Apple GPU numerical comparisons")
    parser.add_argument("--cosmos-shape", action="store_true", help="also check the full 720p attention geometry")
    parser.add_argument("--binary", type=Path, default=ROOT / "local/out/cosmos-attention-blas")
    parser.add_argument("--report", type=Path, default=ROOT / "local/out/cosmos-attention-blas-results.json")
    args = parser.parse_args()
    if args.cosmos_shape and not args.gpu:
        parser.error("--cosmos-shape requires --gpu")
    report = {"status": "running", "gpu_requested": args.gpu,
              "scope": "experimental tiled SDPA worker, not a completed pretrained Cosmos inference"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    try:
        report["sha256"] = {(str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)):
                            hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in (Path(__file__).resolve(), ROOT / "local/cosmos_attention_blas.cu",
                                         args.binary.resolve(), ROOT / "build/libcumetal.dylib")}
        report["protocol_cases"] = protocol_tests(args.binary)
        print(f"PASS {len(report['protocol_cases'])} CPU-only protocol cases", flush=True)
        if args.gpu:
            report["gpu_cases"] = gpu_tests(args.binary)
            report["gpu_negative_cases"] = ["finite-input-intermediate-overflow-rejected"]
            if args.cosmos_shape:
                report["cosmos_shape"] = cosmos_shape_test(args.binary)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(args.report)


if __name__ == "__main__":
    main()
