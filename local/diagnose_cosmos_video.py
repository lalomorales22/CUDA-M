#!/usr/bin/env python3
"""Same-seed pretrained MPS/CuMetal comparison with sampled CPU SDPA references."""
import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import json
import os
from pathlib import Path
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np
import torch
import torch.nn.functional as F

from cosmos_cumetal import CuMetalAttention, CuMetalBlasAttention, blas_attention_plan, OUT
from cosmos_predict2_contract import (NVIDIA_SOURCE_REVISION, experimental_ab2_scheduler,
                                     experimental_posterior_mode)
from run_cosmos_video import MODEL, guardrail_weight_staging, make_safety_checker


def differences(actual, reference):
    a, b = actual.detach().float().cpu(), reference.detach().float().cpu()
    delta = a - b
    return {"finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
            "max_abs": float(delta.abs().max()),
            "rms": float(delta.square().mean().sqrt()),
            "relative_rms": float(delta.square().mean().sqrt() /
                                  b.square().mean().sqrt().clamp_min(1e-12))}


def frame_differences(actual, reference):
    delta = actual.astype(np.float32) - reference.astype(np.float32)
    return {"max_abs": float(np.abs(delta).max()), "mean_abs": float(np.abs(delta).mean())}


def contact_sheet(directory):
    from PIL import Image, ImageDraw
    backends = [backend for backend in ("mps", "cumetal", "cumetal-blas")
                if (directory / f"{backend}-frames.npy").is_file()]
    arrays = [np.load(directory / f"{backend}-frames.npy") for backend in backends]
    height, width = arrays[0].shape[1:3]
    sheet = Image.new("RGB", (width * 3, (height + 24) * len(arrays)), "white")
    draw = ImageDraw.Draw(sheet)
    labels = {"mps": "MPS", "cumetal": "CUDA-M", "cumetal-blas": "CUDA-M tiled BLAS (experimental)"}
    for row, (backend, frames) in enumerate(zip(backends, arrays, strict=True)):
        for column, index in enumerate((0, len(frames) // 2, len(frames) - 1)):
            image = Image.fromarray(frames[index].astype(np.uint8))
            sheet.paste(image, (column * width, row * (height + 24) + 24))
            draw.text((column * width + 4, row * (height + 24) + 4), f"{labels[backend]}: frame {index}", fill="black")
    sheet.save(directory / "comparison.png")


@contextmanager
def denoised_prediction_adapter(scheduler):
    """Experimental x0 -> Euler derivative adapter; preserve the baseline scheduler."""
    original = scheduler.step

    def step(denoised, timestep, sample, *args, **kwargs):
        index = scheduler.step_index
        if index is None:
            index = scheduler.index_for_timestep(timestep)
        sigma = scheduler.sigmas[index].to(device=sample.device, dtype=torch.float32)
        if not torch.isfinite(sigma) or sigma <= 0:
            raise ValueError("denoised prediction needs a finite positive sigma")
        derivative = (sample.float() - denoised.float()) / sigma
        return original(derivative, timestep, sample, *args, **kwargs)

    scheduler.step = step
    try:
        yield
    finally:
        scheduler.step = original


class _AttentionAudit:
    reference_query_limit = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step = 0
        self.sampled_steps = {}
        self.audit = []

    def __call__(self, query, key, value, attn_mask=None, dropout_p=0.0,
                 is_causal=False, scale=None, enable_gqa=False):
        if attn_mask is not None or dropout_p != 0.0 or is_causal or scale is not None or enable_gqa:
            raise NotImplementedError("audit only qualifies default unmasked non-causal attention")
        actual = super().__call__(query, key, value, attn_mask, dropout_p,
                                  is_causal, scale, enable_gqa)
        count = self.sampled_steps.get(self.step, 0)
        if count < 2:
            self.sampled_steps[self.step] = count + 1
            reference_query, reference_actual = query, actual
            indices = None
            if self.reference_query_limit is not None and query.shape[2] > self.reference_query_limit:
                indices = torch.linspace(0, query.shape[2] - 1, self.reference_query_limit,
                                         device=query.device).long()
                reference_query = query.index_select(2, indices)
                reference_actual = actual.index_select(2, indices)
            q, k, v = [t.detach().float().cpu() for t in (reference_query, key, value)]
            cpu = F.scaled_dot_product_attention(q, k, v)
            mps = F.scaled_dot_product_attention(reference_query, key, value)
            mps_fp32 = F.scaled_dot_product_attention(reference_query.float(), key.float(), value.float())
            record = {"backend": self.backend_name, "step": self.step, "call": self.calls,
                      "query_shape": list(query.shape), "key_shape": list(key.shape),
                      "reference_query_indices": indices.cpu().tolist() if indices is not None else None,
                      "reference_query_count": reference_query.shape[2],
                      "dtype": str(query.dtype),
                      "cumetal_vs_cpu_fp32": differences(reference_actual, cpu),
                      "mps_vs_cpu_fp32": differences(mps, cpu),
                      "mps_fp32_vs_cpu_fp32": differences(mps_fp32, cpu),
                      "cumetal_vs_cpu_rounded": differences(reference_actual, cpu.to(query.dtype)),
                      "cumetal_vs_mps": differences(reference_actual, mps)}
            self.audit.append(record)
            print("ATTENTION " + json.dumps(record), flush=True)
        return actual


class AuditedAttention(_AttentionAudit, CuMetalAttention):
    backend_name = "cumetal"


class AuditedBlasAttention(_AttentionAudit, CuMetalBlasAttention):
    backend_name = "cumetal-blas"
    # Full key context with sampled query rows avoids a huge CPU score matrix.
    reference_query_limit = 16


def create_attention_bridges(backends):
    factories = {"cumetal": AuditedAttention, "cumetal-blas": AuditedBlasAttention}
    return {name: factories[name]() for name in backends if name in factories}


def attention_evidence(bridges):
    return {"sampled_attention": [event for bridge in bridges.values() for event in bridge.audit],
            "cumetal_calls": sum(bridge.calls for bridge in bridges.values()),
            "apple_gpu_launches": sum(len(bridge.launches) for bridge in bridges.values()),
            "gpu_launch_count_scope": "runtime kernel records only; cuBLAS application receipts are not kernel provenance",
            "attention_backends": {name: {"calls": bridge.calls, "kernel_launches": len(bridge.launches),
                "application_completion_receipts": bridge.completion_receipts}
                for name, bridge in bridges.items()}}


@contextmanager
def capture_native_attention(records):
    from diffusers.models.transformers import transformer_cosmos
    original = transformer_cosmos.dispatch_attention_fn

    def dispatch(query, key, value, *args, **kwargs):
        result = original(query, key, value, *args, **kwargs)
        if len(records) < 2:
            records.append({"query_shape": list(query.shape), "dtype": str(query.dtype),
                            "device": str(query.device),
                            "finite": bool(torch.isfinite(result).all())})
        return result

    transformer_cosmos.dispatch_attention_fn = dispatch
    try:
        yield
    finally:
        transformer_cosmos.dispatch_attention_fn = original


def diagnostic_geometry(frames, height, width):
    if not 5 <= frames <= 93 or (frames - 1) % 4:
        raise ValueError("frames must be 4n+1 between 5 and 93")
    if (not 64 <= height <= 704 or not 64 <= width <= 1280 or
            height % 16 or width % 16):
        raise ValueError("height must be 64-704 and width 64-1280, both divisible by 16")
    # Predict2 2B: VAE compression (4,8,8), patch size (1,2,2), 16 heads.
    latent_frames = (frames - 1) // 4 + 1
    tokens = latent_frames * (height // 16) * (width // 16)
    score_elements = 16 * tokens * tokens
    blas_plans = {}
    for name, keys in (("self_attention", tokens), ("cross_attention", 512)):
        try:
            blas_plans[name] = blas_attention_plan(1, 16, tokens, keys, 128, 128)
        except NotImplementedError:
            blas_plans[name] = None
    return {"latent_frames": latent_frames, "self_attention_tokens": tokens,
            "dense_score_bytes": score_elements * 4,
            "cumetal_worker_supported": tokens <= 4096 and score_elements <= 67108864,
            "cumetal_blas_worker_supported": all(plan is not None for plan in blas_plans.values()),
            "cumetal_blas_plans": blas_plans}


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=OUT / "robot-welding-input.jpg")
    parser.add_argument("--prompt", default="A small toy robot moves gently across a tabletop, fixed camera.")
    parser.add_argument("--revision", default="f50c09f5d8ab133a90cac3f4886a6471e9ba3f18")
    parser.add_argument("--steps", type=int, default=8, help="denoising-loop model evaluations")
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--native-rope", action="store_true",
                        help="use unscaled temporal positions, matching NVIDIA Predict2's disabled FPS modulation")
    parser.add_argument("--posterior-mode", action="store_true",
                        help="experimental deterministic VAE conditioning; changes diffusion RNG consumption")
    parser.add_argument("--native-ab2", action="store_true",
                        help="experimental AB2: N evaluations = N-1 transitions plus one clean pass; 36 matches NVIDIA's 35+1")
    parser.add_argument("--export-fps", type=int, default=8,
                        help="playback rate only; does not change transformer temporal positions")
    parser.add_argument("--output-dir", type=Path, default=OUT / "cosmos-diagnostic")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--visual-quality", help="record an inspection assessment with --summarize-only")
    parser.add_argument("--backends", nargs="+", choices=("mps", "cumetal", "cumetal-blas"),
                        default=["mps", "cumetal"], help="cumetal-blas is an explicit experimental tiled cuBLAS/MPS backend")
    parser.add_argument("--denoised-euler", action="store_true",
                        help="experimental denoised-prediction to Euler-derivative adapter")
    args = parser.parse_args(argv)
    if len(set(args.backends)) != len(args.backends):
        parser.error("backends must be unique to preserve separate evidence")
    try:
        geometry = diagnostic_geometry(args.frames, args.height, args.width)
    except ValueError as error:
        parser.error(str(error))
    if args.native_ab2 and args.denoised_euler:
        parser.error("--native-ab2 consumes the existing derivative; do not combine it with --denoised-euler")
    minimum, maximum = (2, 36) if args.native_ab2 else (1, 35)
    if not minimum <= args.steps <= maximum or not 1 <= args.export_fps <= 60:
        parser.error(f"steps must be {minimum}-{maximum} for this sampler and export FPS must be 1-60")
    if not args.summarize_only:
        if (args.output_dir / "comparison.json").exists():
            parser.error("output directory already contains evidence; select a new directory")
        if not args.image.is_file():
            parser.error("an existing conditioning image is required")
        if args.visual_quality:
            parser.error("visual assessment is only accepted with --summarize-only")
        if "cumetal" in args.backends and not geometry["cumetal_worker_supported"]:
            parser.error("geometry exceeds the bounded CuMetal attention worker; use --backends mps for this diagnostic")
        if "cumetal-blas" in args.backends and not geometry["cumetal_blas_worker_supported"]:
            parser.error("geometry exceeds the experimental tiled cuBLAS memory/operation bounds")
    return args


def contract_metadata(args):
    differences = []
    if args.native_ab2:
        differences = [
            "N evaluations perform N-1 transitions plus the final clean prediction; NVIDIA N sampling steps use N+1 evaluations",
            "scheduler tensor arithmetic is FP32 instead of NVIDIA's default FP64; scalar coefficients use Python FP64",
            "earlier Diffusers BF16 x0 rounding is unchanged",
        ]
    if args.posterior_mode:
        differences.append("posterior mode does not consume the generator; same seed no longer means baseline initial noise")
    if differences:
        differences.append("no end-to-end NVIDIA equivalence or usable-video claim; geometry, kernels, and other pipeline contracts remain unqualified")
    adapter = "experimental_native_ab2_final_clean" if args.native_ab2 else (
        "experimental_x0_to_euler_derivative" if args.denoised_euler else "none")
    return {"sampler_adapter": adapter, "model_evaluations": args.steps,
            "model_evaluation_definition": "one denoising-loop evaluation with two CFG transformer forwards",
            "transition_steps": args.steps - 1, "final_clean_prediction": args.native_ab2,
            "conditioning_latents": "posterior_mode" if args.posterior_mode else "posterior_sample",
            "contract_reference_revision": NVIDIA_SOURCE_REVISION if args.native_ab2 or args.posterior_mode else None,
            "contract_experiment_differences": differences}


def main():
    args = parse_arguments()
    if args.summarize_only:
        path = args.output_dir / "comparison.json"
        result = json.loads(path.read_text())
        if "mps" in result["runs"]:
            comparisons = result.setdefault("comparisons_to_mps", {})
            for backend in result["runs"]:
                if backend != "mps":
                    comparisons.setdefault(backend, {})["frames_difference"] = frame_differences(
                        np.load(args.output_dir / f"{backend}-frames.npy"),
                        np.load(args.output_dir / "mps-frames.npy"))
            if "cumetal" in result["comparisons_to_mps"]:
                result["frames_difference"] = result["comparisons_to_mps"]["cumetal"]["frames_difference"]
        contact_sheet(args.output_dir)
        # Older harness runs retained the baseline's final step counter at
        # the first CuMetal call; the call IDs disambiguate those two samples.
        for sample in result.get("sampled_attention", []):
            if sample["call"] <= 2:
                sample["step"] = 0
        if args.visual_quality:
            result["visual_quality"] = args.visual_quality
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items()
                          if k not in ("sampled_attention", "native_attention")}, indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "comparison.json"
    report = {"status": "running", "model": MODEL, "revision": args.revision,
              "prompt": args.prompt, "image_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
              "seed": 20261009, "frames": args.frames, "height": args.height, "width": args.width,
              "steps": args.steps, "dtype": "torch.bfloat16", "runs": {},
              "temporal_fps": None if args.native_rope else 16, "export_fps": args.export_fps,
              "temporal_rope": "native_unscaled" if args.native_rope else "diffusers_default_1.5x",
              "geometry": diagnostic_geometry(args.frames, args.height, args.width),
              **contract_metadata(args),
              "cpu_arithmetic_fallback": False,
              "cpu_reference_scope": "sampled attention only; not full pretrained CPU inference"}
    started = time.monotonic()
    bridges = {}

    def save_report():
        report["elapsed_seconds"] = time.monotonic() - started
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    try:
        bridges = create_attention_bridges(args.backends)
        from diffusers import Cosmos2VideoToWorldPipeline
        from diffusers.utils import export_to_video, load_image
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable")
        save_report()
        with guardrail_weight_staging():
            pipeline = Cosmos2VideoToWorldPipeline.from_pretrained(
                MODEL, revision=args.revision, dtype=torch.bfloat16,
                use_safetensors=True, trust_remote_code=False, local_files_only=True,
                cache_dir=OUT / "hf-cache", safety_checker=make_safety_checker())
        pipeline.enable_sequential_cpu_offload(device="mps")
        image = load_image(str(args.image.resolve()))
        native_attention = []
        latent_runs = {}
        for backend in args.backends:
            run_started = time.monotonic()
            latents = []
            bridge = bridges.get(backend)
            if bridge is not None:
                bridge.step = 0

            def callback(_pipeline, step, timestep, tensors):
                tensor = tensors["latents"].detach().float().cpu()
                if not torch.isfinite(tensor).all():
                    raise RuntimeError("non-finite denoising latents")
                latents.append(tensor)
                if bridge is not None:
                    bridge.step = step + 1
                print(f"STEP {backend} {step + 1}/{args.steps} rms={tensor.square().mean().sqrt().item():.6g}",
                      flush=True)
                report["progress"] = {"backend": backend, "step": step + 1, "steps": args.steps,
                                      "latent_rms": float(tensor.square().mean().sqrt())}
                save_report()
                return tensors

            torch.manual_seed(report["seed"])
            context = bridge.diffusers() if bridge is not None else capture_native_attention(native_attention)
            print("RUN " + backend, flush=True)
            adapter = experimental_ab2_scheduler(pipeline) if args.native_ab2 else (
                denoised_prediction_adapter(pipeline.scheduler) if args.denoised_euler else nullcontext())
            posterior = experimental_posterior_mode(pipeline.vae) if args.posterior_mode else nullcontext()
            with torch.inference_mode(), context, adapter, posterior:
                frames = pipeline(image=image,
                    prompt=args.prompt,
                    height=report["height"], width=report["width"], num_frames=report["frames"],
                    num_inference_steps=args.steps, fps=report["temporal_fps"],
                    generator=torch.Generator(device="cpu").manual_seed(report["seed"]),
                    callback_on_step_end=callback).frames[0]
            array = np.asarray(frames)
            if (len(latents) != args.steps or len(frames) != report["frames"] or
                    not np.isfinite(array).all() or not array.std()):
                raise RuntimeError("missing, blank, or non-finite video")
            output = args.output_dir / f"{backend}.mp4"
            export_to_video(frames, str(output), fps=report["export_fps"])
            np.save(args.output_dir / f"{backend}-frames.npy", array)
            latent_runs[backend] = latents
            np.savez(args.output_dir / f"{backend}-latents.npz",
                     **{f"step_{i}": t.numpy() for i, t in enumerate(latents)})
            report["runs"][backend] = {"output": str(output.resolve()),
                "elapsed_seconds": time.monotonic() - run_started, "frame_std": float(array.std())}
            save_report()
        if (any(bridge.calls == 0 for bridge in bridges.values()) or
                "mps" in args.backends and not native_attention):
            raise RuntimeError("each selected attention path must execute")
        if "mps" in args.backends:
            b = np.load(args.output_dir / "mps-frames.npy")
            report["comparisons_to_mps"] = {}
            for backend in bridges:
                a = np.load(args.output_dir / f"{backend}-frames.npy")
                report["comparisons_to_mps"][backend] = {
                    "latents": [dict(step=i + 1, **differences(x, y)) for i, (x, y) in
                                enumerate(zip(latent_runs[backend], latent_runs["mps"], strict=True))],
                    "frames_difference": frame_differences(a, b)}
            if "cumetal" in report["comparisons_to_mps"]:
                report.update(report["comparisons_to_mps"]["cumetal"])
        contact_sheet(args.output_dir)
        report.update(status="execution_passed", native_attention=native_attention,
                      **attention_evidence(bridges),
                      visual_quality="requires inspection; no usable-video claim")
    except Exception as error:
        report.update(status="failed", error=str(error), **attention_evidence(bridges))
        raise
    finally:
        save_report()
    print("COMPARISON " + str(report_path.resolve()), flush=True)


if __name__ == "__main__":
    main()
