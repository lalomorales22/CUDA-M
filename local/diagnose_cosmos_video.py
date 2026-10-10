#!/usr/bin/env python3
"""Same-seed pretrained MPS/CuMetal comparison with sampled CPU SDPA references."""
import argparse
from contextlib import contextmanager, nullcontext
import json
import os
from pathlib import Path
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np
import torch
import torch.nn.functional as F

from cosmos_cumetal import CuMetalAttention, OUT
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
    backends = [backend for backend in ("mps", "cumetal")
                if (directory / f"{backend}-frames.npy").is_file()]
    arrays = [np.load(directory / f"{backend}-frames.npy") for backend in backends]
    height, width = arrays[0].shape[1:3]
    sheet = Image.new("RGB", (width * 3, (height + 24) * len(arrays)), "white")
    draw = ImageDraw.Draw(sheet)
    labels = {"mps": "MPS", "cumetal": "CUDA-M"}
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


class AuditedAttention(CuMetalAttention):
    def __init__(self):
        super().__init__()
        self.step = 0
        self.sampled_steps = {}
        self.audit = []

    def __call__(self, query, key, value, *args, **kwargs):
        actual = super().__call__(query, key, value, *args, **kwargs)
        count = self.sampled_steps.get(self.step, 0)
        if count < 2:
            self.sampled_steps[self.step] = count + 1
            q, k, v = [t.detach().float().cpu() for t in (query, key, value)]
            # The Cosmos dispatcher supplies only unmasked, non-causal attention.
            if args and (args[0] is not None or args[1] != 0 or args[2]):
                raise RuntimeError("audit only qualifies unmasked non-causal attention")
            cpu = F.scaled_dot_product_attention(q, k, v)
            mps = F.scaled_dot_product_attention(query, key, value)
            mps_fp32 = F.scaled_dot_product_attention(query.float(), key.float(), value.float())
            record = {"step": self.step, "call": self.calls,
                      "query_shape": list(query.shape), "key_shape": list(key.shape),
                      "dtype": str(query.dtype),
                      "cumetal_vs_cpu_fp32": differences(actual, cpu),
                      "mps_vs_cpu_fp32": differences(mps, cpu),
                      "mps_fp32_vs_cpu_fp32": differences(mps_fp32, cpu),
                      "cumetal_vs_cpu_rounded": differences(actual, cpu.to(query.dtype)),
                      "cumetal_vs_mps": differences(actual, mps)}
            self.audit.append(record)
            print("ATTENTION " + json.dumps(record), flush=True)
        return actual


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=OUT / "robot-welding-input.jpg")
    parser.add_argument("--revision", default="f50c09f5d8ab133a90cac3f4886a6471e9ba3f18")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=OUT / "cosmos-diagnostic")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--visual-quality", help="record an inspection assessment with --summarize-only")
    parser.add_argument("--backends", nargs="+", choices=("mps", "cumetal"), default=["mps", "cumetal"])
    parser.add_argument("--denoised-euler", action="store_true",
                        help="experimental denoised-prediction to Euler-derivative adapter")
    args = parser.parse_args()
    if args.summarize_only:
        path = args.output_dir / "comparison.json"
        result = json.loads(path.read_text())
        if set(result["runs"]) == {"mps", "cumetal"}:
            result["frames_difference"] = frame_differences(
                np.load(args.output_dir / "cumetal-frames.npy"),
                np.load(args.output_dir / "mps-frames.npy"))
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
    if not args.image.is_file() or not 1 <= args.steps <= 35:
        parser.error("an existing conditioning image and 1-35 steps are required")
    if args.visual_quality:
        parser.error("visual assessment is only accepted with --summarize-only")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "comparison.json"
    report = {"status": "running", "model": MODEL, "revision": args.revision,
              "seed": 20261009, "frames": 9, "height": 192, "width": 320,
              "steps": args.steps, "dtype": "torch.bfloat16", "runs": {},
              "sampler_adapter": "experimental_x0_to_euler_derivative" if args.denoised_euler else "none",
              "cpu_arithmetic_fallback": False,
              "cpu_reference_scope": "sampled attention only; not full pretrained CPU inference"}
    started = time.monotonic()
    bridge = AuditedAttention()

    def save_report():
        report["elapsed_seconds"] = time.monotonic() - started
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    try:
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
            bridge.step = 0

            def callback(_pipeline, step, timestep, tensors):
                tensor = tensors["latents"].detach().float().cpu()
                if not torch.isfinite(tensor).all():
                    raise RuntimeError("non-finite denoising latents")
                latents.append(tensor)
                bridge.step = step + 1
                print(f"STEP {backend} {step + 1}/{args.steps} rms={tensor.square().mean().sqrt().item():.6g}",
                      flush=True)
                return tensors

            torch.manual_seed(report["seed"])
            context = bridge.diffusers() if backend == "cumetal" else capture_native_attention(native_attention)
            print("RUN " + backend, flush=True)
            adapter = denoised_prediction_adapter(pipeline.scheduler) if args.denoised_euler else nullcontext()
            with torch.inference_mode(), context, adapter:
                frames = pipeline(image=image,
                    prompt="A small toy robot moves gently across a tabletop, fixed camera.",
                    height=report["height"], width=report["width"], num_frames=report["frames"],
                    num_inference_steps=args.steps,
                    generator=torch.Generator(device="cpu").manual_seed(report["seed"]),
                    callback_on_step_end=callback).frames[0]
            array = np.asarray(frames)
            if (len(latents) != args.steps or len(frames) != report["frames"] or
                    not np.isfinite(array).all() or not array.std()):
                raise RuntimeError("missing, blank, or non-finite video")
            output = args.output_dir / f"{backend}.mp4"
            export_to_video(frames, str(output), fps=8)
            np.save(args.output_dir / f"{backend}-frames.npy", array)
            latent_runs[backend] = latents
            np.savez(args.output_dir / f"{backend}-latents.npz",
                     **{f"step_{i}": t.numpy() for i, t in enumerate(latents)})
            report["runs"][backend] = {"output": str(output.resolve()),
                "elapsed_seconds": time.monotonic() - run_started, "frame_std": float(array.std())}
            save_report()
        if ("cumetal" in args.backends and bridge.calls == 0 or
                "mps" in args.backends and not native_attention):
            raise RuntimeError("each selected attention path must execute")
        if set(args.backends) == {"mps", "cumetal"}:
            report["latents"] = [dict(step=i + 1, **differences(a, b)) for i, (a, b) in
                                 enumerate(zip(latent_runs["cumetal"], latent_runs["mps"], strict=True))]
            a = np.load(args.output_dir / "cumetal-frames.npy")
            b = np.load(args.output_dir / "mps-frames.npy")
            report["frames_difference"] = frame_differences(a, b)
        contact_sheet(args.output_dir)
        report.update(status="execution_passed", native_attention=native_attention,
                      sampled_attention=bridge.audit, cumetal_calls=bridge.calls,
                      apple_gpu_launches=len(bridge.launches),
                      visual_quality="requires inspection; no usable-video claim")
    except Exception as error:
        report.update(status="failed", error=str(error), sampled_attention=bridge.audit,
                      cumetal_calls=bridge.calls, apple_gpu_launches=len(bridge.launches))
        raise
    finally:
        save_report()
    print("COMPARISON " + str(report_path.resolve()), flush=True)


if __name__ == "__main__":
    main()
