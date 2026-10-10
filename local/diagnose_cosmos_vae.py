#!/usr/bin/env python3
"""Cached-only Wan VAE precision and temporal reconstruction diagnostic."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
MODEL = "nvidia/Cosmos-Predict2-2B-Video2World"
REVISION = "f50c09f5d8ab133a90cac3f4886a6471e9ba3f18"


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_final_latents(path, shape):
    with np.load(path, allow_pickle=False) as archive:
        if not archive.files or any(not re.fullmatch(r"step_\d+", key) for key in archive.files):
            raise ValueError("latent archive must contain only step_N arrays")
        indices = sorted(int(key[5:]) for key in archive.files)
        if indices != list(range(len(indices))):
            raise ValueError("latent archive steps must be contiguous from zero")
        array = archive[f"step_{indices[-1]}"]
        if array.shape != shape or array.dtype.kind != "f" or not np.isfinite(array).all():
            raise ValueError("final latents have invalid shape, dtype, or non-finite values")
        return torch.from_numpy(array.copy()).float(), indices[-1]


def denormalize_latents(latents, config, sigma_data=1.0):
    if not math.isfinite(sigma_data) or sigma_data <= 0:
        raise ValueError("sigma_data must be finite and positive")
    mean = torch.tensor(config.latents_mean, device=latents.device, dtype=torch.float32)
    std = torch.tensor(config.latents_std, device=latents.device, dtype=torch.float32)
    if (mean.numel() != latents.shape[1] or std.numel() != latents.shape[1]
            or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any()):
        raise ValueError("invalid per-channel latent normalization")
    return latents.float() * std.view(1, -1, 1, 1, 1) / sigma_data + mean.view(1, -1, 1, 1, 1)


def frame_metrics(actual, reference):
    actual = np.asarray(actual, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    if actual.shape != reference.shape or actual.ndim != 4 or actual.shape[-1] != 3:
        raise ValueError("metrics require matching THWC RGB arrays")
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError("non-finite decoded frames")
    records = []
    for i, (a, b) in enumerate(zip(actual, reference, strict=True)):
        delta = a - b
        mse = float(np.square(delta).mean())
        records.append({"frame": i, "finite": True, "mae": float(np.abs(delta).mean()),
                        "rmse": math.sqrt(mse), "max_abs": float(np.abs(delta).max()),
                        "psnr_db": 10 * math.log10(4 / mse) if mse else None,
                        "psnr_exact": mse == 0,
                        "clipped_fraction": float((np.abs(a) >= 0.999).mean())})
    return records


def tensor_frames(tensor, frames, height, width):
    if tuple(tensor.shape) != (1, 3, frames, height, width):
        raise ValueError("VAE returned unexpected output shape")
    result = tensor.detach().float().cpu().permute(0, 2, 3, 4, 1).numpy()[0]
    if not np.isfinite(result).all():
        raise ValueError("VAE returned non-finite values")
    return result.copy()


def contact_sheet(rows, destination):
    from PIL import Image, ImageDraw

    frames, height, width, _ = rows[0][1].shape
    sheet = Image.new("RGB", (width * 3, (height + 26) * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for row, (label, array) in enumerate(rows):
        for column, frame in enumerate((0, frames // 2, frames - 1)):
            pixels = np.rint((array[frame].clip(-1, 1) + 1) * 127.5).astype(np.uint8)
            x, y = column * width, row * (height + 26)
            draw.text((x + 4, y + 4), f"{label}: frame {frame}", fill="black")
            sheet.paste(Image.fromarray(pixels), (x, y + 26))
    sheet.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=OUT / "robot-welding-input.jpg")
    parser.add_argument("--latents", type=Path, default=OUT / "cosmos-diagnostic/mps-latents.npz")
    parser.add_argument("--output-dir", type=Path, default=OUT / "cosmos-vae-diagnostic")
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--dtypes", choices=("bf16", "fp32"), nargs="+", default=["bf16", "fp32"])
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=320)
    args = parser.parse_args()
    if args.frames < 5 or args.frames > 17 or (args.frames - 1) % 4:
        parser.error("frames must be 5, 9, 13, or 17")
    if any(dim < 64 or dim > 320 or dim % 16 for dim in (args.height, args.width)):
        parser.error("dimensions must be 64-320, divisible by 16")
    if len(set(args.dtypes)) != len(args.dtypes):
        parser.error("dtypes must not repeat")
    if not args.image.is_file() or not args.latents.is_file():
        parser.error("existing image and retained latent archive are required")
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable; CPU fallback is not allowed")
    # Refuse to replace an earlier evidence set, including a failed run.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "results.json"
    if report_path.exists():
        parser.error("output directory already contains results; select a new directory")
    report = {"status": "running", "model": MODEL, "revision": args.revision,
              "component": "AutoencoderKLWan", "device": args.device,
              "frames": args.frames, "height": args.height, "width": args.width,
              "cpu_arithmetic_fallback": False,
              "scope": "VAE diagnostic only; no transformer or safety-qualified generated video",
              "reconstruction_posterior": "mode; no sampling noise",
              "image_sha256": file_hash(args.image), "latents_sha256": file_hash(args.latents),
              "runs": {}}
    started = time.monotonic()

    def save():
        report["elapsed_seconds"] = time.monotonic() - started
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    rows = []
    results = {}
    try:
        from diffusers import AutoencoderKLWan
        from diffusers.video_processor import VideoProcessor
        from diffusers.utils import load_image

        shape = (1, 16, (args.frames - 1) // 4 + 1, args.height // 8, args.width // 8)
        normalized, step = load_final_latents(args.latents, shape)
        report["retained_latent_step_index"] = step
        processor = VideoProcessor(vae_scale_factor=8)
        image = load_image(str(args.image.resolve()))
        pixels = processor.preprocess(image, args.height, args.width).unsqueeze(2)
        repeated = pixels.repeat(1, 1, args.frames, 1, 1)
        reference = tensor_frames(repeated, args.frames, args.height, args.width)
        rows.append(("Input repeated", reference))
        for precision in args.dtypes:
            run_started = time.monotonic()
            dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[precision]
            vae = None
            print(f"VAE {args.device} {precision}: loading cached weights", flush=True)
            try:
                # Only this bounded VAE resides on the device, never the transformer/text encoder.
                vae = AutoencoderKLWan.from_pretrained(
                    MODEL, subfolder="vae", revision=args.revision, dtype=dtype,
                    use_safetensors=True, local_files_only=True, cache_dir=OUT / "hf-cache")
                vae.eval().requires_grad_(False).to(args.device)
                with torch.inference_mode():
                    encoded = vae.encode(repeated.to(device=args.device, dtype=dtype)).latent_dist.mode()
                    reconstruction = tensor_frames(vae.decode(encoded).sample,
                                                   args.frames, args.height, args.width)
                    del encoded
                    raw = denormalize_latents(normalized.to(args.device), vae.config)
                    decoded = tensor_frames(vae.decode(raw.to(dtype)).sample,
                                            args.frames, args.height, args.width)
                    del raw
                record = {"elapsed_seconds": time.monotonic() - run_started,
                          "model_parameter_dtype": str(vae.dtype),
                          "reconstruction_vs_input": frame_metrics(reconstruction, reference),
                          "retained_decode_finite": True,
                          "retained_decode_per_frame_rms": np.sqrt(
                              np.square(decoded).mean(axis=(1, 2, 3))).tolist()}
                report["runs"][precision] = record
                np.save(args.output_dir / f"{precision}-reconstruction.npy", reconstruction)
                np.save(args.output_dir / f"{precision}-retained-decode.npy", decoded)
                rows.extend([(f"{precision} reconstruction", reconstruction),
                             (f"{precision} retained decode", decoded)])
                results[precision] = (reconstruction, decoded)
                contact_sheet(rows, args.output_dir / "comparison.png")
                save()
                print(f"VAE {precision}: {record['elapsed_seconds']:.2f}s, reconstruction frame RMSE "
                      + str([round(x["rmse"], 5) for x in record["reconstruction_vs_input"]]), flush=True)
            finally:
                if vae is not None:
                    vae.clear_cache()
                    vae.to("cpu")
                    del vae
                gc.collect()
                if args.device == "mps":
                    torch.mps.empty_cache()
        if set(results) == {"bf16", "fp32"}:
            report["bf16_vs_fp32"] = {
                "reconstruction": frame_metrics(results["bf16"][0], results["fp32"][0]),
                "retained_decode": frame_metrics(results["bf16"][1], results["fp32"][1])}
        report["status"] = "execution_passed; inspect reconstruction and retained-decode quality"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        save()
    print("VAE report: " + str(report_path.resolve()), flush=True)


if __name__ == "__main__":
    main()
