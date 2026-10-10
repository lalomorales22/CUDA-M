#!/usr/bin/env python3
"""Compare retained Cosmos diagnostic runs without loading or rerunning models."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch

from diagnose_cosmos_video import differences, frame_differences

ROOT = Path(__file__).resolve().parent.parent
CORE_FIELDS = ("model", "revision", "prompt", "image_sha256", "seed", "frames", "height", "width",
               "steps", "dtype", "temporal_fps", "export_fps", "sampler_adapter")
OPTIONAL_FIELDS = ("temporal_rope", "model_evaluations", "model_evaluation_definition", "transition_steps",
                   "final_clean_prediction", "conditioning_latents", "contract_reference_revision",
                   "contract_experiment_differences")
BACKENDS = ("mps", "cumetal", "cumetal-blas")


def portable(path):
    return os.path.relpath(Path(path).resolve(), ROOT)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_contract(baseline, candidate):
    for label, report in (("baseline", baseline), ("candidate", candidate)):
        if report.get("status") != "execution_passed":
            raise ValueError(f"{label} must be a completed execution_passed diagnostic")
        missing = [key for key in CORE_FIELDS if key not in report]
        if missing:
            raise ValueError(f"{label} is missing required contract fields: {', '.join(missing)}")
        if report.get("cpu_arithmetic_fallback") is not False:
            raise ValueError(f"{label} must explicitly disable CPU arithmetic fallback")
        for key in ("frames", "height", "width", "steps", "export_fps"):
            if type(report[key]) is not int or report[key] <= 0:
                raise ValueError(f"{label} {key} must be a positive integer")
    mismatches = [key for key in CORE_FIELDS if baseline[key] != candidate[key]]
    mismatches += [key for key in OPTIONAL_FIELDS
                   if key in baseline and key in candidate and baseline[key] != candidate[key]]
    if mismatches:
        raise ValueError("contract mismatch: " + ", ".join(mismatches))
    return {"matching_core_fields": {key: baseline[key] for key in CORE_FIELDS},
            "matching_optional_fields": {key: baseline[key] for key in OPTIONAL_FIELDS
                                         if key in baseline and key in candidate},
            "missing_optional_contract_metadata": {
                label: [key for key in OPTIONAL_FIELDS if key not in report]
                for label, report in (("baseline", baseline), ("candidate", candidate))},
            "identity_limit": "matching recorded fields do not establish complete execution-contract identity"}


def load_frames(path, report):
    frames = np.load(path, allow_pickle=False)
    expected = (report["frames"], report["height"], report["width"], 3)
    if (frames.shape != expected or frames.dtype.kind not in "uif"
            or not np.isfinite(frames).all() or frames.min() < 0 or frames.max() > 255):
        raise ValueError("frames must be finite THWC RGB in [0,255] with report-matching shape")
    return frames


def load_latents(path, steps):
    with np.load(path, allow_pickle=False) as archive:
        keys = [f"step_{i}" for i in range(steps)]
        if len(archive.files) != steps or set(archive.files) != set(keys):
            raise ValueError("latent archive must contain exactly contiguous step_0 through step_N arrays")
        arrays = [archive[key].copy() for key in keys]
    shape = arrays[0].shape
    if (len(shape) != 5 or any(dim <= 0 for dim in shape)
            or any(a.shape != shape or a.dtype.kind != "f" or not np.isfinite(a).all() for a in arrays)):
        raise ValueError("latents must be finite floating point arrays with one consistent nonempty 5D shape")
    return arrays


def codec_metadata(path):
    executable = shutil.which("ffprobe")
    if executable is None or not path.is_file():
        return {"inspected": False, "reason": "ffprobe unavailable" if executable is None else "video unavailable"}
    try:
        result = subprocess.run([executable, "-v", "error", "-select_streams", "v:0", "-show_entries",
                                 "stream=codec_name,width,height,nb_frames,r_frame_rate,avg_frame_rate,duration:format=duration",
                                 "-of", "json", str(path)], check=True, capture_output=True, text=True, timeout=15)
        return {"inspected": True, "metadata": json.loads(result.stdout)}
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        return {"inspected": False, "reason": str(error)}


def make_sheet(baseline, candidate, backends, destination):
    from PIL import Image, ImageDraw

    count, height, width, _ = baseline.shape
    tile_width = min(width, 480)
    tile_height = round(height * tile_width / width)
    sheet = Image.new("RGB", (tile_width * 3, (tile_height + 24) * 2), "white")
    draw = ImageDraw.Draw(sheet)
    names = {"mps": "MPS", "cumetal": "CUDA-M", "cumetal-blas": "CUDA-M cuBLAS"}
    for row, (array, backend) in enumerate(zip((baseline, candidate), backends, strict=True)):
        for column, frame in enumerate((0, count // 2, count - 1)):
            x, y = column * tile_width, row * (tile_height + 24)
            draw.text((x + 4, y + 4), f"{names[backend]}: frame {frame}", fill="black")
            image = Image.fromarray(np.rint(array[frame]).astype(np.uint8))
            image = image.resize((tile_width, tile_height), Image.Resampling.LANCZOS)
            sheet.paste(image, (x, y + 24))
    sheet.save(destination)


def compare_runs(baseline_dir, candidate_dir, baseline_backend, candidate_backend, output_dir):
    baseline_dir, candidate_dir, output_dir = map(Path, (baseline_dir, candidate_dir, output_dir))
    if baseline_backend not in BACKENDS or candidate_backend not in BACKENDS:
        raise ValueError("unsupported backend")
    report_path = output_dir / "comparison.json"
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError("output directory already exists; select a new output directory")
    directories = (baseline_dir, candidate_dir)
    backends = (baseline_backend, candidate_backend)
    reports = [json.loads((directory / "comparison.json").read_text()) for directory in directories]
    contract = validate_contract(*reports)
    for report, backend in zip(reports, backends, strict=True):
        if backend not in report.get("runs", {}):
            raise ValueError(f"completed report has no selected backend: {backend}")
    frames = [load_frames(directory / f"{backend}-frames.npy", report)
              for directory, backend, report in zip(directories, backends, reports, strict=True)]
    latents = [load_latents(directory / f"{backend}-latents.npz", report["steps"])
               for directory, backend, report in zip(directories, backends, reports, strict=True)]
    if any(a.shape != b.shape for a, b in zip(*latents, strict=True)):
        raise ValueError("baseline and candidate latent shapes differ")
    numerical = [dict(step_index=i, shape=list(a.shape),
                      **differences(torch.from_numpy(b), torch.from_numpy(a)))
                 for i, (a, b) in enumerate(zip(*latents, strict=True))]
    pixel_metrics = [dict(frame=i, **frame_differences(b, a))
                     for i, (a, b) in enumerate(zip(*frames, strict=True))]
    result = {"status": "compared", "baseline": {"directory": portable(baseline_dir), "backend": baseline_backend},
              "candidate": {"directory": portable(candidate_dir), "backend": candidate_backend},
              **contract, "cpu_arithmetic_fallback": False, "models_rerun": False,
              "numerical_parity": "not assessed; no pass tolerance defined",
              "human_assessment": "pending visual inspection",
              "difference_direction": "candidate minus baseline",
              "latent_step_differences": numerical, "per_frame_rgb_difference_0_255": pixel_metrics,
              "all_frames_rgb_difference_0_255": frame_differences(frames[1], frames[0]),
              "frame_array_shapes": [list(array.shape) for array in frames],
              "frame_array_dtypes": [str(array.dtype) for array in frames],
              "codec": {}, "artifact_sha256": {}}
    for label, directory, backend in zip(("baseline", "candidate"), directories, backends, strict=True):
        result["codec"][label] = codec_metadata(directory / f"{backend}.mp4")
        for filename in ("comparison.json", f"{backend}-frames.npy", f"{backend}-latents.npz", f"{backend}.mp4"):
            path = directory / filename
            if path.is_file():
                result["artifact_sha256"][portable(path)] = digest(path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    make_sheet(*frames, backends, output_dir / "comparison.png")
    result["contact_sheet"] = portable(output_dir / "comparison.png")
    result["artifact_sha256"][result["contact_sheet"]] = digest(output_dir / "comparison.png")
    with report_path.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-backend", choices=BACKENDS, default="mps")
    parser.add_argument("--candidate-backend", choices=BACKENDS, default="cumetal-blas")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare_runs(args.baseline, args.candidate, args.baseline_backend, args.candidate_backend, args.output_dir)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps({"report": portable(args.output_dir / "comparison.json"),
                      "latent_steps": len(result["latent_step_differences"]),
                      "rgb_difference": result["all_frames_rgb_difference_0_255"],
                      "numerical_parity": result["numerical_parity"]}, indent=2))


if __name__ == "__main__":
    main()
