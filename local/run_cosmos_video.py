#!/usr/bin/env python3
"""Attempt pretrained Cosmos video inference with CuMetal attention and MPS math."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import time

# Unsupported MPS arithmetic must surface as an error instead of a CPU fallback.
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import GatedRepoError

from cosmos_cumetal import CuMetalAttention, OUT

MODEL = "nvidia/Cosmos-Predict2-2B-Video2World"


@contextmanager
def guardrail_weight_staging():
    """Load CUDA-tagged RetinaFace weights without requiring a CUDA device."""
    import torch
    import cosmos_guardrail.cosmos_guardrail as guardrail
    from cosmos_guardrail.cosmos_utils import _check_keys, _remove_prefix

    original = guardrail.load_model

    def load_model(model, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = _remove_prefix(state.get("state_dict", state), "module.")
        _check_keys(model, state)
        model.load_state_dict(state, strict=True)
        return model

    guardrail.load_model = load_model
    try:
        yield
    finally:
        guardrail.load_model = original


def make_safety_checker():
    from cosmos_guardrail.cosmos_guardrail import CosmosSafetyChecker, Qwen3Guard

    class AppleSafetyChecker(CosmosSafetyChecker):
        @property
        def device(self):
            return next(self.nn_models[0].parameters()).device

        @property
        def dtype(self):
            return next(self.nn_models[0].parameters()).dtype

        def to(self, *args, **kwargs):
            for model in self.nn_models:
                model.to(*args, **kwargs)
            return self

    checker = AppleSafetyChecker()
    for model in checker.nn_models:
        if isinstance(model, Qwen3Guard):
            # Upstream catches inference errors and returns safe. Refuse instead.
            model.is_safe = model.extract_label_and_categories
    return checker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--prompt", default="A small toy robot moves gently across a tabletop, fixed camera.")
    parser.add_argument("--frames", type=int, default=9)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--output", type=Path, default=OUT / "cosmos-video.mp4")
    args = parser.parse_args()
    if args.frames < 5 or args.frames > 17 or (args.frames - 1) % 4:
        parser.error("initial video gate supports 5, 9, 13, or 17 frames")
    if not 1 <= args.steps <= 35 or any(d < 64 or d > 320 or d % 16 for d in (args.height, args.width)):
        parser.error("initial gate needs 1-35 steps and dimensions 64-320 divisible by 16")
    OUT.mkdir(parents=True, exist_ok=True)
    report_path = OUT / "cosmos-video-results.json"
    report = {"status": "running", "model": MODEL, "attention_backend": "CuMetal",
              "gate": "execution_only; visual quality requires separate inspection",
              "other_tensor_operations": "PyTorch MPS; sequential weight staging on CPU",
              "cpu_arithmetic_fallback": False, "fp32_attention_accumulation": True,
              "frames": args.frames, "steps": args.steps, "size": [args.width, args.height]}
    bridge = None
    started = time.monotonic()
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        info = HfApi().model_info(MODEL)
        report["revision"] = info.sha
        config = hf_hub_download(MODEL, "model_index.json", revision=info.sha, cache_dir=OUT / "hf-cache")
        report["pipeline_config"] = json.loads(Path(config).read_text())
        if args.preflight:
            report["status"] = "preflight_passed"
            print("Checkpoint access verified; inference has not run.")
            return
        if not args.image or not args.image.is_file():
            parser.error("--image must name an existing conditioning image")
        import numpy as np
        import torch
        from diffusers import Cosmos2VideoToWorldPipeline
        from diffusers.utils import export_to_video, load_image
        if not torch.backends.mps.is_available():
            raise RuntimeError("Apple GPU MPS backend is unavailable")
        bridge = CuMetalAttention()
        print("Execution: CUDA-M attention, PyTorch MPS other tensor math; FP32 attention accumulation.", flush=True)
        with guardrail_weight_staging():
            safety_checker = make_safety_checker()
            pipeline = Cosmos2VideoToWorldPipeline.from_pretrained(MODEL, revision=info.sha,
                dtype=torch.bfloat16, use_safetensors=True, trust_remote_code=False,
                cache_dir=OUT / "hf-cache", safety_checker=safety_checker)
        # Keep the large text encoder's weights staged without running its math on the CPU.
        pipeline.enable_sequential_cpu_offload(device="mps")
        image = load_image(str(args.image.resolve()))
        with torch.inference_mode(), bridge.diffusers():
            frames = pipeline(image=image, prompt=args.prompt, height=args.height, width=args.width,
                              num_frames=args.frames, num_inference_steps=args.steps,
                              generator=torch.Generator(device="cpu").manual_seed(20261009)).frames[0]
        array = np.asarray(frames)
        if len(frames) != args.frames or not np.isfinite(array).all() or not array.std():
            raise RuntimeError("pipeline returned missing, non-finite, or blank frames")
        if bridge.calls == 0:
            raise RuntimeError("pipeline did not execute CUDA-M attention")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(frames, str(args.output.resolve()), fps=8)
        report.update(status="passed", output=str(args.output.resolve()), attention_calls=bridge.calls,
                      launches=bridge.launches)
        print("Generated: " + str(args.output.resolve()))
    except GatedRepoError as error:
        report.update(status="blocked_model_access", error=str(error))
        raise RuntimeError("Hugging Face denied access to a required checkpoint: " + str(error)) from None
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        if bridge is not None:
            report.update(attention_calls=bridge.calls, launches=bridge.launches)
        report["elapsed_seconds"] = time.monotonic() - started
        report_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
