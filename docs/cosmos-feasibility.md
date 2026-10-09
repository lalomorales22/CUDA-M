# Cosmos on CUDA-M

## Verified on 2026-10-09

Apple M4 Mac mini, 32 GiB memory, macOS 27.0, LLVM 22.1.7, Metal compiler
32023.921, PyTorch 2.14.1, Diffusers 0.41.0.

`./cuda-mac test` passes the original six numerical Apple-GPU launches.
`./cuda-mac cosmos-test` verifies 13 attention calls / 39 Apple-GPU launches.
Five FP32 cases cover multiple batches/heads, unequal sequence lengths, causal
masks, custom scale, and 128-channel heads. Maximum absolute error against
PyTorch CPU: 8.35e-7. FP16/BF16 inputs explicitly use FP32 accumulation and
return their original dtype; observed errors: 0.0009765625 / 0.00390625.

Diffusers' actual `CosmosTransformer3DModel` passes a reduced random-weight
forward check: one block, two heads, two latent frames, self and cross attention.
With other tensor arithmetic on MPS and attention on CuMetal, maximum absolute
error against CPU is 8.35e-7. **This is not pretrained generation.**
Nine negative checks reject unsupported modes, invalid data, and malformed
worker input. Successful calls require completed, exact Apple-GPU provenance
for all three kernels. Reports record source, binary, and runtime hashes.

A pretrained `Cosmos-Predict2-2B-Video2World` execution completed on this Mac:
nine frames, 320x192, eight denoising steps, about 347 seconds. All 896 attention
calls produced 2,688 exact successful Apple M4 kernel launch records. The MP4
decodes to nine changing, nonblank RGB frames, but visual inspection shows
severely distorted generated frames. **This passes execution, not visual quality.**
The cause is not isolated; low test settings and backend parity still need
controlled comparison. Evidence and output are in `local/out/cosmos-video-results.json`,
`local/out/cosmos-video.mp4`, and `local/out/cosmos-video-contact-sheet.png`.

## Implementation limits

`local/cosmos_cumetal.py` stages PyTorch tensors through CPU memory to the CUDA
executable in `local/cosmos_attention.cu`. Attention arithmetic runs on the Apple
GPU through CuMetal. The candidate video runner uses MPS for other model math.
This is an explicit bridge, not a replacement for PyTorch's CUDA device backend.

The inference-only worker starts a process per call and materializes scores.
It is not performance-tuned. Batch * heads is bounded to 128, sequences to 4096,
channel widths to 128, and scores to 64 million FP32 values. Explicit masks,
dropout, GQA, and backward are refused. The scoped Cosmos dispatch replacement
must not run concurrently with other inference in the same Python process.

The pretrained target is `nvidia/Cosmos-Predict2-2B-Video2World`. Local login
and target/guardrail access succeed; roughly 14 GB of primary weights downloaded.
The full small-test pipeline, guardrail checks, and video export ran successfully.
Production-size MPS coverage, peak memory usage, performance tuning, and usable
visual quality remain unverified. Safety checks are not bypassed.

Local guardrail portability fixes stage CUDA-tagged RetinaFace weights on CPU
with `weights_only=True` and strict state-dict validation, expose the checker's
actual parameter device/dtype, and preserve `.to()` chaining. Qwen safety-check
exceptions propagate instead of upstream's fail-open return. Three download-free
tests cover weight mapping, invalid checkpoints, patch restoration, device/dtype,
and refusal on inference errors (`local/test_cosmos_video.py`).

## Reproduce

The Python 3.14 arm64 environment is isolated in ignored `local/out/cosmos-venv`:

```bash
python3 -m venv local/out/cosmos-venv
local/out/cosmos-venv/bin/python -m pip install --dry-run --ignore-installed --only-binary=:all: --report local/out/cosmos-dependencies.json -r local/cosmos-requirements.txt
```

Audit the report before installing:

```bash
local/out/cosmos-venv/bin/python local/audit_cosmos_dependencies.py local/out/cosmos-dependencies.json --install
```

Unresolved advisories prevent installation. For the narrow reviewed NLTK
exception below, explicitly add `--allow-advisory GHSA-8mgp-746c-j5xp
--reason "Reviewed Cosmos tokenization-only use; no affected artifact APIs"`.
Reassess that exception for your workload; other findings still block installation.
The pinned environment requires Python 3.14 on arm64 macOS; it is not qualified
on other Python versions. Install public Xcode Metal tools and the build
dependencies listed in `local/README.md`, then run `./cuda-mac build` first.
Authenticate with `hf auth login` and obtain approval for both
`nvidia/Cosmos-Predict2-2B-Video2World` and `nvidia/Cosmos-1.0-Guardrail`.
Model downloads require roughly 16 GB beyond the Python environment and build.
In the prepared environment:

```bash
./cuda-mac cosmos-test
local/out/cosmos-venv/bin/python local/test_cosmos_video.py
python3 local/test_cosmos_dependency_audit.py
local/out/cosmos-venv/bin/python local/run_cosmos_video.py --preflight
local/out/cosmos-venv/bin/python local/run_cosmos_video.py --image /absolute/path/input.png --prompt "A small toy robot moves gently across a tabletop, fixed camera."
```

The candidate runner defaults to nine frames, 320x192, eight steps. These small
settings are an initial compatibility gate, not a promise of model quality.
It retains the upstream checker and uses sequential MPS weight offload.
CPU arithmetic fallback is disabled; RNG, tokenization, staging, and encoding
remain host work. Preflight only checks checkpoint access, not inference.

## Security review

The 62 resolved Python versions were checked against OSV before installation.
PyTorch 2.14.1 and Diffusers 0.41.0 exceed the reviewed checkpoint-loading and
remote-code fix versions. The runner requests built-in pipelines, safetensors,
and no remote custom code. Reports remain under `local/out/`.

NLTK 3.10.3 has [CVE-2026-81726 / GHSA-8mgp-746c-j5xp](https://github.com/advisories/GHSA-8mgp-746c-j5xp),
with no published patched version at this check. It was installed with a
documented local-workload exception after inspecting cosmos-guardrail 0.3.2:
that code uses `word_tokenize` and `WordNetLemmatizer` with fixed NVIDIA cache
paths, rather than the affected artifact train/save/load APIs accepting
caller-controlled paths. The finding remains recorded; this exception does
not approve those APIs or imply safety for other NLTK uses.

## Compiler regression

CUDA attention exposed LLVM's `-Inf` spelling leaking into Metal. The typed
NVVM importer now emits non-finite FP32 constants using exact raw bits,
preserving infinity signs and NaN payloads. A unit test and registered GPU
functional gate cover positive/negative infinity, NaN, and signed zero.
The executable test harness now handles empty arrays on macOS Bash 3.2.
