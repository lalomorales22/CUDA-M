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

### Same-seed diagnostic

A follow-up holds the checkpoint revision, image, prompt, seed, BF16 dtype,
nine frames, 320x192 size, and eight steps fixed. Native MPS attention and
CuMetal attention both complete, with other model math on MPS and no automatic
CPU arithmetic fallback. The CuMetal run executes 896 calls / 2,688 verified
Apple-GPU launches. Native MPS takes 79.82 seconds; CuMetal takes 313.37 seconds
including sampled audits. These are local timings, not a controlled benchmark.

Both videos show severe distortion after the conditioning frame. On 16 sampled
self/cross attention calls, maximum relative RMS error against the CPU FP32
reference rounded to BF16 is `0.000151435` (0.0152%). Against native MPS it is
`0.000142592` (0.0143%). Those samples do not qualify every layer or full-model
parity: latent relative RMS differences grow from `0.00259` after step one to
`0.31560` at the end. Mean decoded RGB difference is 17.14 on the 0-255 scale.

This isolates a visual failure that also occurs without CuMetal attention;
it does not prove all CuMetal arithmetic correct or establish full-model parity.
Full pretrained CPU inference was not attempted. CPU references cover sampled
attention and the separate reduced random-weight transformer test.

An explicit experimental denoised-prediction-to-Euler-derivative adapter was
tested on MPS. It worsens the video and drives the final latent RMS to 119.894,
so **it is rejected as a fix** and is not enabled in the normal video runner.
The installed pipeline matches the official Diffusers v0.41.0 source. Next
quality investigations should check trained dimensions/frame counts, step count,
precision, and the model/pipeline contract rather than infer a generic CUDA gap.
See the [portable diagnostic snapshot](../local/verification/2026-10-09/cosmos-diagnostic-summary.json).

```bash
./cuda-mac cosmos-test
local/out/cosmos-venv/bin/python local/test_cosmos_diagnostic.py
local/out/cosmos-venv/bin/python local/diagnose_cosmos_video.py --image /absolute/path/input.png
local/out/cosmos-venv/bin/python local/diagnose_cosmos_video.py --summarize-only
```

The diagnostic uses cached pretrained weights (`local_files_only=True`) and
retains the safety checker. Outputs, step latents, samples, and the comparison
sheet stay in ignored `local/out/cosmos-diagnostic/`; previous video evidence
is preserved. The original seven download-free tests check metrics, unsigned frame arithmetic,
dispatch restoration, the experimental adapter's analytic step, and zero-sigma
refusal. The adapter is available only as an explicitly selected experiment.

### Recognizable 720p MPS clip

A cached-only native MPS run now produces a recognizable welding scene across
all nine inspected frames at **1280x704**, eight denoising steps, and unscaled
temporal rotary positions (`--native-rope`). The clip is 0.5625 seconds at
16 fps. The scene remains coherent with changing smoke and sparks, but visible
softness and texture artifacts remain. This is a tiny visual smoke test, not
production quality or prompt-adherence qualification: the earlier toy-robot
prompt was deliberately held fixed despite the welding input image.

The run took 555.00 seconds including conditioning, inference, decoding and
safety checks (561.29 seconds overall). All other model math and attention used
PyTorch MPS; CPU arithmetic fallback was disabled. **This result contains zero
CuMetal attention calls and is not proof of CUDA-M pretrained parity.**
The earlier source-compiled CuMetal video remains distorted at its small size.

Separate 35-step, 320x192 runs remain distorted, both with the original temporal
positions and with unscaled positions. VAE reconstruction works in BF16 and
FP32 across nine frames; decoding the earlier bad latents stays distorted in
both precisions. These observations narrow the problem but do not isolate a
single cause. See the [quality evidence](../local/verification/2026-10-09/COSMOS-QUALITY.md).

```bash
local/out/cosmos-venv/bin/python local/diagnose_cosmos_video.py \
  --backends mps --native-rope --steps 8 --height 704 --width 1280 \
  --frames 9 --export-fps 16 --image /absolute/path/input.png \
  --output-dir local/out/cosmos-720p-rerun
```

The output directory must not already contain a comparison report. This command
requires the cached model/guardrail environment described below; it downloads
no weights and retains the safety checker. Its output is `mps.mp4` in that
directory. It was tested on this 32 GiB M4 only.

`--native-rope` follows NVIDIA Predict2's disabled FPS modulation. The installed
Diffusers pipeline otherwise passes 16 fps to a rotary embedding based on 24 fps,
scaling temporal positions by 1.5. This mismatch is also tracked in
[Diffusers issue 14768](https://github.com/huggingface/diffusers/issues/14768).
Playback fps is independent of this transformer setting.

Two further flags, `--posterior-mode` and `--native-ab2`, reproduce selected
parts of NVIDIA's initial-release conditioning/sampler contract. Their analytic
and synthetic-pipeline tests pass, but no pretrained video quality result is
claimed for them. AB2 with 36 evaluations means 35 transitions plus a final
clean prediction; 35 evaluations preserves the earlier 35-evaluation budget.
Earlier BF16 rounding and FP32-versus-FP64 arithmetic differences remain.
The older `--denoised-euler` experiment is not compatible with AB2 and remains a
rejected fix: the installed pipeline already performs that derivative conversion.

The [bounded REA investigation](../local/verification/2026-10-09/COSMOS-REA.md)
did not produce a new successful Cosmos debugger trace. A separate normal
execution passed its independent numerical test; it is not a substitute for
the missing instrumented evidence.

## Implementation limits

`local/cosmos_cumetal.py` stages PyTorch tensors through CPU memory to the CUDA
executable in `local/cosmos_attention.cu`. Attention arithmetic runs on the Apple
GPU through CuMetal. The candidate video runner uses MPS for other model math.
This is an explicit bridge, not a replacement for PyTorch's CUDA device backend.

The original inference-only worker starts a process per call and materializes scores.
It is not performance-tuned. Batch * heads is bounded to 128, sequences to 4096,
channel widths to 128, and scores to 64 Mi FP32 values. Explicit masks,
dropout, GQA, and backward are refused. The scoped Cosmos dispatch replacement
must not run concurrently with other inference in the same Python process.

### Experimental tiled attention

`local/cosmos_attention_blas.cu` adds a separate source-compiled worker for this
named workload. Query tiles use the existing clean-room cuBLAS shim for two
public MPS matrix multiplications and a CUDA-to-Metal row-softmax kernel.
The original three-kernel dense worker remains available and unchanged.

Standalone attention at `[batch, heads, queries, keys, D, Dv] =
[1, 16, 10560, 10560, 128, 128]` passes against a bounded, query-tiled native
MPS FP32 reference covering every key and query. Maximum absolute difference
is `5.44e-7`, relative RMS `2.10e-6`. The score workspace is 165 MiB versus
6.65 GiB for a full score matrix, a size the original worker refuses.
A local invocation took 2.12 seconds; this includes
process/staging work and is not a full-model benchmark. Ten additional GPU
shapes and an intermediate-overflow rejection pass, alongside 24 CPU-only
protocol cases.

The large test records 42 completed softmax kernel launches and 84 synchronous
cuBLAS calls. **Only softmax has runtime GPU kernel provenance.** GEMM completion
is an application-level receipt backed by the current cuBLAS-to-MPS source path,
not an instrumented GEMM launch trace. The explicit `cumetal-blas` diagnostic
backend checks both evidence types separately. This does not qualify a full
pretrained video at this size or the default 93-frame model geometry.

```bash
./cuda-mac compile local/cosmos_attention_blas.cu -o local/out/cosmos-attention-blas
local/out/cosmos-venv/bin/python local/test_cosmos_attention_blas.py --gpu --cosmos-shape
local/out/cosmos-venv/bin/python local/verify_cosmos_blas_bridge.py
```

See [worker evidence and limits](../local/verification/2026-10-09/COSMOS-BLAS.md).
The [bridge qualification](../local/verification/2026-10-09/cosmos-blas-bridge-summary.json)
also passes FP32/FP16/BF16 CPU/MPS staging, causal/custom-scale attention, and
a reduced random-weight Cosmos transformer on CPU and MPS. Its 11 attention
calls produce 18 softmax runtime launch records and receipts for 36 GEMMs;
the MPS transformer differs from native MPS by at most `2.38e-7`.
No compiler/runtime change or new library shim is involved. Masks, dropout,
GQA, and backward remain unsupported; operation, tensor and workspace budgets
are checked before execution. The Python bridge still stages through host
memory and starts one subprocess per attention call.

The pretrained target is `nvidia/Cosmos-Predict2-2B-Video2World`. Local login
and target/guardrail access succeed; roughly 14 GB of primary weights downloaded.
The full small-test pipeline, guardrail checks, and video export ran successfully.
The follow-up above produces a recognizable, very short 720p MPS clip. Longer
clips, production quality, peak memory usage, performance tuning, and pretrained
CuMetal parity at that size remain unverified. Safety checks are not bypassed.

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
