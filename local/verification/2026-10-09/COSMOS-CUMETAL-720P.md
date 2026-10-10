# Pretrained Cosmos With CUDA-M Attention

`Cosmos-Predict2-2B-Video2World` completed a nine-frame, 1280x704 pretrained
run on Apple M4 using the explicit `cumetal-blas` backend. Attention uses
query tiling, the CuMetal cuBLAS shim backed by MPS matrix multiplication,
and CUDA softmax compiled through CuMetal to Metal. Other model arithmetic
runs in PyTorch MPS with sequential CPU weight staging. Automatic CPU
arithmetic fallback is disabled and the pipeline safety checks are retained.
This is a scoped attention bridge, not a replacement PyTorch CUDA backend.

## Execution Evidence

The run made 896 attention calls: 448 self-attention and 448 cross-attention,
each using 42 query tiles of up to 256 rows. The report retains a validated
count of **37,632 runtime softmax kernel launches** and **896 application
completion receipts**, reporting **75,264 strided-batched GEMM API calls**.
All receipts report completion on Apple M4. Raw softmax event records were
validated during execution but were not retained. GEMM application receipts
are not an independent GPU trace and are not counted as kernel provenance.

Candidate model time was 1,532.27 seconds, with 1,536.99 seconds total. The
earlier native MPS baseline took 555.00 seconds for its model run. These are
local observations with diagnostic overhead, not a controlled benchmark.

The candidate uses eight denoising-loop evaluations with two classifier-free
guidance transformer forwards each, seven scheduler transitions, and no final
clean-prediction pass. Conditioning uses a sampled VAE posterior; native,
unscaled temporal RoPE is selected with `fps=None`. Export is separately 16 FPS.

## Numerical Comparison

Sixteen sampled attention records cover two calls at each of eight evaluations,
with only 16 selected query rows per record. Maximum relative RMS difference
against CPU FP32 attention rounded to BF16 is `0.000157204` (0.01572%). Against
native MPS attention it is `0.000157129`. Both have maximum absolute difference
`0.001953125`. These samples do not qualify every layer or full-model CPU parity.

An offline comparison checks all eight saved latent arrays and nine retained
RGB frame arrays before MP4 encoding, without rerunning either model. These
RGB metrics are not measured from MP4-decoded pixels. Final latent relative RMS difference against
the earlier MPS baseline is `0.05008569` (5.0086%); maximum over the eight steps
is `0.06284782`. Mean absolute RGB difference is `1.3007964` on the 0-255 scale,
with maximum pixel-channel difference `91`. The conditioning frame is identical.
**No numerical parity pass is claimed: no pass tolerance was defined.**

The saved model/revision, image hash, prompt, seed, geometry, eight-step count,
BF16 dtype, temporal FPS/RoPE, export FPS and sampler-adapter fields match.
The baseline lacks explicit `model_evaluations`, `model_evaluation_definition`,
`transition_steps`, `final_clean_prediction`, `conditioning_latents`,
`contract_reference_revision` and `contract_experiment_differences` metadata.
Matching recorded fields does not fill those gaps or establish an identical
complete execution contract. The baseline's exact source/binary identity was
not captured.

The candidate identity manifest was recorded **during inference**, not before
launch or from process memory. All 13 recorded on-disk hashes were rechecked
after completion and still match. This strengthens artifact identity without
turning the manifest into a prelaunch or process-memory attestation.

## Visual Result

The MP4 probes as H.264, 1280x704, nine frames, 16 FPS, **0.5625 seconds**.
The controlling run observed a full ffmpeg decode with exit code 0. Assistant
inspection covered all nine MP4-decoded frames and the MPS comparison sheet:
the industrial robot-welding scene is recognizable and coherent, with changing
smoke and sparks. Pronounced softness and textured or banded appearance remain.
The severe multicolor distortion seen at small resolution is absent.

This is assistant inspection, not a human-user quality assessment or a
production-quality/duration qualification. The toy-robot tabletop prompt was
kept fixed for the comparison despite the industrial welding input image, so
prompt adherence is not qualified. Full pretrained CPU inference, all-layer
equivalence, longer video generation and broad CUDA application compatibility
remain separate gates.

## Retained Artifacts

The [portable summary](cosmos-cumetal-720p-summary.json) records metrics, hashes,
counts, contract gaps and limitations. Raw artifacts are local and ignored:

- `local/out/cosmos-cumetal-blas-720p-8/comparison.json`
- `local/out/cosmos-cumetal-blas-720p-8/run-identity.json`
- `local/out/cosmos-cumetal-blas-720p-8/cumetal-blas.mp4`
- `local/out/cosmos-cumetal-blas-720p-8/all-frames.png`
- `local/out/cosmos-mps-vs-cumetal-blas-720p-8/comparison.json`
- `local/out/cosmos-mps-vs-cumetal-blas-720p-8/comparison.png`

Raw reports retain their original pending-inspection fields; this separate
summary records the subsequent assistant visual inspection.
