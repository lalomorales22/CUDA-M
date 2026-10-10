# Cosmos Quality Isolation

These local experiments investigate the distorted video from
`nvidia/Cosmos-Predict2-2B-Video2World`, revision
`f50c09f5d8ab133a90cac3f4886a6471e9ba3f18`, on Apple M4.
Execution success and visual quality remain separate gates. Raw artifacts live
in ignored `local/out/`; the summaries here retain portable evidence.

## Thirty-five-step baseline

Native MPS attention, BF16 model weights, the same welding image, prompt and
seed `20261009`, nine frames, and 320x192 were rerun with 35 denoising steps
instead of eight. The standard scheduler was used without the experimental
adapter. Safety checks were retained and CPU arithmetic fallback was disabled.

The run completed in 275.14 seconds, but inspection of frames 0, 4, and 8 still
shows severe distortion after the conditioning frame. Increasing the step count
alone did not fix this small-resolution sample. This run does not exercise the
CuMetal attention bridge: the report contains zero CuMetal calls and zero
CuMetal provenance launch records. It is an MPS baseline, not CUDA-M parity.

Evidence:

- `local/out/cosmos-35step-diagnostic/comparison.json`, SHA-256
  `390fe9634a771ea107c7a4f7695d1373d1be5ee3af423159a11a2fd7c3e31c2e`
- `local/out/cosmos-35step-diagnostic/comparison.png`, SHA-256
  `7123283458031e28ccdef8b9b0ce075f1b889159afae3f9fc6123b15a99c45ae`
- `local/out/cosmos-35step-diagnostic/mps.mp4`
- `local/out/cosmos-35step-diagnostic/mps-latents.npz`

## VAE isolation

The checkpoint uses `AutoencoderKLWan`. A cached-only diagnostic loads that
component alone, reconstructs nine repeated copies of the input using the
posterior mode, and separately decodes the same retained final normalized
latents from the eight-step MPS baseline. It repeats both operations in BF16
and FP32, releasing the VAE between runs. No transformer or safety-qualified
video generation is involved in this diagnostic.

Both precisions reconstruct the repeated input faithfully across all nine
frames. Both decode the retained generated latents into the same severe
distortion. All outputs are finite.

| Measurement | BF16 | FP32 |
| --- | ---: | ---: |
| Maximum reconstruction frame RMSE | 0.037952 | 0.037851 |
| Minimum reconstruction frame PSNR | 34.44 dB | 34.46 dB |
| Local component run time | 8.04 s | 12.28 s |

Metrics use decoded RGB in `[-1,1]`; PSNR uses a peak-to-peak range of 2.
BF16 versus FP32 retained decode has maximum frame RMSE `0.004877` and minimum
frame PSNR `52.26 dB`. These differences do not explain the visible failure.
Gross temporal reconstruction or decode-precision failure is therefore an
unlikely explanation for this sample. This does not establish all MPS operation
correctness: full CPU VAE parity and full pretrained CPU inference were not run.

The [portable VAE summary](cosmos-vae-summary.json) records hashes, metrics,
scope, and limitations. The inspected contact sheet and full per-frame metrics
are `local/out/cosmos-vae-diagnostic/comparison.png` and
`local/out/cosmos-vae-diagnostic/results.json`.

```bash
local/out/cosmos-venv/bin/python local/test_cosmos_vae.py
local/out/cosmos-venv/bin/python local/diagnose_cosmos_vae.py --output-dir local/out/cosmos-vae-rerun
```

Seven CPU-only, download-free tests cover normalization inversion, invalid
normalization, metrics, non-finite values, layout, numeric latent-step ordering,
and malformed latent archives. The diagnostic requires cached weights and
refuses to overwrite a directory containing an earlier report.

## Native temporal RoPE

A further 320x192, nine-frame, 35-step MPS run used `fps=None` for native,
unscaled temporal rotary position embeddings instead of the pipeline's default
FPS-conditioned scaling. Export remained separate at 16 FPS. The run completed
in 273.46 seconds, but inspected frames 0, 4, and 8 remain severely distorted
after the conditioning frame. Native temporal RoPE alone did not fix this
small-resolution sample.

Its original report and contact sheet are
`local/out/cosmos-native-rope-35/comparison.json` and
`local/out/cosmos-native-rope-35/comparison.png`.

## Higher-resolution MPS result

The next native-RoPE run used 1280x704, nine frames, and eight steps with the
same checkpoint, image, prompt, seed, and BF16 precision. It completed in
555.00 seconds for the model run, 561.29 seconds total. Safety checks remained
enabled; CPU arithmetic fallback remained disabled.

Inspection of all nine frames shows a coherent welding scene, changing sparks
and smoke, and visible motion. The severe multicolor distortion of the small
tests is absent. Soft and banded texture remains. This is a nominal
**0.5625-second diagnostic at 16 export FPS**, not production-quality or
production-duration qualification.

The retained frame array is finite `uint8` RGB with shape `[9,704,1280,3]`.
Mean adjacent-frame absolute difference is `3.7714` on the 0-255 scale. That
confirms changing pixels; it is not a semantic quality or physical-correctness
score. Final latent RMS is `0.566849`.

**This result uses native MPS attention, not CUDA-M attention.** It records
zero CuMetal calls and zero CuMetal provenance launches. Its 10,560 attention
tokens exceed the current dense CuMetal worker's bounded scope; materializing
the corresponding FP32 attention scores would require 7,136,870,400 bytes.

The unchanged prompt describes a toy robot on a tabletop while the conditioning
image shows industrial welding. Keeping it fixed preserves the earlier
baseline, but **prompt adherence is not qualified**. The two native-RoPE runs
also differ in both resolution and step count, so resolution alone has not been
causally isolated. Full pretrained CPU inference and full-model CUDA-M parity
remain untested at this geometry.

The [portable quality summary](cosmos-quality-summary.json) records exact
metrics, artifact hashes, inspection, and limits. Evidence:

- `local/out/cosmos-native-rope-720p-8/comparison.json`
- `local/out/cosmos-native-rope-720p-8/all-frames.png`
- `local/out/cosmos-native-rope-720p-8/mps.mp4`
- `local/out/cosmos-native-rope-720p-8/mps-frames.npy`
- `local/out/cosmos-native-rope-720p-8/mps-latents.npz`

Raw reports are unchanged and retain their original requires-inspection field;
this document and the portable summary record the subsequent visual inspection.
