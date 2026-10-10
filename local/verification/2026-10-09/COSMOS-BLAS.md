# Cosmos Tiled Attention Qualification

CUDA-M's experimental `local/cosmos_attention_blas.cu` passed standalone
FP32 attention checks on the Apple M4. This qualifies the attention geometry
needed by a 720p, 9-frame Cosmos run, not a complete pretrained inference or
Omniverse compatibility.

## Reproduce

From the repository root, using the existing Cosmos environment:

```bash
./cuda-mac compile local/cosmos_attention_blas.cu -o local/out/cosmos-attention-blas
local/out/cosmos-venv/bin/python local/test_cosmos_attention_blas.py
local/out/cosmos-venv/bin/python local/test_cosmos_attention_blas.py --gpu --cosmos-shape
local/out/cosmos-venv/bin/python local/verify_cosmos_blas_bridge.py
```

The first test command is CPU-only. `--gpu` explicitly enables numerical GPU
tests; `--cosmos-shape` adds the large target. Compilation links `libcumetal.dylib`,
which already exports the clean-room cuBLAS entry points in this checkout;
no NVIDIA headers, separate cuBLAS package, or new dependency is required.

## Observed Results

- 24 strict protocol and CPU-only validation cases passed.
- 10 positive GPU cases passed, including causal offsets across multiple
  query tiles, a final partial tile, multihead/batch strides, custom scale,
  differing value width, and an adaptive 128-query workspace tile.
- Finite inputs whose dot product overflows were refused without output.
- The full target `[B,H,Q,K,D,Dv] = [1,16,10560,10560,128,128]` passed against
  native MPS SDPA. The reference used 256-query tiles with every key retained,
  not a dense CPU score matrix. All 21,626,880 output elements were compared.
- Target maximum absolute error: `5.438923835754395e-7`.
- Target relative RMS error: `2.0967063392163254e-6`.
- Target wall time: worker `2.116924125002697 s`; native MPS reference,
  including staging and result copies, `0.8339954170078272 s`.
- Target score workspace: `43,253,760` FP32 elements, `173,015,040` bytes
  (`165 MiB`), versus `7,136,870,400` bytes for all scores at once.
- The target emitted 42 completed GPU softmax launches and a completion
  receipt for 84 cuBLAS/MPS calls. The final query tile contained 64 rows.
- Observed native-reference MPS driver allocation high-water sample:
  `1,082,621,952` bytes. Harness child-process RSS high-water mark:
  `636,485,632` bytes. The latter includes all test subprocesses, is not
  isolated target-worker RSS, and does not measure GPU allocation.

These are single-run timings, not a stable benchmark. The ten smaller positive
cases contributed 17 softmax launches and 34 completed GEMM calls. The positive
suite plus target therefore contains 59 traced softmax launches and 118
application-reported completed GEMM calls; the deliberate overflow case is
excluded from those positive-result totals.

The separate [Python bridge qualification](cosmos-blas-bridge-summary.json)
passes FP32/FP16/BF16 tensors staged from CPU and MPS, causal/custom-scale
attention, and an actual reduced random-weight Cosmos transformer on CPU/MPS.
It has 11 attention calls, 18 traced softmax launches, and application receipts
for 36 GEMMs. The MPS transformer's maximum difference from native MPS is
`2.384185791015625e-7`. This is not pretrained generation. The bridge harness
refuses to overwrite its existing report; use `--report` for a new destination.

The original bridge was rerun after the shared Python validation refactor:
13 attention calls and 39 exact Apple-GPU launch records still pass, including
its reduced transformer checks. The separate local report is
`local/out/cosmos-attention-regression-tiled-bridge.json`; older evidence was
not overwritten.

## Provenance Boundary

QK and SV use `cublasSgemmStridedBatched` through CUDA-M's existing public
`MPSMatrixMultiplication` backend. Softmax is CUDA source compiled to Metal;
every query tile has runtime `CUMETAL_PROVENANCE` with `device=apple_gpu`,
`launch_success=true`, `semantic_quality=exact`, and Apple device identity.

The existing cuBLAS backend does not emit an independent per-GEMM runtime
trace. `COSMOS_BLAS_COMPLETION` is an application receipt issued only after
synchronous successful API calls, finite-output validation, buffer cleanup,
and successful output flush. Its GEMM count must not be presented as 84
independently traced Metal kernels.

## Bounds And Limits

The worker caps combined heads at 128, sequence lengths at 16,384, channel
widths at 128, all input/output tensors combined at 128 Mi FP32 elements,
multiply-accumulates at `549,755,813,888`, and score workspace at 64 Mi FP32
elements. Query tiles are at most 256 rows and shrink to satisfy the workspace
bound. It rejects nonfinite inputs, malformed dimensions, truncated or trailing
input, and nonfinite output. Causal masking uses the global query index.

The original simple worker is unchanged. Masks, dropout, grouped-query
attention, backward propagation, arbitrary shapes, and general CUDA binary
compatibility are not established by this result.

## Exact Artifacts

Portable metrics and hashes: [cosmos-blas-summary.json](cosmos-blas-summary.json).
The ignored local report `local/out/cosmos-attention-blas-results.json` retains
every positive launch event and application receipt.

| Artifact | SHA-256 |
| --- | --- |
| `local/cosmos_attention_blas.cu` | `71cc251df9d956d0eb95af0b8ba495a580911bc37eefefdb1729a2dd0d996f2e` |
| `local/test_cosmos_attention_blas.py` | `1316a3ff4bcedf991422a76ebde3cbffd4123bfba1794f013c8d58fe003177c9` |
| `local/out/cosmos-attention-blas` | `f4b51d533a5191e9de4e8a14f8d23528a2aeaf2e6ad29f94a18cc65499f76ae0` |
| `build/libcumetal.dylib` | `25dc16129a6791b18a7cab8636ea2e7b5c5345ca0c0085379afd8bf6d826fe52` |
