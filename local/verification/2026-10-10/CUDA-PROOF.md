# CUDA Source Proof Ladder

Recorded 2026-10-10 on Apple M4 Mac mini, 32 GiB, macOS 27.0.
CUDA-M base: `c709c6770502e4fac4d72ff22b11cbfb0f1dc664`.
These are source-first tests against CuMetal's clean-room headers and runtime,
not execution of NVIDIA's precompiled CUDA binaries.

| Gate | Observed evidence | Boundary |
| --- | --- | --- |
| Official NVIDIA `deviceQuery` | One device named `Apple M4`; `Result = PASS`. | Runtime API enumeration/property compatibility only; no compute kernel executes. |
| Official NVIDIA `vectorAdd.cu` | All 50,000 results pass the upstream check; one completed generic-PTX Apple M4 launch. | Real CUDA-source GPU execution for this kernel, not broad CUDA compatibility or a speed benchmark. |
| Heavy GPU workload | FP32 `2048x2048x2048` matrix multiplication passes; three separately traced completed Apple M4 launches. | Source-compiled GPU execution is verified, but this kernel is slower than the measured optimized Apple baselines. |

## Pinned Source And Security

Official [NVIDIA/cuda-samples v12.8](https://github.com/NVIDIA/cuda-samples/releases/tag/v12.8)
resolves to commit `db3eea23946bca2e90a75eca2b5b3e07158a9e11`, checked through
[GitHub's tag API](https://api.github.com/repos/NVIDIA/cuda-samples/git/ref/tags/v12.8).
The orchestrating session fetched only these unmodified files via pinned
`raw.githubusercontent.com` HTTPS URLs after the advisory check:

- [Samples/1_Utilities/deviceQuery/deviceQuery.cpp](https://github.com/NVIDIA/cuda-samples/blob/db3eea23946bca2e90a75eca2b5b3e07158a9e11/Samples/1_Utilities/deviceQuery/deviceQuery.cpp)
- [Samples/0_Introduction/vectorAdd/vectorAdd.cu](https://github.com/NVIDIA/cuda-samples/blob/db3eea23946bca2e90a75eca2b5b3e07158a9e11/Samples/0_Introduction/vectorAdd/vectorAdd.cu)
- [Common/helper_cuda.h](https://github.com/NVIDIA/cuda-samples/blob/db3eea23946bca2e90a75eca2b5b3e07158a9e11/Common/helper_cuda.h)
- [Common/helper_string.h](https://github.com/NVIDIA/cuda-samples/blob/db3eea23946bca2e90a75eca2b5b3e07158a9e11/Common/helper_string.h)
- [LICENSE](https://github.com/NVIDIA/cuda-samples/blob/db3eea23946bca2e90a75eca2b5b3e07158a9e11/LICENSE)

They remain external fixtures in ignored
`local/out/cuda-proof-2026-10-10/vendor/cuda-samples`; no third-party source was
copied into the tracked tree. This is a five-file fixture, not a complete sample
checkout or qualification of the full cuda-samples sweep. No CUDA Toolkit,
NVIDIA driver, or new package was installed. The existing local compilers,
CuMetal build, and compiler wrappers were reused.
After acquisition, all five local Git blob hashes and byte sizes matched
GitHub Contents API metadata at the pinned commit. SHA-256s are retained below;
the portable [audit summary](cuda-proof-summary.json) records the blob identities.

The bounded pre-download lookup found no published cuda-samples-specific
advisory: the [repository advisory API](https://api.github.com/repos/NVIDIA/cuda-samples/security-advisories)
returned `[]`; NVD queries for [cuda-samples](https://services.nvd.nist.gov/rest/json/cves/2.0?keywordSearch=cuda-samples)
and [CUDA samples](https://services.nvd.nist.gov/rest/json/cves/2.0?keywordSearch=CUDA%20samples)
each returned zero results. NVIDIA's [2022](https://github.com/NVIDIA/product-security/blob/main/2022/README.md),
[2023](https://github.com/NVIDIA/product-security/blob/main/2023/README.md),
[2024](https://github.com/NVIDIA/product-security/blob/main/2024/README.md),
[2025](https://github.com/NVIDIA/product-security/blob/main/2025/README.md), and
[2026](https://github.com/NVIDIA/product-security/blob/main/2026/README.md)
security indexes had no Samples bulletin. This is not a security certification,
complete dependency audit, or proof that no vulnerabilities exist.

## Device Query Is API Evidence

`local/out/cuda-proof-2026-10-10/deviceQuery.log` records:

```text
Detected 1 CUDA Capable device(s)
Device 0: "Apple M4"
CUDA Driver Version / Runtime Version          12.0 / 12.0
CUDA Capability Major/Minor version number:    8.0
(001) Multiprocessors, (064) CUDA Cores/MP:    64 CUDA Cores
Result = PASS
```

The formatting above is condensed; the retained log has the complete output.
The device name comes from Metal, but CUDA version/SM labels describe CuMetal's
compatibility surface. The one multiprocessor is CuMetal's conservative
cooperative-progress partition; the sample derives its core count from an
NVIDIA SM lookup table. It is not a measurement of M4 hardware cores. Clock,
memory-bus, and cache figures also include synthetic or conservative runtime
values, not a hardware characterization. See `runtime/rt/cuda_runtime.cpp`,
`cudaGetDeviceProperties`.

The upstream `CUDART static linking` banner is fixed text; this executable
links to `libcumetal.dylib`. The sample launches no GPU compute work and even
prints `PASS` with zero devices if enumeration succeeds. This run additionally
records one device and its Apple name, but does not prove all reported features
or successful NVIDIA-product execution.

The supplied [how-to forum thread](https://forums.developer.nvidia.com/t/how-to-run-devicequery/54624)
discusses building version-matched sample source. The
[`devicequery zu` thread](https://forums.developer.nvidia.com/t/devicequery-zu/213753)
concerns old Windows `printf` formatting, not a GPU-compute test.

## Official CUDA Kernel Evidence

`local/out/cuda-proof-2026-10-10/vectorAdd.log` records `Test PASSED` and exactly
one completed launch of `_Z9vectorAddPKfS0_Pfi`:

```text
source=generic_ptx provenance=generic_ptx_lowering semantic_quality=exact
device=apple_gpu device_name="Apple M4" math_mode=safe
compile_cache_hit=false launch_success=true unsupported_reason=""
grid=(196,1,1) block=(256,1,1) duration_ns=9750
```

The unmodified source allocates/copies buffers, launches the ordinary `<<<>>>`
kernel, copies results back, and checks 50,000 element-wise sums at `1e-5`
tolerance. The trace establishes the execution path; its `exact` label alone
is not numerical proof. The upstream check and trace are complementary. Its
single 9.75-microsecond dispatch is not an end-to-end timing or a CPU speedup.
Workload specializations and LLMC CPU emulation were explicitly disabled.

## Reproduction

These are the recorded commands with the absolute workspace prefix normalized
to `ROOT`. Run from the repository root with the existing build/toolchain and
pinned fixtures present. The command/environment record comes from the
orchestrating session; the text logs do not independently capture all parent
environment variables. Select new log/output names to preserve prior evidence.

```bash
ROOT="$PWD"
OUT="$ROOT/local/out/cuda-proof-2026-10-10"
SAMPLES="$OUT/vendor/cuda-samples"

xcrun clang++ -std=c++17 -O2 -DCUDART_VERSION=CUDA_VERSION \
  -I"$ROOT/runtime/api" -I"$SAMPLES/Common" \
  "$SAMPLES/Samples/1_Utilities/deviceQuery/deviceQuery.cpp" \
  -L"$ROOT/build" -lcumetal -Wl,-rpath,"$ROOT/build" \
  -o "$OUT/deviceQuery"
env CUMETAL_TRACE_GPU=1 CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS=0 \
  CUMETAL_ENABLE_LLMC_CPU_EMULATION=0 CUMETAL_DISABLE_LLMC_EMULATION=1 \
  "$OUT/deviceQuery" > "$OUT/deviceQuery.log" 2>&1

env PATH="$ROOT/local/toolchain:$PATH" \
  CUMETAL_TOOL_XCRUN="$ROOT/local/toolchain/xcrun" \
  CUMETAL_CUDA_SAMPLES_DIR="$SAMPLES" CUMETAL_BUILD_DIR="$ROOT/build" \
  CUMETAL_PTX_BACKEND=cumetal-ir CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS=0 \
  CUMETAL_ENABLE_LLMC_CPU_EMULATION=0 CUMETAL_DISABLE_LLMC_EMULATION=1 \
  bash tests/cuda_projects/run_cuda_samples_vectoradd_gpu.sh "$ROOT" \
  > "$OUT/vectorAdd.log" 2>&1
```

The host-only query needs the explicit `CUDART_VERSION` define because the
current clean-room header otherwise defines it only for CUDA compilation.
The existing vectorAdd gate uses LLVM CUDA compilation with `-nocudainc
-nocudalib`, clean-room headers, and `libcumetal`; it supplies a fresh temporary
cache and `CUMETAL_TRACE_GPU=1` internally. It requires both the upstream success
message and a completed exact generic-PTX Apple-GPU dispatch, rejecting a logged
CPU fallback/stub. This does not use upstream CMake's NVIDIA Toolkit discovery.

## Heavy Workload

The existing `tools/matmul_bench/test.sh` contract completes successfully:
six shapes exercise 16 CUDA-source variants, including rectangular tails and
small matrices. The aligned-only `vectorized` variant is explicitly skipped on
four unaligned shapes; negative cases reject unaligned explicit selection,
zero dimensions, and an unknown variant. This is not 16 passes on every shape.

The heavy case uses FP32 `M=K=N=2048`, about 17.18 billion floating-point
operations per multiplication. `matmul-gpu-trace.log` records three completed
`block64_32` Apple M4 launches: two warmups and one timed iteration. All have
`source=metallib`, `provenance=precompiled_metallib`, `semantic_quality=exact`,
`launch_success=true`, and empty unsupported reasons. This is separate from
the untraced timing rounds; their kernel counts are not independently attested.

The build compiles `demos/matmul/kernels.cu` with `cumetalc --backend=cumetal-ir
--emit=metallib`. Its host harness uses an explicit `cumetalKernel_t` descriptor
with `cudaLaunchKernel`, not CUDA `<<<>>>` syntax; the official vectorAdd gate
above covers that ordinary launch syntax separately.

The host initializes output with a NaN sentinel, checks all 4,194,304 output
values for finiteness and against an Accelerate FP32 SGEMM reference, and checks
64 reference dot products independently in CPU double precision. The output
tolerance is `2e-5*K + 2e-4*abs(reference)`; the reference spot checks use the
same form with the double result. The traced heavy output records `bad=0`,
`max_abs=0`, and `relative_l2=0`. That is a result for these seeded inputs, not
general bit-exact or FP64 compatibility.

The measurement driver ran nine variants in three rotated rounds, five timed
iterations per process after warmup. All 27 records completed successfully.
Times below are the median of the three per-round means, not pooled iteration
medians. They use synchronized host wall time and exclude compilation,
allocation, transfers, and output verification.

| Path | Median round mean |
| --- | ---: |
| CUDA-source `block64_32` through typed metallib | 12.919 ms |
| Apple Accelerate SGEMM | 9.348 ms |
| Direct Apple MPS matrix multiplication | 6.211 ms |
| CuMetal cuBLAS-compatible API routed to MPS | 5.641 ms |

The compiled kernel is about 1.38 times slower than Accelerate in this run,
not a 10x speedup. The cuBLAS-compatible MPS row is an existing library-routing
path, not NVIDIA cuBLAS or a CUDA compiler performance result. Accelerate is
the platform-optimized host BLAS baseline, not a naive scalar loop. These
measurements qualify this M4 workload only, not M1 or NVIDIA performance.
The [portable audit summary](cuda-proof-summary.json) records timing rows,
controlled commands, validation checks, and retained artifact identities.

Heavy-run commands, using `ROOT` and `OUT` from the reproduction block above:

```bash
RUN_ENV=(CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS=0
  CUMETAL_ENABLE_LLMC_CPU_EMULATION=0 CUMETAL_DISABLE_LLMC_EMULATION=1
  CUMETAL_MSL_MATH_MODE=safe)
env "${RUN_ENV[@]}" CUMETAL_TRACE_GPU=0 \
  PATH="$ROOT/local/toolchain:$PATH" \
  CUMETAL_TOOL_XCRUN="$ROOT/local/toolchain/xcrun" \
  CUMETAL_BUILD_DIR="$ROOT/build" CUMETAL_MATMUL_OUT="$OUT/matmul" \
  bash tools/matmul_bench/test.sh > "$OUT/matmul-contract.log" 2>&1
env "${RUN_ENV[@]}" CUMETAL_TRACE_GPU=1 \
  "$OUT/matmul/matmul_bench" "$OUT/matmul/kernels.metallib" \
  2048 2048 2048 1 block64_32 > "$OUT/matmul-gpu-trace.log" 2>&1
env "${RUN_ENV[@]}" CUMETAL_TRACE_GPU=0 \
  "$ROOT/local/out/cosmos-venv/bin/python" tools/matmul_bench/measure.py \
  "$OUT/matmul" --output "$OUT/matmul-measure" --size 2048 --iterations 5 \
  > "$OUT/matmul-measure.log" 2>&1
```

The three commands exited zero. Independent offline reconciliation matched all
27 timing rows to their logs: 135 timed calls, with 24 full GPU-output checks
and three Accelerate timing-only processes. The small-shape contract has 104
full GPU-output checks (92 CUDA-kernel cases and 12 MPS/cuBLAS cases), plus six
Accelerate timing rows. Timing-call counts are derived from the harness, not
independently captured kernel provenance.

## Product And Project Boundaries

Passing these gates proves supported CUDA source can be compiled and executed
through CuMetal on Apple Silicon. It does not establish arbitrary NVIDIA SASS
binary execution, a general PyTorch CUDA backend, full CUDA library parity,
native RTX/OptiX support, or native Omniverse/Isaac Sim operation. See the
[workload ledger](../../../docs/nvidia-workloads.md) for separately qualified
Cosmos, Warp, and PhysX subsets and remaining product requirements.

[OpenFPM PR #18](https://github.com/mosaic-group/openfpm/pull/18) describes a
separate Metal implementation using MoltenVK/SPIR-V while retaining `.cu`
sources, with its own dependencies and reported validation. It is relevant
source-porting precedent, not evidence that CUDA-M ran OpenFPM. No OpenFPM
download, build, or execution was performed for these gates.

## Retained Artifact Identity

Source paths below are relative to the external `vendor/cuda-samples` fixture;
log paths are relative to `local/out/cuda-proof-2026-10-10`.

| Artifact | SHA-256 |
| --- | --- |
| `Samples/1_Utilities/deviceQuery/deviceQuery.cpp` | `7b7a761007477cef62b74b132f6fae7f9216aa1311bad56c360387afc9805ffc` |
| `Samples/0_Introduction/vectorAdd/vectorAdd.cu` | `9a156c7c4f1ab0759d859af95c8dab55f2280d867edca7daee4546d63effb583` |
| `Common/helper_cuda.h` | `2bf5b66851d6ea49ee0d8e960fb58dbabfc117a6a0b034d084392bfd08a6c2cb` |
| `Common/helper_string.h` | `616ac8b2588a6e73aed207f85faf0da5e617404390df1790f6637b1042dbcc94` |
| `LICENSE` | `b3e40c5bfed1fca5c62d2c1f2208bf51f8d2c910219f94c443f657ace9001be3` |
| `deviceQuery.log` | `1997c807f8b4598e4f4803b69046e865c715dd19a18ee60d926eda38d55b4caa` |
| `vectorAdd.log` | `aeeed10227d671fba8ffc3b89dea5d218b3ae3e310b9e432e35ed281065d5b3d` |
| `matmul-contract.log` | `62371410ff19ec71cb39493861edd0b962f2b79848f7a5dee4a727dcd7833b37` |
| `matmul-gpu-trace.log` | `010be6a459030401fea06fd047d2e0aaedd365987bff5ba3fc3302c081a9619e` |
| `matmul-measure/results.json` | `68cff91611bc089150561e30a64b489eb0ac20bf0757002cd30557345c1d27ff` |
