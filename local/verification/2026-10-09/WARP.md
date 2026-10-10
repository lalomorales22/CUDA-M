# Warp Qualification

This note preserves the before-fix measurements and separately records the
focused post-fix qualification from the 2026-10-09/10 local session. Upstream
unittest outcomes and Apple-GPU launch evidence are distinct. The results do
not establish broad Warp support or Omniverse/Isaac Sim readiness.

## Host And Build

- Apple M4, 32 GiB unified memory (`34359738368` bytes).
- macOS 27.0, build `26A428`; existing Python 3.14.7 Cosmos environment.
- CUDA-M base commit: `156c232d1a4f69f3e9ab46b9cdf1d64643e99581`.
- NVIDIA Warp v1.12.0, exact upstream commit
  `e6c3ba2d54bb048115760b5cd7a4bb2573329ae7`.
- Existing patches: `scripts/warp-patches/0001-cumetal-macos-build.patch`
  and `0002-volume-builder-include-new.patch`.
- Patched Warp build: `--cuda-path=<repository>/build/cumetal-cuda-toolkit
  --no-use-libmathdx --no-standalone --jobs=2`, using
  `local/out/cosmos-venv/bin/python`. CPU JIT and MathDx are unavailable.
- No new packages were installed. Advisory lookup, upstream download risks,
  actual clone history, and wrapper mitigation: [security note](WARP-SECURITY.md).

The first patch explicitly enables the CuMetal toolkit on Darwin; this is not
the upstream macOS CPU-only configuration. Warp's displayed CUDA toolkit,
driver, `sm_80`, and available-memory values are compatibility-shim reporting,
not evidence of NVIDIA hardware. Build settings are retained in the ignored
`local/out/warp-build-first.log`.

## Controlled Runner

The following parent-process environment was recorded by the orchestrating
session. It is not independently recoverable from `summary.json`, which saves
worker commands, interpreter identity, selected modules, deadlines, and outcomes.
From the CUDA-M repository root, the historical baseline invocation was:

```bash
ROOT="$PWD"
export PATH="$ROOT/local/toolchain:$PATH"
export CUMETAL_TOOL_XCRUN="$ROOT/local/toolchain/xcrun"
export CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS=0
export CUMETAL_ENABLE_LLMC_CPU_EMULATION=0
export CUMETAL_DISABLE_LLMC_EMULATION=1
export CUMETAL_TRACE_GPU=1
export CUMETAL_CACHE_DIR="$ROOT/local/out/warp-m4-first-cache"
export WARP_CACHE_PATH="$ROOT/local/out/warp-m4-baseline-jit-cache"
export DYLD_LIBRARY_PATH="$ROOT/build"
export CUMETAL_PYTHON="$ROOT/local/out/cosmos-venv/bin/python"
export CUMETAL_WARP_DIR="$ROOT/local/out/warp-cumetal"
export CUMETAL_WARP_RESULTS="$ROOT/local/out/warp-m4-baseline"
export CUMETAL_WARP_TIMEOUT=120
bash scripts/run_warp_tests_cumetal.sh \
  test_copy test_dense test_devices test_enum test_import \
  test_module_hashing test_triangle_closest_point
```

The expanded invocation used `warp-m4-expanded-jit-cache` and
`warp-m4-expanded` for the two group-specific paths, with modules
`test_conditional test_vec_lite test_vec_assign_copy test_atomic_bitwise`.
`CUMETAL_BUILD_DIR` was not explicitly set; the normal `build` tree was used.
For a rerun, choose new result/cache directories: the runner refuses to
overwrite existing evidence. A 120-second process deadline does not guarantee
cancellation of already submitted GPU work. Do not run the whole suite blindly.

## Pre-Fix Results

Baseline and expanded outcomes come from their structured summaries and
per-module `.result.json` sidecars. Full-vector and isolated-slicing outcomes
come from the unittest text logs only; they did not produce those sidecars.
There were no recorded skips, expected failures, or unexpected successes in
the two structured groups. Overlapping tests across runs must not be added as
unique coverage.

| Group / module | Tests | Passed | Failed | Errors | Validated GPU launches |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline: `test_copy` | 1 | 1 | 0 | 0 | 0 |
| Baseline: `test_dense` | 1 | 1 | 0 | 0 | 0 |
| Baseline: `test_devices` | 6 | 6 | 0 | 0 | 0 |
| Baseline: `test_enum` | 5 | 5 | 0 | 0 | 5 |
| Baseline: `test_import` | 1 | 1 | 0 | 0 | 1 |
| Baseline: `test_module_hashing` | 1 | 1 | 0 | 0 | 0 |
| Baseline: `test_triangle_closest_point` | 1 | 1 | 0 | 0 | 1 |
| **Baseline total** | **16** | **16** | **0** | **0** | **7** |
| Expanded: `test_conditional` | 21 | 19 | 1 | 1 | 0 |
| Expanded: `test_vec_lite` | 3 | 3 | 0 | 0 | 3 |
| Expanded: `test_vec_assign_copy` | 3 | 2 | 1 | 0 | 15 |
| Expanded: `test_atomic_bitwise` | 3 | 3 | 0 | 0 | 18 |
| **Expanded total** | **30** | **27** | **2** | **1** | **36** |
| Full `test_vec`, separate text-log run | 44 | 12 | 22 | 10 | 35 |
| Isolated slicing-backward, separate text-log run | 1 | 0 | 1 | 0 | 3 |

Launch counts were independently parsed from the logs, including records
following unittest progress text. Every counted event had `event=kernel_launch`,
`device=apple_gpu`, `device_name="Apple M4"`, `launch_success=true`,
`semantic_quality=exact`, and an empty `unsupported_reason`.
Baseline's seven records are `precompiled_metallib`. Expanded has 29
`precompiled_metallib` and seven `generic_ptx_lowering` records; full-vector has
11 and 24 respectively; isolated slicing has two and one respectively.

`semantic_quality=exact` labels the selected lowering path. It is not an
independent numerical oracle and does not override a failing upstream
assertion. Allocation/fill helper launches also count, so one launch is not
one passing test. Zero-launch baseline modules supply API/host-side evidence
only, not qualification of a GPU kernel. A `cuda_0` test name alone proves
neither that a kernel ran nor that its output is correct.

## Pre-Fix Failure Boundaries

- **Conditional false confidence:** 19 reported passes accompany zero launch
  records. The first build fails with `typed vprintf requires a constant format
  string`. Warp's `Module.load()` remembers failed builds and returns `None`
  on subsequent attempts; `wp.launch()` can then return without execution.
  Several upstream kernel tests rely on device `expect` output rather than
  copied-back results. These reported passes cannot qualify conditionals.
  The host-checked array test actually returned `[0,0,0]` instead of `[1,2,3]`.
- **Metal argument limit:** generated vector backward code includes a signature
  with 45 arguments (launch bounds plus 22 primal and 22 adjoint arrays).
  The current one-argument-per-buffer lowering exceeds Metal's 31 slots.
  The full-vector log rejects `buffer(31)` and above, requiring indices 0-30.
  This is a compilation blocker, not a successful numerical result.
- **Typed printing gap:** the typed backend cannot yet lower this nonconstant
  `vprintf` format path from Warp's builtins. Silencing diagnostics or returning
  success would not fix the contract; device assertions must remain meaningful.
- **Slicing numerical failure:** the isolated slicing-backward test still fails
  after three validated launches. Its checked output was `[0,0,0,0]`, expected
  `[1,2,0,-1]`. It reproduces independently of the full-vector module's other
  compile failures. The focused post-fix rerun is recorded separately below.
- **Optional driver surface:** startup reports unavailable capture/graph,
  graphics interop, IPC, and array entry points, including
  `cuStreamGetCaptureInfo`, `cuStreamUpdateCaptureDependencies`,
  `cuGraphNodeGetDependentNodes`, `cuGraphNodeGetType`, `cuGraphicsMapResources`,
  `cuIpcGetMemHandle`, and `cuArrayCreate`, with CUDA error 500. They did not
  prevent the selected passing modules from running, but remain unsupported
  coverage rather than successful stubs or general compatibility.

## Pre-Fix Artifact Identity

Large generated logs and caches remain under ignored `local/out/`; this
portable note is intended for source control. These SHA-256 values identify the pre-fix inputs
to this assessment, not future reruns or subsequently rebuilt binaries.

| Local artifact | SHA-256 |
| --- | --- |
| `local/out/warp-m4-baseline/summary.json` | `67fd0e65e43928224a8dc195947d2370b4b9ad1e12f1de6ee5f15981520c4f09` |
| `local/out/warp-m4-expanded/summary.json` | `0c43e02fe8885322c091862c9389aa285139cb1dc5de1948cac1bf37ce13d59e` |
| `local/out/warp-build-first.log` | `b5fbdd409ce47e3993f8d4ca76587c085a1749a83813ec90424e7233c4b9bc14` |
| `local/out/warp-vec-first.log` | `94d4c09c4e1c375bbe1a2c6a456b2e0c71c9fc419b5d81d77dba0e3f72da8391` |
| `local/out/warp-slicing-isolated.log` | `ee2bc87b2150ae6bbf054cb77763520e8eef63746c65499590a9dc3b787e2e27` |
| `local/out/warp-m4-expanded/test_conditional.log` | `6700cf464513a927491154f08351acd809a23757c050bad5e4e183d12e23b038` |
| `local/out/warp-m4-expanded/test_vec_assign_copy.log` | `345bcdcdc3c03cf7fdcdbd6b320ffb5bae4ee61bb4d971f0c8cfdecf731264c2` |
| `local/out/warp-m4-expanded/test_vec_lite.log` | `6348ae9658293c5cc36fd6858e1224ac933306d0479a3bf2a184c261aed12a36` |
| `local/out/warp-m4-expanded/test_atomic_bitwise.log` | `87d37b5cc0a2b64b7f10a2bc42c9e5e72df9c6ad120a50e02251e848a143b71f` |
| `local/out/warp-m4-baseline/test_triangle_closest_point.log` | `561a0088fd4f6fa43d57fe4197d3cb71b3e10788cea6f0e3678df97dc6a989a3` |

## Compiler-Owned Argument ABI

The fix supplies compiler-owned argument metadata for generic PTX JIT launches,
carrying explicit pointer/scalar/aggregate kind and byte width from the selected
lowering backend to runtime registration. Registration cache schema
`cumetal-registration-jit-v25-argument-authority` and metadata magic `CUMETA03`
preserve this contract across cold/warm cache paths and invalidate older cache
metadata. Hidden runtime bindings, including promoted constants, are excluded
from the caller's argument list.

A scalar remains opaque bits even if its value equals a live allocation's
address. A declared pointer that cannot resolve to an allocation is refused,
not reinterpreted as a small scalar. Aggregate pointer relocation remains
separate. Legacy scalar `.f16` lowering refuses an incompatible ABI rather
than publishing misleading metadata. Opt-in workload specializations retain
their existing fallback and provenance labels; this change does not relabel
them as generic lowering. Native/AOT argument handling is unchanged.

The old Warp slicing failure was exposed through the generic PTX fill/helper
path; the fix is not a Warp-kernel-name special case. The following measurements
qualify the bounded behavior after rebuilding, not all uses of these APIs.

## Post-Fix Warp Run

The separate `local/out/warp-m4-verified/summary.json` reports all 10 selected
modules passing: **25 tests passed, zero failures, errors, skips, expected
failures, or unexpected successes**. Its independently audited portable
metrics, launch categories, and artifact hashes are in
[warp-summary.json](warp-summary.json).

| Module | Tests passed | Validated GPU launches |
| --- | ---: | ---: |
| `test_copy` | 1 | 0 |
| `test_dense` | 1 | 0 |
| `test_devices` | 6 | 0 |
| `test_enum` | 5 | 5 |
| `test_import` | 1 | 1 |
| `test_module_hashing` | 1 | 0 |
| `test_triangle_closest_point` | 1 | 1 |
| `test_vec_lite` | 3 | 3 |
| `test_vec_assign_copy` | 3 | 16 |
| `test_atomic_bitwise` | 3 | 18 |
| **Total** | **25** | **44** |

All 44 launch records satisfy the same exact/completed/Apple-M4 checks used
above: 32 precompiled forward test kernels, four precompiled backward test
kernels, and eight generic PTX memtile helpers. There were no invalid records;
ten valid records followed unittest progress prefixes. Assignment/copy now has
16 launches rather than the historical 15 because an additional int64
`memtile_value` helper executes. Numerical confidence comes from the upstream
assertions passing, not from the trace label alone.

This run includes the previously failing slicing-backward case within
`test_vec_assign_copy`. It excludes `test_conditional` and the full `test_vec`
module: their typed-printing/large-argument and broader coverage limits remain
unqualified. The zero-launch modules still provide only API/host-side evidence.

## Post-Fix Regressions

- Focused Release CTest: **4/4 passed**, no skips, in
  `local/out/warp-final-focused.xml`: RoPE Neox, registration scalar64,
  registration argument-index validation, and PTX-to-Metal lowering.
- Focused Debug with binary shim ON: **20/20 passed**, no skips, approximately
  14.93 seconds wall time, in `local/out/warp-debug-focused.xml`. This is a
  focused configuration gate, not the full Debug suite.
- Original Cosmos bridge replay: **13 attention calls, 39 validated GPU
  launches, and nine negative checks**, status `passed`, in
  `local/out/cosmos-attention-warp-regression.json`.
- Tiled Cosmos bridge replay: **11 attention calls, 18 softmax GPU launches,
  36 application-reported GEMM calls**, status `passed`, CPU arithmetic fallback
  disabled, in `local/out/cosmos-blas-bridge-warp-regression.json`.
- The orchestrating session also completed 59 CPU-only tests, plus the build
  wrapper's 31 cases and runner's 13 cases. These counts may overlap other
  gates and must not be summed into a unique-test total.

The Cosmos replays use bounded attention cases and reduced random-weight
transformers. **No full pretrained Cosmos generation was rerun for this fix.**
Application GEMM receipts are not independent runtime kernel traces.

An earlier, pre-final Release CTest run (`local/out/warp-release-ctest.xml`)
recorded 425 passes, three skips, and one RoPE failure in 704.08 seconds. Its
nested Phase 4 probe recorded 183/185 because of RoPE and a compile timeout;
the main libdevice test passed in 17.69 seconds. This historical run is not a
clean final-suite result and must not be combined with the focused successes.

## Final Full-Suite Gate

The final Release build, with binary shim OFF and source registration ON,
completed the full CTest invocation with exit 0: **426 top-level passes, three
skips, zero top-level failures**, 429 registered tests, in **686.94 seconds**.
This includes all seven PhysX gates, 46 typed-PTX corpus cases and 45 native-AOT
corpus cases. The compiler backend matrix matched its reviewed manifest; this
includes expected refusals, not universal backend compatibility.

The three skipped gates require inputs unavailable in this checkout:

- `conformance_cuda_samples_sweep`: external cuda-samples checkout absent.
- `conformance_cuda_samples_vectoradd_gpu`: external cuda-samples checkout absent.
- `conformance_compiler_backend_matrix_versions`: CUDA Clang 21 and 23 absent.

The full invocation's nested Phase 4 gate passed its configured 90% threshold
but recorded **184/185**, with `functional_cuda_projects_libdevice_math` timing
out while compiling at 120.04 seconds. The same test passed later in the
top-level run in 17.71 seconds. This is not a 185/185 result for that invocation.
The full untruncated CTest detail log was preserved before further tests; the
JUnit file truncates long successful-test output and omits this nested summary.

A separate post-suite replay used an explicit 300-second per-case deadline and
a **100% required-pass threshold**. It completed **185/185**, no failures or
skips, with exit 0. Caches were not cleared for this recheck; it does not qualify
the original 120-second deadline or eliminate cold-compile timeout risk. No
compiler/runtime code changed between the full run and the recheck.

Commands from the repository root:

```bash
ROOT="$PWD"
env PATH="$ROOT/local/toolchain:$PATH" \
  CUMETAL_TOOL_XCRUN="$ROOT/local/toolchain/xcrun" \
  PHYSX_REPO="$ROOT/local/out/PhysX" CUMETAL_PHYSX_BUILD_JOBS=2 \
  ctest --test-dir build --output-on-failure -j1 \
  --output-junit "$ROOT/local/out/warp-release-final-ctest.xml" \
  > local/out/warp-release-final-ctest.log 2>&1

env PATH="$ROOT/local/toolchain:$PATH" \
  CUMETAL_TOOL_XCRUN="$ROOT/local/toolchain/xcrun" \
  CUMETAL_CONFORMANCE_SINGLE_TEST_TIMEOUT=300 \
  bash tests/conformance/run_conformance_suite.sh "$ROOT/build" 100 \
  tests/conformance/phase4_functional_manifest.txt \
  > local/out/warp-phase4-strict-recheck.log 2>&1
```

Select new output filenames when reproducing to preserve the prior evidence.
These gates reuse the installed toolchain and existing PhysX checkout; they do
not install the missing external/toolchain inputs. Focused Debug shim-ON results
above remain a separate configuration gate, not a full Debug-suite run.

### Regression Artifact Identity

| Local artifact | SHA-256 |
| --- | --- |
| `local/out/warp-release-final-ctest.xml` | `a660e9990166d54966c4e152752969e9b842725651c43cd6c310f55a12fc6a21` |
| `local/out/warp-release-final-ctest.log` | `ea426f6a398af8f2b65cf039459319a26ddd190bd24c2511ba58896c33da89f4` |
| `local/out/warp-release-final-ctest-detail.log` | `3de3779cbd0a9b4d72ce73a32e12155017b31e4a02e0a9cc77db2bb7b625c2bb` |
| `local/out/warp-phase4-strict-recheck.log` | `2ee22fdbdd0b0f4046c7adfaa007fe704a3c3bb90ac6d4559975fa2740ab2cac` |
| `local/out/warp-final-focused.xml` | `dc57302dec36845af6d2746b09e2d17991c8ee19990d0e94a98cee46e6a59bb0` |
| `local/out/warp-debug-focused.xml` | `79850d581fb4aad2208f8c636ff114822f374a9d1b09e47b49ab0b10f7fa42d3` |
| `local/out/cosmos-attention-warp-regression.json` | `e3c6579108d9b50768a6268cfc967164ee69f1ff371429c79a3c2bebcd657905` |
| `local/out/cosmos-blas-bridge-warp-regression.json` | `87e80a3256e60d695c38987ffc0f18a9aea7dd75ae1aec12797956babce1ab41` |
