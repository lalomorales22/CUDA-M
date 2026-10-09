# M4 publication verification

Date: 2026-10-09. Apple M4 Mac mini, 32 GiB, macOS 27.0.
Release build, LLVM 22.1.7, Metal compiler 32023.921, binary shim OFF.

## Passing focused gates

- `cmake --build build -j2`: full build completed.
- `./cuda-mac test`: six numerically checked Apple-GPU launches.
- `./cuda-mac cosmos-test`: 13 attention calls, 39 Apple-GPU launches,
  nine negative checks, reduced CPU/MPS Cosmos transformer comparisons.
- `python3 local/test_trace_validation.py`: two tests.
- `python3 local/test_cosmos_dependency_audit.py`: four download-free tests.
- `local/out/cosmos-venv/bin/python local/test_cosmos_video.py`: three
  download-free guardrail portability/refusal tests.
- Isolated environment `pip check`: no broken requirements.
- `git diff --check`: passed.

Pretrained video execution completed with 896 attention calls / 2,688 Apple M4
kernel launches. Generated frames are severely distorted. This is execution
evidence, not usable-video qualification; see [the Cosmos snapshot](cosmos-summary.json).

## Full CTest Replay

Command (using this repository's public Metal toolchain discovery wrapper):

```bash
env PATH="$PWD/local/toolchain:$PATH" \
  CUMETAL_TOOL_XCRUN="$PWD/local/toolchain/xcrun" \
  ctest --test-dir build --output-on-failure -j2
```

Final replay: **410 passed, 11 skipped, 3 failed / 424**, 298.62 seconds.
CTest exited 8; this is **not a green full-suite result**.
Full logs remain local in ignored `local/out/full-ctest-final.log`.

Three failing registrations share one compiler gap:

- `functional_typed_direct_system_atomics`
- `conformance_compiler_backend_matrix`
- `conformance_cuda_projects_native_aot_corpus`

Direct source compilation refuses LLVM's
`llvm.nvvm.atomic.add.gen.i.sys.i32.p0`. The existing importer has no mapping
for that intrinsic; the non-finite constant correction does not add one.
The same system-atomics workload passes through the PTX path.
Measured compiler matrix: direct typed IR 50/51, legacy PTX 46/51, typed PTX 49/51.
Native-AOT corpus: 44 passes and one compile error. Typed-PTX numerical corpus passed.
No failing gate was removed, waived, or converted into a success.

Skipped registrations:

- `conformance_physx_grb`
- `conformance_physx_grb_friction`
- `conformance_physx_grb_multibody`
- `conformance_physx_grb_stacked`
- `conformance_physx_grb_box`
- `conformance_physx_grb_trimesh`
- `conformance_physx_pbd_recreations`
- `functional_runtime_cooperative_launch`
- `conformance_cuda_samples_sweep`
- `conformance_compiler_backend_matrix_versions`
- `conformance_cuda_samples_vectoradd_gpu`

The first replay exposed missing-reference test ordering, Bash 3.2 empty-array
failures, and documentation drift. Those were corrected, then the full suite
was replayed. The Metal reference parser/load, typed printf, CI report, and
documentation-consistency tests all passed in the final run. CTest now brings
in reference generation even for a parser-only selection.

This publishes an experimental supported-source workflow, not a release claiming
universal CUDA compatibility, all supported Clang/Xcode versions, or production
Cosmos quality. Model weights, authentication, Python environments, generated
binaries, caches, and videos are not included in Git.
