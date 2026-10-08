# Apple Silicon CUDA verification — 2026-10-08

## Questions and results

| Question | Result | Evidence |
| --- | --- | --- |
| Can this machine build the compiler and runtime? | Answered: Release arm64 build, LLVM/NVVM importer enabled | Local `out/configure.log`, `out/build.log` |
| Can ordinary CUDA source execute correct work on its GPU? | Answered for six selected launches: exact numerical and output-boundary checks passed | [Committed results](verification/2026-10-08/results.json) |
| Does the compiled runtime expose CUDA APIs and link Metal? | Observed statically | REA static record below |
| Does execution actually reach a Metal compute dispatch? | Observed under LLDB | REA runtime record below |
| Will arbitrary CUDA applications or M4 Macs work? | Unresolved; outside this validation scope | Upstream known gaps and per-application testing required |

## Local environment and identity

- Apple M1, eight GPU cores; macOS 26.4.1, build 25E253.
- Xcode 26.6, build 17F113; host Apple Clang 21; Homebrew LLVM 23.1.0.
- Apple Metal compiler 32023.883, component 17F109.
- CuMetal commit `fdf403ebdd14a57565fd167d675e2a9910a72c35`, Release, binary alias OFF.
- Runtime SHA-256: `6907fa27cf6b2957c44bcf42c220786d012fdef49b7a431207e8f7e74eaba08f`.
- Independent test executable SHA-256: `a3540653452d8e6b74eb9c275a9d38501cdf49f1a92347919781e635a5fa2e33`.

The REA subjects were checked against the executable/runtime identities in the
numerical report. Later rebuilds can change these identities; this report is a
dated observation. `./cuda-mac test` regenerates numerical evidence, not REA captures.

## REA static evidence

Record: `ev_e79dbaaf5fa76c82cf1ad17118d5f4dcc3da738b9f8ae7f00ca026ea71bdb82e`.
[Portable summary](verification/2026-10-08/rea-summary.json), `static` section.
The full original export remains local at `local/out/rea-static-evidence.json`;
it is excluded from Git because it contains machine-specific paths and raw metadata.

`inspect_macho` observed an arm64 Mach-O runtime exporting `cudaMalloc`,
`cudaMemcpy`, `cudaLaunchKernel`, and `cudaDeviceSynchronize`, with load
dependencies on Metal and MetalPerformanceShaders, among other libraries.
This establishes the built binary's interface and dependency structure.
Static imports alone do not establish execution or correctness. Symbol coverage
can omit stripped or toolchain-hidden symbols. No native decompilation was used;
the available REA native inspection tools were sufficient for this question.

## REA runtime evidence

Record: `ev_c344be2b39f8f7d7873893f6ab0190699d6cd28d8f5a11f2ebdd9b69ffb5dfdf`.
[Portable summary](verification/2026-10-08/rea-summary.json), `runtime` section.
The full original export remains local at `local/out/rea-runtime-evidence.json`.
The committed summary is derived from the exports; it is not a canonical REA bundle.

`observe_native_calls` launched the exact independent test executable under LLDB,
with breakpoints for `cudaMalloc` and `dispatchThreadgroups:threadsPerThreadgroup:`.
The recorded events include seven allocation entries and two compute-dispatch
entries in `AGXG13GFamilyComputeContext`, Apple's M1 Metal implementation.
The dispatch backtraces include this chain:

```text
CUDA __device_stub__smoke_saxpy
  → cudaLaunch
  → cudaLaunchKernel
  → cumetal::metal_backend::launch_kernel
  → AGXG13GFamilyComputeContext dispatchThreadgroups:threadsPerThreadgroup:
```

The process exited 0 and printed `ALL CHECKS PASSED`, plus five completed Apple
GPU launch records for the independent cases. The REA event stream is evidence
that the dispatch boundary was reached; it is not an exhaustive call count.
It records fewer dispatch entries than the five kernel-completion records, and
lists additional breakpoint stops. Do not treat the two recorded entries as
the total work performed.

REA's limitations apply: entry observations do not capture return values,
floating-point or stack-passed arguments; breakpoints slow execution; executable
path identity was checked but no immutable loaded-image digest was established.
No timing claim is made from this debugger run.

## Numerical verification

[The dated results](verification/2026-10-08/results.json) contain six successful Apple M1 kernel-completion
records across the upstream vector-add example and the independent fixture.
Every record reports `device=apple_gpu`, `launch_success=true`, and
`semantic_quality=exact`; numerical results are checked separately in host code.
CPU emulation was explicitly disabled for these runs.

The independent fixture checked SAXPY sizes 1/257/100003, all 1517 outputs of
a rectangular matrix product, and all 391 shared-memory reduction block sums.
Each case also checked 512 output canaries. The reduction's final accumulation
was performed on the CPU after validating every GPU block sum.

This supports correctness of the specific FP32 kernels and launch shapes tested.
It does not establish full CUDA parity, application compatibility, NVIDIA speed
parity, full floating-point semantics, or behavior on another Apple chip.
