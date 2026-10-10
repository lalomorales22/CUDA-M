# M4 CUDA-to-Metal runtime trace

REA 6.3.0 successfully traced the independent smoke executable on Apple M4,
macOS 27.0, after the user approved the local debugger authorization prompt.
The earlier unapproved attempt timed out without events; it is not positive
evidence. No OS security setting was changed by the installation workflow.

## Observed result

[Portable evidence summary](rea-summary.json), runtime Evidence
`ev_5c44539a0250691e3adcb603e333cedfebe31e485a2574f0bd8810de68638c3e`:

- Process exited 0 and was confirmed terminated.
- 14 `cudaMalloc`, five `cudaLaunchKernel`, and five Metal dispatch entries.
- All five dispatch entries were in `AGXG16GFamilyComputeContext`, module
  `AGXMetalG16G_B0`, with CuMetal frames in their caller stacks.
- Five successful Apple M4 GPU-completion records, validated with the existing
  `cuda-mac` trace validator. CPU emulation was explicitly disabled.
- All numerical assertions and output canaries passed: SAXPY sizes 1, 257,
  100003; rectangular matrix multiplication; 391 reduction block partials.
  The final reduction accumulation is intentionally performed on the CPU.
- Output was complete and untruncated; no unresolved breakpoints, event limit,
  or resource limit. One `SIGSTOP` was retained among other stops.

The representative SAXPY caller chain was:

```text
CUDA device stub
  -> cudaLaunch
  -> cudaLaunchKernel
  -> cumetal::metal_backend::launch_kernel
  -> Metal dispatchThreadgroups:threadsPerThreadgroup:
```

The observed Apple driver class is an implementation of the public Metal
dispatch API, not an API imported or invoked privately by CuMetal.

Static Evidence
`ev_be3a5b1c4ec0f390930d1044593f1a55283bf70ce1adf16943bf219517ae021b`
independently records an arm64 runtime exporting the selected CUDA APIs and
linking Metal/MetalPerformanceShaders. Artifact hashes in both records still
matched the local executable/runtime after the trace.

## Reproduce

Install and audit a pinned REA release separately; it is not a build dependency.
Build and run `./cuda-mac test` first. Then, from the repository root, with
`rea` on PATH:

```bash
rea inspect-macho "$PWD/build/libcumetal.dylib" --format json
rea observe-native-calls "$PWD/local/out/apple_silicon_smoke" '{
  "breakpoints": [
    {"kind":"function","name":"cudaMalloc","module":"libcumetal.dylib"},
    {"kind":"function","name":"cudaLaunchKernel","module":"libcumetal.dylib"},
    {"kind":"objc-method","selector":"dispatchThreadgroups:threadsPerThreadgroup:"}
  ],
  "environment": {
    "CUMETAL_TRACE_GPU":"1",
    "CUMETAL_ENABLE_LLMC_CPU_EMULATION":"0",
    "CUMETAL_DISABLE_LLMC_EMULATION":"1"
  },
  "duration_ms":30000,
  "max_events":128,
  "argument_registers":0,
  "backtrace_frames":8
}' --format json
```

Handle any macOS authorization prompt locally. Do not interpret a timeout or
missing events as a successful trace. Retain and check process outcome, output,
coverage, and limitations alongside the event list.

## Scope and evidence retention

This verifies the selected source-compiled kernels and CUDA-to-Metal boundary.
It does not qualify universal CUDA, full application compatibility, Cosmos video
quality, or Omniverse/Isaac Sim. No decompiler was used or installed.

Breakpoint entries are not kernel completions. In this bounded run their
counts agree with separate post-completion runtime records and numerical
assertions. The selector resolved to 15 candidate locations, not 15 executed
calls. No arguments or return values were inspected. Inlining/direct calls can
escape entry breakpoints. Debugger timings are not performance measurements.
Selected executable pathname/module identity was stable, but REA did not hash
the exact mapped image, so loaded-image identity remains an explicit unknown.

The portable JSON is a derived summary, not a canonical Evidence bundle. Full
records remain in ignored `local/out/rea-2026-10-09/runtime-static.json` and
`runtime-calls-approved.json`. The failed first attempt is retained separately
as `runtime-calls.json`; it was not overwritten by the successful retry.
