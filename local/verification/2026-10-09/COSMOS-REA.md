# Cosmos attention worker: numerics passed, new debugger trace unresolved

On October 9, 2026 (local time), the unchanged `local/out/cosmos-attention`
worker completed a deterministic FP32 attention request on Apple M4. Its three
completion records identified `cosmos_scores`, `cosmos_softmax`, and
`cosmos_values` as successful exact Apple-GPU launches. CPU emulation was
explicitly disabled.

The request used `[B,H,Q,K,D,Dv] = [1,2,17,23,32,16]`, non-causal attention,
float32 sine/cosine fixture inputs, and scale `1/sqrt(32)`. All 544 results
matched an independent JavaScript float64 attention calculation within
`atol=2e-5, rtol=2e-4`; maximum absolute error was `1.0617974827153986e-7`.
Executable and runtime hashes were unchanged before and after.

## What REA did not establish

The REA 6.3.0 call-path attempt did **not** capture a CUDA or Metal call.
Evidence `ev_7f7945eaa2cbbd3a87f7d564dd43827f57a492ced3d68c4169e175eb769df360`
reported a 30-second duration limit, zero events, two unresolved breakpoints,
empty stdout/stderr, and confirmed process termination. No output file was
created. It is partial, failed observation evidence, not a compatibility pass.

REA's current native-call API supplies `/dev/null` as stdin and has no request
field for input data. A small owned C launcher therefore opened the retained
fixture and output paths before an intended `exec` of the existing worker.
The failed trace did not establish that the launcher reached `main` or `exec`.
A separate direct LLDB diagnostic also produced no output; on terminating
its owned target, LLDB reported `dyld::__open + 8`. A loader stall before `main`
is an inference from these observations, not a captured `main` breakpoint.
This narrows the observed boundary but does not establish the cause. No authorization prompt was
established, no new permission was requested, and no OS security setting was
changed. The separate diagnostic's target, LLDB, and debugserver were confirmed
gone, and the REA session was closed.

The same launcher and fixture ran immediately without LLDB, producing the
passing native result above. The prior [smoke call-path proof](REA.md) remains
valid for its own target only; this attempt does not extend it to Cosmos.

## Retained artifacts

[Portable result and limitations](cosmos-rea-summary.json) contains the full
selected-file hashes, numeric tolerances, counts, and cleanup outcome.
Ignored local artifacts are in `local/out/rea-cosmos-2026-10-09/`:

- `rea-failed-harness.bundle.json`: canonical REA failed-observation bundle.
- `launch_attention.c`, `launch_attention`: test-owned stdin/output launcher.
- `fixture.mjs`, `input.bin`: deterministic generation and independent checker.
- `output.bin`, `numerics.json`: successful uninstrumented output and validation.
- `native-run.log`: runtime stderr transcribed from the successful tool result.

The launcher uses `O_EXCL` for its output and refuses to overwrite a file.
The fixture/checker likewise refuses to replace its generated artifacts.
This was one bounded kernel request, not a full video run. It proves neither
video quality nor broad Cosmos/Omniverse/Isaac Sim compatibility. Breakpoint
success and numerical success must continue to be reported separately.
