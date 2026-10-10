# PhysX on Metal

A local single-file PHP application that runs the existing source-patched
PhysX 5.6.1 rigid-body snippet through CUDA-M and renders its actual captured
poses in Three.js. Each new run executes native CPU and Apple-GPU simulations;
the browser replays their captured transforms, it does not simulate physics.

## Run

Requires the existing Apple Silicon CUDA-M/PhysX build, PHP 8.1+ with
`proc_open`, and Node/npm for the two frontend libraries. No PHP packages,
database, paid API, remote compute, or model download is needed.

From the repository root:

```bash
# Once, if the pinned patched PhysX build is not already present:
env PATH="$PWD/local/toolchain:$PATH" \
  CUMETAL_TOOL_XCRUN="$PWD/local/toolchain/xcrun" \
  PHYSX_REPO="$PWD/local/out/PhysX" CUMETAL_PHYSX_BUILD_JOBS=2 \
  scripts/physx-patches/build_physx_cumetal_grb_macos.sh

npm --prefix demos/physx-lab ci --ignore-scripts
bash demos/physx-lab/run.sh
```

Open <http://127.0.0.1:8766>. Set `PHYSX_LAB_PORT` to use another free port.
The app binds to loopback only. Do not expose PHP's development server publicly.
If the source checkout is missing, follow the pinned checkout instructions in
[the NVIDIA workload ledger](../../docs/nvidia-workloads.md) first.

Three.js `0.186.1` and Lucide `1.54.0` are pinned with integrity hashes in the
lockfile. Both have no runtime dependencies. Before the October 9, 2026 install,
OSV exact-version checks and npm audit found no known advisories; npm verified
both registry signatures. This is a dated check, not a security guarantee.
Repeat `npm --prefix demos/physx-lab audit` before future installs.
Assets are served locally, with no CDN requests. Upstream libraries retain
their MIT (Three.js) and ISC (Lucide) licenses inside their installed packages.

## Supported scenes

| Scene | Native work | CPU comparison |
| --- | --- | --- |
| Rolling sphere | 240 steps at 60 Hz, friction plus a separate frictionless GPU control | All 13 state components at every step; relative `1e-3`, absolute `1e-5` |
| Mesh crossing | 30 frictionless steps across a coplanar triangle seam | All 13 components; relative `1e-3`, absolute `1e-5` |
| Two-box stack | 30 frictionless steps, two unit boxes | All 13 components; relative `2e-2`, absolute `1e-5` |

The relative term scales by `max(abs(CPU), 1)`, matching the existing multibody
gate. Every position, quaternion, linear velocity, and angular velocity must
be finite and within its limit. Quaternions must be normalized. Rolling also
checks friction response and the constant-velocity frictionless control.

Every GPU record must report a successful exact Apple-GPU metallib launch.
Scene-specific contact/solver/integration kernels are required. The application
refuses truncated or reordered states, missing work, CPU fallback, native
errors, timeouts, and comparison failures. It never publishes a failed run as
the latest successful capture. Failed logs are retained for diagnosis.

The green body is the CuMetal GPU result. The gold body is a separately run
frictionless control, offset sideways only for rendering. The optional coral
wireframe is the CPU reference at the same captured pose. Playback speed, camera,
and grid controls change visualization only. Native run time includes CPU/control
runs, synchronous launches, trace logging, and validation; it is not a GPU
performance benchmark.

## Boundaries

This is the repository's selected patched PhysX source port, not stock binary
compatibility, full PhysX, or Isaac Sim. ABP broadphase and host orchestration
remain on CPU; GPU contact/dynamics/solver/integration use CUDA-M/Metal. Native
device-address mode and per-launch synchronization are required. PBD cloth is
deliberately excluded because its documented spring constraints are ineffective.

The mesh-friction experiment was also excluded: checking all 13 components
at every step found an angular-velocity mismatch at step 22 on M4. The older
mesh gate checks poses and final friction invariants, not full intermediate
velocity parity. Its success does not contradict this stricter failure.
No tolerance was widened to hide that result. The viewer's frictionless mesh
case passes the stricter gate.

Each capture keeps TSVs, logs, a result JSON, artifact hashes, and measured
metrics under ignored `out/<run-id>/`. `out/latest.json` is atomically replaced
after validation. Captures persist until explicitly removed by the operator.
Native runs are serialized, limited to fixed scene arguments, and bounded by
time/output limits. Same-origin POSTs require a session token; the router
serves only the page, allowlisted assets, and JSON APIs, not arbitrary files.

## Verify

```bash
php demos/physx-lab/test.php
php -l demos/physx-lab/index.php
ctest --test-dir build -R '^unit_documentation_consistency$' --output-on-failure
```

The 33 PHP tests are CPU-only parser/validator, subprocess, and persistence
checks, including negative paths. Running the three scenes in the app is the
actual native numerical/device gate. The [verification snapshot](../../local/verification/2026-10-09/physx-lab-summary.json)
records all three passing captures, the excluded mesh-friction failure, and
desktop/mobile Playwright checks with nonblank/moving canvas pixel evidence.
