# NVIDIA workload ledger

Updated 2026-10-09. Local target: Apple M4 Mac mini, 32 GiB, macOS 27.0.
Source-first CUDA recompilation is the supported direction, not arbitrary
NVIDIA binary or product compatibility.

## Product boundaries

| Workload | Current boundary |
| --- | --- |
| Cosmos Predict2 | Bounded attention bridge and reduced transformer checks pass. Pretrained nine-frame execution completes, but footage is severely distorted; usable-video quality is not qualified. |
| PhysX 5.6.1 | All seven existing patched-source conformance gates pass on M4. Results apply only to the selected port, not the whole SDK or Isaac Sim. |
| Omniverse Kit | Native macOS Apple-GPU execution is not supported by upstream requirements and has not been demonstrated here. |
| Isaac Sim | Native macOS execution is not supported by upstream requirements and has not been demonstrated here. |

[Isaac Sim requirements](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/requirements.html)
list x86 Ubuntu/Windows and an NVIDIA GPU with RT cores; the minimum listed
GPU is RTX 4080. Even NVIDIA A100/H100 devices without RT cores are excluded.
[Omniverse technical requirements](https://docs.omniverse.nvidia.com/dev-guide/latest/common/technical-requirements.html)
list Windows/Linux targets, including selected Linux arm64 RTX systems, not
macOS Apple GPUs. Checked 2026-10-09. A Mac streaming client is not a native
simulation engine. CUDA-M does not implement the ELF/Windows ABI, RTX renderer,
Vulkan/DX12/OptiX stack, or proprietary product modules.

## PhysX source experiment

Source: NVIDIA-Omniverse/PhysX tag `107.3-physx-5.6.1`, commit
`5ca9f472105a90d70d957c243cb0ef36fe251a9f`. The existing repository patch series
is applied; algorithms and workload coverage are intentionally bounded.
See [patch instructions](../scripts/physx-patches/README.md).
Downloaded NVIDIA sources remain in ignored `local/out/PhysX` and are not
redistributed in CUDA-M's clean-room headers.

Before cloning, checked the [release](https://github.com/NVIDIA-Omniverse/PhysX/releases/tag/107.3-physx-5.6.1),
[repository security page](https://github.com/NVIDIA-Omniverse/PhysX/security),
and [NVIDIA product security](https://www.nvidia.com/en-us/product-security/).
No PhysX-specific advisory was found in that check. This is not a security
certification or a claim that no vulnerabilities exist. No driver, kernel
extension, or new package installation is required for this source experiment.

Initial build stopped before GPU execution because LLVM 22's partial standard
`float.h` include hid `FLT_MAX` and `FLT_EPSILON`. Loading floating limits
before `math.h` fixes that frontend blocker. A native-AOT GPU regression checks
the limits, while host/device static assertions cover IEEE float/double metadata.
The runtime build completes with 126 generated Metal libraries. The rigid-body
demo executes on the Apple GPU; its 30-step CPU/GPU comparison passes at
relative tolerance `1e-3`, absolute tolerance `1e-5`, with 960 verified GPU
launches. All seven existing PhysX gates pass in a separate 29.05-second replay:
rigid body, friction, multibody, stacked, box, triangle mesh, and PBD recreations.
The PBD gate checks selected physical invariants; it is not a full CPU/GPU
parity test for every cloth/particle feature. No gate was waived or modified.
Portable evidence is in [the PhysX snapshot](../local/verification/2026-10-09/physx-summary.json).

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --branch 107.3-physx-5.6.1 \
  https://github.com/NVIDIA-Omniverse/PhysX.git local/out/PhysX
env PATH="$PWD/local/toolchain:$PATH" \
  CUMETAL_TOOL_XCRUN="$PWD/local/toolchain/xcrun" \
  PHYSX_REPO="$PWD/local/out/PhysX" CUMETAL_PHYSX_BUILD_JOBS=2 \
  scripts/physx-patches/build_physx_cumetal_grb_macos.sh
env PATH="$PWD/local/toolchain:$PATH" \
  CUMETAL_TOOL_XCRUN="$PWD/local/toolchain/xcrun" \
  PHYSX_REPO="$PWD/local/out/PhysX" CUMETAL_PHYSX_BUILD_JOBS=2 \
  ctest --test-dir build --output-on-failure -R '^conformance_physx_'
```

## Next gates

1. Broaden the passing 51-file compiler matrix with external source workloads;
   retain negative pointer/scope coverage and numerical execution checks.
2. Broaden PhysX workloads beyond these passing patched subsets, retaining
   numerical comparisons and actual Metal kernel launch evidence.
3. Expand source workload coverage using Warp and the CUDA samples manifest.
4. Diagnose Cosmos backend numerical/visual parity before enlarging the model
   or presenting it as usable video generation.

No new library shim should be added without a named workload and its gate.
