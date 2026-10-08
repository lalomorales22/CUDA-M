# CUDA-M local setup

This is a local source build of [CuMetal](https://github.com/Lulzx/cuda-metal),
plus a Mac-specific toolchain wrapper and independent numerical tests.
The compiler/runtime are upstream CuMetal, not a newly written NVIDIA CUDA driver.

Verified on **Apple M1, macOS 26.4.1**, on October 8, 2026:
ordinary `.cu` programs compile and execute selected FP32 kernels on the Apple GPU.
All six launches passed their CPU reference checks and Apple-GPU provenance checks.
REA also observed CUDA calls reaching Apple's Metal compute implementation.
See [the evidence report](EVIDENCE.md).

## Run it

```bash
cd ~/Software/cuda-mac
./cuda-mac test
```

This recompiles both programs every time, checks their output and GPU traces,
and writes `results.json` and compilation/runtime logs to `local/out/`.
The [committed verification snapshot](verification/2026-10-08/results.json) records
the original M1 run and remains available in a fresh clone.
It exits unsuccessfully for missing work, incorrect results, CPU execution,
approximate kernels, or failed launches. The report records source, executable,
and runtime SHA-256 identities. A failed rerun replaces the old success status.

Compile and run the included CUDA example yourself:

```bash
cd ~/Software/cuda-mac
./cuda-mac compile samples/vectorAdd/vectorAdd.cu -o local/out/my-vector-add
CUMETAL_TRACE_GPU=1 ./local/out/my-vector-add
```

Use `./cuda-mac compile your-file.cu -o your-program` for another supported
CUDA program. The output links to this checkout's runtime, so keep the checkout
in place. Existing NVIDIA executables and arbitrary CUDA applications are not
made compatible by this command.

## What was tested

| Work | Coverage |
| --- | --- |
| Upstream CUDA vector addition | Every one of 16,384 outputs |
| Independent SAXPY | Sizes 1, 257, and 100,003; partial blocks |
| Independent matrix multiply | 37×29 times 29×41; all 1,517 outputs; 2D blocks |
| Independent shared-memory reduction | 100,003 inputs; all 391 GPU block sums; barriers and partial final block |
| Output bounds | 512 untouched tail values after each independent case |

Inputs use small signed binary fractions. The checked calculations have exact
FP32 representations; comparison also rejects NaN and infinity. Reduction's
last step combines verified block sums on the CPU and is labeled that way.
These are correctness checks, not performance benchmarks or broad CUDA conformance.

## Build and toolchain

```bash
cd ~/Software/cuda-mac
./cuda-mac build
./cuda-mac doctor
python3 local/test_trace_validation.py
```

The local wrapper uses two build jobs and builds only the compiler, runtime,
doctor, and CUDA toolchain helpers. It requires native arm64 macOS, Python 3.8+,
CMake, Ninja, upstream LLVM, LZ4, Zstd, and Apple's Metal tools. This Mac already
had the Homebrew build dependencies. LLVM defaults to `/opt/homebrew/opt/llvm`;
set `LLVM_DIR` for a different CMake package path. The pinned VF64 submodule is
fetched when missing. Nothing is installed globally by `./cuda-mac build`.

On this Mac, Xcode 26.6 first-launch components were inconsistent and Metal was
missing. Apple's `xcodebuild -runFirstLaunch` completed successfully, followed by
`xcodebuild -downloadComponent MetalToolchain` (17F109, about 688 MB).
The installed Metal compiler is version 32023.883.

The default Xcode `metal` launcher still reports a missing toolchain even though
the installed component runs directly. `local/toolchain/xcrun` resolves that
component's `metal` and `metallib` executables for this project's processes.
Other tools go through `/usr/bin/xcrun`. The wrapper discovers the mounted path
each time; it does not hardcode this machine's temporary mount name or change
shell startup files. If several components are mounted and Xcode cannot choose,
set `CUDA_MAC_METAL_BIN` to the intended component's `usr/bin` directory.
Re-run Apple's download command if the component is unavailable after a reboot.

When run through a restricted agent sandbox, compilation or GPU execution may
require access outside that sandbox. The numerical verification here used that
access. Normal Terminal use does not need a separate agent approval.

## Boundaries

This is experimental compatibility through Metal. It does not make PyTorch's
CUDA backend, arbitrary CUDA extensions, NVIDIA machine code (SASS), tensor-core
instructions, or every CUDA library work on a Mac. Some upstream library paths
use the CPU. Native NVIDIA FP64 behavior is not available from the Apple GPU.
The optional binary alias was disabled for this Release build. M4 execution and
performance have not been tested here.

See upstream [known gaps](../docs/known-gaps.md) before choosing an application.
The next useful step is compiling a specific CUDA workload and expanding its
numerical checks. The full upstream test suite was not run; no compiler or runtime
source changes were made in this checkout.

## Files added locally

- `cuda-mac`: build, doctor, compile, and verification commands.
- `local/toolchain/xcrun`: installed Metal component discovery.
- `local/apple_silicon_smoke.cu`: complete ordinary CUDA test program.
- `local/test_trace_validation.py`: rejects misleading success evidence.
- `local/EVIDENCE.md`: findings, identities, and REA limitations.

Upstream revision: `fdf403ebdd14a57565fd167d675e2a9910a72c35` (CuMetal 0.6.0).
VF64 revision: `8dae385a16493282e2a3e90e4cd1aa1250b36dc5`.
Upstream code retains its Apache-2.0 license.
