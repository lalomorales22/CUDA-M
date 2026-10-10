# Warp patch workflow

These patches target NVIDIA Warp `v1.12.0`, commit
`e6c3ba2d54bb048115760b5cd7a4bb2573329ae7`.

Warp needs two upstream changes before it builds against CuMetal. They belong
in NVIDIA's repository, so this repo carries them as patches rather than a
fork, and applies them to a pinned clone:

```bash
scripts/build_warp_cumetal.sh
```

That clones `https://github.com/NVIDIA/warp.git` at `v1.12.0` into
`../warp-cumetal` (override with `CUMETAL_WARP_DIR`), applies the patch set,
generates the CuMetal CUDA toolkit shim, and compiles `libwarp`'s CUDA sources
through it. Pass `--clone-only` to stop after patching, or `--build` to hand
off to Warp's own `build_lib.py`.

## Build Without Extra Downloads

The `--build` path defaults to `CUMETAL_WARP_STANDALONE=0` and passes both
`--no-standalone` and `--no-use-libmathdx`. It uses an existing Python
interpreter and its installed dependencies; this wrapper does not install
Python packages. The default builds Warp's native CUDA library but does not
build `warp-clang`, so Warp CPU kernel JIT is unavailable in a fresh checkout.
CUDA device testing is separate from CPU JIT support. An existing stale
`warp-clang` library is not deleted by this option.

For example, after provisioning the existing interpreter and NumPy:

```bash
CUMETAL_WARP_DIR=/path/to/warp-cumetal \
CUMETAL_PYTHON=/path/to/existing/python \
CUMETAL_JOBS=2 \
bash scripts/build_warp_cumetal.sh --build
```

Standalone CPU JIT is an explicit opt-in and requires an existing LLVM prefix:

```bash
CUMETAL_WARP_STANDALONE=1 \
CUMETAL_WARP_LLVM_PATH=/path/to/existing/llvm \
CUMETAL_PYTHON=/path/to/existing/python \
bash scripts/build_warp_cumetal.sh --build
```

The wrapper validates `CUMETAL_WARP_STANDALONE` as exactly `0` or `1` before
cloning or patching. With `1`, the LLVM path must already contain `include/`
and `lib/`; it is passed explicitly as `--llvm-path` so Warp does not use its
default Packman LLVM download. This validates the local prefix layout, not
LLVM version or link compatibility, which Warp's build must still establish.
The canonical LLVM prefix may contain only ASCII letters, digits, `_`, `.`,
`/`, `+`, or `-`. Spaces and shell metacharacters are rejected before fetching:
although this wrapper preserves structured arguments, pinned Warp constructs
shell commands from the LLVM include/library paths internally.
Setting an LLVM path with standalone disabled is rejected rather than ignored.
The source checkout is still cloned when missing; reuse a previously patched
checkout to avoid that initial source download too.

The CPU-only wrapper contract test mocks all external commands:

```bash
python3 tests/unit/check_warp_build_contract_test.py scripts/build_warp_cumetal.sh
```

To patch a checkout you already have:

```bash
scripts/warp-patches/apply_warp_patches.sh /path/to/warp
```

The application script is idempotent and rejects any other Warp revision.

The first patch makes Warp build and run with CUDA enabled on macOS. It:

- guards `crt.h`'s barebones-Clang include on `WP_CUMETAL`, since CuMetal drives
  Clang with a real CUDA header set;
- adds an `__APPLE__` branch to `cuda_util.cpp` that `dlopen`s `libcuda.dylib`
  or `libcumetal.dylib` for the driver surface Warp resolves through
  `cuGetProcAddress`;
- honours an explicit `--cuda-path` on Darwin, where `build_lib.py` otherwise
  discards it because no CUDA toolkit can exist;
- adds a CuMetal branch to `build_dll.py` that reads the toolkit's
  `version.json`, defines `WP_CUMETAL=1`, and links `-lcuda` with the toolkit's
  library directory on the rpath, instead of the NVRTC and PTX compiler static
  libraries CuMetal does not ship separately;
- carries `WP_CUMETAL=1` into the JIT NVRTC options in `warp.cu`, so a
  runtime-compiled kernel sees the same `crt.h` branch the static library did;
- returns `cubin` from `Device.get_cuda_output_format` on Darwin. Warp otherwise
  picks PTX whenever the driver is at least as new as the toolkit, and CuMetal
  lowers CUDA source to a Metal library with no PTX to hand back at any version.

The second patch includes `<new>` in `volume_builder.cu` for its device-side
placement `new`. nvcc includes it implicitly and Clang does not, so the file
fails to compile through any Clang-driven toolchain, CuMetal's included.

## What runs

`scripts/build_warp_cumetal.sh --build` links `warp/bin/libwarp.dylib` against
`libcumetal`, and `warp.init()` enumerates `cuda:0`. A `@wp.kernel` over
`wp.array(dtype=wp.vec3)` compiles through NVRTC and returns correct results.
Warp's own `warp/tests` are a long way from green on that device; see
`docs/warp-feasibility.md` for the measured baseline.

## What compiles

All 11 of `libwarp`'s `.cu` files compile through CuMetal as of 2026-09-05, and
`build_warp_cumetal.sh` fails if any of them regresses. Getting the last five
there needed, on the CuMetal side: `.cuh` spellings of the device-wide CUB
headers, `cub::DoubleBuffer` with the matching `DeviceRadixSort` overloads, a
`DeviceSegmentedRadixSort`, and a `cub::BlockReduce` that works in device code
rather than only on the host. See `docs/warp-feasibility.md` for what that
result does and does not claim -- in particular, the CUB device-wide algorithms
are host-backed, so this is correctness rather than device parallelism.
