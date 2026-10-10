# Warp Security And Build Provenance

This is a bounded pre-download advisory lookup and build-path review from the
2026-10-09 local session, not a security certification or a full dependency audit.
GPU correctness and measured execution evidence are recorded separately.

## Source Identity

- Upstream: [NVIDIA/warp](https://github.com/NVIDIA/warp).
- Release: [v1.12.0](https://github.com/NVIDIA/warp/releases/tag/v1.12.0).
- Exact Git commit: `e6c3ba2d54bb048115760b5cd7a4bb2573329ae7`.
- The release page displays a verified tag. Independently, the local checkout
  at `local/out/warp-cumetal` returned the exact commit above with
  `git rev-parse HEAD`; the subsequent local patches do not change HEAD.
- The actual clone was performed without `GIT_LFS_SKIP_SMUDGE`. No claim is
  made that LFS assets were excluded. The pinned
  [.gitattributes](https://github.com/NVIDIA/warp/blob/e6c3ba2d54bb048115760b5cd7a4bb2573329ae7/.gitattributes)
  declares LFS filters for several asset types. Skipping their download is a
  future source-only checkout precaution, not evidence about this clone.

## Advisory Lookup

No published NVIDIA Warp-specific advisory/CVE match was found for
`warp-lang==1.12.0` in these primary-source checks:

| Source | Observed result |
| --- | --- |
| [NVIDIA/warp advisories](https://github.com/NVIDIA/warp/security/advisories) | No published advisories. |
| [GitHub advisory API](https://api.github.com/advisories?ecosystem=pip&affects=warp-lang%401.12.0&per_page=100) | `[]`. |
| [OSV query API](https://api.osv.dev/v1/query) | POST `{"package":{"name":"warp-lang","ecosystem":"PyPI"},"version":"1.12.0"}` returned `{}`. |
| [NVD query API](https://services.nvd.nist.gov/rest/json/cves/2.0?keywordSearch=NVIDIA%20Warp) | `totalResults: 0`. |

Cloudflare WARP and Warp terminal advisories are different products and were
excluded. Empty results do not exclude unpublished, unindexed, or transitive
dependency vulnerabilities; they are not a statement that this old release is
currently supported or vulnerability-free.

## Build Actually Used

The build used the existing `local/out/cosmos-venv/bin/python` environment and
existing NumPy, Xcode compiler, CuMetal build, and generated toolkit. No new
Python packages, LLVM, Packman, or libmathdx were installed for this build.
The effective arguments were:

```text
build_lib.py --cuda-path=<repository>/build/cumetal-cuda-toolkit \
  --no-use-libmathdx --no-standalone --jobs=2
```

`local/out/warp-build-first.log` records `cuda=True`, `standalone=False`,
`use_libmathdx=False`, `build_llvm=False`, `llvm_path=None`, and `jobs=2`.
This was a patched CuMetal CUDA build, not the unmodified upstream macOS
CPU-only build. The repository's `scripts/warp-patches/0001-cumetal-macos-build.patch`
honors an explicit CUDA toolkit on Darwin, supplies `WP_CUMETAL`, and adds
CuMetal driver/link/JIT handling; `0002-volume-builder-include-new.patch` adds
the needed C++ header. Upstream otherwise disables CUDA on Darwin in
[build_lib.py](https://github.com/NVIDIA/warp/blob/e6c3ba2d54bb048115760b5cd7a4bb2573329ae7/build_lib.py#L455).

CPU JIT is unavailable because `warp-clang` was not built. MathDx is disabled.
Neither choice establishes general Warp, NVIDIA CUDA, or Omniverse compatibility.

## Default Download Path And Mitigation

Upstream standalone mode defaults on. Without an explicit existing LLVM path,
[build_llvm.py](https://github.com/NVIDIA/warp/blob/e6c3ba2d54bb048115760b5cd7a4bb2573329ae7/build_llvm.py#L29)
fetches Packman's `clang+llvm-warp` package
`15.0.7-darwin-aarch64-macos11`. The separate `--build-llvm` route can clone
LLVM `21.1.0`; neither route was used here.

The pinned [Packman bootstrap](https://github.com/NVIDIA/warp/blob/e6c3ba2d54bb048115760b5cd7a4bb2573329ae7/tools/packman/packman#L87)
downloads Python `3.10.5-1-macos-x86_64` over HTTP, extracts it without a digest
check in that shell path, then executes the interpreter. This is a concrete
build-chain concern, not a claim of a published CVE or observed compromise.
Packman-common has a separate
[pinned SHA256 check](https://github.com/NVIDIA/warp/blob/e6c3ba2d54bb048115760b5cd7a4bb2573329ae7/tools/packman/bootstrap/install_package.py#L131),
which does not protect the preceding Python bootstrap.

The updated `scripts/build_warp_cumetal.sh` avoids that path by defaulting to
`--no-standalone` and always passing `--no-use-libmathdx`. CPU JIT opt-in requires
`CUMETAL_WARP_STANDALONE=1` and an already existing
`CUMETAL_WARP_LLVM_PATH` containing `include/` and `lib/`; the wrapper refuses
missing, contradictory, or shell-unsafe LLVM path settings before fetch/build.
It passes `--llvm-path` rather than downloading LLVM. This validates the path,
not LLVM ABI compatibility or security. The source clone itself still uses the
configured remote/tag and may access the network.

For initial source execution, the pinned
[pyproject.toml](https://github.com/NVIDIA/warp/blob/e6c3ba2d54bb048115760b5cd7a4bb2573329ae7/pyproject.toml#L14)
requires NumPy at runtime. Packaging adds setuptools/build/wheel, and example
extras add other dependencies; these were not needed or installed for this run.
