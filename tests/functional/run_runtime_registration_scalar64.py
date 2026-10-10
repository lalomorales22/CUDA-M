#!/usr/bin/env python3
"""Check compiler-owned registration ABI across cold, warm, and corrupt caches."""

import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile


def main():
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        print("SKIP: xcrun not installed")
        return 77
    for tool in ("metal", "metallib"):
        if subprocess.run([xcrun, "--find", tool], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL, timeout=10).returncode != 0:
            print(f"SKIP: xcrun {tool} not available")
            return 77
    binary = sys.argv[1]
    with tempfile.TemporaryDirectory(prefix="cumetal-scalar64-") as temporary:
        for backend in ("legacy", "cumetal-ir"):
            cache = Path(temporary) / backend
            env = dict(os.environ, CUMETAL_CACHE_DIR=str(cache), CUMETAL_TRACE_GPU="1",
                       CUMETAL_PTX_BACKEND=backend, CUMETAL_DEBUG_REGISTRATION="1",
                       CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS="0",
                       CUMETAL_ENABLE_LLMC_CPU_EMULATION="0", CUMETAL_DISABLE_LLMC_EMULATION="1")

            def run(label, expected):
                result = subprocess.run([binary], env=env, text=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, timeout=90)
                print(f"== {backend}: {label}\n{result.stdout}", end="")
                assert result.returncode == 0, result.returncode
                assert result.stdout.count("REGISTRATION_SCALAR64_OK") == 1
                launches = [dict(token.split("=", 1) for token in shlex.split(line)[1:])
                            for line in result.stdout.splitlines()
                            if line.startswith("CUMETAL_PROVENANCE event=kernel_launch ")]
                assert len(launches) == 6, "invalid pointer calls must not dispatch"
                assert all(record.get("kernel") == "copy_scalar_bits" and
                           record.get("device") == "apple_gpu" and
                           record.get("semantic_quality") == "exact" and
                           record.get("launch_success") == "true" and
                           int(record.get("duration_ns", -1)) >= 0 for record in launches)
                assert expected in result.stdout, f"missing {expected}"

            run("cold", "jit cache miss:")
            run("warm", "jit metadata cache hit:")
            metadata, = (cache / "registration-jit").glob("*.metadata")
            original = metadata.read_bytes()
            assert original[:8] == b"CUMETA03"
            # This fixture has no printf formats: argument count starts at 16.
            assert struct.unpack_from("<III", original, 8) == (8, 0, 2)
            corruptions = {
                "unknown authority flags": original[:8] + struct.pack("<I", 16) + original[12:],
                "non-authoritative argument payload": original[:8] + struct.pack("<I", 0) + original[12:],
                "truncated argument": original[:-1],
                "unknown argument kind": original[:20] + struct.pack("<I", 3) + original[24:],
                "zero argument size": original[:24] + struct.pack("<I", 0) + original[28:],
                "excess argument count": original[:16] + struct.pack("<I", 4097) + original[20:],
            }
            for label, corrupt in corruptions.items():
                metadata.write_bytes(corrupt)
                run(label, "jit metadata cache miss:")
                assert metadata.read_bytes() == original, "metadata was not repaired"
            run("repaired warm", "jit metadata cache hit:")
    print("PASS: compiler argument ABI, scalar bits, pointer rejection, and cache validation")


if __name__ == "__main__":
    sys.exit(main())
