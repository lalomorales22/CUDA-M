#!/usr/bin/env python3
"""Preserve explicit workload-specialization ABI and provenance across JIT reuse."""

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
    expected = {
        "_ZL9rope_neoxILb1ELb0EffEvPKT1_PT2_iiiiiiiiiiPKifff14rope_corr_dimsfPKfPKxi",
        "_ZL9rope_neoxILb1ELb0EfDF16_EvPKT1_PT2_iiiiiiiiiiPKifff14rope_corr_dimsfPKfPKxi",
    }
    with tempfile.TemporaryDirectory(prefix="cumetal-rope-neox-") as temporary:
        env = dict(os.environ, CUMETAL_CACHE_DIR=temporary, CUMETAL_TRACE_GPU="1",
                   CUMETAL_DEBUG_REGISTRATION="1", CUMETAL_PTX_BACKEND="legacy",
                   CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS="1",
                   CUMETAL_ENABLE_LLMC_CPU_EMULATION="0", CUMETAL_DISABLE_LLMC_EMULATION="1")
        for phase in ("cold", "warm"):
            result = subprocess.run(sys.argv[1:3], env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=90)
            print(f"== {phase}\n{result.stdout}", end="")
            if result.returncode == 77:
                return 77
            assert result.returncode == 0, result.returncode
            assert "PASS: exact forward no-FF GPT-NeoX RoPE float/half outputs" in result.stdout
            launches = [dict(token.split("=", 1) for token in shlex.split(line)[1:])
                        for line in result.stdout.splitlines()
                        if line.startswith("CUMETAL_PROVENANCE event=kernel_launch ")]
            assert len(launches) == 2 and {record.get("kernel") for record in launches} == expected
            assert all(record.get("device") == "apple_gpu" and
                       record.get("source") == "specialized_msl" and
                       record.get("provenance") == "workload_specialization" and
                       record.get("semantic_quality") == "exact" and
                       record.get("launch_success") == "true" and
                       int(record.get("duration_ns", -1)) >= 0 for record in launches)
            event = "jit cache miss:" if phase == "cold" else "jit metadata cache hit:"
            assert result.stdout.count(event) == 2, f"missing two {phase} cache events"
            sidecars = list((Path(temporary) / "registration-jit").glob("*.metadata"))
            assert len(sidecars) == 2
            for sidecar in sidecars:
                data = sidecar.read_bytes()
                assert data[:8] == b"CUMETA03"
                assert struct.unpack_from("<III", data, 8) == (0, 0, 0), "template ABI is not authoritative"
    print("PASS: cold/warm specialized RoPE ABI and provenance preserved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
