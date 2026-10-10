#!/usr/bin/env bash
# Run NVIDIA Warp's device-filtered unittest modules with bounded processes.
# Usage: bash scripts/run_warp_tests_cumetal.sh [test_vec ...]
# No arguments discovers every test_*.py module; live testing should select a
# bounded subset. The historical suite included crashes and GPU timeouts.
#
# Environment overrides:
#   CUMETAL_WARP_DIR       Warp checkout (default: ../warp-cumetal)
#   CUMETAL_WARP_RESULTS   NEW directory for logs/summary.json (default: build/warp-tests)
#   CUMETAL_WARP_TIMEOUT   per-module seconds (default: 300)
#   CUMETAL_PYTHON         interpreter for runner and Warp (default: python3/python)
#
# Only nonempty, entirely successful unittest runs pass. Skips and expected
# failures are separate, nonzero outcomes, as are crashes and missing results.
# cuda_0 filtering is not GPU provenance; validate device evidence separately.
# Process deadlines do not guarantee cancellation of outstanding GPU work.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${CUMETAL_PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
    for candidate in python3 python; do
        command -v "${candidate}" >/dev/null 2>&1 && { PYTHON="${candidate}"; break; }
    done
fi
[[ -n "${PYTHON}" ]] || { echo "ERROR: no python interpreter on PATH." >&2; exit 2; }
exec "${PYTHON}" "${SCRIPT_DIR}/run_warp_tests_cumetal.py" "$@"
