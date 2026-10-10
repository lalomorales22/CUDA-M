#!/usr/bin/env python3
"""Bounded Warp unittest orchestration; test results are not GPU provenance."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parent.parent
COUNTS = ("tests_run", "passed", "failures", "errors", "skipped", "expected_failures", "unexpected_successes")


class CountingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed = 0

    def addSuccess(self, test):
        super().addSuccess(test)
        self.passed += 1


def worker(module, result_path):
    # Printed/partial unittest summaries are not completion evidence. A crash
    # or os._exit before unittest returns cannot produce this receipt.
    runner = unittest.TextTestRunner(resultclass=CountingResult)
    program = unittest.main(module=None, argv=["unittest", "-k", "cuda_0", "warp.tests." + module],
                            testRunner=runner, exit=False)
    result = program.result
    counts = {"tests_run": result.testsRun, "passed": result.passed,
              "failures": len(result.failures), "errors": len(result.errors),
              "skipped": len(result.skipped), "expected_failures": len(result.expectedFailures),
              "unexpected_successes": len(result.unexpectedSuccesses)}
    with Path(result_path).open("x") as stream:
        json.dump({"module": module, "completed": True, **counts}, stream)
        stream.write("\n")
    return 0 if result.wasSuccessful() else 1


class RunInterrupted(Exception):
    def __init__(self, signum):
        self.signum = signum


def interrupt(signum, _frame):
    # The first stop request owns shutdown. Further requests must not interrupt
    # group termination, reaping, or the final evidence write.
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, signal.SIG_IGN)
    raise RunInterrupted(signum)


def kill_group(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def load_result(path, module):
    try:
        result = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError("missing or invalid structured unittest result") from error
    if (not isinstance(result, dict) or result.get("module") != module or result.get("completed") is not True
            or any(type(result.get(key)) is not int or result[key] < 0 for key in COUNTS)
            or result["passed"] > result["tests_run"]):
        raise ValueError("invalid structured unittest result fields")
    return {key: result[key] for key in COUNTS}


def run_module(module, warp_dir, output_dir, timeout):
    log_path, result_path = output_dir / (module + ".log"), output_dir / (module + ".result.json")
    command = [sys.executable, str(Path(__file__).resolve()), "--worker", module, str(result_path)]
    env = dict(os.environ, PYTHONPATH=str(warp_dir))
    record = {"module": module, "status": "running", "command": command,
              "log": log_path.name, "result": result_path.name, "timeout_seconds": timeout}
    started = time.monotonic()
    process = None
    try:
        with log_path.open("x") as log:
            process = subprocess.Popen(command, cwd=warp_dir, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                record["returncode"] = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                record["status"] = "timed_out"
    except RunInterrupted as error:
        record.update(status="interrupted", signal=error.signum)
    except OSError as error:
        record.update(status="launch_error", error=str(error))
    finally:
        if process is not None:
            # Reap the parent and kill compiler/other descendants still in its
            # session, including after a crash or an early successful exit.
            try:
                kill_group(process)
            except RunInterrupted as error:
                # The first signal can arrive during ordinary timeout/exit
                # cleanup rather than during wait(); finish reaping either way.
                record.update(status="interrupted", signal=error.signum)
                kill_group(process)
            record["returncode"] = process.returncode
        record["elapsed_seconds"] = time.monotonic() - started
    try:
        record["tests"] = load_result(result_path, module)
    except ValueError as error:
        record["result_error"] = str(error)
    if record["status"] == "running":
        counts = record.get("tests")
        if record["returncode"] != 0:
            record["status"] = "crashed" if record["returncode"] < 0 else "failed"
        elif counts is None:
            record["status"] = "missing_result"
        elif counts["failures"] or counts["errors"] or counts["unexpected_successes"]:
            record["status"] = "failed"
        elif counts["tests_run"] == 0:
            record["status"] = "no_tests"
        elif counts["skipped"] or counts["expected_failures"]:
            record["status"] = "incomplete"
        elif counts["passed"] != counts["tests_run"]:
            record["status"] = "incomplete"
        else:
            record["status"] = "passed"
    (output_dir / (module + ".exit")).write_text(str(record.get("returncode", "not_started")) + "\n")
    return record


def save_summary(path, report):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modules", nargs="*")
    args = parser.parse_args(argv)
    warp_dir = Path(os.environ.get("CUMETAL_WARP_DIR", ROOT.parent / "warp-cumetal")).expanduser().resolve()
    output_dir = Path(os.environ.get("CUMETAL_WARP_RESULTS", ROOT / "build/warp-tests")).expanduser().absolute()
    try:
        timeout = float(os.environ.get("CUMETAL_WARP_TIMEOUT", "300"))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if not (warp_dir / "warp/tests/__init__.py").is_file():
            raise ValueError("no Warp checkout at " + str(warp_dir))
        modules = args.modules or sorted(path.stem for path in (warp_dir / "warp/tests").glob("test_*.py"))
        if not modules:
            raise ValueError("no test modules discovered")
        if any(not re.fullmatch(r"test_[A-Za-z0-9_]+", module) for module in modules):
            raise ValueError("module names must be test_ followed by letters, digits or underscores")
        if len(set(modules)) != len(modules):
            raise ValueError("duplicate modules would overwrite evidence")
        if output_dir.exists() or output_dir.is_symlink():
            raise ValueError("results directory already exists; select a new directory")
        output_dir.mkdir(parents=True, exist_ok=False)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    report = {"schema_version": 1, "status": "running", "warp_directory": str(warp_dir),
              "python": sys.executable, "python_version": sys.version, "filter": "cuda_0",
              "requested_modules": modules, "timeout_seconds": timeout, "modules": [],
              "evidence_scope": "upstream unittest outcomes only; cuda_0 names and successful exits do not prove Apple-GPU execution",
              "pass_policy": "every selected module must execute tests with no failures, errors, skips, expected failures or unexpected successes",
              "timeout_scope": "process-group termination; outstanding GPU work may not be cancelled"}
    path = output_dir / "summary.json"
    handlers = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    exit_code = 1
    try:
        save_summary(path, report)
        for module in modules:
            record = run_module(module, warp_dir, output_dir, timeout)
            report["modules"].append(record)
            print(f"{record['status']}: {module}", flush=True)
            save_summary(path, report)
            if record["status"] == "interrupted":
                raise RunInterrupted(record["signal"])
        report["status"] = "passed" if all(r["status"] == "passed" for r in report["modules"]) else "failed"
        exit_code = 0 if report["status"] == "passed" else 1
    except RunInterrupted as error:
        report.update(status="interrupted", signal=error.signum)
        exit_code = 128 + error.signum
    except OSError as error:
        report.update(status="failed", error=str(error))
    finally:
        report["not_run"] = modules[len(report["modules"]):]
        report["test_totals"] = {key: sum(record.get("tests", {}).get(key, 0) for record in report["modules"])
                                 for key in COUNTS}
        save_summary(path, report)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    print(f"{report['status']}: {len(report['modules'])}/{len(modules)} modules; {path}", flush=True)
    return exit_code


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--worker":
        sys.exit(worker(sys.argv[2], sys.argv[3]))
    sys.exit(main())
