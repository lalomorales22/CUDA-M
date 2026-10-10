#!/usr/bin/env python3
"""CPU-only runner checks using a temporary fake Warp unittest package."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/run_warp_tests_cumetal.sh"
PASS = "import unittest\nclass Case(unittest.TestCase):\n    def test_ok_cuda_0(self): pass\n"


class WarpRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="warp runner ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.warp = self.root / "checkout with spaces"
        self.tests = self.warp / "warp/tests"
        self.tests.mkdir(parents=True)
        (self.warp / "warp/__init__.py").touch()
        (self.tests / "__init__.py").touch()
        self.output = self.root / "results with spaces"
        self.env = dict(os.environ, CUMETAL_PYTHON=sys.executable,
                        CUMETAL_WARP_DIR=str(self.warp), CUMETAL_WARP_RESULTS=str(self.output),
                        CUMETAL_WARP_TIMEOUT="5")

    def module(self, name, source=PASS):
        (self.tests / (name + ".py")).write_text(source)

    def invoke(self, *modules, env=None):
        return subprocess.run(["bash", str(RUNNER), *modules], cwd=self.root,
                              env=env or self.env, capture_output=True, text=True, timeout=15)

    def summary(self):
        return json.loads((self.output / "summary.json").read_text())

    def test_pass_filter_and_relative_paths(self):
        self.module("test_good", PASS + "    def test_bad_cpu(self): self.fail('CPU tests must be filtered')\n")
        env = dict(self.env, CUMETAL_WARP_DIR=self.warp.name, CUMETAL_WARP_RESULTS=self.output.name,
                   CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS="0", WARP_CACHE_PATH="caller-cache")
        self.module("test_env", "import os, unittest\nclass Case(unittest.TestCase):\n"
                    "    def test_env_cuda_0(self):\n"
                    "        self.assertEqual(os.environ['WARP_CACHE_PATH'], 'caller-cache')\n"
                    "        self.assertEqual(os.environ['CUMETAL_ENABLE_WORKLOAD_SPECIALIZATIONS'], '0')\n")
        run = self.invoke("test_good", "test_env", env=env)
        self.assertEqual(run.returncode, 0, run.stderr + run.stdout)
        report = self.summary()
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["test_totals"]["passed"], 2)
        self.assertEqual(report["test_totals"]["tests_run"], 2)
        self.assertEqual(report["not_run"], [])
        self.assertIn("do not prove Apple-GPU", report["evidence_scope"])
        self.assertEqual((self.output / "test_good.exit").read_text(), "0\n")

    def test_failure_import_error_and_later_success_propagate(self):
        self.module("test_fail", PASS.replace("pass\n", "self.fail('failure')\n"))
        self.module("test_import", "raise ImportError('import failed')\n")
        self.module("test_good")
        run = self.invoke("test_fail", "test_import", "test_good")
        self.assertEqual(run.returncode, 1)
        report = self.summary()
        self.assertEqual([r["status"] for r in report["modules"]], ["failed", "failed", "passed"])
        self.assertEqual(report["test_totals"]["passed"], 1)
        self.assertEqual(report["test_totals"]["failures"], 1)
        self.assertEqual(report["test_totals"]["errors"], 1)

    def test_zero_skipped_and_expected_failure_are_not_passes(self):
        self.module("test_zero", PASS.replace("cuda_0", "cpu"))
        self.module("test_skip", PASS.replace("    def test", "    @unittest.skip('not supported')\n    def test"))
        self.module("test_expected", PASS.replace("    def test", "    @unittest.expectedFailure\n    def test")
                    .replace("pass\n", "self.fail('expected')\n"))
        self.module("test_unexpected", PASS.replace("    def test", "    @unittest.expectedFailure\n    def test"))
        run = self.invoke("test_zero", "test_skip", "test_expected", "test_unexpected")
        self.assertEqual(run.returncode, 1)
        report = self.summary()
        self.assertEqual([r["status"] for r in report["modules"]],
                         ["no_tests", "incomplete", "incomplete", "failed"])
        self.assertEqual(report["test_totals"]["passed"], 0)
        for key in ("skipped", "expected_failures", "unexpected_successes"):
            self.assertEqual(report["test_totals"][key], 1)

    def test_mixed_success_and_skip_is_incomplete(self):
        self.module("test_mixed", PASS + "    @unittest.skip('missing')\n    def test_skip_cuda_0(self): pass\n")
        self.assertEqual(self.invoke("test_mixed").returncode, 1)
        self.assertEqual(self.summary()["modules"][0]["status"], "incomplete")
        self.assertEqual(self.summary()["test_totals"]["passed"], 1)

    def test_printed_summary_and_zero_exit_are_not_completion(self):
        self.module("test_spoof", "import os\nprint('Ran 42 tests in 0.1s\\n\\nOK', flush=True)\nos._exit(0)\n")
        self.assertEqual(self.invoke("test_spoof").returncode, 1)
        self.assertEqual(self.summary()["modules"][0]["status"], "missing_result")
        self.assertEqual(self.summary()["test_totals"]["passed"], 0)

    def test_crash_and_nonzero_after_structured_success_are_not_passed(self):
        self.module("test_crash", "import os, signal\nos.kill(os.getpid(), signal.SIGTERM)\n")
        self.module("test_late_exit", "import atexit, os\natexit.register(lambda: os._exit(7))\n" + PASS)
        self.assertEqual(self.invoke("test_crash", "test_late_exit").returncode, 1)
        records = self.summary()["modules"]
        self.assertEqual([r["status"] for r in records], ["crashed", "failed"])
        self.assertEqual(records[0]["returncode"], -signal.SIGTERM)
        self.assertEqual(records[1]["returncode"], 7)
        self.assertEqual(records[1]["tests"]["passed"], 1)

    def test_default_discovery_and_empty_discovery(self):
        empty = self.invoke()
        self.assertEqual(empty.returncode, 2)
        self.assertIn("no test modules", empty.stderr)
        self.assertFalse(self.output.exists())
        self.module("test_b")
        self.module("test_a")
        self.assertEqual(self.invoke().returncode, 0)
        self.assertEqual(self.summary()["requested_modules"], ["test_a", "test_b"])

    def test_invalid_module_names_duplicates_and_missing_module(self):
        for modules in (("../test_bad",), ("test_a", "test_a"), ("x",)):
            with self.subTest(modules=modules):
                self.assertEqual(self.invoke(*modules).returncode, 2)
                self.assertFalse(self.output.exists())
        self.assertEqual(self.invoke("test_missing").returncode, 1)
        self.assertEqual(self.summary()["modules"][0]["status"], "failed")

    def test_invalid_timeouts_and_missing_checkout(self):
        self.module("test_good")
        for value in ("0", "-1", "nan", "inf", "bad"):
            with self.subTest(timeout=value):
                self.assertEqual(self.invoke("test_good", env=dict(self.env, CUMETAL_WARP_TIMEOUT=value)).returncode, 2)
                self.assertFalse(self.output.exists())
        run = self.invoke("test_good", env=dict(self.env, CUMETAL_WARP_DIR=str(self.root / "absent")))
        self.assertEqual(run.returncode, 2)
        self.assertFalse(self.output.exists())

    def test_existing_results_and_dangling_symlink_are_preserved(self):
        self.module("test_good")
        self.output.mkdir()
        stale = self.output / "test_good.log"
        stale.write_text("retained evidence")
        self.assertEqual(self.invoke("test_good").returncode, 2)
        self.assertEqual(stale.read_text(), "retained evidence")
        self.assertEqual(list(self.output.iterdir()), [stale])
        dangling = self.root / "dangling"
        target = self.root / "missing-target"
        dangling.symlink_to(target, target_is_directory=True)
        run = self.invoke("test_good", env=dict(self.env, CUMETAL_WARP_RESULTS=str(dangling)))
        self.assertEqual(run.returncode, 2)
        self.assertTrue(dangling.is_symlink())
        self.assertFalse(target.exists())

    def long_module(self, name):
        self.module(name, "import os, pathlib, subprocess, sys, time, unittest\n"
                    "class Case(unittest.TestCase):\n"
                    "    def test_wait_cuda_0(self):\n"
                    "        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                    "        pathlib.Path('pids.txt').write_text(str(os.getpid()) + ' ' + str(child.pid))\n"
                    "        time.sleep(60)\n")

    def assert_processes_stopped(self):
        pids = (self.warp / "pids.txt").read_text().split()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            states = [subprocess.run(["ps", "-p", pid, "-o", "stat="], capture_output=True,
                                     text=True, check=False).stdout.strip() for pid in pids]
            if all(not state or state.startswith("Z") for state in states):
                return
            time.sleep(0.02)
        self.fail("module process/child still running: " + str(states))

    def test_timeout_kills_process_group_and_continues(self):
        self.long_module("test_timeout")
        self.module("test_good")
        run = self.invoke("test_timeout", "test_good", env=dict(self.env, CUMETAL_WARP_TIMEOUT="1"))
        self.assertEqual(run.returncode, 1, run.stderr)
        self.assertEqual([r["status"] for r in self.summary()["modules"]], ["timed_out", "passed"])
        self.assert_processes_stopped()

    def test_interrupt_kills_group_and_records_unstarted_modules(self):
        self.long_module("test_wait")
        self.module("test_good")
        for signum in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signal=signum):
                self.output = self.root / ("interrupt-" + str(signum))
                pid_path = self.warp / "pids.txt"
                pid_path.unlink(missing_ok=True)
                env = dict(self.env, CUMETAL_WARP_RESULTS=str(self.output), CUMETAL_WARP_TIMEOUT="10")
                process = subprocess.Popen(["bash", str(RUNNER), "test_wait", "test_good"],
                                           cwd=self.root, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                try:
                    deadline = time.monotonic() + 5
                    while not pid_path.exists() and time.monotonic() < deadline and process.poll() is None:
                        time.sleep(0.01)
                    self.assertTrue(pid_path.exists())
                    process.send_signal(signum)
                    stdout, stderr = process.communicate(timeout=5)
                    self.assertEqual(process.returncode, 128 + signum, stdout + stderr)
                    self.assertEqual(self.summary()["status"], "interrupted")
                    self.assertEqual(self.summary()["not_run"], ["test_good"])
                    self.assert_processes_stopped()
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait()

    def test_repeated_interrupts_preserve_cleanup_and_final_report(self):
        self.long_module("test_wait")
        self.module("test_good")
        launcher = (
            "import pathlib, sys, time\n"
            f"sys.path.insert(0, {str(ROOT / 'scripts')!r})\n"
            "import run_warp_tests_cumetal as runner\n"
            f"root = pathlib.Path({str(self.root)!r})\n"
            "def checkpoint(name):\n"
            "    (root / (name + '-ready')).touch()\n"
            "    deadline = time.monotonic() + 5\n"
            "    while not (root / (name + '-release')).exists():\n"
            "        if time.monotonic() > deadline: raise RuntimeError('test checkpoint timeout')\n"
            "        time.sleep(0.005)\n"
            "original_kill, original_save = runner.kill_group, runner.save_summary\n"
            "def delayed_kill(process):\n"
            "    checkpoint('cleanup')\n"
            "    original_kill(process)\n"
            "def delayed_save(path, report):\n"
            "    if report['status'] == 'interrupted': checkpoint('report')\n"
            "    original_save(path, report)\n"
            "runner.kill_group, runner.save_summary = delayed_kill, delayed_save\n"
            "sys.exit(runner.main(['test_wait', 'test_good']))\n"
        )
        process = subprocess.Popen([sys.executable, "-c", launcher], cwd=self.root, env=self.env,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        def wait_for(path):
            deadline = time.monotonic() + 5
            while not path.exists() and time.monotonic() < deadline and process.poll() is None:
                time.sleep(0.005)
            self.assertTrue(path.exists(), str(path))

        try:
            wait_for(self.warp / "pids.txt")
            process.send_signal(signal.SIGINT)
            wait_for(self.root / "cleanup-ready")
            process.send_signal(signal.SIGTERM)
            (self.root / "cleanup-release").touch()
            wait_for(self.root / "report-ready")
            process.send_signal(signal.SIGINT)
            (self.root / "report-release").touch()
            stdout, stderr = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 128 + signal.SIGINT, stdout + stderr)
            self.assertEqual(self.summary()["status"], "interrupted")
            self.assertEqual(self.summary()["modules"][0]["status"], "interrupted")
            self.assertEqual(self.summary()["not_run"], ["test_good"])
            self.assert_processes_stopped()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            pid_path = self.warp / "pids.txt"
            if pid_path.exists():
                try:
                    os.killpg(int(pid_path.read_text().split()[0]), signal.SIGKILL)
                except ProcessLookupError:
                    pass


if __name__ == "__main__":
    unittest.main()
