"""Download-free checks that dependency installation fails closed."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from audit_cosmos_dependencies import main


class DependencyAuditTest(unittest.TestCase):
    def run_audit(self, results, options=()):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            report.write_text(json.dumps({"install": [
                {"metadata": {"name": "example", "version": "1.2.3"}}
            ]}))
            response = io.BytesIO(json.dumps({"results": results}).encode())
            with patch("sys.argv", ["audit", str(report), *options]), \
                 patch("urllib.request.urlopen", return_value=response), \
                 patch("subprocess.run") as install:
                main()
                return install.call_args, json.loads(report.with_suffix(".security.json").read_text())

    def test_known_advisory_refuses_installation(self):
        with self.assertRaisesRegex(RuntimeError, "known vulnerabilities"):
            self.run_audit([{"vulns": [{"id": "GHSA-example"}]}], ["--install"])

    def test_incomplete_response_refuses_installation(self):
        with self.assertRaisesRegex(RuntimeError, "incomplete OSV response"):
            self.run_audit([], ["--install"])

    def test_exception_requires_reason(self):
        with self.assertRaisesRegex(RuntimeError, "requires a reason"):
            self.run_audit([{}], ["--allow-advisory", "GHSA-example"])

    def test_reviewed_exception_installs_exact_versions(self):
        call, report = self.run_audit([{"vulns": [{"id": "GHSA-example"}]}],
                                     ["--allow-advisory", "GHSA-example", "--reason", "reviewed", "--install"])
        self.assertEqual(call.args[0][-1], "example==1.2.3")
        self.assertIn("--no-deps", call.args[0])
        self.assertIn("--only-binary=:all:", call.args[0])
        self.assertEqual(report["reason"], "reviewed")
        self.assertEqual(report["unresolved"], [])
        self.assertEqual(report["findings"][0]["vulns"][0]["id"], "GHSA-example")


if __name__ == "__main__":
    unittest.main()
