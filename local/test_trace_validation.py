"""Verify that missing or misleading device evidence cannot pass the local gate."""
from pathlib import Path
import runpy
import unittest

verify = runpy.run_path(str(Path(__file__).resolve().parents[1] / "cuda-mac"))["verify_trace"]
LINE = ('CUMETAL_PROVENANCE event=kernel_launch kernel="smoke" '
        'semantic_quality=exact device=apple_gpu device_name="Apple M1" launch_success=true')


class EvidenceGateTests(unittest.TestCase):
    def test_valid_evidence(self):
        self.assertEqual(len(verify(LINE + "\nPASS", {"smoke": 1}, "PASS")), 1)

    def test_false_passes_are_rejected(self):
        cases = {
            "no numerical result": LINE,
            "no GPU launch": "PASS",
            "CPU fallback": LINE.replace("apple_gpu", "cpu") + "\nPASS",
            "failed launch": LINE.replace("launch_success=true", "launch_success=false") + "\nPASS",
            "approximate output": LINE.replace("semantic_quality=exact", "semantic_quality=approximate") + "\nPASS",
            "extra launch": LINE + "\n" + LINE + "\nPASS",
            "wrong kernel": LINE.replace('kernel="smoke"', 'kernel="different"') + "\nPASS",
            "unknown device": LINE.replace('device_name="Apple M1"', 'device_name="unknown"') + "\nPASS",
        }
        for name, output in cases.items():
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                verify(output, {"smoke": 1}, "PASS")


if __name__ == "__main__":
    unittest.main()
