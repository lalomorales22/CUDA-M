"""CPU-only checks for comparing retained Cosmos runs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from compare_cosmos_runs import compare_runs, load_frames, load_latents, validate_contract


def report():
    return {"status": "execution_passed", "model": "test", "revision": "revision", "prompt": "test",
            "image_sha256": "image", "seed": 1, "frames": 5, "height": 8, "width": 8, "steps": 2,
            "dtype": "torch.bfloat16", "temporal_fps": None, "export_fps": 16, "sampler_adapter": "none",
            "cpu_arithmetic_fallback": False, "runs": {"mps": {}, "cumetal-blas": {}}}


class CompareTest(unittest.TestCase):
    def test_matching_core_does_not_infer_missing_legacy_metadata(self):
        a, b = report(), report()
        b["conditioning_latents"] = "posterior_sample"
        result = validate_contract(a, b)
        self.assertIn("conditioning_latents", result["missing_optional_contract_metadata"]["baseline"])
        self.assertNotIn("conditioning_latents", result["matching_optional_fields"])

    def test_contract_mismatch(self):
        a, b = report(), report()
        b["seed"] = 2
        with self.assertRaisesRegex(ValueError, "seed"):
            validate_contract(a, b)
        b = report()
        a["conditioning_latents"], b["conditioning_latents"] = "sample", "mode"
        with self.assertRaisesRegex(ValueError, "conditioning_latents"):
            validate_contract(a, b)

    def test_missing_required_fields(self):
        a = report()
        del a["image_sha256"]
        with self.assertRaisesRegex(ValueError, "missing required"):
            validate_contract(a, report())

    def test_running_or_fallback_refused(self):
        for key, value in (("status", "running"), ("cpu_arithmetic_fallback", True), ("steps", False)):
            a = report()
            a[key] = value
            with self.assertRaises(ValueError):
                validate_contract(a, report())

    def test_frame_shape_finite_rgb(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frames.npy"
            good = np.zeros((5, 8, 8, 3), dtype=np.float32)
            for array in (good[0], good[..., :2], good + np.nan, good - 1, good + 256):
                np.save(path, array)
                with self.assertRaises(ValueError):
                    load_frames(path, report())
            np.save(path, good)
            self.assertEqual(load_frames(path, report()).shape, good.shape)

    def test_malformed_latent_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "latents.npz"
            good = np.ones((1, 2, 2, 1, 1), dtype=np.float32)
            for values in ({"step_0": good}, {"step_0": good, "step_2": good},
                           {"step_0": good, "step_1": good + np.nan},
                           {"step_0": good, "step_1": good.astype(np.int32)},
                           {"step_0": good, "step_1": good[..., :0]},
                           {"step_0": good, "step_1": np.ones((2,), dtype=np.float32)}):
                np.savez(path, **values)
                with self.assertRaises(ValueError):
                    load_latents(path, 2)

    def fixtures(self, root, candidate_shape=(1, 2, 2, 1, 1)):
        directories = (root / "baseline", root / "candidate")
        for index, (directory, backend) in enumerate(zip(directories, ("mps", "cumetal-blas"), strict=True)):
            directory.mkdir()
            (directory / "comparison.json").write_text(json.dumps(report()))
            np.save(directory / f"{backend}-frames.npy", np.full((5, 8, 8, 3), (1 - index) * 255, dtype=np.uint8))
            shape = candidate_shape if index else (1, 2, 2, 1, 1)
            np.savez(directory / f"{backend}-latents.npz", **{
                f"step_{i}": np.full(shape, i + index, dtype=np.float32) for i in range(2)})
        return directories

    def test_comparison_unsigned_arithmetic_and_no_parity_claim(self):
        with tempfile.TemporaryDirectory() as directory, patch("compare_cosmos_runs.shutil.which", return_value=None):
            root = Path(directory)
            a, b = self.fixtures(root)
            result = compare_runs(a, b, "mps", "cumetal-blas", root / "out")
            self.assertEqual(result["all_frames_rgb_difference_0_255"]["mean_abs"], 255)
            self.assertEqual(result["per_frame_rgb_difference_0_255"][4]["max_abs"], 255)
            self.assertEqual(result["latent_step_differences"][1]["max_abs"], 1)
            self.assertIn("not assessed", result["numerical_parity"])
            self.assertEqual(result["human_assessment"], "pending visual inspection")
            self.assertTrue((root / "out/comparison.png").is_file())
            with self.assertRaisesRegex(ValueError, "already exists"):
                compare_runs(a, b, "mps", "cumetal-blas", root / "out")

    def test_mismatching_actual_latent_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = self.fixtures(root, (1, 4, 2, 1, 1))
            with self.assertRaisesRegex(ValueError, "latent shapes differ"):
                compare_runs(a, b, "mps", "cumetal-blas", root / "out")

    def test_preexisting_empty_output_directory_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = self.fixtures(root)
            output = root / "out"
            output.mkdir()
            with self.assertRaisesRegex(ValueError, "output directory already exists"):
                compare_runs(a, b, "mps", "cumetal-blas", output)
            self.assertEqual(list(output.iterdir()), [])

    def test_preexisting_png_only_directory_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = self.fixtures(root)
            output = root / "out"
            output.mkdir()
            image = output / "comparison.png"
            image.write_bytes(b"existing evidence")
            with self.assertRaisesRegex(ValueError, "output directory already exists"):
                compare_runs(a, b, "mps", "cumetal-blas", output)
            self.assertEqual(image.read_bytes(), b"existing evidence")
            self.assertFalse((output / "comparison.json").exists())

    def test_output_directory_race_preserves_competing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = self.fixtures(root)
            output = root / "out"

            def competing_creator(_path):
                if not output.exists():
                    output.mkdir()
                    (output / "comparison.png").write_bytes(b"competing evidence")
                return {"inspected": False}

            with patch("compare_cosmos_runs.codec_metadata", side_effect=competing_creator), \
                    patch("compare_cosmos_runs.make_sheet") as make_sheet:
                with self.assertRaises(FileExistsError):
                    compare_runs(a, b, "mps", "cumetal-blas", output)
                make_sheet.assert_not_called()
            self.assertEqual((output / "comparison.png").read_bytes(), b"competing evidence")
            self.assertFalse((output / "comparison.json").exists())

    def test_missing_selected_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = self.fixtures(root)
            with self.assertRaisesRegex(ValueError, "selected backend"):
                compare_runs(a, b, "mps", "cumetal", root / "out")


if __name__ == "__main__":
    unittest.main()
