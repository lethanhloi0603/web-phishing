import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_mllm_subset.py"
SPEC = importlib.util.spec_from_file_location("download_mllm_subset", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def blobs():
    return [
        {"type": "blob", "path": f"data/MMLLM_Benign/{i}.zip", "size": i * 10, "sha": str(i)}
        for i in range(1, 8)
    ]


class DownloadMllmSubsetTests(unittest.TestCase):
    def test_stable_candidates_are_repeatable_and_seeded(self):
        first = MODULE.stable_candidates(blobs(), "data/MMLLM_Benign/", 42, 1_000)
        second = MODULE.stable_candidates(
            list(reversed(blobs())), "data/MMLLM_Benign/", 42, 1_000
        )
        changed = MODULE.stable_candidates(blobs(), "data/MMLLM_Benign/", 43, 1_000)
        self.assertEqual(
            [item["path"] for item in first], [item["path"] for item in second]
        )
        self.assertNotEqual(
            [item["path"] for item in first], [item["path"] for item in changed]
        )

    def test_select_bounded_respects_budget_and_count(self):
        chosen = MODULE.select_bounded(
            blobs(), "data/MMLLM_Benign/", seed=42, count=3,
            max_archive_bytes=1_000, budget_bytes=130,
        )
        self.assertEqual(len(chosen), 3)
        self.assertLessEqual(sum(item["size"] for item in chosen), 130)

    def test_select_bounded_fails_closed_when_count_cannot_fit(self):
        with self.assertRaisesRegex(RuntimeError, "Only"):
            MODULE.select_bounded(
                blobs(), "data/MMLLM_Benign/", seed=42, count=7,
                max_archive_bytes=1_000, budget_bytes=10,
            )

    def test_safe_destination_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                MODULE.safe_destination(
                    Path(directory), "benign",
                    "data/MMLLM_Benign/../../escape.zip",
                )


if __name__ == "__main__":
    unittest.main()
