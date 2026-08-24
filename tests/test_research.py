import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.research import (
    evaluation_data_evidence, sha256_file, source_snapshot, split_integrity,
)


class ResearchIntegrityTests(unittest.TestCase):
    def test_source_snapshot_changes_with_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            code = root / "src" / "model.py"
            code.write_text("value = 1\n", encoding="utf-8")
            first = source_snapshot(root)
            second = source_snapshot(root)
            self.assertEqual(first["sha256"], second["sha256"])
            self.assertEqual(first["file_count"], 1)
            code.write_text("value = 2\n", encoding="utf-8")
            self.assertNotEqual(first["sha256"], source_snapshot(root)["sha256"])
    def frame(self, hashes, templates):
        return pd.DataFrame({
            "html_hash": hashes,
            "template_id": templates,
            "label": [index % 2 for index in range(len(hashes))],
            "generator": ["human"] * len(hashes),
        })

    def test_split_integrity_detects_exact_and_template_overlap(self):
        train = self.frame(["a", "b"], ["t1", "t2"])
        val = self.frame(["c", "d"], ["t3", "t4"])
        test = self.frame(["a", "e"], ["t5", "t2"])
        report = split_integrity(train, val, test)
        self.assertEqual(report["overlap"]["train_test"]["html_hash"], 1)
        self.assertEqual(report["overlap"]["train_test"]["template_id"], 1)

    def test_sha256_file_is_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.txt"
            path.write_text("research artifact", encoding="utf-8")
            self.assertEqual(sha256_file(path), sha256_file(path))
            self.assertEqual(len(sha256_file(path)), 64)

    def test_evaluation_evidence_binds_counts_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.csv"
            frame = pd.DataFrame({
                "label": [0, 1, 1],
                "dataset_provenance": ["external"] * 3,
                "brand": ["unknown", "apple", "microsoft"],
            })
            frame.to_csv(path, index=False)
            evidence = evaluation_data_evidence(frame, path)
            self.assertEqual(evidence["rows"], 3)
            self.assertEqual(evidence["phishing"], 2)
            self.assertEqual(evidence["known_brands"], 2)
            self.assertEqual(evidence["sha256"], sha256_file(path))

    def test_evaluation_evidence_uses_site_label_for_multipage_sites(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.csv"
            path.write_text("evidence", encoding="utf-8")
            frame = pd.DataFrame({
                "label": [1, 0, 0],
                "site_label": [1, 1, 0],
                "codebase_id": ["phish-site", "phish-site", "safe-site"],
            })
            evidence = evaluation_data_evidence(frame, path)
            self.assertEqual(evidence["site_label_conflicts"], 0)
            self.assertEqual(evidence["phishing_sites"], 1)
            self.assertEqual(evidence["benign_sites"], 1)


if __name__ == "__main__":
    unittest.main()
