import csv
import tempfile
import unittest
from pathlib import Path

from scripts.review_llm_audit import ReviewStore


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ReviewStoreTests(unittest.TestCase):
    def test_independent_reviewer_does_not_overwrite_consensus_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = root / "audit.csv"
            llm = root / "llm.csv"
            write_rows(audit, [{
                "sample_id": "s1", "ai_validated": "1", "manual_validated": "0",
                "review_status": "pending", "validation_reason": "",
            }])
            write_rows(llm, [{"sample_id": "s1", "html": "<form></form>"}])
            store = ReviewStore(audit, llm, reviewer_id="reviewer_a")
            store.save_decision("s1", "phishing", "credential form")
            saved = store.audit_rows[0]
            self.assertEqual(saved["reviewer_a_label"], "1")
            self.assertEqual(saved["reviewer_a_reason"], "credential form")
            self.assertTrue(saved["reviewer_a_reviewed_at"])
            self.assertEqual(saved["manual_validated"], "0")
            self.assertNotIn("reviewer_b_label", store.public_row(0))

    def test_invalid_reviewer_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = root / "audit.csv"
            llm = root / "llm.csv"
            write_rows(audit, [{"sample_id": "s1", "ai_validated": "1"}])
            write_rows(llm, [{"sample_id": "s1", "html": "<p>x</p>"}])
            with self.assertRaises(ValueError):
                ReviewStore(audit, llm, reviewer_id="../reviewer")


if __name__ == "__main__":
    unittest.main()
