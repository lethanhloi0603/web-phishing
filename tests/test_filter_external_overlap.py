import unittest

import pandas as pd

from scripts.filter_external_overlap import filter_overlap


class ExternalOverlapFilterTests(unittest.TestCase):
    def test_strict_filter_removes_all_identity_levels(self):
        external = pd.DataFrame({
            "sample_id": ["a", "b", "c", "d"],
            "html_hash": ["h1", "h2", "h3", "h4"],
            "template_id": ["t1", "t2", "t3", "t4"],
            "codebase_id": ["s1", "s2", "s3", "s4"],
        })
        reference = pd.DataFrame({
            "html_hash": ["h1", "other", "other2"],
            "template_id": ["x", "t2", "y"],
            "codebase_id": ["z", "z2", "s3"],
        })
        clean, report = filter_overlap(external, [reference])
        self.assertEqual(clean["sample_id"].tolist(), ["d"])
        self.assertEqual(report["rows_removed"], 3)

    def test_template_overlap_can_be_reported_without_removal(self):
        external = pd.DataFrame({
            "html_hash": ["h"], "template_id": ["shared"], "codebase_id": ["s"]
        })
        reference = pd.DataFrame({
            "html_hash": ["x"], "template_id": ["shared"], "codebase_id": ["y"]
        })
        clean, report = filter_overlap(
            external, [reference], drop_template_overlap=False
        )
        self.assertEqual(len(clean), 1)
        self.assertEqual(report["template_overlap_rows"], 1)


if __name__ == "__main__":
    unittest.main()
