import unittest

import pandas as pd

from src.site_evaluation import aggregate_site_predictions, site_level_evaluation


class SiteEvaluationTests(unittest.TestCase):
    def test_max_aggregation_detects_site_from_one_positive_page(self):
        frame = pd.DataFrame({
            "sample_id": ["a", "b", "c"],
            "codebase_id": ["site-1", "site-1", "site-2"],
            "label": [1, 0, 0],
            "site_label": [1, 1, 0],
            "probability": [0.9, 0.1, 0.2],
        })
        sites = aggregate_site_predictions(frame)
        first = sites.set_index("codebase_id").loc["site-1"]
        self.assertEqual(first["label"], 1)
        self.assertEqual(first["n_pages"], 2)
        self.assertAlmostEqual(first["probability"], 0.9)

    def test_conflicting_site_labels_are_rejected(self):
        frame = pd.DataFrame({
            "codebase_id": ["same", "same"], "label": [0, 1],
            "site_label": [0, 1], "probability": [0.1, 0.9],
        })
        with self.assertRaises(ValueError):
            aggregate_site_predictions(frame)

    def test_site_level_report_is_calibrated(self):
        val = pd.DataFrame({
            "codebase_id": ["v0", "v1", "v2", "v3"],
            "label": [0, 0, 1, 1], "site_label": [0, 0, 1, 1],
            "probability": [0.1, 0.2, 0.8, 0.9],
        })
        test = pd.DataFrame({
            "codebase_id": ["t0", "t1", "t2", "t3"],
            "label": [0, 0, 1, 1], "site_label": [0, 0, 1, 1],
            "probability": [0.05, 0.3, 0.7, 0.95],
        })
        report, _, _ = site_level_evaluation(val, test)
        self.assertIn("target_fpr", report["calibration"])
        self.assertEqual(report["test_sites"], 4)


if __name__ == "__main__":
    unittest.main()
