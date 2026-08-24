import unittest

import numpy as np
import pandas as pd

from scripts.compare_variants import align
from scripts.summarize_experiments import stratified_bootstrap_indices


class ExperimentSummaryTests(unittest.TestCase):
    def test_site_bootstrap_preserves_complete_clusters(self):
        frame = pd.DataFrame({
            "sample_id": ["a1", "a2", "b1", "c1"],
            "codebase_id": ["a", "a", "b", "c"],
            "label": [1, 1, 0, 0],
        })
        sampled = stratified_bootstrap_indices(
            frame, np.random.default_rng(7), cluster_column="codebase_id"
        )
        sampled_ids = frame.iloc[sampled]["sample_id"].tolist()
        self.assertEqual(sampled_ids.count("a1"), sampled_ids.count("a2"))
        self.assertEqual(len(sampled_ids), 4)

    def test_site_bootstrap_rejects_conflicting_cluster_labels(self):
        frame = pd.DataFrame({
            "sample_id": ["a1", "a2", "b1"],
            "codebase_id": ["a", "a", "b"],
            "label": [0, 1, 0],
        })
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            stratified_bootstrap_indices(
                frame, np.random.default_rng(7), cluster_column="codebase_id"
            )

    def test_site_bootstrap_allows_mixed_page_labels_with_site_label(self):
        frame = pd.DataFrame({
            "sample_id": ["a1", "a2", "a3", "b1"],
            "codebase_id": ["a", "a", "a", "b"],
            "label": [0, 1, 0, 0],
            "site_label": [1, 1, 1, 0],
        })
        sampled = stratified_bootstrap_indices(
            frame, np.random.default_rng(7), cluster_column="codebase_id"
        )
        sampled_ids = frame.iloc[sampled]["sample_id"].tolist()
        self.assertEqual(sampled_ids.count("a1"), sampled_ids.count("a2"))
        self.assertEqual(sampled_ids.count("a2"), sampled_ids.count("a3"))
        self.assertEqual(len(sampled_ids), 4)

    def test_page_bootstrap_supports_missing_cluster_column(self):
        frame = pd.DataFrame({"label": [0, 0, 1, 1]})
        sampled = stratified_bootstrap_indices(
            frame, np.random.default_rng(7), cluster_column=None
        )
        labels = frame.iloc[sampled]["label"].tolist()
        self.assertEqual(labels.count(0), 2)
        self.assertEqual(labels.count(1), 2)

    def test_paired_alignment_rejects_cluster_mismatch(self):
        left = [("seed_42", 0.5, pd.DataFrame({
            "sample_id": ["x"], "label": [1], "probability": [0.8],
            "codebase_id": ["site-a"],
        }))]
        right = [("seed_42", 0.5, pd.DataFrame({
            "sample_id": ["x"], "label": [1], "probability": [0.7],
            "codebase_id": ["site-b"],
        }))]
        with self.assertRaisesRegex(ValueError, "Cluster mismatch"):
            align(left, right, cluster_column="codebase_id")


if __name__ == "__main__":
    unittest.main()
