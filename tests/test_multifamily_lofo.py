import unittest

import pandas as pd

from scripts.build_multifamily_lofo import parse_families
from scripts.summarize_multifamily_lofo import (
    metrics_from_decisions,
    select_heldout,
)


class MultiFamilyLofoTests(unittest.TestCase):
    def test_family_parser_normalizes_and_deduplicates(self):
        self.assertEqual(
            parse_families("Qwen, gpt,qwen, Claude"),
            ["qwen", "gpt", "claude"],
        )

    def test_heldout_selection_excludes_human_background(self):
        frame = pd.DataFrame({
            "sample_id": ["q1", "q0", "h1"],
            "generator_family": ["qwen", "qwen", "human"],
            "dataset_domain": ["llm_generated", "llm_generated", "real_world"],
            "label": [1, 0, 1],
            "probability": [0.8, 0.2, 0.9],
        })
        selected = select_heldout(frame, "qwen")
        self.assertEqual(selected["sample_id"].tolist(), ["q1", "q0"])

    def test_micro_metrics_respect_fold_specific_decisions(self):
        result = metrics_from_decisions(
            labels=[1, 0, 1, 0],
            probability=[0.6, 0.4, 0.4, 0.3],
            prediction=[1, 0, 1, 0],
        )
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["fpr"], 0.0)
        self.assertEqual(result["tp"], 2)


if __name__ == "__main__":
    unittest.main()
