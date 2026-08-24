import unittest

import pandas as pd

from src.shortcut_audit import shortcut_report


class ShortcutAuditTests(unittest.TestCase):
    def test_detects_missing_2x2_cells_and_brand_shortcut(self):
        frame = pd.DataFrame({
            "label": [0] * 20 + [1] * 20,
            "source": ["conventional"] * 20 + ["llm"] * 20,
            "brand": ["unknown"] * 20 + ["bluemind"] * 20,
        })
        report = shortcut_report(frame, min_group_size=10, max_group_purity=0.95)
        self.assertIn("human/label=1", report["missing_source_label_cells"])
        self.assertIn("llm/label=0", report["missing_source_label_cells"])
        self.assertIn("brand", report["high_purity_groups"])

    def test_balanced_2x2_has_no_missing_cells(self):
        frame = pd.DataFrame({
            "label": [0, 1, 0, 1] * 10,
            "source": ["human", "human", "llm", "llm"] * 10,
            "brand": ["same"] * 40,
        })
        report = shortcut_report(frame, min_group_size=10, max_group_purity=0.95)
        self.assertEqual(report["missing_source_label_cells"], [])
        self.assertNotIn("brand", report["high_purity_groups"])


if __name__ == "__main__":
    unittest.main()
