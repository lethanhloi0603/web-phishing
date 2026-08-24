import unittest

import pandas as pd

from scripts.build_scenarios import (
    compose_holdout_test, compose_temporal_pool, group_split, group_split_three,
    remove_test_leakage,
)


class ScenarioTests(unittest.TestCase):
    def test_holdout_test_can_include_both_human_labels(self):
        held_out = pd.DataFrame({"html_hash": ["q"], "label": [1]})
        human = pd.DataFrame({"html_hash": ["h0", "h1"], "label": [0, 1]})
        legacy = compose_holdout_test(held_out, human, include_human_phishing=False)
        matched = compose_holdout_test(held_out, human, include_human_phishing=True)
        self.assertEqual(set(legacy["html_hash"]), {"q", "h0"})
        self.assertEqual(set(matched["html_hash"]), {"q", "h0", "h1"})

    def sample_frame(self, count=40):
        return pd.DataFrame({
            "html": [f"<html>{i}</html>" for i in range(count)],
            "html_hash": [f"h{i}" for i in range(count)],
            "template_id": [f"t{i}" for i in range(count)],
            "label": [i % 2 for i in range(count)],
            "source": ["conventional"] * count,
            "generator": ["human"] * count,
            "generator_family": ["human"] * count,
        })

    def test_group_split_has_no_template_overlap_and_both_classes(self):
        train, val = group_split(self.sample_frame(), seed=7)
        self.assertFalse(set(train["template_id"]) & set(val["template_id"]))
        self.assertEqual(train["label"].nunique(), 2)
        self.assertEqual(val["label"].nunique(), 2)

    def test_three_way_split_honors_validation_fraction(self):
        train, val, test = group_split_three(
            self.sample_frame(200), validation_size=0.5, test_size=0.15, seed=3
        )
        self.assertAlmostEqual(len(val) / 200, 0.5, delta=0.05)
        self.assertAlmostEqual(len(test) / 200, 0.15, delta=0.05)
        self.assertFalse(set(train["template_id"]) & set(val["template_id"]))

    def test_temporal_pool_excludes_llm_by_default(self):
        human_train = self.sample_frame(4)
        human_test = self.sample_frame(4).assign(
            html_hash=lambda x: "test-" + x["html_hash"],
            template_id=lambda x: "test-" + x["template_id"],
        )
        llm = self.sample_frame(2).assign(
            html_hash=lambda x: "llm-" + x["html_hash"],
            template_id=lambda x: "llm-" + x["template_id"],
            source="llm",
            generator="qwen",
            generator_family="qwen",
        )
        default_pool = compose_temporal_pool(human_train, human_test, llm)
        mixed_pool = compose_temporal_pool(
            human_train, human_test, llm, include_llm=True
        )
        self.assertNotIn("llm", set(default_pool["source"]))
        self.assertIn("llm", set(mixed_pool["source"]))

    def test_strict_leakage_filter_removes_exact_and_template_matches(self):
        train = self.sample_frame(10)
        val = self.sample_frame(10).assign(
            html_hash=lambda x: "v" + x["html_hash"],
            template_id=lambda x: "v" + x["template_id"],
        )
        test = self.sample_frame(4).copy()
        test.loc[2, "html_hash"] = "new-hash"
        test.loc[2, "template_id"] = "new-template"
        filtered, report = remove_test_leakage(test, train, val, strict_template_isolation=True)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(report["test_rows_removed"], 3)


if __name__ == "__main__":
    unittest.main()
