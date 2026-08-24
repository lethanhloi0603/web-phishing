import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluate_deep import load_or_build_feature_cache
from src.preprocessing import STATIC_FEATURE_NAMES


class EvaluationCacheTests(unittest.TestCase):
    def test_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split = root / "split.csv"
            split.write_text("identity", encoding="utf-8")
            frame = pd.DataFrame({
                "html": ["<html><body><form></form></body></html>"],
                "url": ["https://example.test/login"],
            })
            checkpoint = {
                "max_dom_length": 8,
                "model_config": {"use_dom": True, "use_static": True},
            }
            first = load_or_build_feature_cache(
                frame, split, checkpoint, root / "cache"
            )
            second = load_or_build_feature_cache(
                frame, split, checkpoint, root / "cache"
            )
            self.assertEqual(first[0].shape, (1, len(STATIC_FEATURE_NAMES)))
            self.assertEqual(first[1].shape, (1, 8))
            np.testing.assert_array_equal(first[0], second[0])
            np.testing.assert_array_equal(first[1], second[1])
            self.assertEqual(len(list((root / "cache").glob("*.npz"))), 1)


if __name__ == "__main__":
    unittest.main()
