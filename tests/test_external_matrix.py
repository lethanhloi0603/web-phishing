import importlib.util
from pathlib import Path
import tempfile
import unittest

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_external_matrix.py"
SPEC = importlib.util.spec_from_file_location("evaluate_external_matrix", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ExternalMatrixTests(unittest.TestCase):
    def test_evaluation_command_declares_external_role(self):
        command = MODULE.evaluation_command(
            "model.pt", "val.csv", "external.csv", "metrics.json", "pred.csv",
            32, 0.01, 0.95,
        )
        index = command.index("--evaluation-role")
        self.assertEqual(command[index + 1], "external")

    def test_external_frame_requires_independent_sites(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "external.csv"
            frame = pd.DataFrame({
                "label": [0, 0, 1, 1],
                "codebase_id": ["b", "b", "p", "p"],
                "dataset_provenance": ["external"] * 4,
            })
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "sites"):
                MODULE.validate_external_frame(frame, path, min_sites=2)


if __name__ == "__main__":
    unittest.main()
