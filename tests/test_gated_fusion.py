import unittest

import torch

from src.model import PhishGenModel


class GatedFusionTests(unittest.TestCase):
    def test_dom_static_gates_sum_to_one(self):
        model = PhishGenModel(
            use_html=False,
            use_dom=True,
            use_static=True,
            max_dom_length=8,
            dom_hidden_size=16,
            dom_layers=1,
            dom_heads=4,
            static_feature_size=5,
            static_hidden_size=8,
            fusion_hidden_size=12,
            projection_size=6,
            fusion_type="gated",
            modality_dropout=0.0,
        )
        model.eval()
        output = model(
            torch.zeros((3, 1), dtype=torch.long),
            torch.ones((3, 1), dtype=torch.long),
            torch.randint(1, 20, (3, 8)),
            torch.ones((3, 8), dtype=torch.long),
            torch.randn(3, 5),
        )
        self.assertEqual(tuple(output["logits"].shape), (3,))
        self.assertEqual(tuple(output["branch_weights"].shape), (3, 2))
        torch.testing.assert_close(
            output["branch_weights"].sum(dim=1), torch.ones(3)
        )

    def test_invalid_fusion_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            PhishGenModel(use_html=False, fusion_type="unknown")
        with self.assertRaises(ValueError):
            PhishGenModel(use_html=False, modality_dropout=1.0)


if __name__ == "__main__":
    unittest.main()
