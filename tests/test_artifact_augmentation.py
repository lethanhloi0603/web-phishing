import unittest

import pandas as pd
import torch

from src.data import WebPageDataset
from src.preprocessing import STATIC_FEATURE_NAMES, apply_artifact_transform
from scripts.train_deep import artifact_consistency_loss, paired_stochastic_forwards


class ArtifactAugmentationTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({
            "html": [
                '<html><!--generator--><body><form id="login" class="card">'
                '<input type="password"></form></body></html>'
            ],
            "url": ["https://example.test/login"],
            "label": [1],
            "generator": ["qwen"],
            "generator_family": ["qwen"],
            "source": ["llm"],
        })

    def test_transform_validation(self):
        with self.assertRaises(ValueError):
            apply_artifact_transform("<html></html>", "unknown")

    def test_train_augmentation_recomputes_static_features(self):
        dataset = WebPageDataset(
            self.frame(),
            use_html=False,
            use_dom=True,
            use_static=True,
            artifact_augmentation_probability=1.0,
            artifact_augmentation_modes=("comments_removed",),
            augmentation_seed=7,
        )
        first = dataset[0]
        second = dataset[0]
        self.assertEqual(first["artifact_augmentation"], "comments_removed")
        self.assertEqual(first["label"].item(), 1.0)
        torch.testing.assert_close(first["static_features"], second["static_features"])
        self.assertFalse(torch.equal(
            first["static_features"], torch.tensor(dataset.static_features[0])
        ))

    def test_validation_default_stays_original(self):
        dataset = WebPageDataset(
            self.frame(), use_html=False, use_dom=False, use_static=False
        )
        self.assertEqual(dataset[0]["artifact_augmentation"], "original")

    def test_clean_view_is_returned_for_consistency_training(self):
        dataset = WebPageDataset(
            self.frame(),
            use_html=False,
            use_dom=True,
            use_static=True,
            artifact_augmentation_probability=1.0,
            artifact_augmentation_modes=("comments_removed",),
            return_clean_view=True,
        )
        item = dataset[0]
        self.assertEqual(item["artifact_applied"].item(), 1.0)
        self.assertIn("clean_dom_ids", item)
        self.assertIn("clean_static_features", item)
        self.assertFalse(torch.equal(item["static_features"], item["clean_static_features"]))

    def test_consistency_loss_ignores_unmodified_rows(self):
        augmented = {
            "logits": torch.tensor([0.0, 2.0], requires_grad=True),
            "embedding": torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True),
        }
        clean = {
            "logits": torch.tensor([10.0, -2.0], requires_grad=True),
            "embedding": torch.tensor([[-1.0, 0.0], [1.0, 0.0]], requires_grad=True),
        }
        loss = artifact_consistency_loss(
            augmented, clean, torch.tensor([0.0, 1.0])
        )
        self.assertGreater(loss.item(), 0.0)
        loss.backward()
        self.assertEqual(float(augmented["logits"].grad[0]), 0.0)

    def test_paired_forwards_share_dropout_masks(self):
        class StochasticModel(torch.nn.Module):
            def forward(
                self, input_ids, attention_mask, dom_ids, dom_mask,
                static_features, grl_strength=1.0,
            ):
                embedding = torch.nn.functional.dropout(
                    static_features, p=0.5, training=self.training
                )
                return {
                    "embedding": embedding,
                    "logits": embedding.sum(dim=1),
                }

        model = StochasticModel().train()
        values = torch.ones(8, 32)
        batch = {
            "input_ids": torch.zeros(8, 1, dtype=torch.long),
            "attention_mask": torch.ones(8, 1, dtype=torch.long),
            "dom_ids": torch.zeros(8, 1, dtype=torch.long),
            "dom_mask": torch.ones(8, 1, dtype=torch.long),
            "static_features": values,
            "clean_input_ids": torch.zeros(8, 1, dtype=torch.long),
            "clean_attention_mask": torch.ones(8, 1, dtype=torch.long),
            "clean_dom_ids": torch.zeros(8, 1, dtype=torch.long),
            "clean_dom_mask": torch.ones(8, 1, dtype=torch.long),
            "clean_static_features": values.clone(),
        }
        augmented, clean = paired_stochastic_forwards(
            model, batch, torch.device("cpu")
        )
        torch.testing.assert_close(augmented["embedding"], clean["embedding"])

    def test_precomputed_dom_and_static_features_are_used(self):
        static = torch.arange(
            len(STATIC_FEATURE_NAMES), dtype=torch.float32
        ).numpy()[None, :]
        dom = torch.tensor([[3, 4, 0, 0]], dtype=torch.long).numpy()
        dataset = WebPageDataset(
            self.frame(), use_html=False, use_dom=True, use_static=True,
            max_dom_length=4,
            static_mean=[0.0] * len(STATIC_FEATURE_NAMES),
            static_std=[1.0] * len(STATIC_FEATURE_NAMES),
            precomputed_static_features=static,
            precomputed_dom_ids=dom,
        )
        item = dataset[0]
        torch.testing.assert_close(
            item["static_features"], torch.tensor(static[0])
        )
        torch.testing.assert_close(item["dom_ids"], torch.tensor(dom[0]))


if __name__ == "__main__":
    unittest.main()
