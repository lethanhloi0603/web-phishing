import unittest

import numpy as np
import torch

from src.calibration import calibrated_evaluation, select_threshold
from src.metrics import clopper_pearson_upper
from src.losses import generator_aware_supcon_loss, phishing_domain_alignment_loss


class CalibrationAndLossTests(unittest.TestCase):
    def test_threshold_uses_validation_labels(self):
        threshold, metrics = select_threshold(
            [0, 0, 1, 1], [0.05, 0.2, 0.8, 0.95], mode="target_fpr", target_fpr=0.0
        )
        self.assertGreater(threshold, 0.2)
        self.assertEqual(metrics["fpr"], 0.0)

    def test_threshold_rejects_single_class_validation(self):
        with self.assertRaises(ValueError):
            select_threshold([1, 1], [0.8, 0.9])

    def test_exact_upper_bound_can_certify_zero_of_299(self):
        self.assertLessEqual(clopper_pearson_upper(0, 299, 0.95), 0.01)
        self.assertGreater(clopper_pearson_upper(0, 298, 0.95), 0.01)

    def test_confident_threshold_reports_whether_target_is_certified(self):
        labels = np.asarray([0] * 299 + [1])
        probability = np.asarray([0.1] * 299 + [0.9])
        threshold, metrics = select_threshold(
            labels,
            probability,
            mode="target_fpr_confident",
            target_fpr=0.01,
            fpr_confidence=0.95,
        )
        self.assertGreater(threshold, 0.1)
        self.assertTrue(metrics["target_fpr_certified"])
        self.assertLessEqual(metrics["fpr_upper_bound"], 0.01)

        report = calibrated_evaluation(
            [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9],
            [0, 1], [0.2, 0.8], target_fpr=0.01,
        )
        self.assertFalse(
            report["target_fpr_confident"]["validation"]["target_fpr_certified"]
        )

    def test_confident_threshold_maximizes_recall_within_certified_set(self):
        labels = np.asarray([0] * 1066 + [1] * 20)
        probability = np.asarray(
            [0.8] * 5 + [0.1] * 1061 + [0.9] * 10 + [0.7] * 10
        )
        threshold, metrics = select_threshold(
            labels,
            probability,
            mode="target_fpr_confident",
            target_fpr=0.01,
            fpr_confidence=0.95,
        )
        self.assertEqual(metrics["fp"], 5)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertLessEqual(metrics["fpr_upper_bound"], 0.01)
        self.assertLessEqual(threshold, 0.7)

    def test_losses_are_finite_on_small_batch(self):
        embeddings = torch.randn(6, 8, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1, 1, 1])
        generators = torch.tensor([0, 0, 1, 2, 1, 2])
        templates = torch.tensor([1, 2, 3, 4, 5, 6])
        domains = torch.tensor([0, 0, 0, 0, 1, 1])
        supcon = generator_aware_supcon_loss(embeddings, labels, generators, templates)
        alignment = phishing_domain_alignment_loss(embeddings, labels, domains)
        self.assertTrue(torch.isfinite(supcon))
        self.assertTrue(torch.isfinite(alignment))
        (supcon + alignment).backward()
        self.assertIsNotNone(embeddings.grad)


if __name__ == "__main__":
    unittest.main()
