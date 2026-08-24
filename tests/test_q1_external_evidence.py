import unittest

from scripts.q1_readiness import (
    group_recall,
    primary_training_identity_issues,
    training_configuration_signature,
    validate_external_report,
)


class ExternalEvidenceTests(unittest.TestCase):
    def report(self, provenance="phishoracle", role="external"):
        return {
            "evaluation_role": role,
            "evaluation_data": {
                "sha256": "external-hash",
                "phishing": 150,
                "benign": 150,
                "phishing_sites": 150,
                "benign_sites": 150,
                "site_label_conflicts": 0,
                "dataset_provenance": [provenance],
                "dataset_license": ["CC-BY-4.0"],
                "dataset_citation": ["https://doi.org/10.example/dataset"],
            },
            "overall": {"auprc": 0.9},
            "checkpoint_sha256": "primary-checkpoint",
        }

    def test_independent_external_report_passes_evidence_validation(self):
        issues = validate_external_report(
            "metrics.json", self.report(), {"phreshphish"}, "primary-hash"
        )
        self.assertEqual(issues, [])

    def test_primary_result_cannot_masquerade_as_external(self):
        issues = validate_external_report(
            "metrics.json",
            self.report(provenance="phreshphish", role="primary"),
            {"phreshphish"},
            "primary-hash",
        )
        self.assertTrue(any("evaluation_role" in issue for issue in issues))
        self.assertTrue(any("overlaps" in issue for issue in issues))

    def test_small_external_split_is_rejected(self):
        report = self.report()
        report["evaluation_data"]["phishing"] = 10
        issues = validate_external_report(
            "metrics.json", report, {"phreshphish"}, "primary-hash"
        )
        self.assertTrue(any("phishing rows" in issue for issue in issues))

    def test_mixing_primary_negatives_into_external_data_is_rejected(self):
        report = self.report()
        report["evaluation_data"]["dataset_provenance"] = [
            "phishoracle", "phreshphish"
        ]
        issues = validate_external_report(
            "metrics.json", report, {"phreshphish"}, "primary-hash"
        )
        self.assertTrue(any("overlaps" in issue for issue in issues))

    def test_repeated_pages_do_not_masquerade_as_independent_sites(self):
        report = self.report()
        report["evaluation_data"]["phishing_sites"] = 4
        issues = validate_external_report(
            "metrics.json", report, {"phreshphish"}, "primary-hash"
        )
        self.assertTrue(any("phishing sites" in issue for issue in issues))

    def test_external_result_must_use_primary_checkpoint(self):
        issues = validate_external_report(
            "metrics.json",
            self.report(),
            {"phreshphish"},
            "primary-hash",
            primary_checkpoint_hashes={"different-checkpoint"},
        )
        self.assertTrue(any("primary checkpoint" in issue for issue in issues))

    def test_unknown_external_license_is_rejected(self):
        report = self.report()
        report["evaluation_data"]["dataset_license"] = ["NOASSERTION"]
        issues = validate_external_report(
            "metrics.json", report, {"phreshphish"}, "primary-hash"
        )
        self.assertTrue(any("license" in issue for issue in issues))

    def test_heldout_group_recall_supports_binary_and_single_class_groups(self):
        binary = {
            "groups_target_fpr_confident": {
                "generator_family": {"qwen": {"recall": 0.75}}
            }
        }
        single_class = {
            "groups_target_fpr_confident": {
                "generator_family": {"qwen": {"class_recall": 0.625}}
            }
        }
        self.assertEqual(group_recall(binary, "qwen"), 0.75)
        self.assertEqual(group_recall(single_class, "qwen"), 0.625)
        self.assertIsNone(group_recall(binary, "missing"))

    def test_primary_checkpoint_must_bind_exact_training_splits(self):
        report = {
            "checkpoint_schema_version": 3,
            "training_data_sha256": {"train": "train-hash", "val": "val-hash"},
            "source_snapshot_sha256": "source-hash",
        }
        expected = {"train": "train-hash", "val": "val-hash", "test": "test-hash"}
        self.assertEqual(primary_training_identity_issues(report, expected), [])
        report["training_data_sha256"]["val"] = "wrong"
        self.assertTrue(any(
            "val SHA-256" in issue
            for issue in primary_training_identity_issues(report, expected)
        ))

    def test_training_signature_ignores_only_run_identity_fields(self):
        first = {"training_args": {
            "seed": 42, "output": "a.pt", "train": "a.csv", "epochs": 5,
            "lambda_supcon": 0.05,
        }}
        second = {"training_args": {
            "seed": 43, "output": "b.pt", "train": "b.csv", "epochs": 5,
            "lambda_supcon": 0.05,
        }}
        self.assertEqual(
            training_configuration_signature(first),
            training_configuration_signature(second),
        )
        second["training_args"]["lambda_supcon"] = 0.1
        self.assertNotEqual(
            training_configuration_signature(first),
            training_configuration_signature(second),
        )


if __name__ == "__main__":
    unittest.main()
