import unittest

from scripts.download_phreshphish_subset import merge_manifest_entries


class PhreshPhishSubsetDownloadTests(unittest.TestCase):
    def test_legacy_manifest_is_upgraded_and_accumulated(self):
        existing = {
            "local_file": "test-000.parquet",
            "repository_path": "data/test-000.parquet",
            "sha256": "old",
        }
        entry = {
            "local_file": "test-001.parquet",
            "repository_path": "data/test-001.parquet",
            "sha256": "new",
        }
        files = merge_manifest_entries(existing, entry)
        self.assertEqual(
            [item["local_file"] for item in files],
            ["test-000.parquet", "test-001.parquet"],
        )

    def test_existing_entry_is_replaced(self):
        existing = {"files": [{"local_file": "test-001.parquet", "sha256": "old"}]}
        entry = {"local_file": "test-001.parquet", "sha256": "new"}
        self.assertEqual(merge_manifest_entries(existing, entry), [entry])


if __name__ == "__main__":
    unittest.main()
