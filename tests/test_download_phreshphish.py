import unittest

from scripts.download_phreshphish import clean_metadata, site_identifier


class PhreshPhishDownloadTests(unittest.TestCase):
    def test_site_identifier_groups_same_host(self):
        self.assertEqual(
            site_identifier("https://login.example.test/a"),
            site_identifier("login.example.test/b"),
        )
        self.assertNotEqual(
            site_identifier("https://a.example.test"),
            site_identifier("https://b.example.test"),
        )

    def test_clean_metadata_handles_nulls(self):
        self.assertEqual(clean_metadata(None), "unknown")
        self.assertEqual(clean_metadata("null", default=""), "")


if __name__ == "__main__":
    unittest.main()
