import importlib.util
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_zenodo_external.py"
SPEC = importlib.util.spec_from_file_location("download_zenodo_external", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DownloadZenodoExternalTests(unittest.TestCase):
    def test_label_directory(self):
        self.assertEqual(
            MODULE.label_directory("phishing_5001-5151.zip"), "phishing"
        )
        self.assertEqual(
            MODULE.label_directory("not-phishing_5001-5244.zip"), "not-phishing"
        )
        with self.assertRaises(ValueError):
            MODULE.label_directory("unknown.zip")

    def test_checksum_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.bin"
            path.write_bytes(b"abc")
            self.assertEqual(
                MODULE.checksum_file(path, "md5"),
                "900150983cd24fb0d6963f7d28e17f72",
            )

    def test_download_url_supports_current_zenodo_schema(self):
        current = {"key": "sample.zip", "links": {"self": "https://example/x"}}
        legacy = {"key": "sample.zip", "links": {"content": "https://example/y"}}
        self.assertEqual(MODULE.file_download_url(current), "https://example/x")
        self.assertEqual(MODULE.file_download_url(legacy), "https://example/y")
        with self.assertRaises(ValueError):
            MODULE.file_download_url({"key": "broken.zip", "links": {}})

    def test_default_archives_fit_two_gibibytes(self):
        self.assertEqual(len(MODULE.DEFAULT_FILES), 2)
        self.assertIn("not-phishing_5001-5244.zip", MODULE.DEFAULT_FILES)


if __name__ == "__main__":
    unittest.main()
