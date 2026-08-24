import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.download_phish360 import FILES, check_storage, selected_files


class Phish360DownloadTests(unittest.TestCase):
    def test_selection(self):
        self.assertEqual(set(selected_files("both")), {"phish", "legit"})
        self.assertEqual(list(selected_files("phish")), ["phish"])
        self.assertGreater(FILES["legit"]["bytes"], FILES["phish"]["bytes"])

    def test_storage_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            usage = shutil_usage = type("Usage", (), {"free": 100})()
            with patch("scripts.download_phish360.shutil.disk_usage", return_value=usage):
                with self.assertRaises(RuntimeError):
                    check_storage(Path(directory), 50, 75)


if __name__ == "__main__":
    unittest.main()
