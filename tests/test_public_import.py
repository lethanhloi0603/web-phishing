import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import zipfile

import pandas as pd

import scripts.import_public_dataset as public_import
from scripts.import_public_dataset import (
    deterministic_class_cap, import_csv, import_directory, import_parquet,
    import_parquet_directory,
    infer_path_brand, normalized_brand, normalized_label, select_one_member_per_site,
    url_site_identifier,
)
from scripts.build_scenarios import filter_validated_llm
from scripts.prepare_llm_repo import infer_codebase, infer_generator


class PublicDatasetImportTests(unittest.TestCase):
    def test_normalized_label(self):
        self.assertEqual(normalized_label("phishing"), 1)
        self.assertEqual(normalized_label("benign"), 0)
        with self.assertRaises(ValueError):
            normalized_label("uncertain")

    def test_generic_or_missing_brand_is_unknown(self):
        self.assertEqual(normalized_brand("phish"), "unknown")
        self.assertEqual(normalized_brand(pd.NA), "unknown")
        self.assertEqual(normalized_brand("Microsoft"), "Microsoft")

    def test_directory_import_infers_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "phishing").mkdir()
            (root / "benign").mkdir()
            (root / "phishing" / "a.html").write_text("<html><form><input type='password'></form></html>" * 3, encoding="utf-8")
            (root / "benign" / "b.html").write_text("<html><main>official home page</main></html>" * 3, encoding="utf-8")
            args = argparse.Namespace(input=str(root), min_chars=20, max_chars=10000, source="published", generator="human_kit")
            frame = import_directory(args)
            self.assertEqual(set(frame["label"]), {0, 1})
            self.assertEqual(set(frame["dataset_provenance"]), {"published"})
            self.assertEqual(set(frame["dataset_domain"]), {"adversarial_external"})
            self.assertEqual(set(frame["manual_validated"]), {0})
            self.assertEqual(set(frame["published_label"]), {1})

    def test_csv_import_maps_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.csv"
            pd.DataFrame({"page": ["<html>safe content</html>" * 5], "class": ["benign"]}).to_csv(path, index=False)
            args = argparse.Namespace(
                input=str(path), html_column="page", label_column="class", url_column="",
                brand_column="", date_column="", min_chars=20, max_chars=10000,
                source="published", generator="human_kit",
            )
            frame = import_csv(args)
            self.assertEqual(frame.iloc[0]["label"], 0)

    def test_parquet_import_groups_pages_by_url_host(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.parquet"
            pd.DataFrame({
                "full_html": ["<html>one</html>" * 10, "<html>two</html>" * 10],
                "class": ["phish", "phish"],
                "URL": ["https://login.example.test/a", "https://login.example.test/b"],
            }).to_parquet(path, index=False)
            args = argparse.Namespace(
                input=str(path), html_column="full_html", label_column="class",
                url_column="URL", site_column="", brand_column="", date_column="",
                min_chars=20, max_chars=10000, source="phreshphish",
                generator="human_kit",
            )
            frame = import_parquet(args)
            self.assertEqual(len(frame), 2)
            self.assertEqual(frame["codebase_id"].nunique(), 1)
            self.assertEqual(
                frame.iloc[0]["codebase_id"],
                url_site_identifier("phreshphish", "login.example.test/x"),
            )

    def test_parquet_read_is_class_capped_and_clipped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.parquet"
            pd.DataFrame({
                "html": [f"<html>{i}</html>" * 20 for i in range(12)],
                "label": ["benign"] * 6 + ["phish"] * 6,
                "url": [f"https://site-{i}.test" for i in range(12)],
            }).to_parquet(path, index=False)
            args = argparse.Namespace(
                input=str(path), html_column="html", label_column="label",
                url_column="url", site_column="", brand_column="",
                date_column="", min_chars=20, max_chars=80,
                max_per_class=2, sample_seed=5, parquet_batch_size=2,
                source="external", generator="human_kit",
            )
            frame = import_parquet(args)
            self.assertEqual(frame.groupby("label").size().to_dict(), {0: 2, 1: 2})
            self.assertLessEqual(frame["html"].str.len().max(), 80)

    def test_missing_explicit_site_falls_back_to_url_host(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.parquet"
            pd.DataFrame({
                "full_html": ["<html>one</html>" * 10, "<html>two</html>" * 10],
                "class": ["phish", "phish"],
                "URL": ["https://a.test/one", "https://b.test/two"],
                "Domain": [None, None],
            }).to_parquet(path, index=False)
            args = argparse.Namespace(
                input=str(path), html_column="full_html", label_column="class",
                url_column="URL", site_column="Domain", brand_column="",
                date_column="", min_chars=20, max_chars=10000,
                source="external", generator="human_kit",
            )
            frame = import_parquet(args)
            self.assertEqual(frame["codebase_id"].nunique(), 2)

    def test_deterministic_class_cap_is_order_independent(self):
        frame = pd.DataFrame({
            "label": [0] * 5 + [1] * 5,
            "html_hash": [f"hash-{index}" for index in range(10)],
        })
        first = deterministic_class_cap(frame, 2, 7)
        second = deterministic_class_cap(frame.sample(frac=1, random_state=9), 2, 7)
        self.assertEqual(first.groupby("label").size().to_dict(), {0: 2, 1: 2})
        self.assertEqual(set(first["html_hash"]), set(second["html_hash"]))

    def test_parquet_directory_combines_and_caps_classes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for label in ("benign", "phish"):
                pd.DataFrame({
                    "full_html": [f"<html>{label}-{i}</html>" * 10 for i in range(4)],
                    "class": [label] * 4,
                    "URL": [f"https://{label}-{i}.test/login" for i in range(4)],
                }).to_parquet(root / f"{label}.parquet", index=False)
            args = argparse.Namespace(
                input=str(root), html_column="full_html", label_column="class",
                url_column="URL", site_column="", brand_column="", date_column="",
                min_chars=20, max_chars=10000, max_per_class=2, sample_seed=11,
                source="external", generator="human_kit",
            )
            frame = import_parquet_directory(args)
            self.assertEqual(frame.groupby("label").size().to_dict(), {0: 2, 1: 2})
            self.assertTrue(frame["file_path"].str.contains(".parquet:row:", regex=False).all())

    def test_main_drops_mixed_label_sites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "mixed.csv"
            output = root / "normalized.csv.gz"
            pd.DataFrame({
                "html": ["<html>safe</html>" * 10, "<html>phish</html>" * 10],
                "label": ["benign", "phish"],
                "url": ["https://same.test/a", "https://same.test/b"],
            }).to_csv(source, index=False)
            argv = [
                "import_public_dataset.py", "--input", str(source), "--format", "csv",
                "--source", "mixed", "--out", str(output),
            ]
            with patch("sys.argv", argv):
                with self.assertRaises(RuntimeError):
                    public_import.main()

    def test_directory_import_can_infer_brand_from_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            page = root / "phishing" / "microsoft" / "login.html"
            page.parent.mkdir(parents=True)
            page.write_text("<html><form><input type='password'></form></html>" * 3, encoding="utf-8")
            args = argparse.Namespace(
                input=str(root), min_chars=20, max_chars=10000,
                source="published", generator="human_kit", brand_from_path=True,
            )
            frame = import_directory(args)
            self.assertEqual(frame.iloc[0]["brand"], "microsoft")

    def test_brand_inference_ignores_layout_and_label_folders(self):
        self.assertEqual(
            infer_path_brand(Path("dataset/phishing/apple/pages/index.html")),
            "apple",
        )

    def test_zip_directory_import_reads_html_without_extracting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "phishing" / "amazon.zip"
            archive_path.parent.mkdir(parents=True)
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(
                    "site/login.html",
                    "<html><form><input type='password'></form></html>" * 3,
                )
                archive.writestr("site/screenshot.png", b"not-an-image-for-this-test")
            args = argparse.Namespace(
                input=str(root), min_chars=20, max_chars=10000,
                source="multimodal_external", generator="human_kit",
                brand_from_path=True,
            )
            frame = import_directory(args)
            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.iloc[0]["label"], 1)
            self.assertEqual(frame.iloc[0]["brand"], "amazon")
            self.assertIn("amazon.zip!", frame.iloc[0]["file_path"])
            self.assertEqual(frame.iloc[0]["codebase_id"], frame.iloc[0]["codebase_id"])
            self.assertEqual(len(frame.iloc[0]["codebase_id"]), 24)

    def test_zip_pages_from_same_site_share_codebase(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "phishing" / "batch.zip"
            archive_path.parent.mkdir(parents=True)
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("site-a/index.html", "<html><body>A</body></html>" * 5)
                archive.writestr("site-a/login.html", "<html><body>B</body></html>" * 5)
                archive.writestr("site-b/index.html", "<html><body>C</body></html>" * 5)
            args = argparse.Namespace(
                input=str(root), min_chars=20, max_chars=10000,
                source="published", generator="human_kit",
            )
            frame = import_directory(args)
            self.assertEqual(len(frame), 3)
            self.assertEqual(frame["codebase_id"].nunique(), 2)

    def test_one_page_per_site_prefers_original_over_asset_html(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "phishing" / "batch.zip"
            archive_path.parent.mkdir(parents=True)
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("site/assets/noise.html", "<html>noise</html>" * 10)
                archive.writestr("site/clean.html", "<html>clean</html>" * 10)
                archive.writestr("site/original.html", "<html>original</html>" * 10)
            args = argparse.Namespace(
                input=str(root), min_chars=20, max_chars=10000,
                source="published", generator="human_kit",
                one_page_per_site=True,
                preferred_page_names="original.html,index.html,clean.html",
            )
            frame = import_directory(args)
            self.assertEqual(len(frame), 1)
            self.assertIn("original", frame.iloc[0]["html"])

    def test_large_member_is_clipped_instead_of_using_max_chars_as_size_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "phishing" / "batch.zip"
            archive_path.parent.mkdir(parents=True)
            page = "<html><body>large page</body></html>" * 20_000
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("site/original.html", page)
            args = argparse.Namespace(
                input=str(root), min_chars=20, max_chars=1_000,
                max_member_bytes=5_000_000,
                source="published", generator="human_kit",
                one_page_per_site=True,
                preferred_page_names="original.html,index.html,clean.html",
            )
            frame = import_directory(args)
            self.assertEqual(len(frame), 1)
            self.assertEqual(len(frame.iloc[0]["html"]), 1_000)

    def test_threatadv_folder_names_are_positive(self):
        from scripts.import_public_dataset import infer_path_label

        self.assertEqual(infer_path_label(Path("APW-Lab/apple/index.html")), 1)
        self.assertEqual(infer_path_label(Path("APW-Wild/paypal/index.html")), 1)

    def test_codebase_group_ends_at_prompt(self):
        codebase, codebase_id, prompt = infer_codebase(Path("GPT/5-Codex/prompt3/public/login.html"))
        self.assertEqual(codebase, "GPT/5-Codex/prompt3")
        self.assertEqual(prompt, "prompt3")
        self.assertEqual(len(codebase_id), 24)

    def test_generator_keeps_model_variant_and_family(self):
        generator, family = infer_generator(Path("codebases/GPT/5-Codex/prompt3/public/login.html"))
        self.assertEqual(generator, "gpt_5-codex")
        self.assertEqual(family, "gpt")

    def test_published_labels_are_opt_in_and_not_manual(self):
        frame = pd.DataFrame({
            "manual_validated": [0, 0],
            "published_label": [1, 0],
            "sample_id": ["a", "b"],
        })
        selected = filter_validated_llm(frame, allow_published_labels=True)
        self.assertEqual(selected["sample_id"].tolist(), ["a"])
        self.assertEqual(int(selected.iloc[0]["manual_validated"]), 0)


if __name__ == "__main__":
    unittest.main()
