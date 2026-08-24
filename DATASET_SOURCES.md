# Dataset sources and experimental roles

The project keeps dataset origin separate from the binary model label. A released paper label is recorded as `published_label=1` and `label_provenance=published_dataset` (or `published_generation_task`); it is never rewritten as `manual_validated=1`.

| Domain | Dataset | Intended role | Local status |
|---|---|---|---|
| `real_world` | [PhreshPhish](https://arxiv.org/abs/2507.10854) | Conventional benign/phishing train data and realistic background negatives | Streaming subset already supported |
| `real_world` | [Phish360](https://doi.org/10.3390/app16020751) | Provenance-disjoint external HTML/URL evaluation | 10,748-page Parquet release downloaded; deterministic 1,942-page clean subset prepared |
| `llm_generated` | [Assessing Spear-Phishing Website Generation](https://arxiv.org/abs/2602.13363) | Primary LOGO/LOFO capability benchmark across coding-agent families | 413 unique pages from 137 observed successful codebases normalized locally |
| `adversarial_external` | [PhishOracle](https://github.com/LetsBeSecure/PhishOracle-Project) | External robustness test against generated adversarial phishing pages | Adapter ready; dataset not downloaded |
| `adversarial_external` | [Multimodal LLM Phishing Detection](https://github.com/jehleekr/multimodal_llm_phishing_detection) | External HTML/text/screenshot benchmark | Adapter ready; dataset not downloaded |
| `adversarial_external` | [Phishpedia](https://github.com/lindsey98/Phishpedia) | Cross-brand phishing HTML/screenshot benchmark | Sparse code repository downloaded; 30k Google Drive dataset not downloaded |
| `adversarial_external` | [ThreatAdvPhish](https://github.com/hihey54/www24_threatAdvPhish) | Visual evasion benchmark across APW-Lab/APW-Wild and legitimate pages | Downloaded locally; released experimental pages are screenshots, so do not report them as HTML-model evidence |
| `adversarial_external` | [Zenodo Phishing Website Dataset](https://zenodo.org/records/8041387) | Optional cross-brand stress test | Small phishing chunk downloaded; large benign transfer stopped to preserve disk; upstream license is `NOASSERTION` |

The two large public archives are not downloaded automatically. This avoids silently consuming tens of gigabytes and makes license/access review an explicit research step.

## Normalize the published LLM codebases

```powershell
python scripts\prepare_llm_repo.py `
  --repo external\assessingSpearPhishing `
  --brand bluemind `
  --label-unit page `
  --out data\llm_phish_published_page.csv
```

Outputs:

- `llm_phish_published_page.csv`: page-level model input.
- `llm_phish_published_page_codebases.csv`: codebase-level generation summary.
- `llm_phish_published_page_audit.csv`: optional defensive review sheet.
- `llm_phish_published_page.csv.manifest.json`: paper reference, counts and hashes.

The normalized dataset carries two label units. `page_label=1` only for credential-entry pages; auxiliary pages such as dashboards and 404 pages use `page_label=0`. `site_label=1` marks the complete codebase released by the source spear-phishing study. The model trains on page labels by default, while site-level evaluation groups pages by `codebase_id` and uses the maximum page probability. These published labels do not claim independent human confirmation of malicious intent.

## Import another published dataset

Directory layout with `phishing/` and `benign/` folders:

```powershell
python scripts\import_public_dataset.py `
  --input external\PhishOracle-Dataset `
  --format directory `
  --source phishoracle `
  --dataset-domain adversarial_external `
  --dataset-version 2025-paper-release `
  --citation "ACM DTRAP 2025, DOI or repository URL" `
  --license unknown `
  --brand-from-path `
  --out data\external\phishoracle.csv
```

CSV input with different column names:

```powershell
python scripts\import_public_dataset.py `
  --input external\dataset.csv `
  --format csv `
  --source published_dataset_name `
  --dataset-domain adversarial_external `
  --html-column html_content `
  --label-column class `
  --url-column url `
  --brand-column brand `
  --out data\external\published_dataset_name.csv
```

Always check the sidecar manifest and the source dataset license before including results in a thesis or publication.

### Storage-bounded Phish360 external test

The paper states that Phish360 is available for academic use. It does not expose
a standalone SPDX dataset license, so cite the paper, do not redistribute the
raw Parquet files, and retain the download manifest.

```powershell
python scripts\download_phish360.py `
  --class both `
  --output-dir data\external_phish360_raw `
  --min-free-gb 6

python scripts\import_public_dataset.py `
  --input data\external_phish360_raw `
  --format parquet-directory `
  --source phish360 `
  --dataset-domain real_world `
  --dataset-version crossphire-2026 `
  --label-provenance phish360_published_label `
  --citation https://doi.org/10.3390/app16020751 `
  --license academic-use-only `
  --generator human_kit `
  --html-column full_html `
  --label-column Class `
  --url-column URL `
  --site-column Domain `
  --brand-column brand `
  --max-chars 200000 `
  --max-per-class 1000 `
  --sample-seed 42 `
  --out data\external_phish360_balanced2k.csv.gz
```

The importer groups pages by domain, removes exact HTML duplicates, and drops
every domain carrying conflicting class labels. The current normalized subset
has 1,942 pages (997 benign, 945 phishing), 1,819 sites, 1,913 templates and
zero site-label conflicts. A final strict filter removes six rows sharing a DOM
template fingerprint with the primary scenario:

```powershell
python scripts\filter_external_overlap.py `
  --input data\external_phish360_balanced2k.csv.gz `
  --reference `
    data\scenario_lofo_qwen_page_matched_calibrated\train.csv.gz `
    data\scenario_lofo_qwen_page_matched_calibrated\val.csv.gz `
    data\scenario_lofo_qwen_page_matched_calibrated\test.csv.gz `
  --out data\external_phish360_calibrated_strict.csv.gz
```

The strict external test has 1,936 pages and zero HTML, template or site overlap.

Directory import also reads bounded HTML members inside ZIP archives without
extracting them to disk. Folder names `APW-Lab`, `APW-Wild` and
`Unperturbed Phishing` are recognized as positive labels when HTML is actually
present. The public ThreatAdvPhish user-study folders currently contain
screenshots rather than HTML and therefore require a separately trained visual
branch. Use `--brand-from-path` only after checking that the source layout
actually uses brand directories.

## Build the published-label LOFO benchmark

```powershell
python scripts\build_scenarios.py `
  --phresh-train data\phresh_train.csv `
  --phresh-test data\phresh_test.csv `
  --llm data\llm_phish_published_page.csv `
  --require-validated `
  --allow-published-labels `
  --scenario lofo `
  --holdout-family qwen `
  --strict-template-isolation `
  --compression gzip `
  --out-dir data\scenario_lofo_qwen_page
```

Current calibrated result: 2,169 train rows, 2,052 validation rows and 1,815 test
rows. Test contains 33 Qwen credential-positive pages, 34 Qwen
auxiliary-negative pages, and both labels for conventional pages so source is
not a perfect label shortcut. At site level, the Qwen positives represent 23
codebases and one known brand. Exact-HTML and template overlap are zero.

## Build an external-validation scenario

After normalizing Phish360, evaluate the exact five primary checkpoints while
retaining primary-validation thresholds:

```powershell
python scripts\evaluate_external_matrix.py `
  --checkpoints-root outputs\experiments\qwen_calibrated_robust5 `
  --variant robust_full `
  --seeds 42,43,44,45,46 `
  --val data\scenario_lofo_qwen_page_matched_calibrated\val.csv.gz `
  --test data\external_phish360_calibrated_strict.csv.gz `
  --output-dir outputs\external\phish360_calibrated_robust5 `
  --min-external-sites-per-class 100 `
  --min-free-gb 6
```
