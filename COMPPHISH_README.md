# CompPhish leakage-aware benchmark

Copyright (c) 2026 Le Thanh Loi. All rights reserved.

This repository snapshot contains the code and compact research evidence for a
leakage-aware phishing-website benchmark on CompPhish v4.

## Final protocol

- Integrity-clean, registered-domain-disjoint Train/Test split.
- 13,203 Train rows and 2,054 Test rows.
- 70 published variables: 47 Brand–URL and 23 HTML/content variables.
- `url_more_than_mean` and `domain_more_than_mean` are recomputed from phishing
  rows inside each OOF training fold and from all Train rows for the final fit.
- Twelve base learners are evaluated independently on Brand–URL, HTML and Full
  views (36 base configurations).
- Boosting and heterogeneous OOF stacking are built independently inside every
  feature view (six stacking configurations).
- Five-fold `StratifiedGroupKFold` inside Train, grouped by registered domain.
- Fixed classification threshold 0.5; Test is used for final evaluation only.
- Full CatBoost is the pre-specified single-model reference.
- TreeSHAP and permutation importance explain Full CatBoost; logistic
  meta-classifier coefficients explain the stacking layer separately.

## Reproduction

See [`COMPPHISH_RUNBOOK.md`](COMPPHISH_RUNBOOK.md) for the PowerShell commands.
Install dependencies from `requirements-compphish.txt`.

The raw CompPhish files, processed datasets, fitted `.joblib` models and large
HTML-pair audit tables are intentionally excluded from Git. Obtain CompPhish
from its official distribution and preserve its licence and citation terms.

## Main scripts

- `scripts/prepare_compphish.py`
- `scripts/prepare_compphish_integrity_split.py`
- `scripts/audit_compphish_leakage.py`
- `scripts/understand_compphish.py`
- `scripts/train_compphish_models.py`
- `scripts/analyze_compphish_results.py`
- `scripts/explain_compphish.py`
- `scripts/build_compphish_chapter4_figures.py`
- `scripts/run_compphish_ablation.py`
- `scripts/compare_compphish_ablations.py`

## Data and model exclusion

The following remain local and must be backed up separately if required:

- `data/raw/`
- `data/processed/`
- fitted model files under `artifacts/**/models/`
- large near-duplicate HTML candidate tables

GitHub is used for source code and compact reproducibility evidence, not as a
replacement for licensed dataset storage.
