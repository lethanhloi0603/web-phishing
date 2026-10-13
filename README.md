# CompPhish leakage-aware phishing detection benchmark

Copyright (c) 2026 Le Thanh Loi. All rights reserved.

This repository contains the reproducible source code and compact research
evidence for a phishing-website detection study on CompPhish v4.

## Research protocol

- Integrity-clean, domain-disjoint Train/Test split.
- 15,257 cleaned samples: 13,203 Train and 2,054 Test.
- 70 published features: 47 Brand-URL and 23 HTML/content features.
- `url_more_than_mean` and `domain_more_than_mean` are recomputed only from
  phishing rows in the relevant training partition.
- Twelve base learners are evaluated independently on Brand-URL, HTML and Full
  feature views, producing 36 base configurations.
- Boosting Stacking and Heterogeneous Stacking are constructed independently
  inside each feature view using five-fold out-of-fold predictions.
- `StratifiedGroupKFold` is grouped by registered domain inside Train.
- The classification threshold is fixed at 0.5 and Test is not used for model
  fitting or configuration decisions.
- Full CatBoost is the feature-level XAI reference model.
- TreeSHAP and permutation importance are used for feature-level explanation.

## Main scripts

| Script | Purpose |
|---|---|
| `scripts/prepare_compphish.py` | Read, clean and normalize the published workbook |
| `scripts/prepare_compphish_integrity_split.py` | Build the integrity-clean Train/Test split |
| `scripts/audit_compphish_leakage.py` | Audit domain, feature-vector and HTML overlap |
| `scripts/understand_compphish.py` | Produce data-understanding tables and figures |
| `scripts/train_compphish_models.py` | Train base learners and branch-aligned OOF stacking models |
| `scripts/analyze_compphish_results.py` | Summarize final model performance |
| `scripts/run_compphish_ablation.py` | Run feature-removal ablations |
| `scripts/compare_compphish_ablations.py` | Compare the ablation variants |
| `scripts/explain_compphish.py` | Generate TreeSHAP and permutation-importance outputs |
| `scripts/build_compphish_chapter3_feature_figure.py` | Build the Chapter 3 feature-structure figure |
| `scripts/build_compphish_chapter4_figures.py` | Build the Chapter 4 result figures |

## Reproduction

Install the dependencies:

```powershell
.\.venv311\Scripts\python.exe -m pip install -r requirements-compphish.txt
```

Run the workflow using the commands documented in
[`COMPPHISH_RUNBOOK.md`](COMPPHISH_RUNBOOK.md).

## Repository scope

The raw CompPhish distribution, processed datasets, fitted model binaries and
large generated audit tables are intentionally excluded from Git. Obtain the
dataset from its official distribution and preserve its licence and citation
terms.

The repository contains source code and selected compact evidence only. It
does not contain or serve live phishing websites, execute archived JavaScript,
or submit forms from the collected HTML pages.

## Dataset citation

CompPhish is distributed through Mendeley Data. Users of this repository must
cite the dataset and its accompanying publication according to the official
dataset page.
