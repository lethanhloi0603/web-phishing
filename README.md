# Phishing website detection on CompPhish

Copyright (c) 2026 Le Thanh Loi. All rights reserved.

This repository contains the code used in my phishing-detection study on the
CompPhish v4 dataset. The experiments compare URL and HTML feature groups,
several supervised learning algorithms, and two stacking configurations.

## Experimental setup

After URL normalization and duplicate removal, the dataset contains 15,257
samples. The Train set contains 13,203 samples and the Test set contains 2,054.
Samples connected by registered domain, an identical feature vector, or
identical HTML are kept in the same partition.

The experiments use all 70 features published with CompPhish:

- **Brand-URL:** 47 URL and brand-jacking features.
- **HTML:** 23 content features.
- **Full:** all 70 features.

The same 12 base learners are trained separately on each feature view. This
produces 36 base-model configurations. Boosting Stacking and Heterogeneous
Stacking are then evaluated within each view. Their second-level inputs are
created from five-fold out-of-fold predictions grouped by registered domain.

The two label-conditioned features, `url_more_than_mean` and
`domain_more_than_mean`, are recalculated inside each training partition. The
decision threshold is fixed at 0.5. Test data are not used to fit models,
choose configurations, or set the threshold.

Full CatBoost is retained as the reference model for feature-level
interpretation. Its predictions are examined with TreeSHAP and permutation
importance.

## Scripts

| Script | What it does |
|---|---|
| `scripts/prepare_compphish.py` | Reads and cleans the published feature workbook |
| `scripts/prepare_compphish_integrity_split.py` | Creates the Train/Test split while keeping related samples together |
| `scripts/audit_compphish_leakage.py` | Checks cross-partition domain, feature-vector, and HTML overlap |
| `scripts/understand_compphish.py` | Produces descriptive tables and figures for the dataset |
| `scripts/train_compphish_models.py` | Trains the base models and branch-specific OOF stacking models |
| `scripts/analyze_compphish_results.py` | Summarizes model performance |
| `scripts/run_compphish_ablation.py` | Runs the feature-removal experiments |
| `scripts/compare_compphish_ablations.py` | Compares the ablation variants |
| `scripts/explain_compphish.py` | Produces the TreeSHAP and permutation-importance results |
| `scripts/build_compphish_chapter3_feature_figure.py` | Draws the feature-group figure used in Chapter 3 |
| `scripts/build_compphish_chapter4_figures.py` | Creates the performance figures used in Chapter 4 |

## Running the experiments

Install the required Python packages:

```powershell
.\.venv311\Scripts\python.exe -m pip install -r requirements-compphish.txt
```

The commands for preparing the data, training the models, running the audits,
and generating the figures are listed in
[`COMPPHISH_RUNBOOK.md`](COMPPHISH_RUNBOOK.md).

## Data and generated files

The CompPhish files are not redistributed in this repository. Processed data,
trained models, and large generated artifacts are also excluded from Git. They
can be recreated locally by following the runbook after obtaining CompPhish
from its official distribution.

The archived HTML is used only for offline integrity checks and feature
analysis. The scripts do not host phishing pages, execute archived JavaScript,
or submit webpage forms.

## CompPhish

The dataset is available from the
[official CompPhish page on Mendeley Data](https://data.mendeley.com/datasets/fmbs4kp9wz/4).
Please follow the licence and citation instructions provided by the dataset
authors.
