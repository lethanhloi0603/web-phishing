# Phishing website detection with CompPhish

Research code and reproducibility evidence for Lê Thành Lợi's phishing-website
detection study using CompPhish v4.

The study evaluates three feature views under the same experimental protocol:

- **Brand–URL:** 47 URL and brand-jacking features;
- **HTML:** 23 features extracted from page content;
- **Full:** all 70 published features.

Twelve base learners are evaluated on each feature view. Two branch-aligned
stacking configurations are then built from five-fold out-of-fold predictions,
giving 42 evaluated configurations in total.

## Dataset protocol

The processed dataset contains 15,257 observations:

| Partition | Observations | Purpose |
|---|---:|---|
| Train | 13,203 | Model fitting and five-fold OOF prediction |
| Test | 2,054 | Final evaluation |

Related observations are assigned to the same partition when they share a
registered domain, an identical feature vector, or identical normalized HTML.
The resulting Train and Test partitions therefore have no exact overlap under
these three checks.

The published variables `url_more_than_mean` and `domain_more_than_mean` are
recalculated only from the applicable training partition. The classification
threshold is fixed at 0.5. Test observations are not used for fitting or
threshold selection.

## Models and explanations

The base-model set contains Logistic Regression, Gaussian Naive Bayes, KNN,
Decision Tree, RBF-SVM, MLP, Random Forest, Extra Trees,
HistGradientBoosting, XGBoost, LightGBM and CatBoost.

The two ensemble variants are:

- **Boosting Stacking:** combines the boosting-family members available in the
  corresponding feature view;
- **Heterogeneous Stacking:** combines models from different learning
  families.

Both use logistic regression as the second-level learner. Full CatBoost is the
feature-level explanation reference. TreeSHAP and permutation importance are
used for global analysis, while representative true-positive, false-negative
and false-positive observations are examined with local TreeSHAP explanations.

## Repository contents

| Path | Purpose |
|---|---|
| `scripts/prepare_compphish.py` | Cleans and normalizes the published feature workbook |
| `scripts/prepare_compphish_integrity_split.py` | Creates the integrity-controlled Train/Test split |
| `scripts/audit_compphish_leakage.py` | Audits domain, feature-vector and HTML overlap |
| `scripts/understand_compphish.py` | Produces descriptive tables and figures |
| `scripts/train_compphish_models.py` | Trains the base learners and OOF stacking models |
| `scripts/analyze_compphish_results.py` | Summarizes predictive performance |
| `scripts/run_compphish_ablation.py` | Runs the feature-removal experiments |
| `scripts/compare_compphish_ablations.py` | Compares the ablation variants |
| `scripts/explain_compphish.py` | Generates global TreeSHAP and permutation results |
| `scripts/explain_compphish_local_cases.py` | Generates local TP, FN and FP explanations |
| `scripts/benchmark_compphish_runtime.py` | Measures training and inference cost |
| `scripts/build_compphish_runtime_comparison_figure.py` | Builds the compact runtime comparison figure |
| `scripts/build_compphish_chapter3_feature_figure.py` | Builds the Chapter 3 feature figure |
| `scripts/build_compphish_chapter4_figures.py` | Builds the Chapter 4 result figures |
| `artifacts/compphish/` | Compact tables, figures and audit summaries retained as evidence |

## Reproduction

Install the dependencies:

```powershell
.\.venv311\Scripts\python.exe -m pip install -r requirements-compphish.txt
```

Run the workflow using the commands in
[`COMPPHISH_RUNBOOK.md`](COMPPHISH_RUNBOOK.md).

## Data availability and safety

The raw CompPhish distribution, processed datasets, archived HTML and fitted
model binaries are not redistributed here. CompPhish is available from the
[official Mendeley Data release](https://data.mendeley.com/datasets/fmbs4kp9wz/4);
users should follow the licence and citation requirements stated by the dataset
authors.

Archived pages are used only for offline integrity checks. The repository does
not host phishing pages, execute archived JavaScript or submit web forms.

## Author

Lê Thành Lợi (2026).
