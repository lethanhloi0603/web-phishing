# CompPhish experimental runbook

This pipeline treats CompPhish v4 as a newly released benchmark and does not
claim a new learning algorithm. Its contribution is a reproducible,
leakage-aware empirical benchmark of three aligned feature branches and their
OOF stacking ensembles, followed by two-level explainability.

## Protocol

1. Normalize URLs, remove conflicting/duplicate normalized URLs, connect exact
   domain/HTML/feature duplicates, and create integrity-clean Train/Test sets.
2. Retain all 70 published features. Recompute `url_more_than_mean` and
   `domain_more_than_mean` inside each OOF training fold; Test is never used.
3. Compare three numeric branches: Brand-URL (47), HTML (23), and Full (70).
4. Run the same 12 prespecified learners in every branch: Logistic Regression,
   Gaussian Naive Bayes, KNN, Decision Tree, RBF-SVM, MLP, Random Forest,
   Extra Trees, HistGradientBoosting, XGBoost, LightGBM, and CatBoost.
5. Generate grouped OOF predictions inside Train. In each branch independently,
   train (a) boosting stacking and (b) heterogeneous stacking. Predictions from
   different branches are not mixed inside a stack.
6. Refit every base learner on all Train rows and evaluate once on Test with
   the fixed decision threshold 0.5.
7. Report accuracy, precision, recall, specificity, F1, balanced accuracy,
   PR-AUC, ROC-AUC, MCC, FPR, FNR, Brier score, ECE, and confusion counts.
   Paired bootstrap and McNemar are outside the thesis protocol.
8. Explain each stacker using its logistic meta-classifier coefficients and
   explain the pre-specified Full CatBoost reference model using TreeSHAP plus
   permutation importance. CatBoost feature explanations are not presented as
   explanations of the complete stacking ensemble.

## Commands (PowerShell)

Install only the dependencies required by these scripts:

```powershell
.\.venv311\Scripts\python.exe -m pip install openpyxl shap matplotlib tldextract xgboost lightgbm catboost
```

Prepare and audit the data:

```powershell
.\.venv311\Scripts\python.exe scripts\prepare_compphish.py `
  --input data\raw\CompPhish\All_Features_threshold90.xlsx `
  --output data\processed\compphish\compphish_clean_split.csv.gz `
  --audit-dir artifacts\compphish\data_audit
```

Re-run the leakage audit under the same final 70-feature protocol. The two
mean-derived indicators are recomputed using Train phishing rows only:

```powershell
.\.venv311\Scripts\python.exe scripts\audit_compphish_leakage.py `
  --data data\processed\compphish\compphish_integrity_split.csv.gz `
  --html-zip data\raw\CompPhish\All_HTML.zip `
  --output-dir artifacts\compphish\leakage_audit_integrity_70 `
  --near-html-distance 3 `
  --nearest-feature-rows 250
```

Run a smoke test first:

```powershell
.\.venv311\Scripts\python.exe scripts\train_compphish_models.py `
  --data data\processed\compphish\compphish_integrity_split.csv.gz `
  --output-dir artifacts\compphish\branch_stacking_smoke `
  --cv-folds 3 `
  --quick `
  --train-limit 3000
```

Run the full experiment after the smoke test succeeds:

```powershell
.\.venv311\Scripts\python.exe scripts\train_compphish_models.py `
  --data data\processed\compphish\compphish_integrity_split.csv.gz `
  --output-dir artifacts\compphish\branch_stacking_full `
  --cv-folds 5 `
  --views brand_url html full `
  --model-set framework `
  --optional-boosters required `
  --threshold 0.5
```

Generate XAI artifacts:

```powershell
.\.venv311\Scripts\python.exe scripts\explain_compphish.py `
  --data data\processed\compphish\compphish_integrity_split.csv.gz `
  --experiment-dir artifacts\compphish\branch_stacking_full `
  --output-dir artifacts\compphish\branch_stacking_xai_70 `
  --feature-model full__catboost `
  --sample-size 1000 `
  --top-k 25 `
  --permutation-repeats 10
```

Generate final descriptive tables and figures:

```powershell
.\.venv311\Scripts\python.exe scripts\analyze_compphish_results.py `
  --experiment-dir artifacts\compphish\branch_stacking_full `
  --output-dir artifacts\compphish\branch_stacking_analysis `
  --seed 42
```

Do not paste terminal prompts such as `(.venv311) PS C:\...>` or old output
back into PowerShell; paste only the commands shown in the code blocks.
