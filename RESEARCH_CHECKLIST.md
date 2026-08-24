# Research-grade checklist

This checklist separates a reproducible thesis experiment from a preliminary demo.

The recommended primary design is now multi-family LOFO rather than a
Qwen-only claim. See `MULTIFAMILY_LOFO.md` for the exact staged commands. The
older Qwen scenario remains a pilot and regression fixture.

## Current hardening status (2026-08-22)

Completed in code:

- The automated regression suite covers data import, leakage controls,
  evaluation identity, confidence bounds, artifact transforms, and model losses.
- PhreshPhish streaming preserves `target` as brand, shuffles deterministically,
  and writes source/license/hash manifests.
- Public imports support brand-aware directory layouts and bounded HTML reads
  inside ZIP files without extracting archives.
- Temporal scenarios are real-world-only by default; LLM pages require the
  explicit `--include-llm-temporal` flag.
- `artifact_consistency` and `robust_full` pair clean and transformed views and
  now synchronize dropout/modality masks, so the consistency loss measures the
  artifact change rather than unrelated stochastic noise.
- Primary/external metrics are bound to split and checkpoint SHA-256 values.
  External evidence must be provenance-disjoint, have at least 100 samples per
  class, use primary checkpoints, and cover five seeds.
- The evidence gate checks confidence-certified recall for the held-out Qwen
  family at both page and site level. Overall human-phishing performance cannot
  substitute for the held-out-family result.
- Multi-seed confidence intervals use a site-clustered stratified bootstrap by
  default, avoiding page-level pseudo-replication for multi-page websites.
- A strict PhreshPhish temporal scenario exists at `data/scenario_temporal_phresh`
  with 3,308 train, 581 validation and 1,735 test pages and zero HTML/template
  overlap. Its char-TF-IDF pilot has AUPRC 0.851 and F1 0.741, but only 0.055
  recall at the confidence-certified 1% FPR threshold.

Still requires external evidence or human work:

- At least 100 independent Qwen-positive pages across at least 3 brands.
- Independent double annotation and an agreement report.
- Five corrected `robust_full` seeds plus five matching Phish360 evaluations
  are not yet complete. Phish360 supplies both HTML classes and is
  provenance-disjoint; its paper grants academic use, but no standalone SPDX
  dataset license was found, so raw files must not be redistributed.

## Mandatory publication gates

- Every LLM phishing sample used in the final tables is independently human-validated.
- Report reviewer instructions, disagreements and inter-rater agreement when two reviewers are available.
- Each held-out generator/family test contains at least 100 phishing samples and preferably 300+ independent templates.
- Report page-level and site/codebase-level tasks separately; a dashboard or 404 page is not a positive phishing page solely because its parent codebase is positive.
- Exact HTML overlap is zero across train, validation and test.
- Template overlap is zero for the strict primary analysis; a non-strict analysis may be reported separately.
- Thresholds and model selection use validation data only.
- A claimed 1% FPR must pass the one-sided 95% Clopper-Pearson upper bound,
  not only the observed validation FPR.
- The primary model includes all branches claimed in the paper.
- Use at least five seeds for the primary table and report paired confidence intervals.
- Keep dataset hashes, environment versions, commands and the source commit with every run.
- Include error analysis, artifact stress tests, subgroup results and limitations.

Dataset roles, provenance rules and public-dataset import commands are documented in `DATASET_SOURCES.md`.

## Dataset gate

Run both integrity/provenance and shortcut gates. Count held-out-family positives
separately so a large human test subset cannot hide a small LOFO sample.

```powershell
python scripts\audit_dataset.py `
  --scenario-dir data\scenario_lofo_qwen_page_matched_calibrated `
  --min-test-positives 100 `
  --min-test-positive-sites 20 `
  --heldout-family qwen `
  --min-heldout-positives 100 `
  --min-heldout-positive-sites 50 `
  --min-heldout-positive-templates 100 `
  --min-heldout-positive-brands 3 `
  --require-template-isolation `
  --require-manual-llm-validation `
  --allow-published-llm-labels `
  --out outputs\audits\scenario_lofo_qwen_page_matched_calibrated.json

python scripts\audit_shortcuts.py `
  --scenario-dir data\scenario_lofo_qwen_page_matched_calibrated `
  --require-source-label-2x2 `
  --fail-on-high-purity `
  --out outputs\audits\scenario_lofo_qwen_page_matched_calibrated_shortcuts.json
```

Exit code `0` means the declared gate passes. Exit code `2` means the experiment must not be used as a primary publication result.

The current matched Qwen scenario passes leakage and shortcut checks but fails
the primary publication gate because it has only 33 held-out Qwen-positive pages,
23 positive sites/templates, one brand, and no independent manual validation of
those Qwen pages. Published labels remain useful for a pilot but do not replace
the thesis team's annotation evidence.
Use it as a pilot/capability benchmark, not as the primary Q1 result.

For a capability benchmark using labels released by the source LLM-generation study, add `--allow-published-llm-labels`. Report this separately from an independently human-validated phishing benchmark.

```powershell
python scripts\audit_dataset.py `
  --scenario-dir data\scenario_lofo_qwen_published `
  --min-test-positives 50 `
  --require-template-isolation `
  --require-manual-llm-validation `
  --allow-published-llm-labels
```

## Primary experiment

```powershell
python scripts\run_experiment_matrix.py `
  --scenario-dir data\scenario_lofo_qwen_page_matched_calibrated `
  --output-dir outputs\experiments\qwen_calibrated_robust5 `
  --variants bce,robust_full `
  --seeds 42,43,44,45,46 `
  --epochs 5 `
  --batch-size 32 `
  --max-dom-length 128 `
  --artifact-augmentation-probability 0.3 `
  --artifact-augmentation-modes comments_removed,identifiers_normalized,visible_text_removed `
  --artifact-tests `
  --artifact-test-variants robust_full `
  --bootstrap-samples 2000 `
  --min-free-gb 6
```

The run manifest records SHA-256 hashes of all splits, Git state when available, and software/hardware versions.

## Paired comparison and error analysis

```powershell
python scripts\compare_variants.py `
  --root outputs\experiments\qwen_calibrated_robust5 `
  --variant-a bce --variant-b robust_full `
  --evaluation target_fpr_confident `
  --bootstrap-samples 10000 `
  --out outputs\experiments\qwen_calibrated_robust5\bce_vs_robust_full.json

python scripts\error_analysis.py `
  --variant-dir outputs\experiments\qwen_calibrated_robust5\robust_full `
  --evaluation predicted_target_fpr_confident `
  --out outputs\experiments\qwen_calibrated_robust5\robust_full_errors.csv
```

## Required tables for a strong thesis or paper

1. Dataset composition and annotation agreement.
2. In-distribution, Human-to-LLM, LOGO, LOFO and temporal results.
3. Full model versus classical and neural baselines.
4. Component and input-branch ablations.
5. Per-generator/family/brand performance with confidence intervals.
6. Paired statistical comparisons against the strongest baseline.
7. Artifact robustness and false-positive/false-negative taxonomy.
8. Runtime, parameter count, memory and inference latency.

No checklist can guarantee a grade or journal acceptance. Novelty, writing, ethics, dataset quality and independent peer review remain decisive.

## One-command Q1 evidence gate

Run this after producing five primary-seed metric files, an annotation-agreement
report and at least one independent external-dataset result. The command fails
closed when evidence is absent; `--advisory` writes the same report but exits 0.

```powershell
python scripts\q1_readiness.py `
  --scenario-dir data\scenario_lofo_qwen_page_matched_calibrated `
  --heldout-family qwen `
  --agreement-json outputs\audits\annotation_agreement.json `
  --metrics `
    outputs\experiments\qwen_calibrated_robust5\robust_full\seed_42\metrics.json `
    outputs\experiments\qwen_calibrated_robust5\robust_full\seed_43\metrics.json `
    outputs\experiments\qwen_calibrated_robust5\robust_full\seed_44\metrics.json `
    outputs\experiments\qwen_calibrated_robust5\robust_full\seed_45\metrics.json `
    outputs\experiments\qwen_calibrated_robust5\robust_full\seed_46\metrics.json `
  --external-metrics `
    outputs\external\phish360_calibrated_robust5\robust_full\seed_42\metrics.json `
    outputs\external\phish360_calibrated_robust5\robust_full\seed_43\metrics.json `
    outputs\external\phish360_calibrated_robust5\robust_full\seed_44\metrics.json `
    outputs\external\phish360_calibrated_robust5\robust_full\seed_45\metrics.json `
    outputs\external\phish360_calibrated_robust5\robust_full\seed_46\metrics.json `
  --out outputs\audits\q1_readiness.json
```

An external metrics file is accepted only when `evaluate_deep.py` was run with
`--evaluation-role external`, contains at least 100 examples of each class, has
valid dataset provenance, is independent of the primary train/validation
provenance, and uses one of the exact primary checkpoint hashes. Five external
seed reports are required by the default policy. Merely renaming an internal
metrics file no longer passes the gate.

Train-only artifact augmentation is deterministic by seed and epoch. It changes
only training HTML and recomputes DOM/static features from the altered page;
validation, test and artifact stress tests remain untouched.

For double annotation, keep two independent columns such as `reviewer_a_label` and `reviewer_b_label`, then run:

```powershell
python scripts\review_llm_audit.py --reviewer-id reviewer_a --port 8765
python scripts\review_llm_audit.py --reviewer-id reviewer_b --port 8766
```

Run the two sessions separately. Reviewer mode hides the other reviewer's labels
and writes reviewer-specific label/reason/timestamp columns without overwriting
`manual_validated`. Resolve disagreements before rebuilding the scenario.

```powershell
python scripts\annotation_agreement.py `
  --audit data\llm_phish_audit_ranked.csv `
  --reviewer-a reviewer_a_label `
  --reviewer-b reviewer_b_label `
  --out outputs\audits\annotation_agreement.json
```
