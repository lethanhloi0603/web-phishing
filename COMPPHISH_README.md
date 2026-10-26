# CompPhish experiment record

This file summarizes the exact protocol represented by the scripts and compact
artifacts in this repository. For the project overview, see
[`README.md`](README.md). For executable commands, see
[`COMPPHISH_RUNBOOK.md`](COMPPHISH_RUNBOOK.md).

## Fixed protocol

- CompPhish v4 with 15,257 cleaned observations.
- Integrity-controlled split: 13,203 Train and 2,054 Test observations.
- 70 published features: 47 Brand–URL and 23 HTML/content features.
- Three feature views: Brand–URL, HTML and Full.
- Twelve base learners per view, producing 36 base configurations.
- Boosting Stacking and Heterogeneous Stacking per view, producing six
  additional configurations.
- Five-fold `StratifiedGroupKFold` inside Train, grouped by registered domain.
- Stacking inputs generated only from out-of-fold Train predictions.
- `url_more_than_mean` and `domain_more_than_mean` recalculated inside each
  applicable training partition.
- Classification threshold fixed at 0.5.
- Test used only for final evaluation.
- Full CatBoost used as the feature-level XAI reference.

## Evidence retained in Git

The repository keeps compact CSV, JSON and PNG outputs needed to trace the
reported results, including performance summaries, leakage-audit summaries,
global and local XAI results, and runtime comparisons.

Raw data, processed datasets, archived HTML, fitted model binaries and large
pairwise audit tables remain local. They are excluded because of dataset
licensing, safety and repository-size constraints.

## Authorship

Research implementation and experiment record: Lê Thành Lợi, 2026.
