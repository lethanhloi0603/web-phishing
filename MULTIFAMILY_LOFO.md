# Multi-family LOFO experiment

This protocol evaluates whether phishing representations generalize to an LLM
family that is absent from training. The primary folds are Claude, Gemini, GPT,
Grok, Mistral and Qwen. Nova and Llama remain exploratory because the current
release does not contain enough pages from both classes.

No script in this document changes `manual_validated`. Source-published labels
are suitable for a pilot only. Final evidence uses the `manual` label policy and
the annotation agreement report.

## 1. Build the pilot folds

```powershell
python scripts\build_multifamily_lofo.py `
  --phresh-train data\phresh_train.csv `
  --phresh-test data\phresh_test.csv `
  --llm data\llm_phish_published_page.csv `
  --families claude,gemini,gpt,grok,mistral,qwen `
  --label-policy published `
  --validation-size 0.5 `
  --strict-template-isolation `
  --include-human-phishing-test `
  --compression gzip `
  --out-root data\scenario_multifamily_lofo_published
```

The command fails when a requested family has fewer than 20 pages from either
class. Each fold has an independent manifest and the root manifest records all
input hashes and commands.

## 2. Run a small pilot

Start with two families and one seed. This verifies runtime and storage without
creating evidence that can be confused with the final experiment.

```powershell
python scripts\run_multifamily_lofo.py `
  --scenario-root data\scenario_multifamily_lofo_published `
  --output-root outputs\experiments\multifamily_lofo_pilot `
  --families gpt,qwen `
  --variants bce,robust_full `
  --seeds 42 `
  --epochs 3 `
  --batch-size 32 `
  --max-dom-length 128 `
  --artifact-tests `
  --min-free-gb 5.5
```

Do not report the pilot as the primary multi-family result.

Optional pilot summary:

```powershell
python scripts\summarize_multifamily_lofo.py `
  --scenario-root data\scenario_multifamily_lofo_published `
  --experiments-root outputs\experiments\multifamily_lofo_pilot `
  --families gpt,qwen `
  --variants bce,robust_full `
  --seeds 42 `
  --bootstrap-samples 1000 `
  --out-dir outputs\summaries\multifamily_lofo_pilot
```

## 3. Complete human annotation

Run two reviewer sessions separately. The interface hides the other reviewer's
decision and writes reviewer-specific columns.

```powershell
python scripts\review_llm_audit.py --reviewer-id reviewer_a --port 8765
python scripts\review_llm_audit.py --reviewer-id reviewer_b --port 8766

python scripts\annotation_agreement.py `
  --audit data\llm_phish_audit_ranked.csv `
  --reviewer-a reviewer_a_label `
  --reviewer-b reviewer_b_label `
  --out outputs\audits\annotation_agreement.json
```

Adjudicate disagreements and set `manual_validated=1` only for reviewed samples
that satisfy the page-label policy. Do not copy source-published or automated
screening decisions into the manual field.

## 4. Build final manual-label folds

```powershell
python scripts\build_multifamily_lofo.py `
  --phresh-train data\phresh_train.csv `
  --phresh-test data\phresh_test.csv `
  --llm data\llm_phish_published_page.csv `
  --llm-audit data\llm_phish_audit_ranked.csv `
  --families claude,gemini,gpt,grok,mistral,qwen `
  --label-policy manual `
  --validation-size 0.5 `
  --strict-template-isolation `
  --include-human-phishing-test `
  --compression gzip `
  --out-root data\scenario_multifamily_lofo_manual
```

If this command rejects a family, annotation coverage is incomplete. Do not
lower the minimum counts merely to make the command pass.

## 5. Run the primary matrix

The full CPU job contains 60 training runs: six families, two variants and five
seeds. Run it only after checking free disk and power settings. The runner is
resumable: completed checkpoints and metric files are skipped unless
`--overwrite` is supplied.

```powershell
python scripts\run_multifamily_lofo.py `
  --scenario-root data\scenario_multifamily_lofo_manual `
  --output-root outputs\experiments\multifamily_lofo_primary `
  --variants bce,robust_full `
  --seeds 42,43,44,45,46 `
  --epochs 5 `
  --batch-size 32 `
  --max-dom-length 128 `
  --artifact-tests `
  --artifact-test-variants robust_full `
  --bootstrap-samples 2000 `
  --min-free-gb 5.5
```

## 6. Summarize out-of-family results

Each LLM page is selected only from its held-out fold. The summarizer rejects a
sample that appears in more than one family fold. It produces per-family,
macro-family, micro OOF, worst-family, page-level and site-level tables.

```powershell
python scripts\summarize_multifamily_lofo.py `
  --scenario-root data\scenario_multifamily_lofo_manual `
  --experiments-root outputs\experiments\multifamily_lofo_primary `
  --variants bce,robust_full `
  --seeds 42,43,44,45,46 `
  --evaluation target_fpr_confident `
  --bootstrap-samples 10000 `
  --out-dir outputs\summaries\multifamily_lofo_primary
```

Use macro-family performance as the primary result, micro OOF as a secondary
result and explicitly report the worst family.

## 7. External Phish360 evaluation

The exact primary checkpoints are evaluated with their own fold validation
thresholds. Phish360 remains external and must not be merged into training.

```powershell
python scripts\evaluate_multifamily_external.py `
  --scenario-root data\scenario_multifamily_lofo_manual `
  --checkpoints-root outputs\experiments\multifamily_lofo_primary `
  --test data\external_phish360_calibrated_strict.csv.gz `
  --output-root outputs\external\multifamily_phish360 `
  --variant robust_full `
  --seeds 42,43,44,45,46 `
  --batch-size 32 `
  --bootstrap-samples 2000 `
  --min-external-sites-per-class 100 `
  --min-free-gb 5.5
```

## 8. Final evidence gate

```powershell
python scripts\multifamily_q1_readiness.py `
  --scenario-root data\scenario_multifamily_lofo_manual `
  --experiments-root outputs\experiments\multifamily_lofo_primary `
  --external-root outputs\external\multifamily_phish360 `
  --summary-json outputs\summaries\multifamily_lofo_primary\summary.json `
  --agreement-json outputs\audits\annotation_agreement.json `
  --variant robust_full `
  --seeds 42,43,44,45,46 `
  --out outputs\audits\multifamily_q1_readiness.json
```

Exit code `0` means the repository's declared evidence policy passed. It is not
a guarantee of acceptance by a Q1 journal. Exit code `2` identifies missing or
invalid evidence and must not be bypassed for the primary result.

## Storage and runtime

The current machine has limited free disk. Keep at least 5.5 GB free during a
run and preferably free 8 GB before starting the full matrix. DOM/static mode
does not download another language model. The full CPU matrix can take roughly
12--24 hours; it is safe to stop between completed runs and invoke the same
command again.
