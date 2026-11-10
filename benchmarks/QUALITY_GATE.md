# Paired binary-quality check

This is an evaluation utility, not a completed model/quantization experiment.
All checks added on November 10, 2026 use synthetic labels. No LLM was run,
no model quality was measured, and no candidate is approved for deployment.

The serving matrix measures a deterministic HTTP fixture. A future real-model
speed comparison also needs independent quality evidence. This utility makes
one small part of that evidence reproducible: align externally graded binary
scores by item ID, count paired gains/losses, and report conservative uncertainty.

## Freeze before scoring

Use a genuinely held-out workload and a fixed grader/rubric. Define the target
population and independent sampling unit, baseline/candidate model revisions,
generation settings, correctness rule, sample size, alpha and acceptable
absolute accuracy loss before examining outcomes. Preserve these in an
immutable experiment plan outside this utility, then retain its hash with the
results. The tool records hashes; it cannot prove chronological preregistration,
the accuracy of model labels, or that the workload was never used in tuning.

The JSON plan has exactly these fields:

```json
{
  "dataset_id": "versioned-held-out-task-name",
  "expected_items": 4000,
  "item_ids_sha256": "<64 lowercase hex characters>",
  "baseline_id": "pinned-model-and-generation-config-a",
  "candidate_id": "pinned-model-and-generation-config-b",
  "alpha": 0.05,
  "margin": 0.05,
  "evidence_kind": "held-out"
}
```

Use `synthetic` instead of `held-out` for fixtures. The digest is SHA-256 of
`json.dumps(sorted(ids), ensure_ascii=False, separators=(",", ":")).encode("utf-8")`.
Freeze the complete IDs and count in advance: comparing only an intersection
would silently discard failures. Both score files must contain exactly the
planned set. Each line is exactly `{"id":"stable-item-id","correct":true}`.
Only JSON booleans are accepted; no missing/duplicate IDs, blank lines, numeric
coercion or extra fields. Preserve failed generations in the workload; apply
the predeclared grading rule rather than dropping them. Prompts from the same
document/user or repeated generations are not independent merely because their
IDs differ. Use an appropriate independent-unit design before this analysis.

```bash
python -m benchmarks.quality_gate --plan frozen-plan.json \
  --baseline baseline-scores.jsonl --candidate candidate-scores.jsonl \
  --output new-quality-report.json
python -m unittest discover -s tests -p test_quality_gate.py -v
```

The output is created exclusively; an existing report is never overwritten.
Input hashes support reproducibility, not proof that files describe actual
model outputs. Inference errors, grader errors and protocol violations require
separate auditing. There are no model/API calls, credentials or downloads.

## Statistical interpretation

Let `D_i = candidate_correct_i - baseline_correct_i`, so `D_i` is in [-1,1].
For independent paired sampling units, the two-sided Hoeffding bound gives
`P(|mean(D)-mean(E[D])| >= r) <= 2 exp(-n*r*r/2)`.
Thus `r = sqrt(2*log(2/alpha)/n)`. Clip the interval to [-1,1]. For an IID target
sample, the estimand is the population difference in binary task accuracy.
Pair dependence within an item is allowed; independence is across items.

This applies the bounded-summand inequality from
[Hoeffding (1963), section 2](https://doi.org/10.1080/01621459.1963.10500830).
It is deliberately conservative and does not exploit the observed paired
variance. The interpretation assumes one fixed comparison and fixed sample
size. No repeated-peeking, adaptive model selection or multiple-comparison
adjustment is implemented. Do not tune the margin after seeing this report.

If the lower bound is at least `-margin`, the result supports noninferiority
**conditional on the sampling/grading/analysis assumptions**. If the upper
bound is below `-margin`, it supports degradation beyond that margin. Otherwise
the result is inconclusive; a point estimate within the margin does not prove
noninferiority. `margin=0.05` means five absolute percentage points, not a 5%
relative change. It is illustrative, not a recommended tolerance.

Even identical scores on 100 items give radius about 0.272 at alpha 0.05.
For a zero observed difference and a 0.05 margin, this bound needs at least
`ceil(2*log(40)/0.05**2) = 2952` independent pairs to support noninferiority.
These are formula examples, not a model-quality finding. A synthetic result
always has `decision_scope=synthetic_fixture_only`.

This binary check does not establish calibration, subgroup safety, robustness,
privacy, semantic answer equivalence, latency or production readiness. A later
quality-speed experiment must retain real responses, pinned model/config and
grader provenance, failure accounting and separately measured serving results.
Do not merge this output with the CPU-fixture matrix as model performance.
