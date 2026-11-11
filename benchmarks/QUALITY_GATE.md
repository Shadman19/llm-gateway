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

## Plan sample size and comparison budgets

For **zero observed paired difference**, this interval clears margin `m` when
`n >= ceil(2*log(2/alpha)/m**2)`. These are illustrative thresholds for this
specific interval and observed difference, **not a statistical power analysis**,
a minimum dataset size for every method, or a promise that a future run passes.

| Absolute margin | One comparison, alpha 0.05 | 20 predeclared comparisons, alpha 0.0025 each |
| --- | ---: | ---: |
| 1 percentage point | 73,778 | 133,693 |
| 2 percentage points | 18,445 | 33,424 |
| 5 percentage points | 2,952 | 5,348 |
| 10 percentage points | 738 | 1,337 |

Reproduce each threshold with Python's standard library:

```python
import math

for comparisons in (1, 20):
    alpha = 0.05 / comparisons
    for margin in (0.01, 0.02, 0.05, 0.10):
        n = math.ceil(2 * math.log(2 / alpha) / margin**2)
        assert math.sqrt(2 * math.log(2 / alpha) / n) <= margin
        assert math.sqrt(2 * math.log(2 / alpha) / (n - 1)) > margin
        print(comparisons, margin, n)
```

For a **fixed, predeclared family** of 20 comparisons, allocating 0.05/20 to
each interval gives at least 95% simultaneous coverage by the union bound,
provided every individual interval meets its assumptions. Independence between
comparisons is not required for that union bound; independent sampling units
within each comparison are still required. The tool does not coordinate a
family: preserve the complete family and alpha allocation outside it, decide
in advance what each interval means, and retain every result. One passing
comparison is not approval of all candidates. This does not fix repeated
peeking, adaptive family selection, invalid graders or dependent prompts.

A distinct item ID does not establish a distinct sampling unit. For example,
40 independently sampled tasks copied 100 times remain 40 independent tasks,
not 4,000. Simply collapsing duplicates is valid only when the independent
unit and equal-weight estimand really are the original tasks; other clustered
designs need their own analysis.

The eight thresholds were checked against `evaluate` at `n-1` and `n` using
artificial identical all-correct labels (16 boundary checks). Every result
retained `synthetic_fixture_only`. This validates the calculation and utility
boundary behavior, not coverage on a real workload or model noninferiority.

## Review failed generations and important slices

Freeze the attempted-task denominator, failure policy and slices before
scoring. A missing response must remain in the workload under the declared
grading rule; intersecting successful IDs can manufacture a quality gain.
Failure-as-incorrect can be appropriate for a particular task-success metric,
but is not a universal utility model for abstention or safety.

Aggregate noninferiority does not establish subgroup noninferiority. For
example, suppose a synthetic comparison has 9,000 A items improving by ten
percentage points and 1,000 B items degrading by thirty points. The overall
difference is positive six points while B regresses sharply. With fixed
within-group differences, changing B's proportion to `w` changes the aggregate
difference to `0.10 - 0.40*w`: zero at 25% and minus five points at 37.5%.
These are invented arithmetic examples, not model results or distribution-shift
guarantees; real shifts can change within-group behavior too.

Predeclare a justified comparison family that includes important slices,
allocate alpha across it, retain every result and ensure each slice has enough
independent units for its intended decision. An inconclusive slice is not
evidence of safety. The utility does not accept group metadata or automatically
coordinate these comparisons. A family-wise adjustment cannot fix invalid
labels, unrepresentative sampling or dependent examples.

Strict JSON validation is a separate layer from scientific validity. Rejecting
duplicate fields/IDs, malformed records and coerced labels protects the score
interface; it cannot establish that a supplied `correct` label is actually
correct. Independently audit the grader, preserve adjudication and rubric
versions, and report disagreement without silently changing the frozen plan.
