"""Paired binary-quality audit; no model, grader, or held-out validity is inferred."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def ids_digest(ids):
    """Hash canonical IDs, independent of score-file order."""
    canonical = json.dumps(sorted(ids), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_scores(path):
    raw = Path(path).read_bytes()
    records = {}
    for line_no, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{path}:{line_no}: blank records are not allowed")
        row = json.loads(line, object_pairs_hook=strict_object)
        if not isinstance(row, dict) or set(row) != {"id", "correct"}:
            raise ValueError(f"{path}:{line_no}: require exactly id and correct")
        item = row["id"]
        if not isinstance(item, str) or not item.strip() or item != item.strip():
            raise ValueError(f"{path}:{line_no}: id must be nonempty unpadded text")
        if type(row["correct"]) is not bool:
            raise ValueError(f"{path}:{line_no}: correct must be a JSON boolean")
        if item in records:
            raise ValueError(f"{path}:{line_no}: duplicate id {item!r}")
        records[item] = row["correct"]
    if not records:
        raise ValueError("empty score file")
    return records, hashlib.sha256(raw).hexdigest()


def validate_plan(plan):
    required = {
        "dataset_id",
        "expected_items",
        "item_ids_sha256",
        "baseline_id",
        "candidate_id",
        "alpha",
        "margin",
        "evidence_kind",
    }
    if not isinstance(plan, dict) or set(plan) != required:
        raise ValueError("plan keys must exactly match the documented schema")
    for key in ("dataset_id", "baseline_id", "candidate_id"):
        if not isinstance(plan[key], str) or not plan[key].strip():
            raise ValueError(f"{key} must be nonempty text")
    if plan["baseline_id"] == plan["candidate_id"]:
        raise ValueError("baseline_id and candidate_id must differ")
    if type(plan["expected_items"]) is not int or plan["expected_items"] < 1:
        raise ValueError("expected_items must be a positive integer")
    digest = plan["item_ids_sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("item_ids_sha256 must be a lowercase SHA-256 hex digest")
    if any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("item_ids_sha256 must be a lowercase SHA-256 hex digest")
    for key in ("alpha", "margin"):
        value = plan[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"{key} must be finite and strictly between zero and one")
    if plan["evidence_kind"] not in ("synthetic", "held-out"):
        raise ValueError("evidence_kind must be synthetic or held-out")


def evaluate(plan, baseline, candidate):
    """Two-sided Hoeffding interval for independent paired differences in [-1,1]."""
    validate_plan(plan)
    if set(baseline) != set(candidate):
        raise ValueError("baseline and candidate IDs differ; never silently intersect")
    if len(baseline) != plan["expected_items"]:
        raise ValueError("item count differs from the frozen plan")
    if ids_digest(baseline) != plan["item_ids_sha256"]:
        raise ValueError("item IDs differ from the frozen plan")
    if any(type(v) is not bool for scores in (baseline, candidate) for v in scores.values()):
        raise ValueError("scores must be boolean")
    n = len(baseline)
    gains = sum(candidate[i] and not baseline[i] for i in baseline)
    losses = sum(baseline[i] and not candidate[i] for i in baseline)
    difference = (gains - losses) / n
    radius = math.sqrt(2 * (math.log(2) - math.log(plan["alpha"])) / n)
    lower, upper = max(-1.0, difference - radius), min(1.0, difference + radius)
    if lower >= -plan["margin"]:
        result = "supports_noninferiority_under_assumptions"
    elif upper < -plan["margin"]:
        result = "supports_regression_beyond_margin_under_assumptions"
    else:
        result = "inconclusive"
    return {
        "schema_version": 1,
        "plan": plan,
        "paired_items": n,
        "baseline_accuracy": sum(baseline.values()) / n,
        "candidate_accuracy": sum(candidate.values()) / n,
        "candidate_only_correct": gains,
        "baseline_only_correct": losses,
        "both_correct": sum(baseline[i] and candidate[i] for i in baseline),
        "both_incorrect": sum(not baseline[i] and not candidate[i] for i in baseline),
        "candidate_minus_baseline": difference,
        "hoeffding_two_sided_interval": [lower, upper],
        "confidence_level": 1 - plan["alpha"],
        "statistical_result": result,
        "decision_scope": (
            "synthetic_fixture_only"
            if plan["evidence_kind"] == "synthetic"
            else "conditional_on_declared_sampling_grading_and_frozen_plan"
        ),
        "limitations": [
            "No automatic verification of independent held-out sampling or frozen analysis choices.",
            "Repeated prompts, shared documents or users can violate independence despite unique IDs.",
            "Correctness labels are supplied externally; grader validity/noise is not established.",
            "One predeclared comparison; no repeated-peeking or multiple-comparison correction.",
            "Binary task accuracy only; not calibration, safety, latency or deployment approval.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        plan_raw = args.plan.read_bytes()
        plan = json.loads(plan_raw, object_pairs_hook=strict_object)
        baseline, baseline_hash = load_scores(args.baseline)
        candidate, candidate_hash = load_scores(args.candidate)
        report = evaluate(plan, baseline, candidate)
        report["input_sha256"] = {
            "plan": hashlib.sha256(plan_raw).hexdigest(),
            "baseline": baseline_hash,
            "candidate": candidate_hash,
        }
        # Exclusive creation prevents an accidental rerun from overwriting prior evidence.
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write("\n")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"quality gate: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
