"""Synthetic checks only: no trained model or external grader is evaluated."""

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from benchmarks.quality_gate import evaluate, ids_digest, load_scores, main


class QualityGateTests(unittest.TestCase):
    def setUp(self):
        self.baseline = {f"item-{i}": i % 2 == 0 for i in range(4000)}
        self.plan = {
            "dataset_id": "synthetic-unit-test",
            "expected_items": 4000,
            "item_ids_sha256": ids_digest(self.baseline),
            "baseline_id": "fixture-a",
            "candidate_id": "fixture-b",
            "alpha": 0.05,
            "margin": 0.05,
            "evidence_kind": "synthetic",
        }

    def test_equal_scores_have_nonzero_uncertainty(self):
        report = evaluate(self.plan, self.baseline, self.baseline)
        expected = math.sqrt(2 * math.log(40) / 4000)
        self.assertAlmostEqual(report["hoeffding_two_sided_interval"][0], -expected)
        self.assertAlmostEqual(report["hoeffding_two_sided_interval"][1], expected)
        self.assertEqual(report["decision_scope"], "synthetic_fixture_only")
        self.assertEqual(report["statistical_result"], "supports_noninferiority_under_assumptions")

    def test_small_equal_sample_is_inconclusive(self):
        scores = {"one": True}
        plan = dict(self.plan, expected_items=1, item_ids_sha256=ids_digest(scores))
        report = evaluate(plan, scores, scores)
        self.assertEqual(report["statistical_result"], "inconclusive")
        self.assertEqual(report["hoeffding_two_sided_interval"], [-1.0, 1.0])

    def test_declared_held_out_is_still_conditional(self):
        r = evaluate(dict(self.plan, evidence_kind="held-out"), self.baseline, self.baseline)
        self.assertEqual(
            r["decision_scope"],
            "conditional_on_declared_sampling_grading_and_frozen_plan",
        )
        self.assertTrue(any("No automatic verification" in s for s in r["limitations"]))

    def test_stricter_confidence_widens_interval(self):
        r = evaluate(self.plan, self.baseline, self.baseline)
        strict = evaluate(dict(self.plan, alpha=0.001), self.baseline, self.baseline)
        self.assertLess(
            strict["hoeffding_two_sided_interval"][0],
            r["hoeffding_two_sided_interval"][0],
        )

    def test_id_hash_handles_unicode_and_boundaries(self):
        self.assertEqual(ids_digest(["α", "b"]), ids_digest(["b", "α"]))
        self.assertNotEqual(ids_digest(["ab", "c"]), ids_digest(["a", "bc"]))

    def test_invalid_cli_plan_creates_no_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = root / "plan.json"
            scores = root / "scores.jsonl"
            output = root / "report.json"
            plan.write_text('{"alpha":0.05,"alpha":0.1}')
            scores.write_text('{"id":"one","correct":true}\n')
            with self.assertRaises(SystemExit):
                main(
                    [
                        "--plan",
                        str(plan),
                        "--baseline",
                        str(scores),
                        "--candidate",
                        str(scores),
                        "--output",
                        str(output),
                    ]
                )
            self.assertFalse(output.exists())

    def test_clear_regression(self):
        report = evaluate(
            self.plan,
            dict.fromkeys(self.baseline, True),
            dict.fromkeys(self.baseline, False),
        )
        self.assertEqual(report["candidate_minus_baseline"], -1)
        self.assertEqual(report["baseline_only_correct"], 4000)
        self.assertEqual(
            report["statistical_result"],
            "supports_regression_beyond_margin_under_assumptions",
        )

    def test_pairing_is_independent_of_file_order(self):
        candidate = dict(reversed(list(self.baseline.items())))
        self.assertEqual(
            evaluate(self.plan, self.baseline, candidate),
            evaluate(self.plan, self.baseline, self.baseline),
        )

    def test_contingency_counts_and_difference(self):
        a = {"a": True, "b": True, "c": False, "d": False}
        b = {"a": True, "b": False, "c": True, "d": False}
        plan = dict(self.plan, expected_items=4, item_ids_sha256=ids_digest(a))
        r = evaluate(plan, a, b)
        self.assertEqual(
            [
                r[k]
                for k in (
                    "both_correct",
                    "both_incorrect",
                    "candidate_only_correct",
                    "baseline_only_correct",
                )
            ],
            [1, 1, 1, 1],
        )
        self.assertEqual(r["candidate_minus_baseline"], 0)

    def test_missing_or_different_ids_are_errors(self):
        with self.assertRaisesRegex(ValueError, "IDs differ"):
            evaluate(self.plan, self.baseline, {"other": True})
        dropped = dict(list(self.baseline.items())[1:])
        with self.assertRaisesRegex(ValueError, "count"):
            evaluate(self.plan, dropped, dropped)
        changed = dict(dropped, replaced=True)
        with self.assertRaisesRegex(ValueError, "frozen plan"):
            evaluate(self.plan, changed, changed)

    def test_reject_invalid_plan_parameters(self):
        for key in ("alpha", "margin"):
            for value in (0, 1, -1, True, float("nan"), float("inf"), "0.05"):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    evaluate(dict(self.plan, **{key: value}), self.baseline, self.baseline)
        for changes in (
            {"expected_items": True},
            {"item_ids_sha256": "z" * 64},
            {"evidence_kind": "production"},
            {"extra": 1},
            {"candidate_id": "fixture-a"},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                evaluate(dict(self.plan, **changes), self.baseline, self.baseline)

    def test_reject_duplicate_coerced_or_malformed_scores(self):
        bad = [
            '{"id":"x","correct":true}\n{"id":"x","correct":false}\n',
            '{"id":"x","correct":1}\n',
            '{"id":" x","correct":true}\n',
            '{"id":"x","correct":true,"ignored":1}\n',
            "\n",
            "",
            "[]\n",
            "oops\n",
            '{"id":"x","correct":true,"correct":false}\n',
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.jsonl"
            for value in bad:
                path.write_text(value)
                with self.subTest(value=value), self.assertRaises(ValueError):
                    load_scores(path)

    def test_cli_provenance_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plan = root / "plan.json"
            scores = root / "scores.jsonl"
            output = root / "report.json"
            plan.write_text(json.dumps(self.plan))
            scores.write_text(
                "".join(
                    json.dumps({"id": k, "correct": v}) + "\n" for k, v in self.baseline.items()
                )
            )
            argv = [
                "--plan",
                str(plan),
                "--baseline",
                str(scores),
                "--candidate",
                str(scores),
                "--output",
                str(output),
            ]
            self.assertEqual(main(argv), 0)
            report = json.loads(output.read_text())
            self.assertEqual(
                report["input_sha256"]["baseline"],
                hashlib.sha256(scores.read_bytes()).hexdigest(),
            )
            saved = output.read_bytes()
            with self.assertRaises(SystemExit) as error:
                main(argv)
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(output.read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
