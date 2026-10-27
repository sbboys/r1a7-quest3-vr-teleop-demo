#!/usr/bin/env python3

import csv
import tempfile
import unittest
from pathlib import Path

import analyze_wire_threading_trials as analyzer


class TrialAnalyzerTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {
                "trial_id": "T001",
                "condition_id": "normal",
                "method_id": "dual",
                "duration_s": "10",
                "first_attempt_success": "1",
                "final_success": "1",
                "failure_code": "",
                "alignment_error_mm": "0.08",
            },
            {
                "trial_id": "T002",
                "condition_id": "normal",
                "method_id": "dual",
                "duration_s": "20",
                "first_attempt_success": "0",
                "final_success": "1",
                "failure_code": "",
                "alignment_error_mm": "0.12",
            },
            {
                "trial_id": "T003",
                "condition_id": "occluded",
                "method_id": "wrist",
                "duration_s": "60",
                "first_attempt_success": "0",
                "final_success": "0",
                "failure_code": "PERCEPTION_WIRE_LOST",
                "alignment_error_mm": "",
            },
        ]

    def test_summary_counts_and_metrics(self):
        result = analyzer.summarize(self.rows)
        self.assertEqual(result["n_trials"], 3)
        self.assertEqual(result["first_attempt_success"]["successes"], 1)
        self.assertEqual(result["final_success"]["successes"], 2)
        self.assertEqual(result["failure_counts"], {"PERCEPTION_WIRE_LOST": 1})
        self.assertEqual(result["metrics"]["alignment_error_mm"]["n"], 2)
        self.assertEqual(result["metrics"]["alignment_error_mm"]["p95_nearest_rank"], 0.12)

    def test_grouped_summary(self):
        result = analyzer.grouped_summary(self.rows, ["method_id"])
        self.assertEqual(result["groups"]["dual"]["n_trials"], 2)
        self.assertEqual(result["groups"]["wrist"]["n_trials"], 1)

    def test_csv_loader_rejects_duplicate_trial_id(self):
        fields = list(self.rows[0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trials.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(self.rows[0])
                writer.writerow(self.rows[0])
            with self.assertRaisesRegex(ValueError, "duplicate trial_id"):
                analyzer.load_rows(path)


if __name__ == "__main__":
    unittest.main()
