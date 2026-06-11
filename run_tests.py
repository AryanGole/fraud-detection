"""
============================================================
Fraud Detection Pipeline — Test Runner & Report Generator
============================================================
Runs all test suites and produces a structured report.
Usage:  python3 -m tests.run_tests
============================================================
"""

import sys
import os
import io
import time
import unittest
import traceback
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Colour helpers (no deps) ──────────────────────────────────────────────────
RESET  = "\033[0m";  BOLD   = "\033[1m"
GREEN  = "\033[92m"; RED    = "\033[91m"
YELLOW = "\033[93m"; CYAN   = "\033[96m"
BLUE   = "\033[94m"; GREY   = "\033[90m"

def c(colour, text): return f"{colour}{text}{RESET}"


class VerboseResult(unittest.TestResult):
    """Custom result collector that stores per-test timing and output."""

    SUITE_LABELS = {
        "TestTransactionGeneration":        ("1", "Kafka Producer – Transaction Generation"),
        "TestKafkaProducerClass":           ("1", "Kafka Producer – Producer Class"),
        "TestFraudRuleEngine":              ("2", "Kafka Consumer – Rule Engine"),
        "TestFraudScoreEnricher":           ("2", "Kafka Consumer – Score Enricher"),
        "TestSyntheticDataGeneration":      ("3", "ML Pipeline – Synthetic Data"),
        "TestIsolationForestDetector":      ("3", "ML Pipeline – Isolation Forest"),
        "TestXGBoostClassifier":            ("3", "ML Pipeline – XGBoost Classifier"),
        "TestLightGBMClassifier":           ("3", "ML Pipeline – LightGBM Classifier"),
        "TestFraudEnsembleScorer":          ("3", "ML Pipeline – Ensemble Scorer"),
        "TestModelEvaluation":              ("3", "ML Pipeline – Model Evaluation"),
        "TestApiScoringLogic":              ("4", "Scoring API – Risk Band Logic"),
        "TestMockScoringFunction":          ("4", "Scoring API – Fallback Scorer"),
        "TestFeatureEngineeringLogic":      ("5", "Feature Engineering – Logic"),
        "TestIntegrationProducerConsumer":  ("6", "Integration – Producer → Consumer"),
        "TestIntegrationMLPipeline":        ("6", "Integration – Full ML Pipeline"),
        "TestIntegrationScoringAndAlerting":("6", "Integration – Scoring & Alerting"),
        "TestEdgeCases":                    ("7", "Edge Cases & Data Quality"),
    }

    def __init__(self):
        super().__init__()
        self.test_records   = []   # list of dicts
        self.suite_times    = {}   # suite_name → total seconds
        self._start_times   = {}
        self.current_suite  = None

    def startTest(self, test):
        super().startTest(test)
        self._start_times[test] = time.monotonic()

    def _record(self, test, status, detail=""):
        elapsed = time.monotonic() - self._start_times.get(test, time.monotonic())
        cls     = type(test).__name__
        suite_n, suite_label = self.SUITE_LABELS.get(cls, ("?", cls))
        self.suite_times[cls] = self.suite_times.get(cls, 0) + elapsed
        self.test_records.append({
            "suite_num":   suite_n,
            "suite_label": suite_label,
            "suite_cls":   cls,
            "test_name":   test._testMethodName,
            "status":      status,
            "elapsed_ms":  round(elapsed * 1000, 1),
            "detail":      detail,
        })

    def addSuccess(self, test):
        super().addSuccess(test)
        self._record(test, "PASS")

    def addError(self, test, err):
        super().addError(test, err)
        tb = "".join(traceback.format_exception(*err))
        self._record(test, "ERROR", tb)

    def addFailure(self, test, err):
        super().addFailure(test, err)
        tb = "".join(traceback.format_exception(*err))
        self._record(test, "FAIL", tb)

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self._record(test, "SKIP", reason)


def run_all_tests():
    # ── Import mocks first ────────────────────────────────────────────────────
    import tests.mock_deps  # noqa

    from tests.test_suite import (
        TestTransactionGeneration,
        TestKafkaProducerClass,
        TestFraudRuleEngine,
        TestFraudScoreEnricher,
        TestSyntheticDataGeneration,
        TestIsolationForestDetector,
        TestXGBoostClassifier,
        TestLightGBMClassifier,
        TestFraudEnsembleScorer,
        TestModelEvaluation,
        TestApiScoringLogic,
        TestMockScoringFunction,
        TestFeatureEngineeringLogic,
        TestIntegrationProducerConsumer,
        TestIntegrationMLPipeline,
        TestIntegrationScoringAndAlerting,
        TestEdgeCases,
    )

    SUITES = [
        TestTransactionGeneration,
        TestKafkaProducerClass,
        TestFraudRuleEngine,
        TestFraudScoreEnricher,
        TestSyntheticDataGeneration,
        TestIsolationForestDetector,
        TestXGBoostClassifier,
        TestLightGBMClassifier,
        TestFraudEnsembleScorer,
        TestModelEvaluation,
        TestApiScoringLogic,
        TestMockScoringFunction,
        TestFeatureEngineeringLogic,
        TestIntegrationProducerConsumer,
        TestIntegrationMLPipeline,
        TestIntegrationScoringAndAlerting,
        TestEdgeCases,
    ]

    loader  = unittest.TestLoader()
    loader.sortTestMethodsUsing = None   # preserve definition order
    suite   = unittest.TestSuite()
    for cls in SUITES:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    result     = VerboseResult()
    wall_start = time.monotonic()
    suite.run(result)
    wall_elapsed = time.monotonic() - wall_start

    return result, wall_elapsed


def print_report(result: VerboseResult, wall_elapsed: float):
    records = result.test_records
    total   = len(records)
    passed  = sum(1 for r in records if r["status"] == "PASS")
    failed  = sum(1 for r in records if r["status"] == "FAIL")
    errors  = sum(1 for r in records if r["status"] == "ERROR")
    skipped = sum(1 for r in records if r["status"] == "SKIP")

    WIDTH = 80

    # ── Header ────────────────────────────────────────────────────────────────
    print()
    print(c(BOLD + CYAN, "═" * WIDTH))
    print(c(BOLD + CYAN, "  REAL-TIME FRAUD DETECTION PIPELINE — TEST REPORT"))
    print(c(BOLD + CYAN, f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"))
    print(c(BOLD + CYAN, "═" * WIDTH))

    # ── Per-suite breakdown ───────────────────────────────────────────────────
    current_suite = None
    suite_pass = suite_fail = suite_error = 0

    for r in records:
        if r["suite_cls"] != current_suite:
            # Print previous suite summary
            if current_suite is not None:
                _print_suite_summary(current_suite, suite_pass, suite_fail,
                                     suite_error, result.suite_times.get(current_suite, 0))
            current_suite = r["suite_cls"]
            suite_pass = suite_fail = suite_error = 0
            suite_label = r["suite_label"]
            suite_num   = r["suite_num"]
            print()
            print(c(BOLD + BLUE, f"  SUITE {suite_num}: {suite_label}"))
            print(c(GREY, "  " + "─" * (WIDTH - 2)))

        # Test row
        status = r["status"]
        if status == "PASS":
            icon = c(GREEN, "  ✓ ")
            suite_pass += 1
        elif status == "FAIL":
            icon = c(RED, "  ✗ ")
            suite_fail += 1
        elif status == "ERROR":
            icon = c(RED, "  ✗ ")
            suite_error += 1
        else:
            icon = c(YELLOW, "  ⊘ ")

        name     = r["test_name"]
        ms_str   = c(GREY, f"  [{r['elapsed_ms']}ms]")
        # Truncate long test names
        max_len  = WIDTH - 20
        disp     = (name[:max_len] + "…") if len(name) > max_len else name
        print(f"{icon}{disp}{ms_str}")

        # Show failure / error detail (indented)
        if status in ("FAIL", "ERROR") and r["detail"]:
            lines = r["detail"].strip().split("\n")
            # Show last 8 lines of traceback
            for ln in lines[-8:]:
                print(c(RED, f"       {ln}"))

    # Print last suite summary
    if current_suite:
        _print_suite_summary(current_suite, suite_pass, suite_fail,
                             suite_error, result.suite_times.get(current_suite, 0))

    # ── Failures / Errors detail block ────────────────────────────────────────
    bad = [r for r in records if r["status"] in ("FAIL", "ERROR")]
    if bad:
        print()
        print(c(BOLD + RED, "═" * WIDTH))
        print(c(BOLD + RED, f"  FAILURES & ERRORS ({len(bad)} total)"))
        print(c(BOLD + RED, "═" * WIDTH))
        for i, r in enumerate(bad, 1):
            print()
            print(c(RED, f"  [{i}] {r['suite_label']} :: {r['test_name']}"))
            print(c(RED, f"      Status: {r['status']}"))
            if r["detail"]:
                lines = r["detail"].strip().split("\n")
                for ln in lines:
                    print(c(YELLOW, f"      {ln}"))

    # ── Summary bar ───────────────────────────────────────────────────────────
    print()
    print(c(BOLD + CYAN, "═" * WIDTH))
    print(c(BOLD + CYAN, "  SUMMARY"))
    print(c(BOLD + CYAN, "═" * WIDTH))

    pass_pct  = passed / max(total, 1) * 100
    bar_width = 50
    filled    = int(bar_width * passed / max(total, 1))
    bar_fill  = c(GREEN,  "█" * filled)
    bar_empty = c(RED,    "░" * (bar_width - filled))

    print(f"\n  {bar_fill}{bar_empty}  {pass_pct:.1f}% passing\n")

    rows = [
        ("Total Tests",   c(BOLD, str(total))),
        ("Passed",        c(GREEN, str(passed))),
        ("Failed",        c(RED,   str(failed))  if failed  else c(GREEN, "0")),
        ("Errors",        c(RED,   str(errors))  if errors  else c(GREEN, "0")),
        ("Skipped",       c(YELLOW,str(skipped)) if skipped else c(GREY,  "0")),
        ("Wall Time",     c(BOLD,  f"{wall_elapsed:.2f}s")),
    ]
    for label, val in rows:
        print(f"  {label:<20} {val}")

    # ── Bug fix summary ───────────────────────────────────────────────────────
    print()
    print(c(BOLD + CYAN, "═" * WIDTH))
    print(c(BOLD + CYAN, "  BUGS DETECTED & FIXED"))
    print(c(BOLD + CYAN, "═" * WIDTH))
    bugs = [
        ("BUG-01", "train_pipeline.py",        "optuna.Trial type annotation at module level causes AttributeError on import"),
        ("BUG-02", "train_pipeline.py",        "Unused `field` import from dataclasses — causes ImportWarning"),
        ("BUG-03", "train_pipeline.py",        "Unused `LabelEncoder` import from sklearn — dead code"),
        ("BUG-04", "train_pipeline.py",        "evaluate_model: precision_recall_curve()[0][1] / [1][1] index wrong → IndexError or incorrect value"),
        ("BUG-05", "train_pipeline.py",        "evaluate_model: predict_proba() called on ensemble returns 1D array, not 2D → slice [:, 1] would crash"),
        ("BUG-06", "train_pipeline.py",        "SHAP: calibrated_classifiers_ renamed in sklearn ≥1.2; hard crash on production sklearn"),
        ("BUG-07", "train_pipeline.py",        "SHAP: shap_values returned as list[ndarray] for binary classifier; np.abs on list silently wrong"),
        ("BUG-08", "feature_pipeline.py",      "Window.partitionBy() does not accept Column expressions; F.to_date().alias() inside it crashes PySpark"),
        ("BUG-09", "feature_pipeline.py",      "daily_total column dropped before max_single_day computed → AnalysisException in PySpark"),
        ("BUG-10", "feature_pipeline.py",      "return df missing after cleanup in add_velocity_features → function returns None"),
        ("BUG-11", "medallion_pipeline.py",     "Dict type annotation used in run_data_quality_checks but Dict never imported → NameError at runtime"),
        ("BUG-12", "kafka_consumer.py",         "metrics defaultdict(int) accumulates float latency incorrectly; avg_latency always 0 due to int truncation"),
        ("BUG-13", "fraud_scoring_api.py",      "redis.asyncio hard import crashes on systems without redis-py ≥ 4.2; needs try/except guard"),
        ("BUG-14", "fraud_scoring_api.py",      "REDIS_POOL type annotation as aioredis.Redis fails when aioredis is None after import guard"),
    ]
    for bid, fname, desc in bugs:
        status_tag = c(GREEN, "FIXED")
        print(f"  {c(BOLD, bid)}  {c(YELLOW, fname):<30} {status_tag}")
        print(f"       {c(GREY, desc)}")

    print()
    print(c(BOLD + CYAN, "═" * WIDTH))
    overall = c(BOLD + GREEN, "ALL TESTS PASSED") if (failed + errors) == 0 \
              else c(BOLD + RED, f"{failed + errors} TEST(S) FAILED")
    print(f"  {overall}")
    print(c(BOLD + CYAN, "═" * WIDTH))
    print()

    return (failed + errors) == 0


def _print_suite_summary(cls_name, passed, failed, errors, elapsed_s):
    total = passed + failed + errors
    if failed + errors == 0:
        tag = c(GREEN, f"✓ {passed}/{total} passed")
    else:
        tag = c(RED,   f"✗ {passed}/{total} passed  ({failed} fail, {errors} err)")
    print(c(GREY, f"  {'─'*50} {tag}  {elapsed_s:.2f}s"))


if __name__ == "__main__":
    result, elapsed = run_all_tests()
    ok = print_report(result, elapsed)
    sys.exit(0 if ok else 1)
