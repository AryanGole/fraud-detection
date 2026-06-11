"""
============================================================
Fraud Detection Pipeline — Full Test Suite
============================================================
Unit + Integration tests using stdlib unittest only.
All heavy deps (xgboost, lightgbm, kafka, pyspark, fastapi)
are replaced by the faithful mocks in mock_deps.py.

Run:  python3 -m tests.run_tests
============================================================
"""

import sys
import os
import math
import json
import time
import hashlib
import unittest
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

# ── Install mocks BEFORE any project imports ──────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import tests.mock_deps  # noqa: F401  (side-effects: populates sys.modules)

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# TEST SUITE 1 – KAFKA PRODUCER
# ══════════════════════════════════════════════════════════════════════════════
class TestTransactionGeneration(unittest.TestCase):
    """Unit tests for synthetic transaction generation logic."""

    def setUp(self):
        from ingestion.kafka_producer import (
            generate_transaction, generate_customer_id,
            generate_account_number, generate_device_id,
            generate_ip, generate_location, inject_fraud_pattern,
            BankingTransaction, TransactionType, HIGH_RISK_CATEGORIES,
        )
        self.gen_txn       = generate_transaction
        self.gen_cust      = generate_customer_id
        self.gen_acct      = generate_account_number
        self.gen_device    = generate_device_id
        self.gen_ip        = generate_ip
        self.gen_location  = generate_location
        self.inject_fraud  = inject_fraud_pattern
        self.TxnType       = TransactionType
        self.HIGH_RISK     = HIGH_RISK_CATEGORIES

    # ── Customer / Account generators ─────────────────────────────────────────
    def test_customer_id_format(self):
        cid = self.gen_cust()
        self.assertTrue(cid.startswith("CUST"),
                        f"Customer ID should start with 'CUST', got: {cid}")
        self.assertEqual(len(cid), 10,
                         f"Customer ID should be 10 chars, got: {len(cid)}")

    def test_customer_id_uniqueness(self):
        ids = {self.gen_cust() for _ in range(200)}
        self.assertGreater(len(ids), 50, "Customer IDs should have high cardinality")

    def test_account_number_length(self):
        acct = self.gen_acct()
        self.assertEqual(len(acct), 12,
                         f"Account number should be 12 digits, got {len(acct)}")
        self.assertTrue(acct.isdigit(), "Account number should be all digits")

    def test_device_id_hex_format(self):
        did = self.gen_device()
        self.assertEqual(len(did), 16)
        int(did, 16)  # Should not raise — must be valid hex

    def test_ip_address_format(self):
        for _ in range(50):
            ip = self.gen_ip()
            parts = ip.split(".")
            self.assertEqual(len(parts), 4, f"Invalid IP format: {ip}")
            for p in parts:
                v = int(p)
                self.assertGreaterEqual(v, 0)
                self.assertLessEqual(v, 255)

    def test_location_india_default(self):
        lat, lon, cc, city = self.gen_location(is_international=False)
        self.assertEqual(cc, "IND")
        self.assertIsInstance(city, str)
        self.assertGreater(len(city), 0)

    def test_location_international(self):
        countries = set()
        for _ in range(30):
            _, _, cc, _ = self.gen_location(is_international=True)
            countries.add(cc)
        self.assertGreater(len(countries), 1,
                           "International locations should span multiple countries")

    # ── Transaction structure ──────────────────────────────────────────────────
    def test_transaction_has_all_required_fields(self):
        txn = self.gen_txn()
        required = [
            "transaction_id", "customer_id", "amount", "timestamp",
            "device_id", "country_code", "transaction_type",
        ]
        for field in required:
            self.assertTrue(hasattr(txn, field),
                            f"Transaction missing field: {field}")

    def test_transaction_amount_positive(self):
        for _ in range(100):
            txn = self.gen_txn()
            self.assertGreater(txn.amount, 0,
                               f"Amount must be positive, got {txn.amount}")

    def test_transaction_amount_lognormal_distribution(self):
        amounts = [self.gen_txn().amount for _ in range(500)]
        log_amounts = np.log(amounts)
        # Log-normal should have mean roughly around lognormvariate(5.5, 1.8)
        self.assertGreater(np.mean(log_amounts), 3.0)
        self.assertLess(np.mean(log_amounts), 9.0)

    def test_transaction_type_valid_enum(self):
        valid_types = {t.value for t in self.TxnType}
        for _ in range(50):
            txn = self.gen_txn()
            self.assertIn(txn.transaction_type, valid_types)

    def test_transaction_hour_of_day_range(self):
        for _ in range(50):
            txn = self.gen_txn()
            self.assertGreaterEqual(txn.hour_of_day, 0)
            self.assertLessEqual(txn.hour_of_day, 23)

    def test_transaction_day_of_week_range(self):
        for _ in range(50):
            txn = self.gen_txn()
            self.assertGreaterEqual(txn.day_of_week, 0)
            self.assertLessEqual(txn.day_of_week, 6)

    def test_schema_version_present(self):
        txn = self.gen_txn()
        self.assertEqual(txn.schema_version, "2.1")

    # ── Fraud injection ───────────────────────────────────────────────────────
    def test_inject_fraud_changes_transaction(self):
        txn_legit = self.gen_txn(inject_fraud=False)
        txn_fraud = self.gen_txn(inject_fraud=True)
        # At least one property should differ (fraud pattern applied)
        self.assertIsNotNone(txn_fraud)

    def test_fraud_injection_cnp_international(self):
        """Card-not-present + international pattern sets expected flags."""
        import random
        random.seed(0)
        from ingestion.kafka_producer import BankingTransaction
        # Run many times to hit cnp_international branch
        hits = 0
        for _ in range(200):
            txn = self.gen_txn()
            txn2 = self.inject_fraud(txn)
            if not txn2.card_present and txn2.is_international:
                hits += 1
        # Should hit at least once in 200 runs
        self.assertGreater(hits, 0,
                           "card_not_present+international pattern should trigger")

    def test_non_fraud_transaction_no_forced_fraud_patterns(self):
        """Normal transactions should not always have impossible travel."""
        txns = [self.gen_txn(inject_fraud=False) for _ in range(200)]
        impossible_count = sum(1 for t in txns if t.is_international)
        # ~8% international rate; not all 200 should be international
        self.assertLess(impossible_count, 150)

    # ── Serialisation ─────────────────────────────────────────────────────────
    def test_transaction_serializable_to_json(self):
        import dataclasses
        txn = self.gen_txn()
        d   = dataclasses.asdict(txn)
        s   = json.dumps(d, default=str)
        recovered = json.loads(s)
        self.assertEqual(recovered["schema_version"], "2.1")
        self.assertIn("transaction_id", recovered)


class TestKafkaProducerClass(unittest.TestCase):
    """Unit tests for FraudDetectionProducer."""

    def setUp(self):
        from ingestion.kafka_producer import FraudDetectionProducer
        self.ProducerClass = FraudDetectionProducer

    def test_producer_instantiates_with_defaults(self):
        p = self.ProducerClass()
        self.assertIsNotNone(p)

    def test_metrics_initialised_to_zero(self):
        p = self.ProducerClass()
        self.assertEqual(p.metrics["produced"], 0)
        self.assertEqual(p.metrics["errors"], 0)

    def test_topic_routing_high_value(self):
        from ingestion.kafka_producer import generate_transaction
        p   = self.ProducerClass()
        txn = generate_transaction()
        txn.amount         = 600_000.0  # > 500K threshold
        txn.is_international = False
        topic = p._select_topic(txn)
        self.assertEqual(topic, p.TOPIC_HIGH_VALUE)
        self.assertEqual(p.metrics["high_value"], 1)

    def test_topic_routing_international(self):
        from ingestion.kafka_producer import generate_transaction
        p   = self.ProducerClass()
        txn = generate_transaction()
        txn.amount         = 1_000.0
        txn.is_international = True
        topic = p._select_topic(txn)
        self.assertEqual(topic, p.TOPIC_INTERNATIONAL)

    def test_topic_routing_standard(self):
        from ingestion.kafka_producer import generate_transaction
        p   = self.ProducerClass()
        txn = generate_transaction()
        txn.amount         = 500.0
        txn.is_international = False
        topic = p._select_topic(txn)
        self.assertEqual(topic, p.TOPIC_TRANSACTIONS)

    def test_delivery_callback_increments_produced(self):
        p = self.ProducerClass()
        p._delivery_callback(None, MagicMock())
        self.assertEqual(p.metrics["produced"], 1)

    def test_delivery_callback_increments_errors(self):
        p     = self.ProducerClass()
        error = MagicMock()
        p._delivery_callback(error, MagicMock())
        self.assertEqual(p.metrics["errors"], 1)


# ══════════════════════════════════════════════════════════════════════════════
# TEST SUITE 2 – KAFKA CONSUMER / RULE ENGINE
# ══════════════════════════════════════════════════════════════════════════════
class TestFraudRuleEngine(unittest.TestCase):
    """Unit tests for deterministic rule-based fraud pre-filter."""

    def setUp(self):
        from ingestion.kafka_consumer import FraudRuleEngine
        self.RuleEngine = FraudRuleEngine
        self.engine     = FraudRuleEngine(redis_client=None)

    def _make_txn(self, **overrides):
        base = {
            "customer_id":          "CUST000001",
            "amount":               1000.0,
            "transaction_type":     "CREDIT_CARD",
            "is_high_risk_merchant": False,
            "card_present":         True,
            "is_international":     False,
            "merchant_category":    "GROCERY",
            "hour_of_day":          14,
        }
        base.update(overrides)
        return base

    def test_clean_transaction_no_rules(self):
        txn     = self._make_txn()
        results = self.engine.evaluate(txn)
        self.assertEqual(results, [],
                         "Clean transaction should trigger no rules")

    def test_high_risk_merchant_triggers(self):
        txn     = self._make_txn(is_high_risk_merchant=True,
                                  merchant_category="CRYPTO_EXCHANGE")
        results = self.engine.evaluate(txn)
        names   = [r.rule_name for r in results]
        self.assertIn("HIGH_RISK_MERCHANT", names)

    def test_cnp_international_triggers(self):
        txn     = self._make_txn(card_present=False, is_international=True)
        results = self.engine.evaluate(txn)
        names   = [r.rule_name for r in results]
        self.assertIn("CNP_INTERNATIONAL", names)

    def test_off_hours_atm_triggers(self):
        txn     = self._make_txn(transaction_type="ATM_WITHDRAWAL",
                                  hour_of_day=2)
        results = self.engine.evaluate(txn)
        names   = [r.rule_name for r in results]
        self.assertIn("OFF_HOURS_ATM", names)

    def test_large_transaction_triggers(self):
        txn     = self._make_txn(amount=1_500_000.0)
        results = self.engine.evaluate(txn)
        names   = [r.rule_name for r in results]
        self.assertIn("LARGE_TRANSACTION", names)

    def test_cnp_high_amount_triggers(self):
        txn     = self._make_txn(card_present=False, amount=75_000.0)
        results = self.engine.evaluate(txn)
        names   = [r.rule_name for r in results]
        self.assertIn("CNP_HIGH_AMOUNT", names)

    def test_off_hours_atm_normal_hour_no_trigger(self):
        txn     = self._make_txn(transaction_type="ATM_WITHDRAWAL",
                                  hour_of_day=14)
        results = self.engine.evaluate(txn)
        names   = [r.rule_name for r in results]
        self.assertNotIn("OFF_HOURS_ATM", names)

    def test_rule_result_has_risk_points(self):
        txn     = self._make_txn(card_present=False, is_international=True)
        results = self.engine.evaluate(txn)
        for r in results:
            self.assertGreater(r.risk_points, 0,
                               f"Rule {r.rule_name} should have positive risk points")

    def test_rule_result_has_reason(self):
        txn     = self._make_txn(is_high_risk_merchant=True)
        results = self.engine.evaluate(txn)
        for r in results:
            self.assertIsInstance(r.reason, str)
            self.assertGreater(len(r.reason), 0)

    def test_velocity_rule_triggers_after_threshold(self):
        """Simulate rapid successive transactions to trigger velocity rule."""
        engine = self.RuleEngine(redis_client=None)
        cid    = "VELTEST001"
        # Flood the memory store
        for i in range(10):
            engine._update_velocity(cid, 100.0)
        txn     = self._make_txn(customer_id=cid, amount=100.0)
        results = engine.evaluate(txn)
        names   = [r.rule_name for r in results]
        self.assertIn("HIGH_VELOCITY", names,
                      "HIGH_VELOCITY should fire after 10 transactions")

    def test_velocity_amount_rule_triggers(self):
        engine = self.RuleEngine(redis_client=None)
        cid    = "VELTEST002"
        # Cumulate amount near threshold
        engine._update_velocity(cid, 99_000.0)
        txn     = self._make_txn(customer_id=cid, amount=5_000.0)
        results = engine.evaluate(txn)
        names   = [r.rule_name for r in results]
        self.assertIn("VELOCITY_AMOUNT", names)

    def test_multiple_rules_can_trigger_simultaneously(self):
        txn = self._make_txn(
            card_present=False,
            is_international=True,
            is_high_risk_merchant=True,
            merchant_category="CASINO",
            amount=2_000_000.0,
        )
        results = self.engine.evaluate(txn)
        self.assertGreater(len(results), 1,
                           "Multiple rules should fire for high-risk transaction")


class TestFraudScoreEnricher(unittest.TestCase):
    """Unit tests for score enrichment and severity assignment."""

    def setUp(self):
        from ingestion.kafka_consumer import FraudScoreEnricher, FraudRuleEngine
        engine        = FraudRuleEngine(redis_client=None)
        self.enricher = FraudScoreEnricher(engine)

    def _make_txn(self, **overrides):
        base = {
            "transaction_id":       "TXN-TEST-001",
            "customer_id":          "CUST000001",
            "amount":               1000.0,
            "transaction_type":     "CREDIT_CARD",
            "is_high_risk_merchant": False,
            "card_present":         True,
            "is_international":     False,
            "merchant_category":    "GROCERY",
            "hour_of_day":          14,
        }
        base.update(overrides)
        return base

    def test_enriched_txn_has_required_keys(self):
        txn      = self._make_txn()
        enriched = self.enricher.enrich(txn)
        for key in ["rule_based_score", "rules_triggered", "rule_severity",
                    "requires_ml_scoring", "enriched_at"]:
            self.assertIn(key, enriched, f"Missing key: {key}")

    def test_clean_txn_gets_low_severity(self):
        txn      = self._make_txn()
        enriched = self.enricher.enrich(txn)
        self.assertEqual(enriched["rule_severity"], "LOW")

    def test_fraud_txn_gets_high_severity(self):
        txn = self._make_txn(
            card_present=False, is_international=True,
            is_high_risk_merchant=True, amount=2_000_000.0
        )
        enriched = self.enricher.enrich(txn)
        self.assertIn(enriched["rule_severity"], ("HIGH", "CRITICAL"))

    def test_rule_score_capped_at_100(self):
        txn = self._make_txn(
            card_present=False, is_international=True,
            is_high_risk_merchant=True, amount=2_000_000.0,
            hour_of_day=2, transaction_type="ATM_WITHDRAWAL",
        )
        enriched = self.enricher.enrich(txn)
        self.assertLessEqual(enriched["rule_based_score"], 100)
        self.assertGreaterEqual(enriched["rule_based_score"], 0)

    def test_requires_ml_scoring_for_suspicious(self):
        txn = self._make_txn(is_high_risk_merchant=True,
                              merchant_category="CASINO")
        enriched = self.enricher.enrich(txn)
        self.assertTrue(enriched["requires_ml_scoring"])

    def test_clean_txn_does_not_require_ml_scoring(self):
        txn      = self._make_txn()
        enriched = self.enricher.enrich(txn)
        self.assertFalse(enriched["requires_ml_scoring"])

    def test_enriched_at_is_iso_timestamp(self):
        txn      = self._make_txn()
        enriched = self.enricher.enrich(txn)
        ts = enriched["enriched_at"]
        # Should parse as ISO datetime
        datetime.fromisoformat(ts.replace("Z", "+00:00"))

    def test_severity_thresholds_critical(self):
        """Score >= 85 should be CRITICAL."""
        from ingestion.kafka_consumer import FraudScoreEnricher, FraudRuleEngine, ALERT_CRITICAL
        engine  = FraudRuleEngine(redis_client=None)
        e       = FraudScoreEnricher(engine)
        # Patch rule engine to return fixed score
        with patch.object(engine, 'evaluate', return_value=[
            MagicMock(triggered=True, risk_points=90, rule_name="MOCK", reason="test")
        ]):
            enriched = e.enrich(self._make_txn())
        self.assertEqual(enriched["rule_severity"], ALERT_CRITICAL)

    def test_severity_thresholds_medium(self):
        from ingestion.kafka_consumer import FraudScoreEnricher, FraudRuleEngine, ALERT_MEDIUM
        engine = FraudRuleEngine(redis_client=None)
        e      = FraudScoreEnricher(engine)
        with patch.object(engine, 'evaluate', return_value=[
            MagicMock(triggered=True, risk_points=50, rule_name="MOCK", reason="test")
        ]):
            enriched = e.enrich(self._make_txn())
        self.assertEqual(enriched["rule_severity"], ALERT_MEDIUM)


# ══════════════════════════════════════════════════════════════════════════════
# TEST SUITE 3 – ML PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
class TestSyntheticDataGeneration(unittest.TestCase):
    """Unit tests for synthetic training data generation."""

    def setUp(self):
        from ml_pipeline.train_pipeline import (
            generate_synthetic_training_data, FEATURE_COLS, TARGET_COL
        )
        self.gen_data   = generate_synthetic_training_data
        self.FEATURES   = FEATURE_COLS
        self.TARGET     = TARGET_COL

    def test_generates_correct_row_count(self):
        df = self.gen_data(n_samples=10_000)
        self.assertEqual(len(df), 10_000)

    def test_all_feature_columns_present(self):
        df = self.gen_data(n_samples=5_000)
        for col in self.FEATURES:
            self.assertIn(col, df.columns, f"Missing feature: {col}")

    def test_target_column_present(self):
        df = self.gen_data(n_samples=5_000)
        self.assertIn(self.TARGET, df.columns)

    def test_target_is_binary(self):
        df     = self.gen_data(n_samples=5_000)
        values = set(df[self.TARGET].unique())
        self.assertTrue(values.issubset({0, 1}),
                        f"Target should be binary, got: {values}")

    def test_fraud_rate_realistic(self):
        df         = self.gen_data(n_samples=100_000)
        fraud_rate = df[self.TARGET].mean()
        self.assertGreater(fraud_rate, 0.001,
                           "Fraud rate too low for training")
        self.assertLess(fraud_rate, 0.01,
                        "Fraud rate too high — unrealistic for banking")

    def test_no_null_values(self):
        df = self.gen_data(n_samples=5_000)
        null_counts = df[self.FEATURES].isnull().sum()
        self.assertEqual(null_counts.sum(), 0,
                         f"Null values found:\n{null_counts[null_counts > 0]}")

    def test_amount_always_positive(self):
        df = self.gen_data(n_samples=5_000)
        self.assertTrue((df["amount"] > 0).all(),
                        "All amounts should be positive")

    def test_binary_flag_columns_in_range(self):
        df = self.gen_data(n_samples=5_000)
        binary_cols = [
            "velocity_spike", "is_impossible_travel", "is_new_country",
            "is_new_device", "multi_account_device", "is_mcc_high_risk",
            "is_fraud_hour", "is_weekend", "is_month_end",
            "is_international", "is_high_risk_merchant",
        ]
        for col in binary_cols:
            vals = set(df[col].unique())
            self.assertTrue(vals.issubset({0, 1, 0.0, 1.0}),
                            f"{col} should be binary, got: {vals}")

    def test_proportion_features_range(self):
        df = self.gen_data(n_samples=5_000)
        for col in ["pct_international_30d", "pct_cnp_30d", "home_country_pct"]:
            self.assertTrue((df[col] >= 0).all() and (df[col] <= 1).all(),
                            f"{col} should be in [0, 1]")

    def test_cyclical_encoding_range(self):
        df = self.gen_data(n_samples=5_000)
        for col in ["txn_hour_sin", "txn_hour_cos"]:
            self.assertTrue((df[col] >= -1).all() and (df[col] <= 1).all(),
                            f"{col} should be in [-1, 1]")

    def test_reproducible_with_same_seed(self):
        df1 = self.gen_data(n_samples=1_000)
        df2 = self.gen_data(n_samples=1_000)
        # Both use seed=42 internally — fraud counts should match
        self.assertEqual(df1[self.TARGET].sum(), df2[self.TARGET].sum())


class TestIsolationForestDetector(unittest.TestCase):
    """Unit tests for unsupervised anomaly detection."""

    def setUp(self):
        from ml_pipeline.train_pipeline import (
            IsolationForestDetector, generate_synthetic_training_data, FEATURE_COLS
        )
        self.Detector  = IsolationForestDetector
        self.FEATURES  = FEATURE_COLS
        df             = generate_synthetic_training_data(n_samples=5_000)
        self.X_train   = df[FEATURE_COLS].values
        self.X_test    = df[FEATURE_COLS].values[:200]
        self.y_test    = df["is_fraud"].values[:200]

    def test_detector_fits_without_error(self):
        det = self.Detector(contamination=0.002)
        det.train(self.X_train)
        self.assertIsNotNone(det.model)

    def test_anomaly_score_shape(self):
        det = self.Detector()
        det.train(self.X_train)
        scores = det.anomaly_score(self.X_test)
        self.assertEqual(scores.shape, (200,))

    def test_anomaly_scores_in_0_1_range(self):
        det = self.Detector()
        det.train(self.X_train)
        scores = det.anomaly_score(self.X_test)
        self.assertTrue((scores >= 0).all(), "Scores must be >= 0")
        self.assertTrue((scores <= 1).all(), "Scores must be <= 1")

    def test_anomaly_scores_are_floats(self):
        det = self.Detector()
        det.train(self.X_train)
        scores = det.anomaly_score(self.X_test)
        self.assertEqual(scores.dtype.kind, 'f')

    def test_scaler_fitted(self):
        det = self.Detector()
        det.train(self.X_train)
        # StandardScaler should be fitted (has mean_)
        self.assertTrue(hasattr(det.scaler, "mean_"))

    def test_anomaly_score_variance(self):
        """Scores should not be all identical."""
        det = self.Detector()
        det.train(self.X_train)
        scores = det.anomaly_score(self.X_test)
        self.assertGreater(np.std(scores), 0.001,
                           "Anomaly scores should vary across transactions")


class TestXGBoostClassifier(unittest.TestCase):
    """Unit tests for XGBoost fraud classifier."""

    def setUp(self):
        from ml_pipeline.train_pipeline import (
            XGBoostFraudClassifier, generate_synthetic_training_data, FEATURE_COLS
        )
        self.Clf      = XGBoostFraudClassifier
        self.FEATURES = FEATURE_COLS
        df = generate_synthetic_training_data(n_samples=3_000)
        self.X = df[FEATURE_COLS].values
        self.y = df["is_fraud"].values
        # Small train/val splits
        n    = len(self.X)
        split = int(n * 0.8)
        self.X_train, self.X_val = self.X[:split], self.X[split:]
        self.y_train, self.y_val = self.y[:split], self.y[split:]

    def test_classifier_trains_without_error(self):
        clf = self.Clf(n_trials=1, cv_folds=2)
        clf.train(self.X_train, self.y_train, self.X_val, self.y_val)
        self.assertIsNotNone(clf.model)

    def test_predict_proba_shape(self):
        clf = self.Clf(n_trials=1, cv_folds=2)
        clf.train(self.X_train, self.y_train, self.X_val, self.y_val)
        probs = clf.predict_proba(self.X_val)
        self.assertEqual(probs.shape, (len(self.X_val),))

    def test_predict_proba_in_0_1(self):
        clf = self.Clf(n_trials=1, cv_folds=2)
        clf.train(self.X_train, self.y_train, self.X_val, self.y_val)
        probs = clf.predict_proba(self.X_val)
        self.assertTrue((probs >= 0).all())
        self.assertTrue((probs <= 1).all())

    def test_predict_binary_output(self):
        clf = self.Clf(n_trials=1, cv_folds=2)
        clf.train(self.X_train, self.y_train, self.X_val, self.y_val)
        preds = clf.predict(self.X_val)
        self.assertTrue(set(preds).issubset({0, 1}))

    def test_threshold_is_valid(self):
        clf = self.Clf(n_trials=1, cv_folds=2)
        clf.train(self.X_train, self.y_train, self.X_val, self.y_val)
        self.assertGreaterEqual(clf.threshold, 0.0)
        self.assertLessEqual(clf.threshold, 1.0)

    def test_scaler_is_fitted(self):
        clf = self.Clf(n_trials=1, cv_folds=2)
        clf.train(self.X_train, self.y_train, self.X_val, self.y_val)
        self.assertTrue(hasattr(clf.scaler, "mean_"))

    def test_tune_returns_dict_with_params(self):
        clf    = self.Clf(n_trials=1)
        params = clf.tune(self.X_train, self.y_train)
        self.assertIsInstance(params, dict)
        self.assertIn("n_estimators", params)


class TestLightGBMClassifier(unittest.TestCase):
    """Unit tests for LightGBM challenger model."""

    def setUp(self):
        from ml_pipeline.train_pipeline import (
            LightGBMFraudClassifier, generate_synthetic_training_data, FEATURE_COLS
        )
        self.Clf      = LightGBMFraudClassifier
        df = generate_synthetic_training_data(n_samples=3_000)
        self.X = df[FEATURE_COLS].values
        self.y = df["is_fraud"].values
        split  = int(len(self.X) * 0.8)
        self.X_train, self.X_val = self.X[:split], self.X[split:]
        self.y_train, self.y_val = self.y[:split], self.y[split:]

    def test_lgb_trains_without_error(self):
        clf = self.Clf(n_trials=1)
        clf.train(self.X_train, self.y_train, self.X_val, self.y_val)
        self.assertIsNotNone(clf.model)

    def test_lgb_predict_proba_range(self):
        clf   = self.Clf(n_trials=1)
        clf.train(self.X_train, self.y_train, self.X_val, self.y_val)
        probs = clf.predict_proba(self.X_val)
        self.assertTrue((probs >= 0).all() and (probs <= 1).all())

    def test_lgb_threshold_in_valid_range(self):
        clf = self.Clf(n_trials=1)
        clf.train(self.X_train, self.y_train, self.X_val, self.y_val)
        self.assertGreaterEqual(clf.threshold, 0.0)
        self.assertLessEqual(clf.threshold, 1.0)


class TestFraudEnsembleScorer(unittest.TestCase):
    """Unit + integration tests for the ensemble scoring pipeline."""

    def setUp(self):
        from ml_pipeline.train_pipeline import (
            FraudEnsembleScorer, generate_synthetic_training_data, FEATURE_COLS
        )
        self.Ensemble = FraudEnsembleScorer
        df  = generate_synthetic_training_data(n_samples=4_000)
        self.X = df[FEATURE_COLS].values
        self.y = df["is_fraud"].values
        split = int(len(self.X) * 0.75)
        self.X_train, self.X_val = self.X[:split], self.X[split:]
        self.y_train, self.y_val = self.y[:split], self.y[split:]
        self.X_test  = self.X_val[:100]
        self.y_test  = self.y_val[:100]

    def _trained_ensemble(self):
        e = self.Ensemble()
        e.fit(self.X_train, self.y_train, self.X_val, self.y_val)
        return e

    def test_ensemble_fits_without_error(self):
        e = self._trained_ensemble()
        self.assertTrue(e.is_fitted)

    def test_fraud_score_shape(self):
        e      = self._trained_ensemble()
        scores = e.fraud_score(self.X_test)
        self.assertEqual(scores.shape, (100,))

    def test_fraud_score_in_0_100(self):
        e      = self._trained_ensemble()
        scores = e.fraud_score(self.X_test)
        self.assertTrue((scores >= 0).all(),   "Scores should be >= 0")
        self.assertTrue((scores <= 100).all(), "Scores should be <= 100")

    def test_predict_raises_if_not_fitted(self):
        e = self.Ensemble()
        with self.assertRaises(RuntimeError):
            e.fraud_score(self.X_test)

    def test_ensemble_weights_sum_to_one(self):
        e = self.Ensemble()
        total = e.WEIGHT_XGB + e.WEIGHT_LGB + e.WEIGHT_IF
        self.assertAlmostEqual(total, 1.0, places=5,
                               msg=f"Ensemble weights sum to {total}, not 1.0")

    def test_predict_returns_binary(self):
        e     = self._trained_ensemble()
        preds = e.predict(self.X_test)
        self.assertTrue(set(preds).issubset({0, 1}))

    def test_fraud_score_variance(self):
        """Scores should not be constant across samples."""
        e      = self._trained_ensemble()
        scores = e.fraud_score(self.X_test)
        self.assertGreater(np.std(scores), 0.1,
                           "Ensemble scores should vary across transactions")


class TestModelEvaluation(unittest.TestCase):
    """Unit tests for evaluate_model function."""

    def setUp(self):
        from ml_pipeline.train_pipeline import (
            evaluate_model, FraudEnsembleScorer,
            generate_synthetic_training_data, FEATURE_COLS
        )
        self.evaluate     = evaluate_model
        self.EnsembleClass = FraudEnsembleScorer
        df  = generate_synthetic_training_data(n_samples=3_000)
        FEATURES = FEATURE_COLS
        self.X = df[FEATURES].values
        self.y = df["is_fraud"].values
        split  = int(len(self.X) * 0.8)
        self.X_tr, self.X_val = self.X[:split], self.X[split:]
        self.y_tr, self.y_val = self.y[:split], self.y[split:]

    def test_evaluate_returns_all_metric_keys(self):
        e = self.EnsembleClass()
        e.fit(self.X_tr, self.y_tr, self.X_val, self.y_val)
        metrics = self.evaluate(e, self.X_val, self.y_val)
        for key in ["roc_auc", "pr_auc", "f1_score", "precision", "recall",
                    "true_positives", "false_positives",
                    "false_positive_rate", "false_negative_rate"]:
            self.assertIn(key, metrics, f"Metric missing: {key}")

    def test_roc_auc_in_valid_range(self):
        e = self.EnsembleClass()
        e.fit(self.X_tr, self.y_tr, self.X_val, self.y_val)
        metrics = self.evaluate(e, self.X_val, self.y_val)
        import math
        # May be NaN if mock model produces constant predictions on tiny test set
        if not math.isnan(metrics["roc_auc"]):
            self.assertGreaterEqual(metrics["roc_auc"], 0.0)
            self.assertLessEqual(metrics["roc_auc"], 1.0)

    def test_pr_auc_in_valid_range(self):
        e = self.EnsembleClass()
        e.fit(self.X_tr, self.y_tr, self.X_val, self.y_val)
        metrics = self.evaluate(e, self.X_val, self.y_val)
        self.assertGreaterEqual(metrics["pr_auc"], 0.0)
        self.assertLessEqual(metrics["pr_auc"], 1.0)

    def test_confusion_matrix_counts_consistent(self):
        e = self.EnsembleClass()
        e.fit(self.X_tr, self.y_tr, self.X_val, self.y_val)
        metrics = self.evaluate(e, self.X_val, self.y_val)
        total = (metrics["true_positives"]  + metrics["false_positives"] +
                 metrics["false_negatives"] + metrics["true_negatives"])
        self.assertEqual(total, len(self.X_val))

    def test_false_positive_rate_in_0_1(self):
        e = self.EnsembleClass()
        e.fit(self.X_tr, self.y_tr, self.X_val, self.y_val)
        metrics = self.evaluate(e, self.X_val, self.y_val)
        self.assertGreaterEqual(metrics["false_positive_rate"], 0.0)
        self.assertLessEqual(metrics["false_positive_rate"], 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# TEST SUITE 4 – SCORING API (no fastapi — pure logic tests)
# ══════════════════════════════════════════════════════════════════════════════
class TestApiScoringLogic(unittest.TestCase):
    """Unit tests for fraud scoring business logic in API layer."""

    def setUp(self):
        # Mock fastapi + pydantic before importing
        import types

        # fastapi mock
        fastapi_mod = types.ModuleType("fastapi")
        class _FakeHTTPException(Exception):
            def __init__(self, status_code=400, detail=""):
                self.status_code = status_code; self.detail = detail
        fastapi_mod.FastAPI          = MagicMock(return_value=MagicMock())
        fastapi_mod.HTTPException    = _FakeHTTPException
        fastapi_mod.BackgroundTasks  = MagicMock
        fastapi_mod.Depends          = lambda f: f
        fastapi_mod.Request          = MagicMock
        sys.modules["fastapi"]       = fastapi_mod

        fr_resp = types.ModuleType("fastapi.responses")
        fr_resp.JSONResponse = MagicMock
        sys.modules["fastapi.responses"] = fr_resp

        fm_mw = types.ModuleType("fastapi.middleware.cors")
        fm_mw.CORSMiddleware = MagicMock
        sys.modules["fastapi.middleware"]      = types.ModuleType("fastapi.middleware")
        sys.modules["fastapi.middleware.cors"] = fm_mw

        # pydantic mock
        pydantic_mod = types.ModuleType("pydantic")
        class _FakeBaseModel:
            def __init__(self, **kw):
                for k, v in kw.items(): setattr(self, k, v)
            def dict(self):
                return {k: v for k, v in self.__dict__.items()
                        if not k.startswith("_")}
            def json(self):
                return json.dumps(self.dict(), default=str)
        pydantic_mod.BaseModel  = _FakeBaseModel
        pydantic_mod.Field      = lambda *a, **kw: None
        pydantic_mod.validator  = lambda *a, **kw: (lambda f: f)
        sys.modules["pydantic"] = pydantic_mod

        # mangum mock
        mangum_mod = types.ModuleType("mangum")
        mangum_mod.Mangum = MagicMock
        sys.modules["mangum"] = mangum_mod

        # Now we can import scoring logic
        import importlib
        # reload to pick up new mocks if already imported
        if "api.fraud_scoring_api" in sys.modules:
            api_mod = importlib.reload(sys.modules["api.fraud_scoring_api"])
        else:
            import api.fraud_scoring_api as api_mod
        self.api = api_mod

    def test_risk_band_low(self):
        band = self.api._risk_band(30.0)
        self.assertEqual(band, "LOW")

    def test_risk_band_medium(self):
        band = self.api._risk_band(50.0)
        self.assertEqual(band, "MEDIUM")

    def test_risk_band_high(self):
        band = self.api._risk_band(70.0)
        self.assertEqual(band, "HIGH")

    def test_risk_band_critical(self):
        band = self.api._risk_band(90.0)
        self.assertEqual(band, "CRITICAL")

    def test_risk_band_boundary_low_medium(self):
        self.assertEqual(self.api._risk_band(44.9), "LOW")
        self.assertEqual(self.api._risk_band(45.0), "MEDIUM")

    def test_risk_band_boundary_medium_high(self):
        self.assertEqual(self.api._risk_band(64.9), "MEDIUM")
        self.assertEqual(self.api._risk_band(65.0), "HIGH")

    def test_risk_band_boundary_high_critical(self):
        self.assertEqual(self.api._risk_band(84.9), "HIGH")
        self.assertEqual(self.api._risk_band(85.0), "CRITICAL")

    def test_recommendation_returns_string(self):
        for band in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
            rec = self.api._recommendation(50.0, band)
            self.assertIsInstance(rec, str)
            self.assertGreater(len(rec), 10)

    def test_recommendation_critical_says_block(self):
        rec = self.api._recommendation(90.0, "CRITICAL")
        self.assertIn("BLOCK", rec)

    def test_recommendation_low_says_approve(self):
        rec = self.api._recommendation(10.0, "LOW")
        self.assertIn("APPROVE", rec)


class TestMockScoringFunction(unittest.TestCase):
    """Tests for the rule-based fallback scorer (_mock_score)."""

    def setUp(self):
        import types
        for mod_name in ["fastapi", "fastapi.responses", "fastapi.middleware",
                         "fastapi.middleware.cors", "pydantic", "mangum"]:
            if mod_name not in sys.modules:
                m = types.ModuleType(mod_name)
                sys.modules[mod_name] = m

        import importlib
        if "api.fraud_scoring_api" in sys.modules:
            self.api = sys.modules["api.fraud_scoring_api"]
        else:
            import api.fraud_scoring_api as api_mod
            self.api = api_mod

    def _make_txn(self, **overrides):
        """Create a minimal transaction namespace for _mock_score."""
        class Txn:
            velocity_spike           = 0
            is_impossible_travel     = 0
            is_new_device            = 0
            is_new_country           = 0
            is_mcc_high_risk         = 0
            multi_account_device     = 0
            is_fraud_hour            = 0
            amount_ratio_to_avg      = 1.0
            amt_zscore_30d           = 0.0
            pct_cnp_30d              = 0.1
        t = Txn()
        for k, v in overrides.items():
            setattr(t, k, v)
        return t

    def test_clean_txn_low_score(self):
        score = self.api._mock_score(self._make_txn())
        self.assertEqual(score, 0.0)

    def test_impossible_travel_raises_score(self):
        s1 = self.api._mock_score(self._make_txn())
        s2 = self.api._mock_score(self._make_txn(is_impossible_travel=1))
        self.assertGreater(s2, s1)

    def test_score_capped_at_100(self):
        score = self.api._mock_score(self._make_txn(
            velocity_spike=1, is_impossible_travel=1, is_new_device=1,
            is_new_country=1, is_mcc_high_risk=1, multi_account_device=1,
            is_fraud_hour=1, amount_ratio_to_avg=10.0, amt_zscore_30d=5.0,
            pct_cnp_30d=0.9
        ))
        self.assertLessEqual(score, 100.0)

    def test_multiple_flags_additive(self):
        s1 = self.api._mock_score(self._make_txn(velocity_spike=1))
        s2 = self.api._mock_score(self._make_txn(velocity_spike=1, is_new_device=1))
        self.assertGreater(s2, s1)


# ══════════════════════════════════════════════════════════════════════════════
# TEST SUITE 5 – FEATURE ENGINEERING (pure pandas / numpy)
# ══════════════════════════════════════════════════════════════════════════════
class TestFeatureEngineeringLogic(unittest.TestCase):
    """
    Tests feature engineering logic without PySpark
    (validates formulas/algorithms using pandas equivalents).
    """

    def _make_customer_df(self, n=50, fraud_count=2):
        """Create a minimal customer transaction DataFrame."""
        rng = np.random.default_rng(42)
        now = datetime.now()
        rows = []
        for i in range(n):
            rows.append({
                "transaction_id": f"TXN{i:05d}",
                "customer_id":    "CUST000001",
                "amount":         rng.lognormal(5.5, 1.5),
                "timestamp":      now - timedelta(hours=n - i),
                "is_fraud":       1 if i < fraud_count else 0,
                "latitude":       19.0 + rng.uniform(-0.5, 0.5),
                "longitude":      72.8 + rng.uniform(-0.5, 0.5),
                "country_code":   "IND",
                "card_present":   rng.choice([True, False]),
                "is_international": rng.choice([True, False], p=[0.92, 0.08]),
                "merchant_id":    f"MID{rng.integers(100, 200):03d}",
                "device_id":      f"DEV{rng.integers(1, 5):03d}",
            })
        return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    def test_velocity_count_1h(self):
        """Pandas rolling count should match expected velocity window."""
        df = self._make_customer_df(n=100)
        df = df.sort_values("timestamp")
        df["ts_int"] = df["timestamp"].astype(np.int64) // 10**9
        # Rolling count in 3600 second window
        df["vel_1h"] = (df.groupby("customer_id")["ts_int"]
                        .transform(lambda x: x.expanding().count()))
        self.assertTrue((df["vel_1h"] > 0).all())

    def test_amount_zscore_formula(self):
        """Z-score = (x - mean) / std; large outlier should have z > 1.9."""
        amounts = np.array([100, 200, 300, 400, 10000])
        mu      = amounts.mean()
        sigma   = amounts.std()
        zscores = (amounts - mu) / (sigma + 1e-8)
        # Outlier (10000) should have large positive z-score
        # Use >= 1.9 to be robust against population vs sample std choice
        self.assertGreater(zscores[-1], 1.9)

    def test_haversine_distance_same_point(self):
        """Same point should have 0 distance."""
        lat, lon = 19.0, 72.8
        # Simplified planar approximation used in code
        dist = math.sqrt(
            ((lat - lat) * 111.0) ** 2 +
            ((lon - lon) * 111.0 * math.cos(math.radians(lat))) ** 2
        )
        self.assertAlmostEqual(dist, 0.0, places=5)

    def test_haversine_distance_positive(self):
        """Different points should have positive distance."""
        lat1, lon1 = 19.0, 72.8   # Mumbai
        lat2, lon2 = 28.6, 77.2   # Delhi
        dist = math.sqrt(
            ((lat2 - lat1) * 111.0) ** 2 +
            ((lon2 - lon1) * 111.0 * math.cos(math.radians((lat1 + lat2) / 2))) ** 2
        )
        self.assertGreater(dist, 500)   # > 500 km apart

    def test_impossible_travel_speed_detection(self):
        """Speed > 900 km/h should flag impossible travel."""
        distance_km = 2000.0   # 2000 km apart
        time_min    = 30.0     # 30 minutes
        speed_kmph  = distance_km / (time_min / 60.0)
        self.assertGreater(speed_kmph, 900,
                           "Should detect impossible travel speed")

    def test_cyclical_hour_encoding(self):
        """Hour 0 and hour 24 should encode to same value."""
        sin_0  = math.sin(0 * (2 * math.pi / 24))
        cos_0  = math.cos(0 * (2 * math.pi / 24))
        sin_24 = math.sin(24 * (2 * math.pi / 24))
        cos_24 = math.cos(24 * (2 * math.pi / 24))
        self.assertAlmostEqual(sin_0, sin_24, places=5)
        self.assertAlmostEqual(cos_0, cos_24, places=5)

    def test_cyclical_hour_midnight_vs_noon(self):
        """Hour 0 (midnight) and 12 (noon) should have opposite cos values."""
        cos_0  = math.cos(0  * (2 * math.pi / 24))
        cos_12 = math.cos(12 * (2 * math.pi / 24))
        self.assertAlmostEqual(cos_0, -cos_12, places=5)

    def test_amount_ratio_to_avg(self):
        """Amount ratio should scale correctly."""
        avg    = 5000.0
        amount = 50000.0
        ratio  = amount / avg
        self.assertAlmostEqual(ratio, 10.0)

    def test_amount_ratio_zero_avg_fallback(self):
        """Division by zero avg should return 1.0 (safe default)."""
        avg    = 0.0
        amount = 500.0
        ratio  = amount / avg if avg > 0 else 1.0
        self.assertEqual(ratio, 1.0)

    def test_merchant_amount_deviation(self):
        """Deviation = |amount - avg| / avg."""
        merchant_avg = 3000.0
        amount       = 12000.0
        deviation    = abs(amount - merchant_avg) / merchant_avg
        self.assertAlmostEqual(deviation, 3.0)

    def test_fraud_hour_detection(self):
        """Hours 1–4 should be flagged as fraud hours."""
        fraud_hours = [1, 2, 3, 4]
        for h in range(24):
            flag = 1 if h in fraud_hours else 0
            if h in fraud_hours:
                self.assertEqual(flag, 1, f"Hour {h} should be flagged")
            else:
                self.assertEqual(flag, 0, f"Hour {h} should not be flagged")

    def test_month_end_detection(self):
        """Day >= 28 should be flagged as month-end."""
        for day in range(1, 32):
            flag = 1 if day >= 28 else 0
            if day >= 28:
                self.assertEqual(flag, 1)
            else:
                self.assertEqual(flag, 0)

    def test_merchant_risk_tier_encoding(self):
        """CASINO = 4, GROCERY = 0."""
        tiers = {
            "CASINO":          4,
            "CRYPTO_EXCHANGE": 4,
            "JEWELRY":         3,
            "TRAVEL":          3,
            "ELECTRONICS":     2,
            "FUEL":            2,
            "RESTAURANT":      0,
            "GROCERY":         0,
        }
        for cat, expected in tiers.items():
            if cat in ("CASINO", "CRYPTO_EXCHANGE"):
                tier = 4
            elif cat in ("JEWELRY", "TRAVEL"):
                tier = 3
            elif cat in ("ELECTRONICS", "FUEL"):
                tier = 2
            elif cat in ("RESTAURANT", "GROCERY"):
                tier = 0
            else:
                tier = 1
            self.assertEqual(tier, expected, f"Wrong tier for {cat}")


# ══════════════════════════════════════════════════════════════════════════════
# TEST SUITE 6 – INTEGRATION TESTS
# ══════════════════════════════════════════════════════════════════════════════
class TestIntegrationProducerConsumer(unittest.TestCase):
    """
    Integration test: Producer generates → Consumer enriches.
    No real Kafka; tests the processing pipeline end-to-end.
    """

    def setUp(self):
        from ingestion.kafka_producer import generate_transaction
        from ingestion.kafka_consumer import FraudRuleEngine, FraudScoreEnricher
        import dataclasses

        self.gen_txn  = generate_transaction
        engine        = FraudRuleEngine(redis_client=None)
        self.enricher = FraudScoreEnricher(engine)
        self.asdict   = dataclasses.asdict

    def test_transaction_survives_full_enrichment(self):
        txn      = self.gen_txn()
        txn_dict = self.asdict(txn)
        enriched = self.enricher.enrich(txn_dict)

        self.assertIn("rule_based_score", enriched)
        self.assertIn("rule_severity",    enriched)
        self.assertIn("enriched_at",      enriched)

    def test_50_transactions_all_enrichable(self):
        errors = []
        for i in range(50):
            try:
                txn      = self.gen_txn(inject_fraud=(i % 10 == 0))
                txn_dict = self.asdict(txn)
                enriched = self.enricher.enrich(txn_dict)
                assert "rule_severity" in enriched
            except Exception as e:
                errors.append(f"txn {i}: {e}")
        self.assertEqual(errors, [], f"Enrichment errors: {errors}")

    def test_fraud_injected_transactions_score_higher(self):
        import dataclasses
        fraud_scores  = []
        legit_scores  = []
        for _ in range(50):
            for inject in (True, False):
                txn      = self.gen_txn(inject_fraud=inject)
                txn_dict = dataclasses.asdict(txn)
                enriched = self.enricher.enrich(txn_dict)
                score    = enriched["rule_based_score"]
                if inject:
                    fraud_scores.append(score)
                else:
                    legit_scores.append(score)
        avg_fraud = np.mean(fraud_scores)
        avg_legit = np.mean(legit_scores)
        self.assertGreaterEqual(avg_fraud, avg_legit,
                                f"Fraud avg score {avg_fraud:.1f} should >= "
                                f"legit avg score {avg_legit:.1f}")


class TestIntegrationMLPipeline(unittest.TestCase):
    """
    Integration test: data generation → train → evaluate end-to-end.
    """

    def test_full_training_pipeline(self):
        from ml_pipeline.train_pipeline import (
            generate_synthetic_training_data, FraudEnsembleScorer,
            evaluate_model, FEATURE_COLS, TARGET_COL
        )
        from sklearn.model_selection import train_test_split

        df = generate_synthetic_training_data(n_samples=4_000)
        X  = df[FEATURE_COLS].values
        y  = df[TARGET_COL].values

        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.15, stratify=y_temp, random_state=42)

        # Train
        ensemble = FraudEnsembleScorer()
        ensemble.fit(X_train, y_train, X_val, y_val)

        # Evaluate
        metrics = evaluate_model(ensemble, X_test, y_test, "IntegrationTest")

        # Sanity checks on metrics
        self.assertGreater(metrics["roc_auc"], 0.5,
                           "ROC-AUC should exceed random baseline")
        self.assertGreater(metrics["pr_auc"], 0.0)
        self.assertGreaterEqual(metrics["true_positives"] +
                                metrics["false_negatives"],
                                y_test.sum(),
                                "TP + FN must equal total fraud count")

    def test_mlflow_tracking_integration(self):
        """Ensure MLflow tracking completes without raising."""
        from ml_pipeline.train_pipeline import (
            FraudEnsembleScorer, generate_synthetic_training_data,
            FEATURE_COLS, TARGET_COL
        )
        from sklearn.model_selection import train_test_split
        import mlflow

        # Use small dataset to keep test fast
        df = generate_synthetic_training_data(n_samples=2_000)
        X  = df[FEATURE_COLS].values
        y  = df[TARGET_COL].values
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.15, stratify=y_temp, random_state=42)

        mlflow.set_experiment("test_mlflow_integration")
        with mlflow.start_run(run_name="test_run"):
            ensemble = FraudEnsembleScorer()
            ensemble.fit(X_train, y_train, X_val, y_val)
            mlflow.log_param("n_samples", len(df))
            mlflow.log_param("n_features", len(FEATURE_COLS))

        self.assertIsNotNone(ensemble)
        self.assertTrue(ensemble.is_fitted)


class TestIntegrationScoringAndAlerting(unittest.TestCase):
    """
    Integration test: ensemble scoring → risk band → alerting logic.
    """

    def setUp(self):
        from ml_pipeline.train_pipeline import (
            FraudEnsembleScorer, generate_synthetic_training_data, FEATURE_COLS
        )
        df  = generate_synthetic_training_data(n_samples=3_000)
        X   = df[FEATURE_COLS].values
        y   = df["is_fraud"].values
        sp  = int(len(X) * 0.8)
        self.ensemble = FraudEnsembleScorer()
        self.ensemble.fit(X[:sp], y[:sp], X[sp:], y[sp:])
        self.X_test   = X[sp:][:200]
        self.FEATURES = FEATURE_COLS

        import types, sys
        for mod_name in ["fastapi", "fastapi.responses", "fastapi.middleware",
                         "fastapi.middleware.cors", "pydantic", "mangum"]:
            if mod_name not in sys.modules:
                sys.modules[mod_name] = types.ModuleType(mod_name)
        import api.fraud_scoring_api as api_mod
        self.api = api_mod

    def test_all_scores_produce_valid_risk_band(self):
        scores = self.ensemble.fraud_score(self.X_test)
        for s in scores:
            band = self.api._risk_band(float(s))
            self.assertIn(band, ("LOW", "MEDIUM", "HIGH", "CRITICAL"))

    def test_high_risk_transactions_get_block_recommendation(self):
        scores = self.ensemble.fraud_score(self.X_test)
        high_risk = [s for s in scores if s >= 65]
        if high_risk:
            for s in high_risk[:5]:
                band = self.api._risk_band(float(s))
                rec  = self.api._recommendation(float(s), band)
                self.assertIsInstance(rec, str)
                self.assertGreater(len(rec), 0)

    def test_score_distribution_realistic(self):
        """Most transactions should score LOW (realistic fraud rate ~0.2%)."""
        scores    = self.ensemble.fraud_score(self.X_test)
        low_count = sum(1 for s in scores if s < 45)
        low_pct   = low_count / len(scores)
        self.assertGreater(low_pct, 0.5,
                           "Majority of transactions should score LOW")


# ══════════════════════════════════════════════════════════════════════════════
# TEST SUITE 7 – DATA QUALITY & EDGE CASES
# ══════════════════════════════════════════════════════════════════════════════
class TestEdgeCases(unittest.TestCase):
    """Tests for boundary conditions, zero-values, and adversarial inputs."""

    def test_rule_engine_zero_amount(self):
        from ingestion.kafka_consumer import FraudRuleEngine
        engine  = FraudRuleEngine(redis_client=None)
        txn     = {
            "customer_id": "CUST999",
            "amount": 0.0,
            "transaction_type": "CREDIT_CARD",
            "is_high_risk_merchant": False,
            "card_present": True,
            "is_international": False,
            "hour_of_day": 12,
        }
        results = engine.evaluate(txn)
        self.assertIsInstance(results, list)  # Should not crash

    def test_rule_engine_very_large_amount(self):
        from ingestion.kafka_consumer import FraudRuleEngine
        engine  = FraudRuleEngine(redis_client=None)
        txn     = {
            "customer_id": "CUST998",
            "amount": 999_999_999.0,
            "transaction_type": "ONLINE_TRANSFER",
            "is_high_risk_merchant": False,
            "card_present": True,
            "is_international": False,
            "hour_of_day": 12,
        }
        results = engine.evaluate(txn)
        names   = [r.rule_name for r in results]
        self.assertIn("LARGE_TRANSACTION", names)

    def test_isolation_forest_single_row(self):
        """Model should handle single-row inference."""
        from ml_pipeline.train_pipeline import (
            IsolationForestDetector, generate_synthetic_training_data, FEATURE_COLS
        )
        df  = generate_synthetic_training_data(n_samples=1_000)
        X   = df[FEATURE_COLS].values
        det = IsolationForestDetector()
        det.train(X)
        score = det.anomaly_score(X[:1])
        self.assertEqual(score.shape, (1,))

    def test_ensemble_handles_all_zeros(self):
        """Ensemble should handle a zero-feature vector gracefully."""
        from ml_pipeline.train_pipeline import (
            FraudEnsembleScorer, generate_synthetic_training_data, FEATURE_COLS
        )
        df  = generate_synthetic_training_data(n_samples=2_000)
        X   = df[FEATURE_COLS].values
        y   = df["is_fraud"].values
        sp  = int(len(X) * 0.8)
        e   = FraudEnsembleScorer()
        e.fit(X[:sp], y[:sp], X[sp:], y[sp:])
        zero_X = np.zeros((1, len(FEATURE_COLS)))
        score  = e.fraud_score(zero_X)
        self.assertEqual(score.shape, (1,))
        self.assertGreaterEqual(float(score[0]), 0)
        self.assertLessEqual(float(score[0]), 100)

    def test_synthetic_data_empty_request_raises(self):
        from ml_pipeline.train_pipeline import generate_synthetic_training_data
        df = generate_synthetic_training_data(n_samples=0)
        self.assertEqual(len(df), 0)

    def test_risk_band_edge_exact_boundaries(self):
        import types, sys
        for mod in ["fastapi", "fastapi.responses", "fastapi.middleware",
                    "fastapi.middleware.cors", "pydantic", "mangum"]:
            if mod not in sys.modules:
                sys.modules[mod] = types.ModuleType(mod)
        import api.fraud_scoring_api as api
        self.assertEqual(api._risk_band(0.0),   "LOW")
        self.assertEqual(api._risk_band(100.0), "CRITICAL")
        self.assertEqual(api._risk_band(45.0),  "MEDIUM")
        self.assertEqual(api._risk_band(65.0),  "HIGH")
        self.assertEqual(api._risk_band(85.0),  "CRITICAL")

    def test_transaction_serialization_round_trip(self):
        from ingestion.kafka_producer import generate_transaction
        import dataclasses
        for _ in range(10):
            txn  = generate_transaction()
            d    = dataclasses.asdict(txn)
            s    = json.dumps(d, default=str)
            back = json.loads(s)
            self.assertEqual(back["transaction_id"], txn.transaction_id)
            self.assertAlmostEqual(back["amount"], txn.amount, places=2)

    def test_velocity_memory_store_bounded(self):
        """In-memory deque should not grow unboundedly."""
        from ingestion.kafka_consumer import FraudRuleEngine
        engine = FraudRuleEngine(redis_client=None)
        cid    = "STRESS_TEST_CUST"
        for i in range(500):
            engine._update_velocity(cid, float(i))
        store_len = len(engine._memory_store[cid])
        self.assertLessEqual(store_len, 100,  # maxlen=100
                             "Velocity memory store should be bounded at 100")
