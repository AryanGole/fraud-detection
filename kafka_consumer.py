"""
============================================================
Fraud Detection Pipeline - Kafka Stream Consumer
============================================================
Consumes banking transactions, applies pre-scoring rules,
and routes to ML scoring engine with sub-100ms SLA.

Author: Senior ML Engineering Team
Version: 2.1.0
"""

import json
import time
import signal
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Callable
from collections import defaultdict, deque
from dataclasses import dataclass

from confluent_kafka import Consumer, Producer, KafkaError, KafkaException
import redis
import structlog

log = structlog.get_logger(__name__)

# ── Alert Severity Levels ─────────────────────────────────────────────────────
ALERT_CRITICAL = "CRITICAL"   # Immediate block  (score >= 85)
ALERT_HIGH     = "HIGH"       # Manual review    (score 65-84)
ALERT_MEDIUM   = "MEDIUM"     # Enhanced monitor (score 45-64)
ALERT_LOW      = "LOW"        # Log only         (score < 45)


# ── Rule Engine ───────────────────────────────────────────────────────────────
@dataclass
class RuleResult:
    rule_name:   str
    triggered:   bool
    risk_points: int
    reason:      str


class FraudRuleEngine:
    """
    Deterministic rule-based pre-filter before ML scoring.
    Rules are calibrated against fraud analyst playbooks.
    """

    VELOCITY_WINDOW_SECONDS = 300   # 5-minute velocity window
    MAX_TXN_PER_5MIN       = 8
    MAX_AMOUNT_PER_5MIN    = 100000  # ₹1L in 5 mins

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        # In-memory fallback if Redis unavailable
        self._memory_store: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.redis = redis_client

    def _get_velocity(self, customer_id: str, window: int = 300) -> Dict:
        """Get transaction count and amount in rolling window."""
        try:
            if self.redis:
                key_count  = f"vel:count:{customer_id}"
                key_amount = f"vel:amount:{customer_id}"
                count  = int(self.redis.get(key_count) or 0)
                amount = float(self.redis.get(key_amount) or 0.0)
                return {"count": count, "amount": amount}
        except Exception:
            pass
        # Fallback: in-memory
        now   = time.time()
        txns  = self._memory_store[customer_id]
        recent = [t for t in txns if (now - t["ts"]) <= window]
        return {
            "count":  len(recent),
            "amount": sum(t["amt"] for t in recent),
        }

    def _update_velocity(self, customer_id: str, amount: float):
        """Update velocity counters."""
        try:
            if self.redis:
                pipe = self.redis.pipeline()
                pipe.incr(f"vel:count:{customer_id}")
                pipe.expire(f"vel:count:{customer_id}", self.VELOCITY_WINDOW_SECONDS)
                pipe.incrbyfloat(f"vel:amount:{customer_id}", amount)
                pipe.expire(f"vel:amount:{customer_id}", self.VELOCITY_WINDOW_SECONDS)
                pipe.execute()
                return
        except Exception:
            pass
        now = time.time()
        self._memory_store[customer_id].append({"ts": now, "amt": amount})

    def evaluate(self, txn: Dict) -> List[RuleResult]:
        """
        Evaluate all fraud rules against a transaction.
        Returns list of triggered rules with risk point contributions.
        """
        results = []
        customer_id = txn.get("customer_id", "")
        amount      = float(txn.get("amount", 0))

        # ── Rule 1: Transaction velocity ─────────────────────────────────────
        velocity = self._get_velocity(customer_id)
        if velocity["count"] >= self.MAX_TXN_PER_5MIN:
            results.append(RuleResult(
                rule_name   = "HIGH_VELOCITY",
                triggered   = True,
                risk_points = 35,
                reason      = f">{self.MAX_TXN_PER_5MIN} txns in 5 min (actual: {velocity['count']})"
            ))

        # ── Rule 2: Velocity amount threshold ────────────────────────────────
        if velocity["amount"] + amount > self.MAX_AMOUNT_PER_5MIN:
            results.append(RuleResult(
                rule_name   = "VELOCITY_AMOUNT",
                triggered   = True,
                risk_points = 30,
                reason      = f"Amount velocity exceeded ₹{self.MAX_AMOUNT_PER_5MIN:,.0f} in 5 min"
            ))

        # ── Rule 3: High-risk merchant category ──────────────────────────────
        if txn.get("is_high_risk_merchant"):
            results.append(RuleResult(
                rule_name   = "HIGH_RISK_MERCHANT",
                triggered   = True,
                risk_points = 20,
                reason      = f"High-risk merchant category: {txn.get('merchant_category')}"
            ))

        # ── Rule 4: Card-not-present + international ──────────────────────────
        if not txn.get("card_present", True) and txn.get("is_international"):
            results.append(RuleResult(
                rule_name   = "CNP_INTERNATIONAL",
                triggered   = True,
                risk_points = 40,
                reason      = "Card-not-present international transaction"
            ))

        # ── Rule 5: Large ATM withdrawal at unusual hour ───────────────────────
        hour   = txn.get("hour_of_day", 12)
        if txn.get("transaction_type") == "ATM_WITHDRAWAL" and hour in range(1, 5):
            results.append(RuleResult(
                rule_name   = "OFF_HOURS_ATM",
                triggered   = True,
                risk_points = 25,
                reason      = f"ATM withdrawal at {hour:02d}:00"
            ))

        # ── Rule 6: Extremely large single transaction ─────────────────────────
        if amount > 1000000:  # > ₹10L
            results.append(RuleResult(
                rule_name   = "LARGE_TRANSACTION",
                triggered   = True,
                risk_points = 15,
                reason      = f"Transaction amount ₹{amount:,.0f} exceeds threshold"
            ))

        # ── Rule 7: New device + large amount ─────────────────────────────────
        # (Simplified: assume first-time device IDs trigger this)
        if not txn.get("card_present") and amount > 50000:
            results.append(RuleResult(
                rule_name   = "CNP_HIGH_AMOUNT",
                triggered   = True,
                risk_points = 30,
                reason      = f"CNP transaction with high amount ₹{amount:,.0f}"
            ))

        # Update velocity counters after evaluation
        self._update_velocity(customer_id, amount)

        return [r for r in results if r.triggered]


# ── Fraud Score Enricher ───────────────────────────────────────────────────────
class FraudScoreEnricher:
    """
    Enriches transactions with rule-based risk scores and prepares
    the message payload for ML scoring downstream.
    """

    def __init__(self, rule_engine: FraudRuleEngine):
        self.rule_engine = rule_engine

    def enrich(self, txn: Dict) -> Dict:
        rules_triggered = self.rule_engine.evaluate(txn)

        rule_score = min(sum(r.risk_points for r in rules_triggered), 100)
        rule_names = [r.rule_name for r in rules_triggered]
        reasons    = [r.reason for r in rules_triggered]

        severity = (
            ALERT_CRITICAL if rule_score >= 85 else
            ALERT_HIGH     if rule_score >= 65 else
            ALERT_MEDIUM   if rule_score >= 45 else
            ALERT_LOW
        )

        txn["rule_based_score"]     = rule_score
        txn["rules_triggered"]      = rule_names
        txn["rule_reasons"]         = reasons
        txn["rule_severity"]        = severity
        txn["requires_ml_scoring"]  = rule_score >= 20  # Only score suspicious ones fully
        txn["enriched_at"]          = datetime.now(timezone.utc).isoformat()

        return txn


# ── Streaming Consumer ─────────────────────────────────────────────────────────
class FraudStreamConsumer:
    """
    Multi-threaded Kafka consumer group for banking transaction streams.
    Processes, enriches, and routes transactions for ML scoring.
    """

    CONSUMER_GROUP = "fraud-detection-pipeline-v2"

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        redis_url: str         = "redis://localhost:6379/0",
        output_bootstrap:str   = "localhost:9092",
    ):
        redis_client = self._init_redis(redis_url)
        rule_engine  = FraudRuleEngine(redis_client)
        self.enricher = FraudScoreEnricher(rule_engine)

        self.consumer = Consumer({
            "bootstrap.servers":          bootstrap_servers,
            "group.id":                   self.CONSUMER_GROUP,
            "auto.offset.reset":          "latest",
            "enable.auto.commit":         False,  # Manual commit for exactly-once
            "max.poll.interval.ms":       300000,
            "session.timeout.ms":         60000,
            "fetch.min.bytes":            1,
            "fetch.wait.max.ms":          500,
        })

        self.producer = Producer({
            "bootstrap.servers":  output_bootstrap,
            "acks":               "all",
            "enable.idempotence": "true",
        })

        self._running = True
        self.metrics: Dict[str, float] = defaultdict(float)
        signal.signal(signal.SIGTERM, self._shutdown)
        signal.signal(signal.SIGINT,  self._shutdown)

    def _init_redis(self, url: str) -> Optional[redis.Redis]:
        try:
            r = redis.from_url(url, decode_responses=True, socket_timeout=2)
            r.ping()
            log.info("Redis connected", url=url)
            return r
        except Exception as e:
            log.warning("Redis unavailable, using in-memory store", error=str(e))
            return None

    def _shutdown(self, *args):
        log.info("Shutdown signal received, draining consumer...")
        self._running = False

    def _process_message(self, msg_value: bytes) -> Optional[Dict]:
        """Deserialize, validate, and enrich a transaction message."""
        try:
            txn = json.loads(msg_value.decode("utf-8"))
            return self.enricher.enrich(txn)
        except (json.JSONDecodeError, KeyError) as e:
            log.error("Message parse error", error=str(e))
            self.metrics["parse_errors"] += 1
            return None

    def _route_enriched(self, enriched: Dict):
        """Route enriched transaction to appropriate downstream topic."""
        key   = enriched.get("customer_id", "unknown").encode("utf-8")
        value = json.dumps(enriched, default=str).encode("utf-8")

        severity = enriched.get("rule_severity", ALERT_LOW)

        if severity in (ALERT_CRITICAL, ALERT_HIGH):
            # Immediate alert topic for real-time blocking
            self.producer.produce("banking.fraud.alerts", key=key, value=value)
            self.metrics["alerts_produced"] += 1

        # Always send to ML scoring queue
        self.producer.produce("banking.fraud.scored", key=key, value=value)
        self.metrics["scored_produced"] += 1

    def consume(self, topics: Optional[List[str]] = None):
        """Main consumer loop - processes transactions with SLA tracking."""
        topics = topics or [
            "banking.transactions.raw",
            "banking.transactions.high_value",
            "banking.transactions.international",
        ]
        self.consumer.subscribe(topics)
        log.info("Consumer started", topics=topics, group=self.CONSUMER_GROUP)

        batch_offsets = {}
        last_log_time = time.monotonic()

        while self._running:
            msg = self.consumer.poll(timeout=1.0)

            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                log.error("Consumer error", error=msg.error())
                self.metrics["consumer_errors"] += 1
                continue

            start_ts = time.monotonic()

            enriched = self._process_message(msg.value())
            if enriched:
                self._route_enriched(enriched)
                self.metrics["processed"] += 1

            # Track latency
            latency_ms = (time.monotonic() - start_ts) * 1000
            self.metrics["total_latency_ms"] += latency_ms
            if latency_ms > 100:  # SLA breach
                self.metrics["sla_breaches"] += 1
                log.warning("SLA breach", latency_ms=round(latency_ms, 2),
                            txn_id=enriched.get("transaction_id") if enriched else None)

            # Batch commit for throughput
            batch_offsets[(msg.topic(), msg.partition())] = msg.offset() + 1
            if len(batch_offsets) >= 100:
                self.consumer.commit(asynchronous=True)
                batch_offsets.clear()

            # Periodic poll flush
            self.producer.poll(0)

            # Periodic metrics log
            now = time.monotonic()
            if now - last_log_time >= 30:
                self._log_metrics()
                last_log_time = now

        # Cleanup
        self.consumer.close()
        self.producer.flush(30)
        self._log_metrics()
        log.info("Consumer shut down cleanly")

    def _log_metrics(self):
        processed = self.metrics["processed"]
        avg_lat   = (self.metrics["total_latency_ms"] / max(processed, 1))
        log.info(
            "Consumer metrics",
            processed      = processed,
            alerts         = self.metrics["alerts_produced"],
            avg_latency_ms = round(avg_lat, 2),
            sla_breaches   = self.metrics["sla_breaches"],
            errors         = self.metrics["consumer_errors"],
        )


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    consumer = FraudStreamConsumer()
    consumer.consume()
