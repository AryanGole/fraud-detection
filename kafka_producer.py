"""
============================================================
Fraud Detection Pipeline - Kafka Transaction Producer
============================================================
Simulates real-time banking transaction streams at enterprise scale.
Supports: Credit Card, UPI, ATM, Mobile Banking, Online Transfer

Author: Senior ML Engineering Team
Version: 2.1.0
"""

import json
import random
import time
import uuid
import logging
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum

from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic
import yaml
import structlog

# ── Logger Setup ──────────────────────────────────────────────────────────────
log = structlog.get_logger(__name__)

# ── Enums ─────────────────────────────────────────────────────────────────────
class TransactionType(str, Enum):
    CREDIT_CARD   = "CREDIT_CARD"
    UPI           = "UPI"
    ATM_WITHDRAWAL = "ATM_WITHDRAWAL"
    MOBILE_BANKING = "MOBILE_BANKING"
    ONLINE_TRANSFER = "ONLINE_TRANSFER"
    NEFT          = "NEFT"
    RTGS          = "RTGS"
    IMPS          = "IMPS"

class TransactionStatus(str, Enum):
    INITIATED  = "INITIATED"
    PENDING    = "PENDING"
    APPROVED   = "APPROVED"
    DECLINED   = "DECLINED"
    FLAGGED    = "FLAGGED"

class RiskChannel(str, Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"

# ── Transaction Schema ─────────────────────────────────────────────────────────
@dataclass
class BankingTransaction:
    transaction_id:    str
    customer_id:       str
    account_number:    str
    transaction_type:  str
    amount:            float
    currency:          str
    merchant_id:       Optional[str]
    merchant_category: Optional[str]
    merchant_name:     Optional[str]
    timestamp:         str
    local_timestamp:   str
    timezone:          str
    device_id:         str
    device_type:       str
    ip_address:        str
    latitude:          float
    longitude:         float
    country_code:      str
    city:              str
    card_present:      bool
    card_bin:          str
    bank_code:         str
    channel:           str
    session_id:        str
    status:            str
    # Behavioral signals
    is_international:  bool
    is_high_risk_merchant: bool
    hour_of_day:       int
    day_of_week:       int
    # Metadata
    schema_version:    str = "2.1"

# ── Synthetic Data Generators ──────────────────────────────────────────────────
MERCHANT_CATEGORIES = [
    "GROCERY", "ELECTRONICS", "FUEL", "RESTAURANT", "TRAVEL",
    "HEALTHCARE", "ENTERTAINMENT", "CLOTHING", "JEWELRY", "CASINO",
    "ONLINE_GAMING", "CRYPTO_EXCHANGE", "MONEY_TRANSFER", "ATM",
    "UTILITY", "INSURANCE", "TELECOM", "EDUCATION"
]

HIGH_RISK_CATEGORIES = {
    "CASINO", "CRYPTO_EXCHANGE", "MONEY_TRANSFER", "ONLINE_GAMING"
}

DEVICE_TYPES = ["MOBILE_ANDROID", "MOBILE_IOS", "DESKTOP", "TABLET", "POS_TERMINAL", "ATM"]

COUNTRY_COORDS = {
    "IND": (20.5937, 78.9629, ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad"]),
    "USA": (37.0902, -95.7129, ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"]),
    "GBR": (55.3781, -3.4360, ["London", "Manchester", "Birmingham", "Glasgow", "Leeds"]),
    "SGP": (1.3521, 103.8198, ["Singapore"]),
    "UAE": (23.4241, 53.8478, ["Dubai", "Abu Dhabi", "Sharjah"]),
    "CHN": (35.8617, 104.1954, ["Beijing", "Shanghai", "Shenzhen", "Guangzhou"]),
}

BANKS = [
    "HDFC", "ICICI", "SBI", "AXIS", "KOTAK",
    "CITI", "HSBC", "BARCLAYS", "JPM", "BofA"
]


def generate_customer_id() -> str:
    return f"CUST{random.randint(100000, 999999):06d}"


def generate_account_number() -> str:
    return f"{random.randint(10000000, 99999999):08d}{random.randint(1000, 9999):04d}"


def generate_device_id() -> str:
    raw = f"{random.getrandbits(64):016x}"
    return hashlib.md5(raw.encode()).hexdigest()[:16].upper()


def generate_ip() -> str:
    # Occasionally generate suspicious IPs from known fraud ranges
    if random.random() < 0.02:  # 2% suspicious IPs
        return f"185.{random.randint(100,200)}.{random.randint(0,255)}.{random.randint(1,254)}"
    return f"{random.randint(1,254)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"


def generate_location(is_international: bool = False):
    if is_international:
        country_code = random.choice(list(COUNTRY_COORDS.keys()))
    else:
        country_code = "IND"  # Primarily India-based bank
    lat_base, lon_base, cities = COUNTRY_COORDS[country_code]
    lat = lat_base + random.uniform(-5, 5)
    lon = lon_base + random.uniform(-5, 5)
    city = random.choice(cities)
    return lat, lon, country_code, city


def inject_fraud_pattern(txn: BankingTransaction) -> BankingTransaction:
    """
    Inject realistic fraud patterns to simulate labeled fraud transactions.
    Used for generating synthetic training data.
    """
    fraud_type = random.choice([
        "card_not_present", "account_takeover", "velocity_abuse",
        "geo_anomaly", "unusual_amount", "off_hours_atm"
    ])

    if fraud_type == "card_not_present":
        txn.card_present = False
        txn.amount = round(random.uniform(500, 50000), 2)
        txn.merchant_category = random.choice(["ELECTRONICS", "JEWELRY"])
        txn.is_international = True

    elif fraud_type == "account_takeover":
        txn.device_id = generate_device_id()  # New unknown device
        txn.ip_address = generate_ip()
        txn.is_international = True
        lat, lon, cc, city = generate_location(is_international=True)
        txn.latitude, txn.longitude, txn.country_code, txn.city = lat, lon, cc, city
        txn.amount = round(random.uniform(10000, 500000), 2)

    elif fraud_type == "velocity_abuse":
        txn.amount = round(random.uniform(100, 5000), 2)  # Multiple small txns
        txn.merchant_category = "MONEY_TRANSFER"
        txn.is_high_risk_merchant = True

    elif fraud_type == "geo_anomaly":
        txn.is_international = True
        lat, lon, cc, city = generate_location(is_international=True)
        txn.latitude, txn.longitude, txn.country_code, txn.city = lat, lon, cc, city

    elif fraud_type == "unusual_amount":
        txn.amount = round(random.uniform(100000, 2000000), 2)

    elif fraud_type == "off_hours_atm":
        txn.transaction_type = TransactionType.ATM_WITHDRAWAL
        txn.hour_of_day = random.choice([1, 2, 3, 4])  # Unusual hours
        txn.amount = round(random.uniform(20000, 200000), 2)

    return txn


def generate_transaction(inject_fraud: bool = False) -> BankingTransaction:
    """Generate a single synthetic banking transaction."""
    now = datetime.now(timezone.utc)
    txn_type = random.choice(list(TransactionType))
    is_international = random.random() < 0.08  # 8% international
    lat, lon, country_code, city = generate_location(is_international)
    merchant_cat = random.choice(MERCHANT_CATEGORIES)

    txn = BankingTransaction(
        transaction_id      = str(uuid.uuid4()),
        customer_id         = generate_customer_id(),
        account_number      = generate_account_number(),
        transaction_type    = txn_type.value,
        amount              = round(random.lognormvariate(5.5, 1.8), 2),  # Log-normal distribution
        currency            = "INR" if country_code == "IND" else random.choice(["USD", "GBP", "EUR", "SGD"]),
        merchant_id         = f"MID{random.randint(1000, 9999):04d}" if txn_type != TransactionType.ATM_WITHDRAWAL else None,
        merchant_category   = merchant_cat,
        merchant_name       = f"MERCHANT_{merchant_cat}_{random.randint(100, 999)}",
        timestamp           = now.isoformat(),
        local_timestamp     = datetime.now().isoformat(),
        timezone            = "Asia/Kolkata" if country_code == "IND" else "UTC",
        device_id           = generate_device_id(),
        device_type         = random.choice(DEVICE_TYPES),
        ip_address          = generate_ip(),
        latitude            = round(lat, 6),
        longitude           = round(lon, 6),
        country_code        = country_code,
        city                = city,
        card_present        = random.random() < 0.65,
        card_bin            = f"{random.choice(['4111', '5200', '3714', '6011', '3530'])}{random.randint(10, 99):02d}",
        bank_code           = random.choice(BANKS),
        channel             = txn_type.value,
        session_id          = str(uuid.uuid4()),
        status              = TransactionStatus.INITIATED.value,
        is_international    = is_international,
        is_high_risk_merchant = merchant_cat in HIGH_RISK_CATEGORIES,
        hour_of_day         = now.hour,
        day_of_week         = now.weekday(),
    )

    if inject_fraud:
        txn = inject_fraud_pattern(txn)

    return txn


# ── Kafka Producer ─────────────────────────────────────────────────────────────
class FraudDetectionProducer:
    """
    Enterprise Kafka producer for banking transaction streams.
    Supports configurable TPS, fraud injection rate, and partition strategies.
    """

    TOPIC_TRANSACTIONS  = "banking.transactions.raw"
    TOPIC_HIGH_VALUE    = "banking.transactions.high_value"
    TOPIC_INTERNATIONAL = "banking.transactions.international"
    TOPIC_FRAUD_ALERTS  = "banking.fraud.alerts"

    def __init__(self, config_path: str = "config/kafka_config.yaml"):
        self.config = self._load_config(config_path)
        self.producer = Producer(self._kafka_conf())
        self.metrics = {
            "produced": 0,
            "fraud_injected": 0,
            "high_value": 0,
            "errors": 0,
        }
        log.info("FraudDetectionProducer initialized", brokers=self.config.get("bootstrap_servers"))

    def _load_config(self, path: str) -> Dict:
        try:
            with open(path) as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            # Default config for standalone execution
            return {
                "bootstrap_servers": "localhost:9092",
                "acks": "all",
                "retries": 3,
                "batch_size": 16384,
                "linger_ms": 5,
                "compression_type": "snappy",
            }

    def _kafka_conf(self) -> Dict:
        return {
            "bootstrap.servers": self.config.get("bootstrap_servers", "localhost:9092"),
            "acks":               str(self.config.get("acks", "all")),
            "retries":            int(self.config.get("retries", 3)),
            "batch.size":         int(self.config.get("batch_size", 16384)),
            "linger.ms":          int(self.config.get("linger_ms", 5)),
            "compression.type":   self.config.get("compression_type", "snappy"),
            "enable.idempotence": "true",
        }

    def _delivery_callback(self, err, msg):
        if err:
            self.metrics["errors"] += 1
            log.error("Message delivery failed", error=str(err))
        else:
            self.metrics["produced"] += 1

    def _select_topic(self, txn: BankingTransaction) -> str:
        """Route transactions to appropriate topics based on risk attributes."""
        if txn.amount > 500000:  # High value threshold: ₹5L
            self.metrics["high_value"] += 1
            return self.TOPIC_HIGH_VALUE
        if txn.is_international:
            return self.TOPIC_INTERNATIONAL
        return self.TOPIC_TRANSACTIONS

    def produce_transaction(self, txn: BankingTransaction) -> None:
        """Produce a single transaction to the appropriate Kafka topic."""
        topic = self._select_topic(txn)
        key   = txn.customer_id.encode("utf-8")  # Partition by customer
        value = json.dumps(asdict(txn), default=str).encode("utf-8")
        self.producer.produce(topic, key=key, value=value, callback=self._delivery_callback)

    def stream_transactions(
        self,
        tps: int = 500,
        duration_seconds: int = 60,
        fraud_rate: float = 0.002,  # 0.2% fraud rate (realistic banking)
    ) -> None:
        """
        Stream transactions at specified TPS for given duration.

        Args:
            tps: Target transactions per second
            duration_seconds: Total stream duration
            fraud_rate: Fraction of transactions to inject fraud patterns
        """
        log.info("Starting transaction stream",
                 tps=tps,
                 duration=duration_seconds,
                 fraud_rate=fraud_rate)

        interval = 1.0 / tps
        start    = time.monotonic()
        count    = 0

        try:
            while (time.monotonic() - start) < duration_seconds:
                inject_fraud = random.random() < fraud_rate
                txn          = generate_transaction(inject_fraud=inject_fraud)

                if inject_fraud:
                    self.metrics["fraud_injected"] += 1

                self.produce_transaction(txn)
                count += 1

                # Flush every 1000 messages for throughput
                if count % 1000 == 0:
                    self.producer.poll(0)
                    elapsed = time.monotonic() - start
                    actual_tps = count / elapsed if elapsed > 0 else 0
                    log.info("Stream progress",
                             count=count,
                             actual_tps=round(actual_tps, 1),
                             elapsed=round(elapsed, 1))

                time.sleep(max(0, interval - 0.0001))

        except KeyboardInterrupt:
            log.info("Stream interrupted by user")
        finally:
            self.producer.flush(timeout=30)
            self._log_metrics()

    def _log_metrics(self):
        log.info("Producer metrics",
                 **self.metrics,
                 fraud_rate_actual=round(
                     self.metrics["fraud_injected"] / max(self.metrics["produced"], 1), 4
                 ))

    @staticmethod
    def create_topics(bootstrap_servers: str = "localhost:9092"):
        """Create required Kafka topics with appropriate configurations."""
        admin = AdminClient({"bootstrap.servers": bootstrap_servers})
        topics = [
            NewTopic("banking.transactions.raw",         num_partitions=12, replication_factor=3),
            NewTopic("banking.transactions.high_value",  num_partitions=6,  replication_factor=3),
            NewTopic("banking.transactions.international",num_partitions=6,  replication_factor=3),
            NewTopic("banking.fraud.alerts",             num_partitions=3,  replication_factor=3),
            NewTopic("banking.fraud.scored",             num_partitions=12, replication_factor=3),
            NewTopic("banking.dlq",                      num_partitions=3,  replication_factor=3),
        ]
        fs = admin.create_topics(topics)
        for topic, future in fs.items():
            try:
                future.result()
                log.info("Topic created", topic=topic)
            except Exception as e:
                log.warning("Topic creation result", topic=topic, message=str(e))


# ── CLI Entry Point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fraud Detection Kafka Producer")
    parser.add_argument("--tps",      type=int,   default=500,  help="Transactions per second")
    parser.add_argument("--duration", type=int,   default=120,  help="Duration in seconds")
    parser.add_argument("--fraud-rate", type=float, default=0.002, help="Fraud injection rate")
    parser.add_argument("--create-topics", action="store_true",   help="Create Kafka topics")
    args = parser.parse_args()

    if args.create_topics:
        FraudDetectionProducer.create_topics()

    producer = FraudDetectionProducer()
    producer.stream_transactions(
        tps=args.tps,
        duration_seconds=args.duration,
        fraud_rate=args.fraud_rate,
    )
