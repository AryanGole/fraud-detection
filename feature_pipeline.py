"""
============================================================
Fraud Detection Pipeline - Feature Engineering
============================================================
42 features across 6 categories:
  1. Transaction Velocity (8 features)
  2. Customer Behavioral Profile (10 features)
  3. Geographic Anomaly Detection (6 features)
  4. Device & Session Fingerprinting (6 features)
  5. Merchant Risk Indicators (7 features)
  6. Temporal Pattern Analysis (5 features)

Author: Senior ML Engineering Team
Version: 2.1.0
"""

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    IntegerType, BooleanType, TimestampType, LongType
)
import logging

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# TRANSACTION SCHEMA
# ══════════════════════════════════════════════════════════════════════════════
TRANSACTION_SCHEMA = StructType([
    StructField("transaction_id",       StringType(),    False),
    StructField("customer_id",          StringType(),    False),
    StructField("account_number",       StringType(),    False),
    StructField("transaction_type",     StringType(),    False),
    StructField("amount",               DoubleType(),    False),
    StructField("currency",             StringType(),    True),
    StructField("merchant_id",          StringType(),    True),
    StructField("merchant_category",    StringType(),    True),
    StructField("timestamp",            TimestampType(), False),
    StructField("device_id",            StringType(),    True),
    StructField("device_type",          StringType(),    True),
    StructField("ip_address",           StringType(),    True),
    StructField("latitude",             DoubleType(),    True),
    StructField("longitude",            DoubleType(),    True),
    StructField("country_code",         StringType(),    True),
    StructField("city",                 StringType(),    True),
    StructField("card_present",         BooleanType(),   True),
    StructField("card_bin",             StringType(),    True),
    StructField("bank_code",            StringType(),    True),
    StructField("channel",              StringType(),    True),
    StructField("is_international",     BooleanType(),   True),
    StructField("is_high_risk_merchant",BooleanType(),   True),
    StructField("hour_of_day",          IntegerType(),   True),
    StructField("day_of_week",          IntegerType(),   True),
    StructField("is_fraud",             IntegerType(),   True),  # Label (0/1)
])

# High-risk merchant categories
HIGH_RISK_MCCS = ["CASINO", "CRYPTO_EXCHANGE", "MONEY_TRANSFER", "ONLINE_GAMING", "ADULT_CONTENT"]

# Fraud-prone hours (1am-4am)
FRAUD_HOURS = [1, 2, 3, 4]

# Weekend days
WEEKEND_DAYS = [5, 6]


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE CATEGORY 1: TRANSACTION VELOCITY
# ══════════════════════════════════════════════════════════════════════════════
def add_velocity_features(df: DataFrame) -> DataFrame:
    """
    Compute rolling transaction counts and amounts over multiple windows.
    
    Features generated:
      - txn_count_1h:     Transaction count in last 1 hour per customer
      - txn_count_24h:    Transaction count in last 24 hours per customer
      - txn_amount_1h:    Total amount in last 1 hour per customer
      - txn_amount_24h:   Total amount in last 24 hours per customer
      - txn_count_7d:     Transaction count in last 7 days per customer
      - amt_zscore_30d:   Z-score of amount vs 30-day customer baseline
      - velocity_spike:   Binary flag if 1h count > 2x 24h hourly rate
      - max_single_day:   Maximum single-day transaction amount (30d lookback)
    """
    log.info("Computing velocity features...")

    # Window specs using Spark window functions (event-time based)
    window_1h  = Window.partitionBy("customer_id").orderBy(F.col("timestamp").cast("long")).rangeBetween(-3600, 0)
    window_24h = Window.partitionBy("customer_id").orderBy(F.col("timestamp").cast("long")).rangeBetween(-86400, 0)
    window_7d  = Window.partitionBy("customer_id").orderBy(F.col("timestamp").cast("long")).rangeBetween(-604800, 0)
    window_30d = Window.partitionBy("customer_id").orderBy(F.col("timestamp").cast("long")).rangeBetween(-2592000, 0)

    df = df.withColumn("txn_count_1h",   F.count("transaction_id").over(window_1h))
    df = df.withColumn("txn_count_24h",  F.count("transaction_id").over(window_24h))
    df = df.withColumn("txn_count_7d",   F.count("transaction_id").over(window_7d))
    df = df.withColumn("txn_amount_1h",  F.sum("amount").over(window_1h))
    df = df.withColumn("txn_amount_24h", F.sum("amount").over(window_24h))

    # Z-score of current transaction amount vs 30-day mean/stddev
    df = df.withColumn("amt_mean_30d",   F.avg("amount").over(window_30d))
    df = df.withColumn("amt_stddev_30d", F.stddev("amount").over(window_30d))
    df = df.withColumn(
        "amt_zscore_30d",
        F.when(F.col("amt_stddev_30d") > 0,
               (F.col("amount") - F.col("amt_mean_30d")) / F.col("amt_stddev_30d")
        ).otherwise(0.0)
    )

    # Velocity spike: 1h transactions > 2x average hourly rate in past 24h
    df = df.withColumn(
        "velocity_spike",
        F.when(
            (F.col("txn_count_24h") > 0) &
            (F.col("txn_count_1h") > (F.col("txn_count_24h") / 24.0) * 2),
            F.lit(1)
        ).otherwise(F.lit(0))
    )

    # Add date column first, then window over it by string name
    df = df.withColumn("_txn_date", F.to_date("timestamp"))
    window_daily = Window.partitionBy("customer_id", "_txn_date")
    df = df.withColumn("daily_total",  F.sum("amount").over(window_daily))
    df = df.withColumn("max_single_day", F.max("daily_total").over(window_30d))
    df = df.drop("_txn_date")

    # Clean up intermediate columns
    df = df.drop("amt_mean_30d", "amt_stddev_30d", "daily_total")

    return df# ══════════════════════════════════════════════════════════════════════════════
# FEATURE CATEGORY 2: CUSTOMER BEHAVIORAL PROFILE
# ══════════════════════════════════════════════════════════════════════════════
def add_behavioral_features(df: DataFrame) -> DataFrame:
    """
    Capture customer spending behavior and deviations from baseline.

    Features generated:
      - avg_txn_amount_30d:        Customer 30-day average transaction
      - median_txn_amount_30d:     Customer 30-day median transaction
      - amount_ratio_to_avg:       Current txn / 30d avg (deviation factor)
      - pct_international_30d:     % of international txns in 30 days
      - pct_cnp_30d:               % of card-not-present txns in 30 days
      - unique_merchants_7d:       Distinct merchants in 7 days
      - unique_countries_30d:      Distinct countries in 30 days
      - typical_amount_band:       Encoded amount band (micro/small/medium/large)
      - days_since_first_txn:      Customer tenure in days
      - customer_risk_score_hist:  Aggregate historical risk score
    """
    log.info("Computing behavioral features...")

    window_30d = Window.partitionBy("customer_id").orderBy(F.col("timestamp").cast("long")).rangeBetween(-2592000, 0)
    window_7d  = Window.partitionBy("customer_id").orderBy(F.col("timestamp").cast("long")).rangeBetween(-604800, 0)
    window_all = Window.partitionBy("customer_id").orderBy("timestamp")

    df = df.withColumn("avg_txn_amount_30d",
                       F.avg("amount").over(window_30d))

    # Approx median using percentile
    df = df.withColumn("median_txn_amount_30d",
                       F.percentile_approx("amount", 0.5).over(window_30d))

    df = df.withColumn(
        "amount_ratio_to_avg",
        F.when(F.col("avg_txn_amount_30d") > 0,
               F.col("amount") / F.col("avg_txn_amount_30d")
        ).otherwise(1.0)
    )

    df = df.withColumn(
        "pct_international_30d",
        F.avg(F.col("is_international").cast("double")).over(window_30d)
    )

    df = df.withColumn(
        "pct_cnp_30d",
        F.avg((~F.col("card_present")).cast("double")).over(window_30d)
    )

    df = df.withColumn("unique_merchants_7d",
                       F.approx_count_distinct("merchant_id").over(window_7d))

    df = df.withColumn("unique_countries_30d",
                       F.approx_count_distinct("country_code").over(window_30d))

    # Amount band encoding
    df = df.withColumn(
        "typical_amount_band",
        F.when(F.col("amount") < 500,    F.lit(0))  # Micro
         .when(F.col("amount") < 5000,   F.lit(1))  # Small
         .when(F.col("amount") < 50000,  F.lit(2))  # Medium
         .when(F.col("amount") < 500000, F.lit(3))  # Large
         .otherwise(F.lit(4))                         # Very Large
    )

    # Customer tenure (days since first transaction)
    df = df.withColumn("first_txn_ts",
                       F.min("timestamp").over(window_all))
    df = df.withColumn(
        "days_since_first_txn",
        F.datediff(F.col("timestamp"), F.col("first_txn_ts"))
    )

    # Historical risk: weighted sum of past triggered rules
    df = df.withColumn(
        "customer_risk_score_hist",
        F.avg(F.coalesce(F.col("rule_based_score"), F.lit(0))).over(window_30d)
        if "rule_based_score" in df.columns
        else F.lit(0.0)
    )

    df = df.drop("first_txn_ts")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE CATEGORY 3: GEOGRAPHIC ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def add_geo_features(df: DataFrame) -> DataFrame:
    """
    Detect geographically anomalous transactions (impossible travel, etc.).

    Features generated:
      - distance_from_last_txn_km:   Great-circle distance from previous txn location
      - time_since_last_txn_min:     Minutes since last transaction
      - implied_travel_speed_kmph:   Implied travel speed (impossible > 1000 km/h)
      - is_impossible_travel:        Flag for physically impossible location change
      - is_new_country:              Flag for first transaction in this country (30d)
      - home_country_pct:            % of txns from customer's primary country (90d)
    """
    log.info("Computing geographic anomaly features...")

    window_prev = Window.partitionBy("customer_id").orderBy("timestamp")
    window_90d  = Window.partitionBy("customer_id").orderBy(F.col("timestamp").cast("long")).rangeBetween(-7776000, 0)

    # Previous transaction location and time
    df = df.withColumn("prev_lat",  F.lag("latitude",  1).over(window_prev))
    df = df.withColumn("prev_lon",  F.lag("longitude", 1).over(window_prev))
    df = df.withColumn("prev_ts",   F.lag("timestamp", 1).over(window_prev))

    # Haversine distance approximation using PySpark built-ins
    # Using simplified planar distance (adequate for ML features)
    R = 6371.0  # Earth radius km
    df = df.withColumn(
        "distance_from_last_txn_km",
        F.when(
            F.col("prev_lat").isNotNull(),
            F.sqrt(
                F.pow((F.col("latitude")  - F.col("prev_lat"))  * F.lit(111.0), 2) +
                F.pow((F.col("longitude") - F.col("prev_lon"))  * F.lit(111.0) *
                      F.cos(F.radians(F.col("latitude"))), 2)
            )
        ).otherwise(F.lit(0.0))
    )

    # Time difference in minutes
    df = df.withColumn(
        "time_since_last_txn_min",
        F.when(
            F.col("prev_ts").isNotNull(),
            (F.col("timestamp").cast("long") - F.col("prev_ts").cast("long")) / 60.0
        ).otherwise(F.lit(0.0))
    )

    # Implied travel speed
    df = df.withColumn(
        "implied_travel_speed_kmph",
        F.when(
            (F.col("time_since_last_txn_min") > 0) &
            (F.col("distance_from_last_txn_km") > 0),
            F.col("distance_from_last_txn_km") /
            (F.col("time_since_last_txn_min") / 60.0)
        ).otherwise(F.lit(0.0))
    )

    # Impossible travel flag (> 900 km/h ≈ commercial aircraft speed)
    df = df.withColumn(
        "is_impossible_travel",
        F.when(F.col("implied_travel_speed_kmph") > 900, F.lit(1))
         .otherwise(F.lit(0))
    )

    # New country flag
    df = df.withColumn(
        "country_txn_count_30d",
        F.count("transaction_id").over(
            Window.partitionBy("customer_id", "country_code")
                  .orderBy(F.col("timestamp").cast("long"))
                  .rangeBetween(-2592000, 0)
        )
    )
    df = df.withColumn(
        "is_new_country",
        F.when(F.col("country_txn_count_30d") <= 1, F.lit(1)).otherwise(F.lit(0))
    )

    # Primary country percentage in 90 days
    df = df.withColumn("total_txn_90d",
                       F.count("transaction_id").over(window_90d))
    df = df.withColumn(
        "home_country_pct",
        F.when(
            F.col("total_txn_90d") > 0,
            F.count(F.when(F.col("country_code") == "IND", 1)).over(window_90d) /
            F.col("total_txn_90d")
        ).otherwise(F.lit(1.0))
    )

    df = df.drop("prev_lat", "prev_lon", "prev_ts", "country_txn_count_30d", "total_txn_90d")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE CATEGORY 4: DEVICE & SESSION FINGERPRINTING
# ══════════════════════════════════════════════════════════════════════════════
def add_device_features(df: DataFrame) -> DataFrame:
    """
    Detect device anomalies and session-based fraud patterns.

    Features generated:
      - device_txn_count_24h:     Transactions from this device in 24h
      - device_customer_count_7d: Distinct customers using this device in 7 days
      - is_new_device:            Flag if device not seen in past 30 days
      - device_fraud_rate_30d:    Historical fraud rate for this device
      - ip_txn_count_1h:          Transactions from this IP in 1h
      - multi_account_device:     Device used by multiple customers
    """
    log.info("Computing device features...")

    window_dev_24h = Window.partitionBy("device_id").orderBy(F.col("timestamp").cast("long")).rangeBetween(-86400, 0)
    window_dev_7d  = Window.partitionBy("device_id").orderBy(F.col("timestamp").cast("long")).rangeBetween(-604800, 0)
    window_dev_30d = Window.partitionBy("device_id").orderBy(F.col("timestamp").cast("long")).rangeBetween(-2592000, 0)
    window_ip_1h   = Window.partitionBy("ip_address").orderBy(F.col("timestamp").cast("long")).rangeBetween(-3600, 0)

    df = df.withColumn("device_txn_count_24h",
                       F.count("transaction_id").over(window_dev_24h))

    df = df.withColumn("device_customer_count_7d",
                       F.approx_count_distinct("customer_id").over(window_dev_7d))

    # New device flag: first time seen in 30 days
    df = df.withColumn("device_txn_count_30d",
                       F.count("transaction_id").over(window_dev_30d))
    df = df.withColumn(
        "is_new_device",
        F.when(F.col("device_txn_count_30d") <= 1, F.lit(1)).otherwise(F.lit(0))
    )

    # Device fraud rate (only meaningful with labeled data)
    if "is_fraud" in df.columns:
        df = df.withColumn("device_fraud_rate_30d",
                           F.avg("is_fraud").over(window_dev_30d))
    else:
        df = df.withColumn("device_fraud_rate_30d", F.lit(0.0))

    df = df.withColumn("ip_txn_count_1h",
                       F.count("transaction_id").over(window_ip_1h))

    # Multi-account device: suspicious if >2 customers on one device
    df = df.withColumn(
        "multi_account_device",
        F.when(F.col("device_customer_count_7d") > 2, F.lit(1)).otherwise(F.lit(0))
    )

    df = df.drop("device_txn_count_30d")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE CATEGORY 5: MERCHANT RISK INDICATORS
# ══════════════════════════════════════════════════════════════════════════════
def add_merchant_features(df: DataFrame) -> DataFrame:
    """
    Assess merchant-level fraud risk based on transaction history.

    Features generated:
      - merchant_txn_count_7d:     Total transactions at merchant in 7 days
      - merchant_fraud_rate_30d:   Fraud rate at this merchant (30d)
      - merchant_avg_amount_30d:   Average transaction amount at merchant
      - merchant_amount_deviation: Current txn deviation from merchant avg
      - merchant_unique_cards_7d:  Distinct cards used at merchant in 7 days
      - merchant_risk_tier:        Encoded merchant risk category (0-4)
      - is_mcc_high_risk:          Binary flag for high-risk MCC
    """
    log.info("Computing merchant risk features...")

    window_mer_7d  = Window.partitionBy("merchant_id").orderBy(F.col("timestamp").cast("long")).rangeBetween(-604800, 0)
    window_mer_30d = Window.partitionBy("merchant_id").orderBy(F.col("timestamp").cast("long")).rangeBetween(-2592000, 0)

    df = df.withColumn("merchant_txn_count_7d",
                       F.count("transaction_id").over(window_mer_7d))

    if "is_fraud" in df.columns:
        df = df.withColumn("merchant_fraud_rate_30d",
                           F.avg("is_fraud").over(window_mer_30d))
    else:
        df = df.withColumn("merchant_fraud_rate_30d", F.lit(0.0))

    df = df.withColumn("merchant_avg_amount_30d",
                       F.avg("amount").over(window_mer_30d))

    df = df.withColumn(
        "merchant_amount_deviation",
        F.when(
            F.col("merchant_avg_amount_30d") > 0,
            F.abs(F.col("amount") - F.col("merchant_avg_amount_30d")) /
            F.col("merchant_avg_amount_30d")
        ).otherwise(F.lit(0.0))
    )

    df = df.withColumn("merchant_unique_cards_7d",
                       F.approx_count_distinct("card_bin").over(window_mer_7d))

    # Merchant risk tier (0 = safest, 4 = highest risk)
    high_risk_list = F.array([F.lit(c) for c in HIGH_RISK_MCCS])
    df = df.withColumn(
        "merchant_risk_tier",
        F.when(F.col("is_high_risk_merchant"), F.lit(4))
         .when(F.col("merchant_category").isin("JEWELRY", "TRAVEL"), F.lit(3))
         .when(F.col("merchant_category").isin("ELECTRONICS", "FUEL"), F.lit(2))
         .when(F.col("merchant_category").isin("RESTAURANT", "GROCERY"), F.lit(0))
         .otherwise(F.lit(1))
    )

    df = df.withColumn(
        "is_mcc_high_risk",
        F.when(F.col("is_high_risk_merchant"), F.lit(1)).otherwise(F.lit(0))
    )

    return df


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE CATEGORY 6: TEMPORAL PATTERN ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
def add_temporal_features(df: DataFrame) -> DataFrame:
    """
    Extract time-based patterns correlated with fraud.

    Features generated:
      - is_fraud_hour:         1 if transaction in 1-4am window
      - is_weekend:            1 if Saturday/Sunday
      - is_month_end:          1 if last 3 days of month (salary fraud)
      - days_since_last_txn:   Days since customer's previous transaction
      - txn_hour_sin/cos:      Cyclical encoding of hour (prevents ordinal bias)
    """
    log.info("Computing temporal features...")

    df = df.withColumn(
        "is_fraud_hour",
        F.when(F.col("hour_of_day").isin(FRAUD_HOURS), F.lit(1)).otherwise(F.lit(0))
    )

    df = df.withColumn(
        "is_weekend",
        F.when(F.col("day_of_week").isin(WEEKEND_DAYS), F.lit(1)).otherwise(F.lit(0))
    )

    df = df.withColumn("day_of_month", F.dayofmonth("timestamp"))
    df = df.withColumn(
        "is_month_end",
        F.when(F.col("day_of_month") >= 28, F.lit(1)).otherwise(F.lit(0))
    )

    # Previous transaction time for this customer
    window_prev = Window.partitionBy("customer_id").orderBy("timestamp")
    df = df.withColumn("prev_txn_ts", F.lag("timestamp", 1).over(window_prev))
    df = df.withColumn(
        "days_since_last_txn",
        F.when(
            F.col("prev_txn_ts").isNotNull(),
            F.datediff(F.col("timestamp"), F.col("prev_txn_ts")).cast("double")
        ).otherwise(F.lit(30.0))  # Default for first transaction
    )

    # Cyclical time encoding (prevents model treating hour 23 as far from hour 0)
    import math
    df = df.withColumn("txn_hour_sin",
                       F.sin(F.col("hour_of_day") * (2 * math.pi / 24)))
    df = df.withColumn("txn_hour_cos",
                       F.cos(F.col("hour_of_day") * (2 * math.pi / 24)))

    df = df.drop("prev_txn_ts", "day_of_month")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MASTER FEATURE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def build_feature_store(
    spark:      SparkSession,
    input_path: str,
    output_path: str,
    mode:       str = "overwrite",
) -> DataFrame:
    """
    Orchestrate all feature engineering stages and persist Gold layer.

    Args:
        spark:       Active SparkSession
        input_path:  Silver layer Delta table path
        output_path: Gold layer output path (Delta format)
        mode:        Write mode ('overwrite' or 'append')

    Returns:
        Enriched DataFrame with all 42 features
    """
    log.info("Starting feature engineering pipeline", input=input_path, output=output_path)

    # Read Silver layer
    df = spark.read.format("delta").load(input_path)
    log.info(f"Loaded {df.count():,} transactions from Silver layer")

    # Apply all feature groups
    df = add_velocity_features(df)
    df = add_behavioral_features(df)
    df = add_geo_features(df)
    df = add_device_features(df)
    df = add_merchant_features(df)
    df = add_temporal_features(df)

    # Select final feature set (42 features + metadata)
    FEATURE_COLUMNS = [
        # Identity
        "transaction_id", "customer_id", "timestamp",
        # Raw features
        "amount", "transaction_type", "is_international",
        "card_present", "is_high_risk_merchant", "hour_of_day", "day_of_week",
        # Velocity (8)
        "txn_count_1h", "txn_count_24h", "txn_count_7d",
        "txn_amount_1h", "txn_amount_24h",
        "amt_zscore_30d", "velocity_spike", "max_single_day",
        # Behavioral (10)
        "avg_txn_amount_30d", "median_txn_amount_30d", "amount_ratio_to_avg",
        "pct_international_30d", "pct_cnp_30d", "unique_merchants_7d",
        "unique_countries_30d", "typical_amount_band",
        "days_since_first_txn", "customer_risk_score_hist",
        # Geographic (6)
        "distance_from_last_txn_km", "time_since_last_txn_min",
        "implied_travel_speed_kmph", "is_impossible_travel",
        "is_new_country", "home_country_pct",
        # Device (6)
        "device_txn_count_24h", "device_customer_count_7d",
        "is_new_device", "device_fraud_rate_30d",
        "ip_txn_count_1h", "multi_account_device",
        # Merchant (7)
        "merchant_txn_count_7d", "merchant_fraud_rate_30d",
        "merchant_avg_amount_30d", "merchant_amount_deviation",
        "merchant_unique_cards_7d", "merchant_risk_tier", "is_mcc_high_risk",
        # Temporal (5)
        "is_fraud_hour", "is_weekend", "is_month_end",
        "days_since_last_txn", "txn_hour_sin", "txn_hour_cos",
    ]

    # Add label if available
    if "is_fraud" in df.columns:
        FEATURE_COLUMNS.append("is_fraud")

    df_features = df.select(*[c for c in FEATURE_COLUMNS if c in df.columns])

    # Fill nulls with safe defaults
    numeric_features = [f.name for f in df_features.schema.fields
                        if f.dataType in (DoubleType(), IntegerType(), LongType())]
    df_features = df_features.fillna(0.0, subset=numeric_features)

    # Write Gold layer
    (df_features
        .write
        .format("delta")
        .mode(mode)
        .partitionBy("transaction_type")
        .option("overwriteSchema", "true")
        .save(output_path))

    feature_count = len([c for c in FEATURE_COLUMNS if c in df_features.columns]) - 3  # excl. ID cols
    log.info(f"Feature store written: {df_features.count():,} records, {feature_count} features",
             output=output_path)

    return df_features


# ── Standalone execution ───────────────────────────────────────────────────────
if __name__ == "__main__":
    spark = (SparkSession.builder
             .appName("FraudDetection-FeatureEngineering")
             .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
             .config("spark.sql.catalog.spark_catalog",
                     "org.apache.spark.sql.delta.catalog.DeltaCatalog")
             .getOrCreate())

    spark.sparkContext.setLogLevel("WARN")

    build_feature_store(
        spark,
        input_path  = "s3a://fraud-detection-bucket/silver/transactions/",
        output_path = "s3a://fraud-detection-bucket/gold/feature_store/",
        mode        = "overwrite",
    )
