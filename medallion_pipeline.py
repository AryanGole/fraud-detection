"""
============================================================
Fraud Detection Pipeline - PySpark Medallion Architecture
============================================================
Bronze → Silver → Gold Data Lake transformations.
Handles deduplication, enrichment, schema evolution,
and data quality validation at each layer.

Author: Senior ML Engineering Team
Version: 2.1.0
"""

from pyspark.sql import SparkSession, DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import *
from delta.tables import DeltaTable
from typing import Dict, Optional
import logging

log = logging.getLogger(__name__)

# ── Spark Session Factory ──────────────────────────────────────────────────────
def create_spark_session(app_name: str = "FraudDetection-ETL",
                         env: str = "local") -> SparkSession:
    """
    Create optimized SparkSession for fraud detection workloads.
    Configured for Delta Lake + AWS S3 access.
    """
    builder = (SparkSession.builder
               .appName(app_name)
               # Delta Lake extensions
               .config("spark.sql.extensions",
                       "io.delta.sql.DeltaSparkSessionExtension")
               .config("spark.sql.catalog.spark_catalog",
                       "org.apache.spark.sql.delta.catalog.DeltaCatalog")
               # Performance
               .config("spark.sql.adaptive.enabled", "true")
               .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
               .config("spark.sql.shuffle.partitions", "200")
               .config("spark.default.parallelism", "200")
               # Memory
               .config("spark.executor.memory", "8g")
               .config("spark.driver.memory",   "4g")
               .config("spark.executor.memoryOverhead", "2g")
               # Delta optimizations
               .config("spark.databricks.delta.optimizeWrite.enabled", "true")
               .config("spark.databricks.delta.autoCompact.enabled", "true")
    )

    if env == "aws":
        builder = (builder
                   .config("spark.hadoop.fs.s3a.impl",
                           "org.apache.hadoop.fs.s3a.S3AFileSystem")
                   .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                           "com.amazonaws.auth.DefaultAWSCredentialsProviderChain"))

    return builder.getOrCreate()


# ══════════════════════════════════════════════════════════════════════════════
# BRONZE LAYER: Raw Ingestion
# ══════════════════════════════════════════════════════════════════════════════
def ingest_bronze(spark: SparkSession,
                  source_path: str,
                  bronze_path: str,
                  source_format: str = "json") -> DataFrame:
    """
    Ingest raw transaction data into Bronze Delta table.
    Adds ingestion metadata; preserves raw data intact.
    
    Idempotent: uses MERGE to avoid duplicates on re-runs.
    """
    log.info(f"Ingesting Bronze layer from: {source_path}")

    # Read source (Kafka S3 sink, batch files, etc.)
    df_raw = (spark.read
              .format(source_format)
              .option("multiline", "true")
              .option("inferSchema", "true")
              .load(source_path))

    # Add ingestion metadata
    df_raw = (df_raw
              .withColumn("_ingestion_ts",   F.current_timestamp())
              .withColumn("_source_path",    F.lit(source_path))
              .withColumn("_record_hash",    F.sha2(F.to_json(F.struct("*")), 256))
              .withColumn("_batch_id",       F.lit(str(spark.sparkContext.applicationId)))
              .withColumn("_partition_date", F.to_date(F.col("timestamp")))
    )

    # MERGE: Upsert to handle reprocessing
    if DeltaTable.isDeltaTable(spark, bronze_path):
        bronze_table = DeltaTable.forPath(spark, bronze_path)
        (bronze_table.alias("target")
         .merge(df_raw.alias("source"),
                "target.transaction_id = source.transaction_id")
         .whenNotMatchedInsertAll()
         .execute())
        log.info(f"Bronze MERGE complete for {source_path}")
    else:
        (df_raw.write
         .format("delta")
         .mode("overwrite")
         .partitionBy("_partition_date")
         .save(bronze_path))
        log.info(f"Bronze table created at {bronze_path}")

    count = spark.read.format("delta").load(bronze_path).count()
    log.info(f"Bronze table records: {count:,}")
    return spark.read.format("delta").load(bronze_path)


# ══════════════════════════════════════════════════════════════════════════════
# SILVER LAYER: Cleansed & Enriched
# ══════════════════════════════════════════════════════════════════════════════
def transform_silver(spark: SparkSession,
                     bronze_path: str,
                     silver_path: str,
                     customer_ref_path: str = None,
                     merchant_ref_path: str = None) -> DataFrame:
    """
    Transform Bronze to Silver:
    - Deduplicate transactions
    - Validate and cast data types
    - Enrich with customer/merchant reference data
    - Apply data quality flags
    - Standardize currencies to INR
    """
    log.info("Transforming Bronze → Silver...")

    df = spark.read.format("delta").load(bronze_path)
    initial_count = df.count()

    # ── 1. Deduplication ─────────────────────────────────────────────────────
    window_dedup = Window.partitionBy("transaction_id").orderBy(
        F.col("_ingestion_ts").desc()
    )
    df = (df
          .withColumn("_row_num", F.row_number().over(window_dedup))
          .filter(F.col("_row_num") == 1)
          .drop("_row_num"))

    dedup_count = df.count()
    log.info(f"Deduplication: {initial_count:,} → {dedup_count:,} records "
             f"({initial_count - dedup_count:,} duplicates removed)")

    # ── 2. Data Type Casting & Validation ─────────────────────────────────────
    df = (df
          .withColumn("timestamp",   F.to_timestamp("timestamp"))
          .withColumn("amount",      F.col("amount").cast(DoubleType()))
          .withColumn("latitude",    F.col("latitude").cast(DoubleType()))
          .withColumn("longitude",   F.col("longitude").cast(DoubleType()))
          .withColumn("hour_of_day", F.hour("timestamp"))
          .withColumn("day_of_week", F.dayofweek("timestamp"))
          .withColumn("week_of_year",F.weekofyear("timestamp"))
          .withColumn("year_month",  F.date_format("timestamp", "yyyy-MM"))
    )

    # ── 3. Data Quality Flags ─────────────────────────────────────────────────
    df = (df
          .withColumn("dq_amount_valid",
                      F.when((F.col("amount") > 0) & (F.col("amount") < 10_000_000),
                             F.lit(True)).otherwise(F.lit(False)))
          .withColumn("dq_timestamp_valid",
                      F.when(F.col("timestamp").isNotNull() &
                             (F.col("timestamp") > F.lit("2020-01-01")),
                             F.lit(True)).otherwise(F.lit(False)))
          .withColumn("dq_customer_valid",
                      F.col("customer_id").isNotNull() &
                      (F.length("customer_id") > 0))
          .withColumn("dq_passed",
                      F.col("dq_amount_valid") &
                      F.col("dq_timestamp_valid") &
                      F.col("dq_customer_valid"))
    )

    dq_pass_rate = df.filter("dq_passed").count() / max(df.count(), 1)
    log.info(f"Data quality pass rate: {dq_pass_rate:.2%}")

    # ── 4. Currency Normalization to INR ──────────────────────────────────────
    # Static FX rates (in production, join to live FX table)
    FX_TO_INR = {
        "USD": 83.50,
        "GBP": 105.20,
        "EUR": 90.15,
        "SGD": 62.40,
        "AED": 22.73,
        "CNY": 11.52,
        "INR": 1.00,
    }

    fx_expr = F.lit(1.0)
    for currency, rate in FX_TO_INR.items():
        fx_expr = F.when(F.col("currency") == currency, F.lit(rate)).otherwise(fx_expr)

    df = (df
          .withColumn("fx_rate_to_inr",    fx_expr)
          .withColumn("amount_inr",         F.round(F.col("amount") * F.col("fx_rate_to_inr"), 2))
    )

    # ── 5. Customer Enrichment (if reference data available) ──────────────────
    if customer_ref_path:
        cust_df = (spark.read.format("delta").load(customer_ref_path)
                   .select("customer_id", "customer_segment", "kyc_status",
                           "account_age_days", "credit_score"))
        df = df.join(cust_df, on="customer_id", how="left")
        log.info("Customer reference data joined")

    # ── 6. Merchant Enrichment ─────────────────────────────────────────────────
    if merchant_ref_path:
        merch_df = (spark.read.format("delta").load(merchant_ref_path)
                    .select("merchant_id", "merchant_name", "merchant_risk_band",
                            "mcc_code", "on_watchlist"))
        df = df.join(merch_df, on="merchant_id", how="left")
        log.info("Merchant reference data joined")

    # ── 7. Write Silver Layer ──────────────────────────────────────────────────
    # Filter to DQ-passed records for main table; preserve failures in quarantine
    df_clean      = df.filter("dq_passed")
    df_quarantine = df.filter("NOT dq_passed")

    quarantine_count = df_quarantine.count()
    if quarantine_count > 0:
        quarantine_path = silver_path.replace("/silver/", "/quarantine/")
        df_quarantine.write.format("delta").mode("append").save(quarantine_path)
        log.warning(f"{quarantine_count:,} records sent to quarantine: {quarantine_path}")

    (df_clean
     .write
     .format("delta")
     .mode("overwrite")
     .partitionBy("year_month", "transaction_type")
     .option("overwriteSchema", "true")
     .save(silver_path))

    log.info(f"Silver table written: {df_clean.count():,} clean records")
    return df_clean


# ══════════════════════════════════════════════════════════════════════════════
# DATA QUALITY MONITORING
# ══════════════════════════════════════════════════════════════════════════════
def run_data_quality_checks(spark: SparkSession, path: str, layer: str = "silver") -> Dict:
    """
    Run automated data quality checks and return metrics.
    Integrates with Great Expectations / alerting systems.
    """
    df = spark.read.format("delta").load(path)

    checks = {}

    # Completeness checks
    for col in ["transaction_id", "customer_id", "amount", "timestamp"]:
        null_rate = df.filter(F.col(col).isNull()).count() / max(df.count(), 1)
        checks[f"{col}_null_rate"] = null_rate
        if null_rate > 0.01:
            log.warning(f"[DQ] {layer}.{col}: null rate {null_rate:.2%} exceeds 1% threshold")

    # Amount range checks
    amount_stats = df.agg(
        F.min("amount").alias("min"),
        F.max("amount").alias("max"),
        F.avg("amount").alias("mean"),
        F.stddev("amount").alias("stddev"),
    ).collect()[0]
    checks["amount_stats"] = dict(amount_stats.asDict())

    # Fraud rate (if labeled)
    if "is_fraud" in df.columns:
        fraud_rate = df.agg(F.avg("is_fraud")).collect()[0][0]
        checks["fraud_rate"] = fraud_rate
        if fraud_rate > 0.05:
            log.warning(f"[DQ] Unusually high fraud rate: {fraud_rate:.2%}")

    # Freshness check
    max_ts = df.agg(F.max("timestamp")).collect()[0][0]
    checks["max_timestamp"] = str(max_ts)

    log.info(f"[DQ] {layer} layer checks complete: {checks}")
    return checks


# ══════════════════════════════════════════════════════════════════════════════
# BATCH SCORING JOB (PySpark)
# ══════════════════════════════════════════════════════════════════════════════
def batch_score_transactions(spark:        SparkSession,
                              feature_path: str,
                              model_path:   str,
                              output_path:  str) -> None:
    """
    Distributed batch scoring of transactions using loaded ML model.
    Processes millions of records using PySpark UDF.
    """
    import mlflow.pyfunc
    import pandas as pd

    log.info("Starting batch scoring job...")

    # Load MLflow model as PySpark UDF
    model = mlflow.pyfunc.load_model(model_path)

    FEATURE_COLS = [
        "txn_count_1h", "txn_count_24h", "txn_count_7d",
        "txn_amount_1h", "txn_amount_24h", "amt_zscore_30d",
        "velocity_spike", "max_single_day",
        "avg_txn_amount_30d", "median_txn_amount_30d", "amount_ratio_to_avg",
        "pct_international_30d", "pct_cnp_30d", "unique_merchants_7d",
        "unique_countries_30d", "typical_amount_band",
        "days_since_first_txn", "customer_risk_score_hist",
        "distance_from_last_txn_km", "time_since_last_txn_min",
        "implied_travel_speed_kmph", "is_impossible_travel",
        "is_new_country", "home_country_pct",
        "device_txn_count_24h", "device_customer_count_7d",
        "is_new_device", "device_fraud_rate_30d",
        "ip_txn_count_1h", "multi_account_device",
        "merchant_txn_count_7d", "merchant_fraud_rate_30d",
        "merchant_avg_amount_30d", "merchant_amount_deviation",
        "merchant_unique_cards_7d", "merchant_risk_tier", "is_mcc_high_risk",
        "is_fraud_hour", "is_weekend", "is_month_end",
        "days_since_last_txn", "txn_hour_sin", "txn_hour_cos",
        "amount", "is_international", "is_high_risk_merchant",
    ]

    # Define Pandas UDF for distributed scoring
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import DoubleType

    @pandas_udf(DoubleType())
    def score_transactions(*cols) -> pd.Series:
        features = pd.concat(list(cols), axis=1)
        features.columns = FEATURE_COLS
        predictions = model.predict(features)
        return pd.Series(predictions)

    # Load feature store
    df = (spark.read
          .format("delta")
          .load(feature_path)
          .fillna(0.0, subset=FEATURE_COLS))

    # Apply scoring UDF
    df_scored = (df
                 .withColumn("fraud_risk_score",
                             score_transactions(*[F.col(c) for c in FEATURE_COLS]))
                 .withColumn("risk_band",
                             F.when(F.col("fraud_risk_score") >= 85, F.lit("CRITICAL"))
                              .when(F.col("fraud_risk_score") >= 65, F.lit("HIGH"))
                              .when(F.col("fraud_risk_score") >= 45, F.lit("MEDIUM"))
                              .otherwise(F.lit("LOW")))
                 .withColumn("scored_at", F.current_timestamp()))

    # Write scored transactions
    (df_scored
     .write
     .format("delta")
     .mode("overwrite")
     .partitionBy("risk_band")
     .save(output_path))

    # Summary metrics
    summary = (df_scored
               .groupBy("risk_band")
               .agg(
                   F.count("*").alias("count"),
                   F.avg("fraud_risk_score").alias("avg_score"),
               )
               .orderBy(F.col("count").desc()))

    log.info("Batch scoring complete:")
    summary.show()


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fraud Detection PySpark ETL")
    parser.add_argument("--stage",      choices=["bronze", "silver", "gold", "score"],
                        required=True)
    parser.add_argument("--source",     type=str, required=True)
    parser.add_argument("--output",     type=str, required=True)
    parser.add_argument("--env",        type=str, default="local")
    parser.add_argument("--model-path", type=str, default=None)
    args = parser.parse_args()

    spark = create_spark_session(env=args.env)

    if args.stage == "bronze":
        ingest_bronze(spark, args.source, args.output)
    elif args.stage == "silver":
        transform_silver(spark, args.source, args.output)
    elif args.stage == "gold":
        from feature_engineering.feature_pipeline import build_feature_store
        build_feature_store(spark, args.source, args.output)
    elif args.stage == "score":
        batch_score_transactions(spark, args.source, args.model_path, args.output)

    spark.stop()
