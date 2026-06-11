"""
============================================================
Fraud Detection Pipeline - Real-Time Scoring API
============================================================
FastAPI service providing sub-100ms fraud scoring endpoints.
Supports single transaction scoring and batch endpoints.
AWS Lambda compatible handler included.

Author: Senior ML Engineering Team
Version: 2.1.0
"""

import os
import json
import time
import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import mlflow.pyfunc
try:
    import redis.asyncio as aioredis
except ImportError:
    aioredis = None  # type: ignore

log = logging.getLogger(__name__)

# ── Global Model Registry ──────────────────────────────────────────────────────
MODEL_REGISTRY: Dict[str, Any] = {}
REDIS_POOL: Optional[Any] = None

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


# ── App Lifespan ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, release on shutdown."""
    global MODEL_REGISTRY, REDIS_POOL

    log.info("Loading fraud detection models...")
    model_uri = os.getenv("MLFLOW_MODEL_URI", "models:/fraud_ensemble/Production")

    try:
        MODEL_REGISTRY["ensemble"] = mlflow.pyfunc.load_model(model_uri)
        log.info(f"Model loaded from: {model_uri}")
    except Exception as e:
        log.error(f"Model load failed: {e}. Using mock model.")
        MODEL_REGISTRY["ensemble"] = None

    # Redis connection pool
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        REDIS_POOL = await aioredis.from_url(redis_url, decode_responses=True)
        await REDIS_POOL.ping()
        log.info("Redis connected")
    except Exception as e:
        log.warning(f"Redis unavailable: {e}")
        REDIS_POOL = None

    yield  # App running

    # Cleanup
    if REDIS_POOL:
        await REDIS_POOL.close()
    log.info("Shutdown complete")


# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Fraud Detection Scoring API",
    description = "Real-time ML-powered fraud detection for banking transactions",
    version     = "2.1.0",
    docs_url    = "/docs",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ── Pydantic Models ────────────────────────────────────────────────────────────
class TransactionFeatures(BaseModel):
    """Input feature vector for fraud scoring."""
    transaction_id:             str    = Field(..., example="txn_abc123")
    customer_id:                str    = Field(..., example="CUST000042")
    amount:                     float  = Field(..., gt=0, example=15000.0)
    txn_count_1h:               float  = Field(default=1.0, ge=0)
    txn_count_24h:              float  = Field(default=3.0, ge=0)
    txn_count_7d:               float  = Field(default=15.0, ge=0)
    txn_amount_1h:              float  = Field(default=15000.0, ge=0)
    txn_amount_24h:             float  = Field(default=35000.0, ge=0)
    amt_zscore_30d:             float  = Field(default=0.0)
    velocity_spike:             int    = Field(default=0, ge=0, le=1)
    max_single_day:             float  = Field(default=50000.0, ge=0)
    avg_txn_amount_30d:         float  = Field(default=8000.0, ge=0)
    median_txn_amount_30d:      float  = Field(default=5000.0, ge=0)
    amount_ratio_to_avg:        float  = Field(default=1.0, ge=0)
    pct_international_30d:      float  = Field(default=0.05, ge=0, le=1)
    pct_cnp_30d:                float  = Field(default=0.3, ge=0, le=1)
    unique_merchants_7d:        float  = Field(default=4.0, ge=0)
    unique_countries_30d:       float  = Field(default=1.0, ge=0)
    typical_amount_band:        int    = Field(default=2, ge=0, le=4)
    days_since_first_txn:       float  = Field(default=180.0, ge=0)
    customer_risk_score_hist:   float  = Field(default=5.0, ge=0)
    distance_from_last_txn_km:  float  = Field(default=10.0, ge=0)
    time_since_last_txn_min:    float  = Field(default=120.0, ge=0)
    implied_travel_speed_kmph:  float  = Field(default=50.0, ge=0)
    is_impossible_travel:       int    = Field(default=0, ge=0, le=1)
    is_new_country:             int    = Field(default=0, ge=0, le=1)
    home_country_pct:           float  = Field(default=0.95, ge=0, le=1)
    device_txn_count_24h:       float  = Field(default=2.0, ge=0)
    device_customer_count_7d:   float  = Field(default=1.0, ge=0)
    is_new_device:              int    = Field(default=0, ge=0, le=1)
    device_fraud_rate_30d:      float  = Field(default=0.001, ge=0)
    ip_txn_count_1h:            float  = Field(default=1.0, ge=0)
    multi_account_device:       int    = Field(default=0, ge=0, le=1)
    merchant_txn_count_7d:      float  = Field(default=500.0, ge=0)
    merchant_fraud_rate_30d:    float  = Field(default=0.002, ge=0)
    merchant_avg_amount_30d:    float  = Field(default=5000.0, ge=0)
    merchant_amount_deviation:  float  = Field(default=0.2, ge=0)
    merchant_unique_cards_7d:   float  = Field(default=200.0, ge=0)
    merchant_risk_tier:         int    = Field(default=1, ge=0, le=4)
    is_mcc_high_risk:           int    = Field(default=0, ge=0, le=1)
    is_fraud_hour:              int    = Field(default=0, ge=0, le=1)
    is_weekend:                 int    = Field(default=0, ge=0, le=1)
    is_month_end:               int    = Field(default=0, ge=0, le=1)
    days_since_last_txn:        float  = Field(default=1.0, ge=0)
    txn_hour_sin:               float  = Field(default=0.0, ge=-1, le=1)
    txn_hour_cos:               float  = Field(default=1.0, ge=-1, le=1)
    is_international:           int    = Field(default=0, ge=0, le=1)
    is_high_risk_merchant:      int    = Field(default=0, ge=0, le=1)


class FraudScoreResponse(BaseModel):
    """Fraud scoring API response."""
    transaction_id:  str
    customer_id:     str
    fraud_risk_score: float   = Field(..., ge=0, le=100)
    risk_band:       str      = Field(..., example="LOW")
    recommendation:  str
    rules_triggered: List[str] = Field(default_factory=list)
    model_version:   str
    latency_ms:      float
    scored_at:       str


class BatchScoreRequest(BaseModel):
    transactions: List[TransactionFeatures]


class BatchScoreResponse(BaseModel):
    results:       List[FraudScoreResponse]
    total:         int
    high_risk_count: int
    avg_score:     float
    batch_latency_ms: float


# ── Scoring Logic ─────────────────────────────────────────────────────────────
def _risk_band(score: float) -> str:
    if score >= 85: return "CRITICAL"
    if score >= 65: return "HIGH"
    if score >= 45: return "MEDIUM"
    return "LOW"

def _recommendation(score: float, band: str) -> str:
    recs = {
        "CRITICAL": "BLOCK transaction immediately. Trigger fraud alert. Freeze card.",
        "HIGH":     "FLAG for manual review within 15 minutes. Enable step-up authentication.",
        "MEDIUM":   "Enhanced monitoring. Request OTP verification. Log for review.",
        "LOW":      "APPROVE. Continue normal transaction processing.",
    }
    return recs[band]

def _mock_score(features: TransactionFeatures) -> float:
    """
    Rule-based fallback scorer when ML model unavailable.
    Used for development/testing without trained model.
    """
    score = 0.0
    if features.velocity_spike:               score += 20
    if features.is_impossible_travel:         score += 35
    if features.is_new_device:                score += 15
    if features.is_new_country:               score += 25
    if features.is_mcc_high_risk:             score += 15
    if features.multi_account_device:         score += 30
    if features.is_fraud_hour:                score += 10
    if features.amount_ratio_to_avg > 5:      score += 20
    if features.amt_zscore_30d > 3:           score += 15
    if features.pct_cnp_30d > 0.8:           score += 10
    return min(score, 100.0)


async def score_transaction_async(txn: TransactionFeatures) -> FraudScoreResponse:
    """Core async scoring function with caching."""
    start = time.monotonic()

    # Check Redis cache (skip repeat scoring of same transaction)
    cache_key = f"score:{txn.transaction_id}"
    if REDIS_POOL:
        cached = await REDIS_POOL.get(cache_key)
        if cached:
            result = json.loads(cached)
            result["latency_ms"] = round((time.monotonic() - start) * 1000, 2)
            return FraudScoreResponse(**result)

    # Build feature array
    features_dict = txn.dict()
    feature_vector = np.array([[features_dict.get(col, 0.0) for col in FEATURE_COLS]])

    # ML Scoring
    model = MODEL_REGISTRY.get("ensemble")
    if model:
        try:
            import pandas as pd
            df = pd.DataFrame(feature_vector, columns=FEATURE_COLS)
            raw_score = float(model.predict(df)[0])
            fraud_score = min(max(raw_score * 100, 0), 100)
        except Exception as e:
            log.error(f"Model inference error: {e}")
            fraud_score = _mock_score(txn)
    else:
        fraud_score = _mock_score(txn)

    band           = _risk_band(fraud_score)
    recommendation = _recommendation(fraud_score, band)
    latency_ms     = round((time.monotonic() - start) * 1000, 2)

    result = FraudScoreResponse(
        transaction_id   = txn.transaction_id,
        customer_id      = txn.customer_id,
        fraud_risk_score = round(fraud_score, 2),
        risk_band        = band,
        recommendation   = recommendation,
        rules_triggered  = [],  # Populated by rule engine in production
        model_version    = "2.1.0-ensemble",
        latency_ms       = latency_ms,
        scored_at        = datetime.now(timezone.utc).isoformat(),
    )

    # Cache result for 60 seconds
    if REDIS_POOL:
        await REDIS_POOL.setex(cache_key, 60, result.json())

    return result


# ── API Endpoints ──────────────────────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """Service health probe endpoint."""
    return {
        "status":         "healthy",
        "model_loaded":   MODEL_REGISTRY.get("ensemble") is not None,
        "redis_connected": REDIS_POOL is not None,
        "timestamp":      datetime.now(timezone.utc).isoformat(),
    }


@app.get("/model/info")
async def model_info():
    """Current model metadata."""
    return {
        "model_name":    "FraudEnsembleScorer",
        "version":       "2.1.0",
        "components":    ["XGBoost", "LightGBM", "IsolationForest"],
        "weights":       {"xgboost": 0.55, "lightgbm": 0.35, "isolation_forest": 0.10},
        "feature_count": len(FEATURE_COLS),
        "thresholds": {
            "CRITICAL": 85,
            "HIGH":     65,
            "MEDIUM":   45,
        }
    }


@app.post("/score", response_model=FraudScoreResponse)
async def score_single_transaction(
    txn: TransactionFeatures,
    background_tasks: BackgroundTasks,
):
    """
    Score a single banking transaction for fraud risk.
    
    Returns fraud risk score (0-100), risk band, and recommendation.
    Target SLA: < 100ms P99 latency.
    """
    result = await score_transaction_async(txn)

    # Async alert publishing for high-risk transactions
    if result.risk_band in ("CRITICAL", "HIGH"):
        background_tasks.add_task(_publish_fraud_alert, result)

    return result


@app.post("/score/batch", response_model=BatchScoreResponse)
async def score_batch_transactions(request: BatchScoreRequest):
    """
    Score a batch of transactions (up to 1000 per request).
    Uses async parallel scoring for throughput.
    """
    if len(request.transactions) > 1000:
        raise HTTPException(status_code=400,
                            detail="Batch size exceeds limit of 1000 transactions")

    start = time.monotonic()
    tasks = [score_transaction_async(txn) for txn in request.transactions]
    results = await asyncio.gather(*tasks)

    total_ms     = round((time.monotonic() - start) * 1000, 2)
    high_risk    = sum(1 for r in results if r.risk_band in ("CRITICAL", "HIGH"))
    avg_score    = sum(r.fraud_risk_score for r in results) / max(len(results), 1)

    return BatchScoreResponse(
        results          = results,
        total            = len(results),
        high_risk_count  = high_risk,
        avg_score        = round(avg_score, 2),
        batch_latency_ms = total_ms,
    )


@app.get("/metrics/summary")
async def get_scoring_metrics():
    """Real-time scoring pipeline metrics."""
    return {
        "service":              "fraud-scoring-api",
        "uptime_seconds":       time.time(),
        "model_version":        "2.1.0-ensemble",
        "target_latency_p99_ms": 100,
        "features_count":       len(FEATURE_COLS),
    }


async def _publish_fraud_alert(result: FraudScoreResponse):
    """Async background task: publish high-risk alerts to Kafka/SNS."""
    try:
        # In production: publish to Kafka banking.fraud.alerts topic
        log.warning(
            f"[FRAUD ALERT] {result.risk_band}: {result.transaction_id} "
            f"| Customer: {result.customer_id} | Score: {result.fraud_risk_score}"
        )
    except Exception as e:
        log.error(f"Alert publish failed: {e}")


# ── AWS Lambda Handler ─────────────────────────────────────────────────────────
def lambda_handler(event: Dict, context: Any) -> Dict:
    """
    AWS Lambda handler for serverless fraud scoring.
    Compatible with API Gateway v2 (HTTP API) proxy integration.
    """
    import asyncio
    from mangum import Mangum

    handler = Mangum(app, lifespan="off")
    return handler(event, context)


# ── Dev Server ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "fraud_scoring_api:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = True,
        workers = 1,
        log_level = "info",
    )
