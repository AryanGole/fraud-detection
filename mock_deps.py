"""
Mock modules for dependencies not available in this environment.
All mocks faithfully replicate the public API used by the codebase.
"""
import sys
import types
import numpy as np
from unittest.mock import MagicMock, patch


def _make_module(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


# ── structlog ─────────────────────────────────────────────────────────────────
structlog = _make_module("structlog")
class _SLogger:
    def info(self, *a, **kw): pass
    def warning(self, *a, **kw): pass
    def error(self, *a, **kw): pass
    def debug(self, *a, **kw): pass
    def bind(self, **kw): return self
structlog.get_logger = lambda *a, **kw: _SLogger()
structlog.stdlib = MagicMock()


# ── confluent_kafka ───────────────────────────────────────────────────────────
confluent_kafka = _make_module("confluent_kafka")
confluent_kafka.Producer = MagicMock
confluent_kafka.Consumer = MagicMock
confluent_kafka.KafkaError = MagicMock
confluent_kafka.KafkaException = Exception

ck_admin = _make_module("confluent_kafka.admin")
ck_admin.AdminClient = MagicMock
ck_admin.NewTopic = MagicMock

confluent_kafka.admin = ck_admin

ck_schema = _make_module("confluent_kafka.schema_registry")
confluent_kafka.schema_registry = ck_schema


# ── redis ─────────────────────────────────────────────────────────────────────
redis = _make_module("redis")
class _FakeRedis:
    def __init__(self): self._store = {}
    def get(self, k): return self._store.get(k)
    def set(self, k, v, ex=None): self._store[k] = v
    def setex(self, k, ttl, v): self._store[k] = v
    def incr(self, k): self._store[k] = int(self._store.get(k, 0)) + 1; return self._store[k]
    def incrbyfloat(self, k, v): self._store[k] = float(self._store.get(k, 0)) + v; return self._store[k]
    def expire(self, k, t): pass
    def ping(self): return True
    def pipeline(self): return _FakePipeline(self)
    def from_url(self, *a, **kw): return self
redis.Redis = _FakeRedis
redis.from_url = lambda *a, **kw: _FakeRedis()
redis.StrictRedis = _FakeRedis

class _FakePipeline:
    def __init__(self, r): self._r = r; self._cmds = []
    def incr(self, k): self._cmds.append(('incr', k)); return self
    def expire(self, k, t): return self
    def incrbyfloat(self, k, v): return self
    def execute(self): return [1]*len(self._cmds)
redis._FakeRedis = _FakeRedis


# ── xgboost ───────────────────────────────────────────────────────────────────
xgboost = _make_module("xgboost")
from sklearn.base import BaseEstimator, ClassifierMixin as _CM
class _FakeXGB(BaseEstimator, _CM):
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                 reg_lambda=0.1, min_child_weight=3, scale_pos_weight=100,
                 use_label_encoder=False, eval_metric="aucpr",
                 tree_method="hist", random_state=42, **kw):
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.learning_rate    = learning_rate
        self.subsample        = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha        = reg_alpha
        self.reg_lambda       = reg_lambda
        self.min_child_weight = min_child_weight
        self.scale_pos_weight = scale_pos_weight
        self.use_label_encoder= use_label_encoder
        self.eval_metric      = eval_metric
        self.tree_method      = tree_method
        self.random_state     = random_state
        self._fitted          = False

    _estimator_type = "classifier"  # required for sklearn 1.8 CalibratedClassifierCV

    def fit(self, X, y, eval_set=None, verbose=None):
        self._fitted  = True
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        n = len(X)
        p = np.clip(np.random.default_rng(42).random(n)*0.1 + 0.02, 0, 1)
        return np.column_stack([1-p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:,1] >= 0.5).astype(int)

    @property
    def feature_importances_(self):
        return np.random.default_rng(0).random(46)
xgboost.XGBClassifier = _FakeXGB


# ── lightgbm ──────────────────────────────────────────────────────────────────
lightgbm = _make_module("lightgbm")
class _FakeLGB:
    def __init__(self, **kw): self.kw = kw
    def fit(self, X, y, eval_set=None, callbacks=None): self._X=X; return self
    def predict_proba(self, X):
        n = len(X); p = np.clip(np.random.default_rng(7).random(n)*0.08 + 0.01, 0, 1)
        return np.column_stack([1-p, p])
    @property
    def feature_importances_(self): return np.random.default_rng(1).random(46)
lightgbm.LGBMClassifier = _FakeLGB
lightgbm.early_stopping = lambda *a, **kw: MagicMock()
lightgbm.log_evaluation = lambda *a, **kw: MagicMock()


# ── imblearn ──────────────────────────────────────────────────────────────────
imblearn = _make_module("imblearn")
ib_over = _make_module("imblearn.over_sampling")
class _FakeSMOTE:
    def __init__(self, **kw): pass
    def fit_resample(self, X, y):
        # Duplicate minority class to simulate SMOTE
        minority_idx = np.where(y == 1)[0]
        if len(minority_idx) == 0: return X, y
        n_target = max(int(len(y) * 0.05), len(minority_idx))
        n_extra  = max(0, n_target - len(minority_idx))
        repeats  = np.random.default_rng(42).choice(minority_idx, size=n_extra, replace=True)
        X_res = np.vstack([X, X[repeats]])
        y_res = np.concatenate([y, y[repeats]])
        return X_res, y_res
ib_over.SMOTE = _FakeSMOTE
imblearn.over_sampling = ib_over
ib_pipe = _make_module("imblearn.pipeline")
ib_pipe.Pipeline = MagicMock
imblearn.pipeline = ib_pipe


# ── shap ─────────────────────────────────────────────────────────────────────
shap = _make_module("shap")
class _FakeShapExplainer:
    def __init__(self, model): pass
    def shap_values(self, X): return np.random.default_rng(99).random((len(X), X.shape[1]))
shap.TreeExplainer = _FakeShapExplainer


# ── optuna ────────────────────────────────────────────────────────────────────
optuna = _make_module("optuna")
class _FakeTrial:
    def suggest_int(self, n, lo, hi): return (lo + hi) // 2
    def suggest_float(self, n, lo, hi, log=False): return (lo + hi) / 2
optuna.logging = MagicMock()
class _FakeStudy:
    def __init__(self):
        self.best_value = 0.82
        self.best_params = {
            "n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "reg_alpha": 0.1,
            "reg_lambda": 0.1, "min_child_weight": 3, "scale_pos_weight": 100,
        }
    def optimize(self, fn, n_trials=1, show_progress_bar=False):
        fn(_FakeTrial())  # Run once
optuna.create_study = lambda **kw: _FakeStudy()
optuna_samplers = _make_module("optuna.samplers")
optuna_samplers.TPESampler = lambda **kw: None
optuna.samplers = optuna_samplers


# ── mlflow ────────────────────────────────────────────────────────────────────
mlflow = _make_module("mlflow")
mlflow.set_experiment = lambda *a: None
mlflow.log_param  = lambda *a, **kw: None
mlflow.log_params = lambda *a, **kw: None
mlflow.log_metrics= lambda *a, **kw: None
mlflow.log_artifact=lambda *a, **kw: None

class _FakeRun:
    class _Info:
        run_id = "mock-run-id-abc123"
    info = _Info()
class _FakeCtx:
    def __enter__(self): return _FakeRun()
    def __exit__(self, *a): return False
mlflow.start_run = lambda **kw: _FakeCtx()

mlflow_xgb = _make_module("mlflow.xgboost")
mlflow_xgb.log_model = lambda *a, **kw: None
mlflow.xgboost = mlflow_xgb

mlflow_lgb = _make_module("mlflow.lightgbm")
mlflow_lgb.log_model = lambda *a, **kw: None
mlflow.lightgbm = mlflow_lgb

mlflow_sk = _make_module("mlflow.sklearn")
mlflow_sk.log_model = lambda *a, **kw: None
mlflow.sklearn = mlflow_sk

mlflow_pyfunc = _make_module("mlflow.pyfunc")
class _FakePyfuncModel:
    def predict(self, df): return np.clip(np.random.default_rng(5).random(len(df))*0.15, 0, 1)
mlflow_pyfunc.load_model = lambda *a, **kw: _FakePyfuncModel()
mlflow.pyfunc = mlflow_pyfunc


# ── yaml ─────────────────────────────────────────────────────────────────────
# (yaml is pyyaml, may not be available)
try:
    import yaml
except ImportError:
    yaml = _make_module("yaml")
    yaml.safe_load = lambda s: {}
