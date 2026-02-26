"""
FastAPI application for the ML Fin-Advisor (SPEC §11).

Provides REST endpoints for transaction classification, expense forecasting,
and budget recommendations.  Model artefacts are loaded once at startup and
shared across requests via ``app.state``.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.serving.cache import CacheClient, CACHE_TTLS
from src.serving.middleware import generate_api_key, verify_api_key

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SERVE_DIR = PROJECT_ROOT / "models" / "serving"


def _load_artefacts(app: FastAPI) -> None:
    """Load all model artefacts into ``app.state``."""
    import joblib

    # ── Classifier ──────────────────────────────────────────────
    lgb_path = SERVE_DIR / "classifier_lgb.joblib"
    meta_path = SERVE_DIR / "classifier_meta.joblib"
    scaler_path = SERVE_DIR / "scaler.joblib"
    le_path = SERVE_DIR / "label_encoder.joblib"
    cols_path = SERVE_DIR / "feature_columns.json"
    tfidf_path = SERVE_DIR / "tfidf_vectorizer.joblib"
    svd_path = SERVE_DIR / "svd_reducer.joblib"

    app.state.classifier = joblib.load(lgb_path) if lgb_path.exists() else None
    app.state.meta_model = joblib.load(meta_path) if meta_path.exists() else None
    app.state.scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    app.state.label_encoder = joblib.load(le_path) if le_path.exists() else None
    app.state.tfidf = joblib.load(tfidf_path) if tfidf_path.exists() else None
    app.state.svd = joblib.load(svd_path) if svd_path.exists() else None

    if cols_path.exists():
        with open(cols_path) as f:
            app.state.feature_cols = json.load(f)
    else:
        app.state.feature_cols = []

    # ── Forecast data ───────────────────────────────────────────
    fc_path = SERVE_DIR / "forecast_results.json"
    if fc_path.exists():
        with open(fc_path) as f:
            app.state.forecast_data = json.load(f)
    else:
        app.state.forecast_data = None

    # ── Budget data ─────────────────────────────────────────────
    budget_path = SERVE_DIR / "budget_results.json"
    if budget_path.exists():
        with open(budget_path) as f:
            app.state.budget_data = json.load(f)
    else:
        app.state.budget_data = None

    # ── Cache client (Redis w/ in-memory fallback) ──────────────
    app.state.cache = CacheClient()

    loaded = [
        name for name, obj in [
            ("classifier", app.state.classifier),
            ("meta_model", app.state.meta_model),
            ("scaler", app.state.scaler),
            ("label_encoder", app.state.label_encoder),
            ("tfidf", app.state.tfidf),
            ("svd", app.state.svd),
            ("forecast_data", app.state.forecast_data),
            ("budget_data", app.state.budget_data),
        ] if obj is not None
    ]
    logger.info("Loaded artefacts: %s", ", ".join(loaded))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load models on startup, clean up on shutdown."""
    logger.info("ML Fin-Advisor serving layer starting up…")
    _load_artefacts(app)
    logger.info("Model artefacts loaded — ready to serve.")
    yield
    logger.info("ML Fin-Advisor serving layer shutting down.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    from src.serving.routes.budget import router as budget_router
    from src.serving.routes.classify import router as classify_router
    from src.serving.routes.forecast import router as forecast_router
    from src.serving.routes.health import router as health_router
    from src.serving.routes.webhooks import router as webhooks_router

    app = FastAPI(
        title="ML Fin-Advisor API",
        description=(
            "Personal Finance Advisor with Behavior Modeling — "
            "Transaction Classification, Expense Forecasting, "
            "and Budget Recommendations."
        ),
        version="1.0.0",
        lifespan=lifespan,
        dependencies=[Depends(verify_api_key)],
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request timing middleware
    @app.middleware("http")
    async def add_timing_header(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
        return response

    # Register routers
    app.include_router(health_router, tags=["Health"])
    app.include_router(classify_router, prefix="/v1", tags=["Classification"])
    app.include_router(forecast_router, prefix="/v1", tags=["Forecasting"])
    app.include_router(budget_router, prefix="/v1", tags=["Budget"])
    app.include_router(webhooks_router, tags=["Webhooks"])

    # ── Admin: generate a new API key (prints to stdout) ────────
    @app.get("/admin/generate-key", tags=["Admin"], include_in_schema=False)
    async def admin_generate_key():
        """Generate a random API key (for local/dev use only)."""
        key = generate_api_key()
        logger.info("Generated new API key: %s", key)
        return {"api_key": key, "note": "Add this to the API_KEYS env var."}

    # ── Subscription status ─────────────────────────────────────
    @app.get("/v1/subscription/status", tags=["Subscription"])
    async def subscription_status(request: Request):
        """Return the caller's plan, usage, and quota."""
        from src.serving.subscriptions import subscription_store, usage_tracker

        sub = getattr(request.state, "subscription", None)
        if sub is None:
            return {
                "plan": "open_access",
                "status": "active",
                "monthly_limit": None,
                "current_usage": 0,
                "note": "No subscription linked — open access mode.",
            }

        return {
            "plan": sub.plan.name,
            "tier": sub.plan.tier.value,
            "status": sub.status,
            "monthly_limit": sub.plan.monthly_limit,
            "current_usage": usage_tracker.current_usage(sub.api_key),
            "remaining": (
                sub.plan.monthly_limit - usage_tracker.current_usage(sub.api_key)
                if sub.plan.monthly_limit is not None
                else None
            ),
            "subscription_id": sub.subscription_id,
            "customer_email": sub.customer_email,
        }

    return app


app = create_app()
