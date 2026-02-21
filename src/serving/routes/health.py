"""
Health-check routes.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health():
    """Basic health check (liveness probe)."""
    return {"status": "healthy", "service": "ml-fin-advisor"}


@router.get("/ready")
async def readiness(req: Request):
    """Readiness probe — checks model artefacts are loaded."""
    state = req.app.state
    checks = {
        "classifier": state.classifier is not None,
        "label_encoder": state.label_encoder is not None,
        "forecast_data": state.forecast_data is not None,
        "budget_data": state.budget_data is not None,
    }
    all_ready = all(checks.values())
    return {
        "status": "ready" if all_ready else "not_ready",
        "components": checks,
    }
