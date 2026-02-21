"""
GET /v1/forecast/{user_id} — Expense forecast endpoint (SPEC §11.2.2).

Returns per-category p10/p50/p90 forecasts from pre-computed model results.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, Request

from src.data.models import CategoryForecast, ForecastResult
from src.utils.constants import CategoryL2

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/forecast/{user_id}", response_model=ForecastResult)
async def get_forecast(
    user_id: str,
    req: Request,
    horizon: int = Query(default=30, ge=7, le=90, description="Forecast horizon in days"),
    categories: str = Query(default="all", description="Comma-separated category filter or 'all'"),
):
    """
    Retrieve probabilistic expense forecasts for a user.

    Returns per-category p10/p50/p90 forecasts, trend indicators,
    and regime annotations.
    """
    state = req.app.state
    cache = state.cache

    # Check cache first (key includes horizon and category filter)
    cache_key_parts = [user_id, str(horizon), categories]
    cached = cache.get("forecasts", *cache_key_parts)
    if cached is not None:
        logger.info("Cache HIT for forecast %s", user_id)
        return ForecastResult(**cached)

    try:
        fc_data = state.forecast_data
        if fc_data is None:
            raise HTTPException(
                status_code=503,
                detail="Forecast data not available. Run the training pipeline first.",
            )

        raw_forecasts = fc_data.get("forecasts", [])

        # Filter by categories if requested
        if categories != "all":
            requested = {c.strip() for c in categories.split(",")}
            raw_forecasts = [f for f in raw_forecasts if f["category"] in requested]

        # Build response forecasts with trend analysis
        response_forecasts = []
        for fc in raw_forecasts:
            p10 = fc.get("p10", 0.0)
            p50 = fc.get("p50", 0.0)
            p90 = fc.get("p90", 0.0)
            actual = fc.get("actual", p50)

            # Scale from whole validation period to requested horizon
            # (the stored values are for the full val period ≈ 60 days)
            scale = horizon / 60.0
            p10_scaled = round(p10 * scale, 2)
            p50_scaled = round(p50 * scale, 2)
            p90_scaled = round(p90 * scale, 2)

            # Determine trend (compare forecast vs actual)
            if actual > 0:
                ratio = p50 / actual
                if ratio > 1.10:
                    trend = "increasing"
                elif ratio < 0.90:
                    trend = "decreasing"
                else:
                    trend = "stable"
            else:
                trend = "stable"

            # Determine spending regime from variance
            spread = (p90 - p10) / max(p50, 1)
            if spread > 1.0:
                regime = "irregular"
            elif ratio > 1.2 if actual > 0 else False:
                regime = "elevated"
            elif ratio < 0.8 if actual > 0 else False:
                regime = "reduced"
            else:
                regime = "normal"

            cat_name = fc["category"]
            try:
                cat_enum = CategoryL2(cat_name)
            except ValueError:
                continue

            response_forecasts.append(CategoryForecast(
                category=cat_enum,
                p10=p10_scaled,
                p50=p50_scaled,
                p90=p90_scaled,
                trend=trend,
                regime=regime,
            ))

        total_p10 = sum(f.p10 for f in response_forecasts)
        total_p50 = sum(f.p50 for f in response_forecasts)
        total_p90 = sum(f.p90 for f in response_forecasts)

        result = ForecastResult(
            user_id=user_id,
            generated_at=datetime.now(timezone.utc),
            horizon_days=horizon,
            forecasts=response_forecasts,
            total_spend={
                "p10": round(total_p10, 2),
                "p50": round(total_p50, 2),
                "p90": round(total_p90, 2),
            },
        )

        # Cache the result
        cache.set("forecasts", *cache_key_parts, value=result.model_dump(mode="json"))

        return result

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Forecast generation failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
