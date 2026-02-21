"""
API route modules.
"""

from src.serving.routes.budget import router as budget_router
from src.serving.routes.classify import router as classify_router
from src.serving.routes.forecast import router as forecast_router
from src.serving.routes.health import router as health_router

__all__ = [
    "budget_router",
    "classify_router",
    "forecast_router",
    "health_router",
]
