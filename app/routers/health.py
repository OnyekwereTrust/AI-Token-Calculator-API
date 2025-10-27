from fastapi import APIRouter
from ..schemas import HealthResponse


router = APIRouter(prefix="/healthz", tags=["health"])


@router.get("/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns service status and number of loaded models.
    """
    try:
        from ..main import pricing_loader
        models_count = len(pricing_loader.pricing_data)
        return HealthResponse(ok=True, models=models_count)
    except Exception:
        return HealthResponse(ok=False, models=0)
