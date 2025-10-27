
from fastapi import APIRouter, Depends, HTTPException

from ..schemas import ModelsResponse

router = APIRouter(prefix="/models", tags=["models"])


def get_pricing_loader():
    """Dependency to get pricing loader."""
    from ..main import pricing_loader
    if pricing_loader is None:
        raise HTTPException(status_code=500, detail="Pricing loader not initialized")
    return pricing_loader


@router.get("/", response_model=ModelsResponse)
async def list_models(loader=Depends(get_pricing_loader)) -> ModelsResponse:
    """
    List all available models with their pricing configurations.

    Returns the current pricing table loaded in memory.
    """
    models = loader.list_models()
    return ModelsResponse(models=models)
