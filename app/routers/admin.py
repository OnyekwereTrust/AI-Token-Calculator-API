from fastapi import APIRouter, HTTPException, Header, Depends
from ..schemas import RefreshResponse


router = APIRouter(prefix="/prices", tags=["admin"])


def verify_api_key(x_api_key: str = Header(None)):
    """Verify API key for admin endpoints."""
    from ..config import settings
    
    if not x_api_key or x_api_key != settings.admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


@router.post("/refresh", response_model=RefreshResponse)
async def refresh_pricing(
    api_key: str = Depends(verify_api_key)
) -> RefreshResponse:
    """
    Refresh pricing configuration from remote URL or local file.
    
    Requires X-API-Key header with valid admin key.
    """
    try:
        from ..main import pricing_loader
        await pricing_loader.load_pricing()
        
        models_count = len(pricing_loader.pricing_data)
        return RefreshResponse(
            success=True,
            models_loaded=models_count,
            message=f"Successfully loaded {models_count} models"
        )
        
    except Exception as e:
        return RefreshResponse(
            success=False,
            models_loaded=0,
            message=f"Failed to refresh pricing: {str(e)}"
        )
