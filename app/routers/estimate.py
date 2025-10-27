
from fastapi import APIRouter, Depends, HTTPException

from ..schemas import (
    BatchEstimateRequest,
    BatchEstimateResponse,
    EstimateRequest,
    EstimateResponse,
)
from ..services.estimator import EstimationService

router = APIRouter(prefix="/estimate", tags=["estimation"])


def get_estimation_service() -> EstimationService:
    """Dependency to get estimation service."""
    from ..main import estimation_service
    if estimation_service is None:
        raise HTTPException(status_code=500, detail="Estimation service not initialized")
    return estimation_service


@router.post("/", response_model=EstimateResponse)
async def estimate_tokens(
    request: EstimateRequest,
    service: EstimationService = Depends(get_estimation_service),
) -> EstimateResponse:
    """
    Estimate token usage and cost for a single request.

    This endpoint calculates:
    - Input token count (system + user + tools)
    - Output token count (user-provided estimate)
    - Total cost breakdown
    - Context utilization warnings
    """
    try:
        return service.estimate(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Estimation failed: {e!s}")


@router.post("/batch", response_model=BatchEstimateResponse)
async def estimate_batch(
    request: BatchEstimateRequest,
    service: EstimationService = Depends(get_estimation_service),
) -> BatchEstimateResponse:
    """
    Estimate token usage and cost for multiple requests.

    Returns individual results plus aggregated totals.
    """
    try:
        results = []
        total_cost = 0
        total_input_tokens = 0
        total_output_tokens = 0

        for req in request.requests:
            result = service.estimate(req)
            results.append(result)
            total_cost += result.cost
            total_input_tokens += result.input_tokens
            total_output_tokens += result.output_tokens

        return BatchEstimateResponse(
            results=results,
            total_cost=total_cost,
            total_input_tokens=total_input_tokens,
            total_output_tokens=total_output_tokens,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch estimation failed: {e!s}")
