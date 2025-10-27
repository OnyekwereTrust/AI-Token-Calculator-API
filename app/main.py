import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .pricing_loader import PricingLoader
from .routers import admin, estimate, health, models
from .services.estimator import EstimationService

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global instances
pricing_loader: PricingLoader = None
estimation_service: EstimationService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global pricing_loader, estimation_service

    # Startup
    logger.info("Starting Token Calculator API...")

    try:
        pricing_loader = PricingLoader(settings.pricing_url)
        await pricing_loader.load_pricing()

        estimation_service = EstimationService(pricing_loader.pricing_data)

        logger.info(f"API started successfully with {len(pricing_loader.pricing_data)} models")

    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down Token Calculator API...")


# Create FastAPI app
app = FastAPI(
    title="Token Calculator API",
    description="Production-ready Token Cost Calculator API for different LLMs",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    openapi_url="/openapi.json",
)

# Add CORS middleware
cors_origins = settings.cors_origins.split(",") if settings.cors_origins != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing."""
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time

    # Log structured request info
    logger.info(
        f"Request: {request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Duration: {process_time:.3f}s",
    )

    return response


# Include routers
app.include_router(estimate.router)
app.include_router(models.router)
app.include_router(admin.router)
app.include_router(health.router)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "diagnostic_id": f"ERR_{int(time.time())}",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=True,
    )
