"""
Psychic Canary API - Production Ready
Thermodynamic Phase Inference for Market Regime Detection
"""
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from datetime import datetime
import logging
import os
import time

load_dotenv()

from .schemas import InferenceRequest, InferenceResponse
from .data import fetch_prices
from .engine import PsychicCanaryEngine
from .auth import get_api_key_user, generate_api_key, ADMIN_KEY, get_rate_limit
from .limiter import limiter, rate_limit_exceeded_handler
from .cache import cache
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("psychic_canary")

# Initialize app
app = FastAPI(
    title="Psychic Canary API",
    description="Thermodynamic Phase Inference for Market Regime Detection",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add rate limiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engine
engine = PsychicCanaryEngine(risk_aversion=1.0, vol_window=30, entropy_window=60)

# Static files - use pathlib for reliable resolution
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

# Request tracking
request_count = {"total": 0, "success": 0, "errors": 0}


# ============== Middleware ==============

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with timing"""
    start = time.time()
    request_count["total"] += 1

    response = await call_next(request)

    duration = (time.time() - start) * 1000
    logger.info(f"{request.method} {request.url.path} | {response.status_code} | {duration:.1f}ms")

    if response.status_code >= 400:
        request_count["errors"] += 1
    else:
        request_count["success"] += 1

    return response


# ============== Routes ==============

@app.get("/")
def home():
    """Serve the demo frontend"""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "engine": "PsychicCanaryEngine v0.2",
        "timestamp": datetime.utcnow().isoformat(),
        "requests": request_count,
        "cache": cache.stats()
    }


@app.post("/infer", response_model=InferenceResponse)
@limiter.limit("30/minute")  # Default limit, overridden by tier
async def infer_regime(
    request: Request,
    body: InferenceRequest,
    user: dict = Depends(get_api_key_user)
):
    """
    Infer market regime from price data.

    Returns thermodynamic phase (exploitation/exploration/crisis_imminent)
    with optimal portfolio weights.
    """
    tickers = [t.upper() for t in body.tickers]
    days = body.days

    # Check cache first
    cached = cache.get(tickers, days)
    if cached:
        logger.info(f"Cache hit for {tickers}")
        return InferenceResponse(**cached)

    try:
        # Fetch and analyze
        prices = fetch_prices(tickers, days)
        result = engine.infer_phase(prices, tickers)

        # Cache the result
        cache.set(tickers, days, result)

        logger.info(f"Inference: {tickers} | Phase: {result['phase']} | User: {user['user']}")

        return InferenceResponse(**result)

    except ValueError as e:
        logger.warning(f"Bad request: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.get("/api/v1/status")
def api_status(user: dict = Depends(get_api_key_user)):
    """Get API status and user info"""
    return {
        "status": "operational",
        "user": user["user"],
        "tier": user["tier"],
        "rate_limit": f"{get_rate_limit(user['tier'])}/minute",
        "version": "0.2.0"
    }


@app.post("/api/v1/keys/generate")
def create_api_key(
    request: Request,
    user_name: str,
    tier: str = "free",
    admin_key: str = None
):
    """
    Generate a new API key (admin only).

    Tiers: free, pro, enterprise
    """
    if admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin key")

    if tier not in ["free", "pro", "enterprise"]:
        raise HTTPException(status_code=400, detail="Invalid tier")

    new_key = generate_api_key(user_name, tier)
    logger.info(f"New API key generated for {user_name} ({tier})")

    return {
        "api_key": new_key,
        "user": user_name,
        "tier": tier,
        "rate_limit": f"{get_rate_limit(tier)}/minute",
        "note": "Save this key - it cannot be retrieved again"
    }


@app.delete("/api/v1/cache")
def clear_cache(admin_key: str = None):
    """Clear the response cache (admin only)"""
    if admin_key != ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin key")

    count = cache.clear()
    logger.info(f"Cache cleared: {count} entries")
    return {"cleared": count}


# ============== Static Files ==============
# Mount after routes so routes take precedence
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ============== Startup ==============

@app.on_event("startup")
async def startup():
    logger.info("=" * 50)
    logger.info("Psychic Canary API Starting")
    logger.info(f"Static dir: {STATIC_DIR} (exists: {STATIC_DIR.exists()})")
    logger.info("=" * 50)


@app.on_event("shutdown")
async def shutdown():
    logger.info("Psychic Canary API Shutting Down")
