"""
Rate Limiting for Psychic Canary API
"""
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
from fastapi.responses import JSONResponse

def get_key_func(request: Request) -> str:
    """
    Get rate limit key based on API key or IP address.
    API key users get their own bucket; anonymous users share IP bucket.
    """
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"key:{api_key[:20]}"  # Use prefix of key
    return f"ip:{get_remote_address(request)}"

# Create limiter instance
limiter = Limiter(key_func=get_key_func)

def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Custom handler for rate limit exceeded"""
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "detail": str(exc.detail),
            "retry_after": "60 seconds"
        }
    )
