"""
API Key Authentication for Psychic Canary
"""
import os
import secrets
import hashlib
from datetime import datetime
from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

# Simple in-memory store (replace with database in production)
# Format: {api_key_hash: {"user": "name", "tier": "free|pro|enterprise", "created": datetime}}
API_KEYS_DB = {}

# Admin key for management (set in .env)
ADMIN_KEY = os.getenv("ADMIN_API_KEY", "admin-secret-key-change-me")

# API Key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Rate limits by tier (requests per minute)
RATE_LIMITS = {
    "free": 10,
    "pro": 100,
    "enterprise": 1000,
    "demo": 30,  # For unauthenticated demo access
}

def hash_key(key: str) -> str:
    """Hash an API key for storage"""
    return hashlib.sha256(key.encode()).hexdigest()

def generate_api_key(user: str, tier: str = "free") -> str:
    """Generate a new API key"""
    key = f"pc_{tier}_{secrets.token_urlsafe(32)}"
    key_hash = hash_key(key)
    API_KEYS_DB[key_hash] = {
        "user": user,
        "tier": tier,
        "created": datetime.utcnow().isoformat(),
        "requests_today": 0,
        "last_request": None
    }
    return key

def validate_api_key(api_key: Optional[str]) -> dict:
    """
    Validate API key and return user info.
    Returns demo tier if no key provided (for frontend demo).
    """
    if api_key is None:
        # Allow demo access without key (for frontend)
        return {"user": "demo", "tier": "demo"}

    key_hash = hash_key(api_key)

    if key_hash not in API_KEYS_DB:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    return API_KEYS_DB[key_hash]

async def get_api_key_user(api_key: Optional[str] = Security(api_key_header)) -> dict:
    """Dependency for routes that need authentication"""
    return validate_api_key(api_key)

def get_rate_limit(tier: str) -> int:
    """Get rate limit for a tier"""
    return RATE_LIMITS.get(tier, RATE_LIMITS["free"])

# Pre-populate with a test key for development
_test_key = generate_api_key("test_user", "pro")
print(f"[AUTH] Test API key generated: {_test_key}")
