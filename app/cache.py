"""
Simple In-Memory Caching for Psychic Canary
Replace with Redis for production at scale.
"""
import hashlib
import json
from datetime import datetime, timedelta
from typing import Optional, Any

class SimpleCache:
    """Thread-safe in-memory cache with TTL"""

    def __init__(self, default_ttl: int = 300):  # 5 min default
        self._cache = {}
        self._default_ttl = default_ttl

    def _make_key(self, tickers: list, days: int) -> str:
        """Create cache key from request params"""
        key_data = f"{sorted(tickers)}:{days}:{datetime.utcnow().date()}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, tickers: list, days: int) -> Optional[dict]:
        """Get cached response if exists and not expired"""
        key = self._make_key(tickers, days)

        if key not in self._cache:
            return None

        entry = self._cache[key]
        if datetime.utcnow() > entry["expires"]:
            del self._cache[key]
            return None

        return entry["data"]

    def set(self, tickers: list, days: int, data: dict, ttl: Optional[int] = None) -> None:
        """Cache response with TTL"""
        key = self._make_key(tickers, days)
        ttl = ttl or self._default_ttl

        self._cache[key] = {
            "data": data,
            "expires": datetime.utcnow() + timedelta(seconds=ttl),
            "created": datetime.utcnow().isoformat()
        }

    def clear(self) -> int:
        """Clear all cached entries, return count cleared"""
        count = len(self._cache)
        self._cache = {}
        return count

    def stats(self) -> dict:
        """Get cache statistics"""
        now = datetime.utcnow()
        valid = sum(1 for v in self._cache.values() if now < v["expires"])
        return {
            "total_entries": len(self._cache),
            "valid_entries": valid,
            "expired_entries": len(self._cache) - valid
        }

# Global cache instance
cache = SimpleCache(default_ttl=300)  # 5 minute cache
