from datetime import datetime, timedelta
from typing import Dict, Any, Optional

class Cache:
    def __init__(self, expiry_minutes=120):
        self.cache = {}
        self.default_expiry = expiry_minutes

    def set(self, key: str, value: any, expiry_minutes: int = None) -> None:
        """Set cache with optional custom expiry"""
        expiry = expiry_minutes or self.default_expiry
        self.cache[key] = {
            'value': value,
            'expires': datetime.now() + timedelta(minutes=expiry)
        }

    def get(self, key: str) -> any:
        """Get from cache if not expired"""
        if key in self.cache:
            if datetime.now() < self.cache[key]['expires']:
                return self.cache[key]['value']
            del self.cache[key]
        return None

    def clear(self) -> None:
        """Clear all cached data"""
        self.cache.clear()

    def remove(self, key: str) -> None:
        """Remove specific key from cache"""
        if key in self.cache:
            del self.cache[key]

    def clear_expired(self) -> None:
        """Clear only expired entries"""
        now = datetime.now()
        expired_keys = [
            key for key, data in self.cache.items()
            if now > data['expires']
        ]
        for key in expired_keys:
            del self.cache[key]

    def get_all(self) -> Dict[str, Any]:
        """Get all non-expired cache entries"""
        self.clear_expired()
        return {key: value for key, data in self.cache.items()}

    def is_expired(self, key: str) -> bool:
        """Check if a cache entry is expired"""
        if key in self.cache:
            return datetime.now() > self.cache[key]['expires']
        return True 