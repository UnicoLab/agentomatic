"""API decorators for authentication, rate limiting, and error handling."""

import time
import functools
from typing import Dict, Any, Optional, Callable
from collections import defaultdict
from datetime import datetime, timedelta
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from loguru import logger
import asyncio


# Rate limiting storage (in production, use Redis)
_rate_limit_storage: Dict[str, Dict[str, Any]] = defaultdict(dict)


class RateLimiter:
    """Rate limiting implementation with sliding window."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    def is_allowed(self, identifier: str) -> tuple[bool, Dict[str, Any]]:
        """Check if request is allowed under rate limit."""
        now = time.time()
        window_start = now - self.window_seconds

        # Get or create client data
        if identifier not in _rate_limit_storage:
            _rate_limit_storage[identifier] = {
                "requests": [],
                "first_request": now
            }

        client_data = _rate_limit_storage[identifier]

        # Remove old requests outside the window
        client_data["requests"] = [
            req_time for req_time in client_data["requests"]
            if req_time > window_start
        ]

        # Check if limit exceeded
        current_requests = len(client_data["requests"])
        allowed = current_requests < self.max_requests

        if allowed:
            client_data["requests"].append(now)

        # Calculate reset time
        if client_data["requests"]:
            oldest_request = min(client_data["requests"])
            reset_time = oldest_request + self.window_seconds
        else:
            reset_time = now + self.window_seconds

        return allowed, {
            "limit": self.max_requests,
            "remaining": max(0, self.max_requests - current_requests - (1 if allowed else 0)),
            "reset": int(reset_time),
            "retry_after": int(reset_time - now) if not allowed else 0
        }


# Security scheme
security = HTTPBearer(auto_error=False)


def get_client_identifier(request: Request) -> str:
    """Get client identifier for rate limiting."""
    # Try to get user ID from auth, fallback to IP
    user_id = getattr(request.state, "user_id", None)
    if user_id:
        return f"user:{user_id}"

    # Get real IP (considering proxies)
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return f"ip:{forwarded_for.split(',')[0].strip()}"

    return f"ip:{request.client.host}"


def rate_limit(max_requests: int = 100, window_seconds: int = 3600):
    """Rate limiting decorator.

    Args:
        max_requests: Maximum requests allowed in the window
        window_seconds: Time window in seconds

    Example:
        @rate_limit(max_requests=10, window_seconds=60)
        async def my_endpoint():
            return {"message": "Hello"}
    """
    limiter = RateLimiter(max_requests, window_seconds)

    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request from args/kwargs
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if not request:
                # Try to get from kwargs
                request = kwargs.get("request")

            if not request:
                logger.warning("Rate limit decorator: Request not found")
                return await func(*args, **kwargs)

            # Check rate limit
            identifier = get_client_identifier(request)
            allowed, rate_info = limiter.is_allowed(identifier)

            if not allowed:
                logger.warning(f"Rate limit exceeded for {identifier}")
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded",
                    headers={
                        "X-RateLimit-Limit": str(rate_info["limit"]),
                        "X-RateLimit-Remaining": str(rate_info["remaining"]),
                        "X-RateLimit-Reset": str(rate_info["reset"]),
                        "Retry-After": str(rate_info["retry_after"])
                    }
                )

            # Add rate limit headers to response
            response = await func(*args, **kwargs)

            # If response has headers attribute, add rate limit info
            if hasattr(response, "headers"):
                response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
                response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])
                response.headers["X-RateLimit-Reset"] = str(rate_info["reset"])

            return response

        return wrapper
    return decorator


def require_auth(optional: bool = False):
    """Authentication decorator.

    Args:
        optional: If True, auth is optional and user info is added to request state

    Example:
        @require_auth()
        async def protected_endpoint(request: Request):
            user_id = request.state.user_id
            return {"user": user_id}
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request from args/kwargs
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if not request:
                request = kwargs.get("request")

            if not request:
                if not optional:
                    raise HTTPException(
                        status_code=500,
                        detail="Internal error: Request not found"
                    )
                return await func(*args, **kwargs)

            # Get authorization header
            auth_header = request.headers.get("authorization")

            if not auth_header:
                if not optional:
                    raise HTTPException(
                        status_code=401,
                        detail="Authorization header required"
                    )
                return await func(*args, **kwargs)

            # Parse bearer token
            try:
                scheme, token = auth_header.split(" ", 1)
                if scheme.lower() != "bearer":
                    raise ValueError("Invalid scheme")
            except ValueError:
                if not optional:
                    raise HTTPException(
                        status_code=401,
                        detail="Invalid authorization header format"
                    )
                return await func(*args, **kwargs)

            # Validate token (simplified - in production use JWT or session validation)
            user_id = await validate_token(token)

            if not user_id:
                if not optional:
                    raise HTTPException(
                        status_code=401,
                        detail="Invalid or expired token"
                    )
                return await func(*args, **kwargs)

            # Add user info to request state
            request.state.user_id = user_id
            request.state.authenticated = True

            return await func(*args, **kwargs)

        return wrapper
    return decorator


async def validate_token(token: str) -> Optional[str]:
    """Validate authentication token.

    This is a simplified implementation. In production:
    - Use JWT validation with proper secret key
    - Check token expiration
    - Validate against database/session store
    - Handle token refresh

    Args:
        token: Bearer token to validate

    Returns:
        User ID if valid, None otherwise
    """
    # Simplified validation - accept any token starting with "user_"
    # In production, implement proper JWT or session validation
    if token.startswith("user_"):
        return token

    # Example JWT-like validation (commented out)
    # try:
    #     payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    #     return payload.get("user_id")
    # except jwt.InvalidTokenError:
    #     return None

    return None


def handle_errors(log_errors: bool = True):
    """Error handling decorator.

    Args:
        log_errors: Whether to log errors

    Example:
        @handle_errors()
        async def my_endpoint():
            raise ValueError("Something went wrong")
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                # Re-raise HTTP exceptions as-is
                raise
            except Exception as e:
                if log_errors:
                    logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)

                # Convert to HTTP exception
                raise HTTPException(
                    status_code=500,
                    detail="Internal server error"
                )

        return wrapper
    return decorator


def require_role(required_roles: list[str]):
    """Role-based access control decorator.

    Args:
        required_roles: List of roles that are allowed to access the endpoint

    Example:
        @require_role(["admin", "moderator"])
        async def admin_endpoint(request: Request):
            return {"message": "Admin only"}
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request from args/kwargs
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if not request:
                request = kwargs.get("request")

            if not request:
                raise HTTPException(
                    status_code=500,
                    detail="Internal error: Request not found"
                )

            # Check if user is authenticated
            if not getattr(request.state, "authenticated", False):
                raise HTTPException(
                    status_code=401,
                    detail="Authentication required"
                )

            # Get user roles (simplified - in production, fetch from database)
            user_id = getattr(request.state, "user_id", None)
            user_roles = await get_user_roles(user_id)

            # Check if user has required role
            if not any(role in user_roles for role in required_roles):
                raise HTTPException(
                    status_code=403,
                    detail="Insufficient permissions"
                )

            return await func(*args, **kwargs)

        return wrapper
    return decorator


async def get_user_roles(user_id: str) -> list[str]:
    """Get user roles from database.

    This is a simplified implementation. In production:
    - Query actual user database
    - Cache roles for performance
    - Handle role inheritance

    Args:
        user_id: User identifier

    Returns:
        List of user roles
    """
    # Simplified role assignment
    if user_id == "user_admin":
        return ["admin", "user"]
    elif user_id == "user_moderator":
        return ["moderator", "user"]
    else:
        return ["user"]


def cache_response(ttl_seconds: int = 300):
    """Response caching decorator.

    Args:
        ttl_seconds: Time to live in seconds

    Example:
        @cache_response(ttl_seconds=60)
        async def expensive_endpoint():
            # This response will be cached for 60 seconds
            return {"data": "expensive_computation"}
    """
    cache_storage = {}

    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"

            # Check cache
            now = time.time()
            if cache_key in cache_storage:
                cached_data, timestamp = cache_storage[cache_key]
                if now - timestamp < ttl_seconds:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cached_data
                else:
                    # Remove expired entry
                    del cache_storage[cache_key]

            # Execute function and cache result
            result = await func(*args, **kwargs)
            cache_storage[cache_key] = (result, now)
            logger.debug(f"Cached result for {func.__name__}")

            return result

        return wrapper
    return decorator


def async_timeout(seconds: float):
    """Timeout decorator for async functions.

    Args:
        seconds: Timeout in seconds

    Example:
        @async_timeout(30.0)
        async def slow_endpoint():
            await asyncio.sleep(60)  # Will timeout after 30 seconds
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=seconds
                )
            except asyncio.TimeoutError:
                logger.warning(f"Timeout in {func.__name__} after {seconds}s")
                raise HTTPException(
                    status_code=504,
                    detail="Request timeout"
                )

        return wrapper
    return decorator


def validate_content_type(allowed_types: list[str]):
    """Content type validation decorator.

    Args:
        allowed_types: List of allowed content types

    Example:
        @validate_content_type(["application/json"])
        async def json_only_endpoint(request: Request):
            return {"message": "JSON only"}
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request from args/kwargs
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if not request:
                request = kwargs.get("request")

            if not request:
                return await func(*args, **kwargs)

            content_type = request.headers.get("content-type", "").split(";")[0]

            if content_type not in allowed_types:
                raise HTTPException(
                    status_code=415,
                    detail=f"Unsupported content type. Allowed: {', '.join(allowed_types)}"
                )

            return await func(*args, **kwargs)

        return wrapper
    return decorator


# Convenience decorator combinations
def api_endpoint(
    auth_required: bool = True,
    rate_limit_requests: int = 100,
    rate_limit_window: int = 3600,
    timeout_seconds: float = 30.0,
    cache_ttl: Optional[int] = None
):
    """Combined decorator for common API endpoint needs.

    Args:
        auth_required: Whether authentication is required
        rate_limit_requests: Rate limit max requests
        rate_limit_window: Rate limit window in seconds
        timeout_seconds: Request timeout
        cache_ttl: Cache TTL in seconds (None to disable)

    Example:
        @api_endpoint(auth_required=False, cache_ttl=60)
        async def public_cached_endpoint():
            return {"data": "public"}
    """
    def decorator(func: Callable):
        # Apply decorators in reverse order (they wrap from inside out)
        wrapped_func = func

        if cache_ttl:
            wrapped_func = cache_response(cache_ttl)(wrapped_func)

        wrapped_func = async_timeout(timeout_seconds)(wrapped_func)
        wrapped_func = rate_limit(rate_limit_requests, rate_limit_window)(wrapped_func)

        if auth_required:
            wrapped_func = require_auth()(wrapped_func)

        wrapped_func = handle_errors()(wrapped_func)

        return wrapped_func

    return decorator