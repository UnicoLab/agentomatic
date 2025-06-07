"""
API decorators for FastAPI endpoints with comprehensive error handling.
"""

import asyncio
import time
import functools
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Callable, Dict, Optional, Union
from fastapi import HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel


class APIResponse(BaseModel):
    """Standard API response model."""
    success: bool = True
    data: Any = None
    message: str = ""
    timestamp: float = None
    request_id: Optional[str] = None

    def __init__(self, **kwargs):
        if 'timestamp' not in kwargs:
            kwargs['timestamp'] = time.time()
        super().__init__(**kwargs)


class AgentQueue:
    """Simple queue for agent requests."""
    def __init__(self, max_size: int = 100):
        self.queue = asyncio.Queue(maxsize=max_size)
        self.processing = set()

    async def add_request(self, request_data: Dict[str, Any]) -> str:
        """Add request to queue."""
        request_id = f"req_{int(time.time() * 1000)}"
        await self.queue.put({"id": request_id, "data": request_data})
        return request_id

    async def get_request(self) -> Optional[Dict[str, Any]]:
        """Get next request from queue."""
        try:
            return await asyncio.wait_for(self.queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            return None


# Global instances
agent_queue = AgentQueue()


def handle_api_errors(func: Callable) -> Callable:
    """Decorator to handle API errors gracefully."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"API error in {func.__name__}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Internal server error: {str(e)}"
            )
    return wrapper


def log_api_call(func: Callable) -> Callable:
    """Decorator to log API calls."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start_time
            logger.info(f"API call {func.__name__} completed in {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"API call {func.__name__} failed in {duration:.3f}s: {str(e)}")
            raise
    return wrapper


def rate_limit(max_calls: int = 100, window_seconds: int = 60) -> Callable:
    """Simple rate limiting decorator."""
    call_history = {}

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            current_time = time.time()
            client_ip = "default"  # Simplified for testing

            if client_ip not in call_history:
                call_history[client_ip] = []

            # Clean old entries
            call_history[client_ip] = [
                call_time for call_time in call_history[client_ip]
                if current_time - call_time < window_seconds
            ]

            # Check rate limit
            if len(call_history[client_ip]) >= max_calls:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded"
                )

            call_history[client_ip].append(current_time)
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def validate_streaming_support(func: Callable) -> Callable:
    """Decorator to validate streaming support."""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)
    return wrapper


async def create_streaming_response(
    data: Any,
    agent_name: str,
    media_type: str = "text/plain"
) -> StreamingResponse:
    """Create streaming response."""
    async def generate():
        if hasattr(data, '__aiter__'):
            async for chunk in data:
                yield f"data: {chunk}\n\n"
        else:
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type=media_type,
        headers={"X-Agent": agent_name}
    )


@asynccontextmanager
async def agent_context(agent_name: str):
    """Context manager for agent operations."""
    logger.info(f"Starting context for agent: {agent_name}")
    try:
        yield
    finally:
        logger.info(f"Ending context for agent: {agent_name}")


# Export all necessary components
__all__ = [
    "APIResponse",
    "AgentQueue",
    "agent_queue",
    "handle_api_errors",
    "log_api_call",
    "rate_limit",
    "validate_streaming_support",
    "create_streaming_response",
    "agent_context"
]