"""JWT / OAuth2 authentication middleware.

Validates bearer tokens from the ``Authorization`` header using PyJWT.
When ``jwks_url`` is configured the middleware fetches signing keys from the
JWKS endpoint; otherwise it decodes without verification (dev-mode).

If the ``PyJWT`` package is not installed the middleware degrades gracefully
by passing all requests through and logging a warning on first use.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

# ---------------------------------------------------------------------------
# Optional PyJWT import — the middleware must work even when jwt is absent.
# ---------------------------------------------------------------------------
try:
    import jwt as _jwt_lib
    from jwt import ExpiredSignatureError, InvalidTokenError

    _HAS_PYJWT = True
except ImportError:  # pragma: no cover
    _jwt_lib = None  # type: ignore[assignment]
    _HAS_PYJWT = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_SKIP_PATHS: set[str] = {
    "/health",
    "/healthz",
    "/readiness",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/",
}


class JWTConfig(BaseModel):
    """Configuration for the :class:`JWTAuthMiddleware`.

    Attributes:
        enabled: Master toggle — when ``False`` the middleware is a no-op.
        jwks_url: JWKS endpoint used to fetch public signing keys.
        issuer: Expected ``iss`` claim.  Empty string skips issuer validation.
        audience: Expected ``aud`` claim.  Empty string skips audience validation.
        algorithms: Accepted signing algorithms.
        header_name: HTTP header that carries the token.
        header_prefix: Token prefix inside the header value (e.g. ``Bearer``).
        skip_paths: Request paths that bypass authentication.
    """

    enabled: bool = False
    jwks_url: str = ""
    issuer: str = ""
    audience: str = ""
    algorithms: list[str] = Field(default_factory=lambda: ["RS256"])
    header_name: str = "Authorization"
    header_prefix: str = "Bearer"
    skip_paths: set[str] = Field(default_factory=lambda: set(_DEFAULT_SKIP_PATHS))


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that validates JWT bearer tokens.

    On successful validation the decoded claims are stored in
    ``request.state.jwt_claims`` and the ``sub`` claim is copied to
    ``request.state.user_id``.

    Args:
        app: ASGI application.
        config: :class:`JWTConfig` with validation parameters.
    """

    def __init__(self, app: Any, *, config: JWTConfig) -> None:
        super().__init__(app)
        self._config = config
        self._jwks_client: Any | None = None
        self._warned_no_pyjwt = False

    # -- helpers -------------------------------------------------------------

    def _get_jwks_client(self) -> Any:
        """Lazily create and cache the :class:`jwt.PyJWKClient`."""
        if self._jwks_client is None and self._config.jwks_url:
            self._jwks_client = _jwt_lib.PyJWKClient(self._config.jwks_url)  # type: ignore[union-attr]
        return self._jwks_client

    @staticmethod
    def _json_401(detail: str) -> JSONResponse:
        return JSONResponse({"detail": detail}, status_code=401)

    # -- dispatch ------------------------------------------------------------

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Authenticate the request or short-circuit with 401."""
        # 0. If disabled, pass through immediately.
        if not self._config.enabled:
            return await call_next(request)

        # 1. Skip configured paths.
        if request.url.path in self._config.skip_paths:
            return await call_next(request)

        # 2. Graceful degradation when PyJWT is missing.
        if not _HAS_PYJWT:
            if not self._warned_no_pyjwt:
                logger.warning(
                    "PyJWT is not installed — JWT authentication is disabled. "
                    "Install it with: pip install PyJWT[crypto]"
                )
                self._warned_no_pyjwt = True
            return await call_next(request)

        # 3. Extract the bearer token.
        auth_header = request.headers.get(self._config.header_name, "")
        prefix = f"{self._config.header_prefix} "
        if not auth_header.startswith(prefix):
            return self._json_401("Missing or malformed Authorization header")

        token = auth_header[len(prefix) :]
        if not token:
            return self._json_401("Missing bearer token")

        # 4. Decode & validate.
        try:
            decode_kwargs: dict[str, Any] = {
                "algorithms": self._config.algorithms,
            }

            if self._config.jwks_url:
                client = self._get_jwks_client()
                signing_key = client.get_signing_key_from_jwt(token)
                decode_kwargs["key"] = signing_key.key
            else:
                # Dev mode — decode without verification.
                decode_kwargs["options"] = {"verify_signature": False}

            if self._config.issuer:
                decode_kwargs["issuer"] = self._config.issuer
            if self._config.audience:
                decode_kwargs["audience"] = self._config.audience

            claims: dict[str, Any] = _jwt_lib.decode(token, **decode_kwargs)  # type: ignore[union-attr]
        except ExpiredSignatureError:
            logger.debug("JWT token expired")
            return self._json_401("Token has expired")
        except InvalidTokenError as exc:
            logger.debug("JWT validation failed: {}", str(exc))
            return self._json_401(f"Invalid token: {exc}")
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected JWT error: {}", str(exc))
            return self._json_401("Authentication error")

        # 5. Attach claims to request state.
        request.state.jwt_claims = claims
        request.state.user_id = claims.get("sub", "")

        # 6. Continue down the middleware stack.
        response: Response = await call_next(request)
        return response
