"""JWT / OAuth2 authentication middleware.

Validates bearer tokens from the ``Authorization`` header using PyJWT.
When ``jwks_url`` is configured the middleware fetches signing keys from the
JWKS endpoint; otherwise it decodes without verification (dev-mode).

Optional DPoP (RFC 9449) proof validation is supported when a ``DPoP`` header
is present, or when ``require_dpop`` is enabled.

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

from agentomatic.middleware.pathutils import path_is_skipped
from agentomatic.security.claims import extract_roles, extract_scopes
from agentomatic.security.dpop import DPoPConfig, DPoPError, validate_dpop

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
    "/studio",
    "/status",
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
        require_signature: When ``True`` the middleware refuses to run in
            signature-disabled dev mode — a ``jwks_url`` (real verification)
            must be configured. Set automatically by the platform under
            ``require_auth_globally`` so forged/unsigned tokens cannot slip
            through the global auth lock.
        require_dpop: When ``True``, every authenticated request must include
            a valid ``DPoP`` proof header.
        dpop_max_age_seconds: Max age for DPoP ``iat``.
        dpop_clock_skew_seconds: Allowed future skew for DPoP ``iat``.
    """

    enabled: bool = False
    jwks_url: str = ""
    issuer: str = ""
    audience: str = ""
    algorithms: list[str] = Field(default_factory=lambda: ["RS256"])
    header_name: str = "Authorization"
    header_prefix: str = "Bearer"
    skip_paths: set[str] = Field(default_factory=lambda: set(_DEFAULT_SKIP_PATHS))
    require_signature: bool = False
    require_dpop: bool = False
    dpop_max_age_seconds: int = 60
    dpop_clock_skew_seconds: int = 5


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that validates JWT bearer tokens.

    On successful validation the decoded claims are stored in
    ``request.state.jwt_claims``, normalized roles/scopes in
    ``request.state.roles`` / ``request.state.scopes``, and the ``sub`` claim
    is copied to ``request.state.user_id``.

    Args:
        app: ASGI application.
        config: :class:`JWTConfig` with validation parameters.
    """

    def __init__(self, app: Any, *, config: JWTConfig) -> None:
        super().__init__(app)
        # Refuse to run signature-disabled when signature verification is
        # required (e.g. under require_auth_globally): unsigned/forged tokens
        # would authenticate every request otherwise.
        if config.enabled and config.require_signature and not config.jwks_url:
            raise ValueError(
                "JWTAuthMiddleware requires signature verification "
                "(require_signature=True) but no jwks_url is configured. "
                "Set JWTConfig.jwks_url to a JWKS endpoint, or disable the "
                "auth-required flag for local dev."
            )
        self._config = config
        self._jwks_client: Any | None = None
        self._warned_no_pyjwt = False
        self._warned_dev_mode = False
        self._dpop_config = DPoPConfig(
            require_dpop=config.require_dpop,
            max_age_seconds=config.dpop_max_age_seconds,
            clock_skew_seconds=config.dpop_clock_skew_seconds,
        )

    # -- helpers -------------------------------------------------------------

    def _get_jwks_client(self) -> Any:
        """Lazily create and cache the :class:`jwt.PyJWKClient`."""
        if self._jwks_client is None and self._config.jwks_url:
            self._jwks_client = _jwt_lib.PyJWKClient(self._config.jwks_url)  # type: ignore[union-attr]
        return self._jwks_client

    @staticmethod
    def _json_401(detail: str) -> JSONResponse:
        return JSONResponse({"detail": detail}, status_code=401)

    def _request_uri(self, request: Request) -> str:
        """Build the absolute URI used for DPoP ``htu`` checks."""
        # Prefer forwarded proto/host when behind a reverse proxy.
        proto = request.headers.get("x-forwarded-proto") or request.url.scheme
        host = request.headers.get("x-forwarded-host") or request.headers.get("host")
        if host:
            return f"{proto}://{host}{request.url.path}"
        return str(request.url.replace(query="", fragment=""))

    # -- dispatch ------------------------------------------------------------

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Authenticate the request or short-circuit with 401."""
        # 0. If disabled, pass through immediately.
        if not self._config.enabled:
            return await call_next(request)

        # 1. Skip configured paths (exact or prefix).
        if path_is_skipped(request.url.path, self._config.skip_paths):
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
                # Dev mode — decode without signature verification, but ALWAYS
                # keep expiry checking on so stale tokens are still rejected.
                if not self._warned_dev_mode:
                    logger.warning(
                        "JWT dev mode: signatures are NOT verified (no jwks_url). "
                        "Do not use in production; forged tokens are accepted. "
                        "Expiry (exp) is still enforced."
                    )
                    self._warned_dev_mode = True
                decode_kwargs["options"] = {
                    "verify_signature": False,
                    "verify_exp": True,
                }

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

        # 5. Optional / required DPoP proof validation.
        try:
            validate_dpop(
                dpop_header=request.headers.get("DPoP") or request.headers.get("dpop"),
                access_token=token,
                access_claims=claims,
                http_method=request.method,
                http_uri=self._request_uri(request),
                config=self._dpop_config,
            )
        except DPoPError as exc:
            logger.debug("DPoP validation failed: {}", str(exc))
            return self._json_401(str(exc))

        # 6. Attach claims + normalized roles/scopes to request state.
        roles = extract_roles(claims)
        scopes = extract_scopes(claims)
        request.state.jwt_claims = claims
        request.state.user_id = claims.get("sub", "")
        request.state.roles = roles
        request.state.scopes = scopes

        # 7. Continue down the middleware stack.
        response: Response = await call_next(request)
        return response
