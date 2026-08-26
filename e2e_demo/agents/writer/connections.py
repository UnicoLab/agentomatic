"""Per-agent connection declaration for the production Docker fixture."""

from agentomatic.connections import HttpConnectionConfig
from agentomatic.endpoints import AuthType, UpstreamAuthConfig

CONNECTIONS = [
    HttpConnectionConfig(
        name="omlx",
        base_url="${OMLX_BASE_URL}",
        auth=UpstreamAuthConfig(type=AuthType.BEARER, api_key="${OMLX_API_KEY}"),
    )
]
