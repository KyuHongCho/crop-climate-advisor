"""NASA POWER climate lookup, exposed as an MCP server.

Wraps the existing `fetch_climate()` client — the HTTP call itself lives in
`climate.py` and is not reimplemented here. This module is a thin adapter that
makes that capability callable by any MCP client.

Run it as a stdio MCP server with:

    python3 -m crop_advisor.mcp_server
"""
from typing import TypedDict

import anyio.to_thread
from mcp.server.fastmcp import FastMCP

from .climate import fetch_climate

mcp = FastMCP("crop-climate")


class ClimateResult(TypedDict):
    """The tool's return shape.

    Declared as a TypedDict so the field names and types reach the client in the
    tool's `outputSchema`. A bare `dict`/`dict[str, Any]` produces an opaque
    `{"type": "object"}` that tells a caller nothing about what comes back.
    """

    latitude: float
    longitude: float
    annual_mean_temp_c: float
    monthly_mean_temp_c: dict[str, float]
    warmest_month_temp_c: float
    annual_precip_mm: float
    source: str


@mcp.tool()
async def get_climate(lat: float, lon: float) -> ClimateResult:
    """Long-term climate normals for a point, from NASA POWER climatology.

    Returns annual and monthly mean temperature (°C) and annual precipitation
    (mm/yr), plus the provenance string identifying the source dataset.
    """
    # fetch_climate uses blocking urllib (_http.py, 30s timeout). FastMCP 1.x
    # runs a sync tool inline on the event loop, so calling it directly would
    # freeze the whole server — every other request, including cancellation —
    # until the upstream responds. Offload it to a worker thread.
    c = await anyio.to_thread.run_sync(fetch_climate, lat, lon)
    return {
        "latitude": c.latitude,
        "longitude": c.longitude,
        "annual_mean_temp_c": c.annual_mean_temp_c,
        "monthly_mean_temp_c": c.monthly_mean_temp_c,
        # warmest_month_temp_c is a @property, not a dataclass field, so
        # dataclasses.asdict() would silently drop it — set it explicitly.
        "warmest_month_temp_c": c.warmest_month_temp_c,
        "annual_precip_mm": c.annual_precip_mm,
        "source": c.source,
    }


if __name__ == "__main__":
    # Guarded: importing this module (as the tests do) must not start the
    # server, or the import would block forever.
    mcp.run()
