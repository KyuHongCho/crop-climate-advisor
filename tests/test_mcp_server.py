"""Tests for the MCP server wrapper around the NASA POWER climate client.

Two layers, both driving the server through the real MCP protocol (an in-memory
client session doing initialize -> list_tools -> call_tool) rather than calling
the tool function directly — calling it directly would prove nothing about MCP.

  * TestGetClimateOffline runs always and never touches the network.
  * TestGetClimateLive hits the real NASA POWER API and is opt-in:
        CROP_ADVISOR_LIVE_TESTS=1 python3 -m unittest discover -s tests
"""
import asyncio
import os
import time
import unittest
from unittest import mock

from mcp.shared.memory import create_connected_server_and_client_session

from crop_advisor.climate import ClimateSummary
from crop_advisor.mcp_server import mcp

EXPECTED_KEYS = {
    "latitude",
    "longitude",
    "annual_mean_temp_c",
    "monthly_mean_temp_c",
    "warmest_month_temp_c",
    "annual_precip_mm",
    "source",
}

FAKE = ClimateSummary(
    latitude=1.5,
    longitude=2.5,
    annual_mean_temp_c=11.11,
    monthly_mean_temp_c={"JAN": 1.0, "JUL": 20.0},
    annual_precip_mm=999.9,
)


class TestGetClimateOffline(unittest.IsolatedAsyncioTestCase):
    """Offline: fetch_climate is stubbed, so no network call is made."""

    async def test_tool_is_registered_with_a_self_describing_output_schema(self):
        async with create_connected_server_and_client_session(mcp) as session:
            result = await session.list_tools()
        names = [t.name for t in result.tools]
        self.assertEqual(names, ["get_climate"])
        # A TypedDict return annotation is what puts the field names and types
        # in the schema. A bare `dict`/`dict[str, Any]` yields an opaque
        # {"type": "object"} that tells a client nothing about the fields.
        schema = result.tools[0].outputSchema
        self.assertIsNotNone(schema, "no outputSchema at all — is the return annotated?")
        self.assertIn("properties", schema, "outputSchema is opaque — no field names")
        self.assertEqual(set(schema["properties"]), EXPECTED_KEYS)
        self.assertEqual(set(schema["required"]), EXPECTED_KEYS)
        self.assertEqual(schema["properties"]["annual_mean_temp_c"]["type"], "number")
        self.assertEqual(schema["properties"]["source"]["type"], "string")

    async def test_a_slow_call_does_not_block_the_server(self):
        """The tool must not run on the event loop, or one slow upstream
        response freezes every other request (including cancellation)."""
        slow_seconds = 1.0

        def slow_fetch(lat, lon):
            time.sleep(slow_seconds)  # blocking, exactly like urllib in _http.py
            return FAKE

        with mock.patch("crop_advisor.mcp_server.fetch_climate", side_effect=slow_fetch):
            async with create_connected_server_and_client_session(mcp) as session:
                started = time.perf_counter()
                call = asyncio.create_task(
                    session.call_tool("get_climate", {"lat": 1.5, "lon": 2.5})
                )
                await asyncio.sleep(0.05)  # let the slow call get in flight
                await session.list_tools()  # a second request, while it runs
                probe_finished = time.perf_counter() - started
                await call

        self.assertLess(
            probe_finished,
            slow_seconds / 2,
            f"second request took {probe_finished:.2f}s while a {slow_seconds}s "
            "tool call was in flight — the tool is blocking the event loop",
        )

    async def test_returns_full_summary_without_touching_the_network(self):
        # Patch the name as bound in mcp_server, NOT crop_advisor.climate:
        # mcp_server does `from .climate import fetch_climate`, so patching the
        # origin module would leave the already-bound reference live.
        with mock.patch("crop_advisor.mcp_server.fetch_climate", return_value=FAKE) as stub:
            async with create_connected_server_and_client_session(mcp) as session:
                res = await session.call_tool("get_climate", {"lat": 1.5, "lon": 2.5})

        stub.assert_called_once_with(1.5, 2.5)
        self.assertFalse(res.isError)
        self.assertEqual(set(res.structuredContent), EXPECTED_KEYS)
        self.assertEqual(res.structuredContent["annual_mean_temp_c"], 11.11)
        # Derived from the @property, which dataclasses.asdict() would drop.
        self.assertEqual(res.structuredContent["warmest_month_temp_c"], 20.0)

    async def test_reports_an_error_when_the_upstream_call_fails(self):
        with mock.patch("crop_advisor.mcp_server.fetch_climate", side_effect=OSError("boom")):
            async with create_connected_server_and_client_session(mcp) as session:
                res = await session.call_tool("get_climate", {"lat": 1.5, "lon": 2.5})
        self.assertTrue(res.isError)


@unittest.skipUnless(
    os.environ.get("CROP_ADVISOR_LIVE_TESTS"),
    "live NASA POWER test; set CROP_ADVISOR_LIVE_TESTS=1 to run",
)
class TestGetClimateLive(unittest.IsolatedAsyncioTestCase):
    """Opt-in: proves the tool returns real NASA POWER data."""

    async def test_london_returns_real_climate_data(self):
        async with create_connected_server_and_client_session(mcp) as session:
            res = await session.call_tool("get_climate", {"lat": 51.5, "lon": -0.13})

        self.assertFalse(res.isError)
        data = res.structuredContent
        self.assertEqual(set(data), EXPECTED_KEYS)
        # London's annual mean is ~10.4 °C; allow drift if NASA revises the
        # climatology, while still failing if the value is obviously wrong.
        self.assertAlmostEqual(data["annual_mean_temp_c"], 10.4, delta=1.5)
        self.assertEqual(len(data["monthly_mean_temp_c"]), 12)
        self.assertGreater(data["warmest_month_temp_c"], data["annual_mean_temp_c"])
        self.assertGreater(data["annual_precip_mm"], 300)
        self.assertIn("NASA POWER", data["source"])


if __name__ == "__main__":
    unittest.main()
