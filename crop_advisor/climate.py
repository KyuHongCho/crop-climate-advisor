"""Location climate via the NASA POWER climatology API (free, public, no auth).

This is the live data source: `mcp_server.py` already wraps it as an MCP tool,
and the standalone CLI also calls this client function directly.
With CROP_ADVISOR_CLIMATE_REPLAY=<dir> set, recorded responses are read from <dir>; nothing goes online.
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path

from ._http import get_json

REPLAY_ENV = "CROP_ADVISOR_CLIMATE_REPLAY"
REPLAY_MISSING = "CROP_ADVISOR_REPLAY_MISSING"

_MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
           "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

_POWER_URL = (
    "https://power.larc.nasa.gov/api/temporal/climatology/point"
    "?parameters=T2M,PRECTOTCORR&community=AG"
    "&longitude={lon}&latitude={lat}&format=JSON"
)


@dataclass
class ClimateSummary:
    latitude: float
    longitude: float
    annual_mean_temp_c: float
    monthly_mean_temp_c: dict          # {"JAN": 4.14, ...}
    annual_precip_mm: float
    source: str = "NASA POWER climatology (T2M, PRECTOTCORR; community=AG)"

    @property
    def warmest_month_temp_c(self) -> float:
        return max(self.monthly_mean_temp_c.values())


def power_url(lat: float, lon: float) -> str:
    return _POWER_URL.format(lat=lat, lon=lon)


def replay_path(replay_dir, lat: float, lon: float) -> Path:
    # float() so 51 and 51.0 name the same file.
    return Path(replay_dir) / f"{float(lat)}_{float(lon)}.json"


def _power_response(lat: float, lon: float) -> dict:
    replay_dir = os.environ.get(REPLAY_ENV)
    if not replay_dir:
        return get_json(power_url(lat, lon))
    path = replay_path(replay_dir, lat, lon)
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        # Never fall back to a live call: a replayed run must not silently go online.
        raise FileNotFoundError(
            f"{REPLAY_MISSING}: no recorded NASA POWER response at {path} "
            f"({REPLAY_ENV}={replay_dir})") from None


def fetch_climate(lat: float, lon: float) -> ClimateSummary:
    data = _power_response(lat, lon)
    param = data["properties"]["parameter"]
    temp = param["T2M"]
    precip = param["PRECTOTCORR"]  # long-term mean, mm/day
    return ClimateSummary(
        latitude=lat,
        longitude=lon,
        annual_mean_temp_c=round(temp["ANN"], 2),
        monthly_mean_temp_c={m: round(temp[m], 2) for m in _MONTHS},
        annual_precip_mm=round(precip["ANN"] * 365.25, 1),  # mm/day -> mm/yr
    )
