"""Location climate via the NASA POWER climatology API (free, public, no auth).

This is the live data source that the MCP tool will wrap in a later stage; for
now it's a plain client function the standalone app calls.
"""
from dataclasses import dataclass

from ._http import get_json

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


def fetch_climate(lat: float, lon: float) -> ClimateSummary:
    data = get_json(_POWER_URL.format(lat=lat, lon=lon))
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
