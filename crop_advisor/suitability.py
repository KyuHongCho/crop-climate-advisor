"""Suitability reasoning: crop requirements vs a location's climate.

Pure functions (no network) so the logic is unit-testable offline.
"""
from dataclasses import dataclass

from .climate import ClimateSummary


@dataclass
class BandAssessment:
    metric: str
    location_value: float
    unit: str
    opt_min: float
    opt_max: float
    status: str        # "optimal" | "survivable" | "unsuitable"
    correction: float  # signed change needed to reach the optimal band; 0 if optimal


@dataclass
class Assessment:
    crop: str
    place: str
    temperature: BandAssessment
    rainfall: BandAssessment
    warmest_month_reaches_opt: bool
    verdict: str


def _assess_band(metric: str, value: float, unit: str, band: dict) -> BandAssessment:
    opt_min, opt_max = band["opt_min"], band["opt_max"]
    abs_min, abs_max = band.get("abs_min"), band.get("abs_max")
    if opt_min <= value <= opt_max:
        status, correction = "optimal", 0.0
    else:
        correction = (opt_min - value) if value < opt_min else -(value - opt_max)
        within_abs = (abs_min is None or value >= abs_min) and (abs_max is None or value <= abs_max)
        status = "survivable" if within_abs else "unsuitable"
    return BandAssessment(metric, value, unit, opt_min, opt_max, status, round(correction, 1))


def assess(crop: dict, climate: ClimateSummary, place: str) -> Assessment:
    temp = _assess_band("temperature", climate.annual_mean_temp_c, "°C", crop["temperature_c"])
    rain = _assess_band("rainfall", climate.annual_precip_mm, "mm/yr", crop["rainfall_mm_yr"])
    warmest_reaches = climate.warmest_month_temp_c >= crop["temperature_c"]["opt_min"]

    name = crop.get("common_name") or crop.get("name") or "crop"
    if temp.status == "optimal" and rain.status == "optimal":
        verdict = f"{name.capitalize()} is a good outdoor match for {place}."
    elif "unsuitable" in (temp.status, rain.status):
        verdict = (f"{name.capitalize()} cannot be grown outdoors at {place} without a "
                   f"controlled-environment chamber.")
    else:
        verdict = (f"{name.capitalize()} is marginal outdoors at {place}; a controlled-environment "
                   f"chamber would need to close the gaps below.")
    return Assessment(name, place, temp, rain, warmest_reaches, verdict)
