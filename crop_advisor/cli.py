"""Command-line entry point for the Stage-0 slice.

Example:
    python3 -m crop_advisor.cli --crop basil --lat 51.5 --lon -0.13 --place London
"""
import argparse

from .claims import cite, classify, conservative, consensus, divergence, temperature_claims
from .climate import fetch_climate
from .ecocrop import available_crops, load_crop
from .suitability import Assessment, _assess_band, assess


def _num(value) -> str:
    """Render a band edge without a pointless trailing '.0' (18.0 -> '18')."""
    f = float(value)
    return str(int(f)) if f.is_integer() else str(f)


def _agreement(claims, unit: str) -> str:
    """One clause describing whether the claims share any window at all.

    Derived from consensus(): if the sources ever did agree, this names the
    window they agree on instead, in the unit it is given.
    """
    claims = tuple(claims)   # len() below, so any iterable must be materialised
    agreed = consensus(claims)
    if agreed is None:
        return f"no single window satisfies all {len(claims)} published sources"
    return f"all {len(claims)} published sources are satisfied by {_num(agreed[0])}–{_num(agreed[1])} {unit}"


def _disagreement_line(value, unit, claims) -> str:
    """Name the sources that put `value` on opposite sides of their optimal band.

    Precondition: both directions are present (the caller checks divergence()).
    A source that calls the site optimal is not a party to the disagreement, so
    it is not named here; its own row is printed above.
    """
    above = ", ".join(c.source for c in claims if classify(value, c) == "above")
    below = ", ".join(c.source for c in claims if classify(value, c) == "below")
    return (f"Sources disagree: {value} {unit} is above optimal for {above}; "
            f"below optimal for {below}.")


def _format(a: Assessment, crop: dict, climate) -> str:
    claims = temperature_claims(crop)
    lines = []
    lines.append(f"Crop-Climate Advisor — {a.crop} @ {a.place}")
    lines.append("=" * 56)
    lines.append(f"Location climate (NASA POWER): annual mean {climate.annual_mean_temp_c} °C, "
                 f"warmest month {climate.warmest_month_temp_c} °C, "
                 f"annual precip {climate.annual_precip_mm} mm")
    lines.append(f"Crop needs — temperature: {_agreement(claims, a.temperature.unit)}; "
                 f"every claim is listed below.")
    lines.append(f"Crop needs — rainfall (FAO ECOCROP id {crop.get('ecocrop_id')}): "
                 f"opt {crop['rainfall_mm_yr']['opt_min']}–{crop['rainfall_mm_yr']['opt_max']} mm/yr")
    lines.append("")
    for b in (a.temperature, a.rainfall):
        if b.metric == "temperature":
            lines.append(f"  {b.metric:<12} {b.location_value} {b.unit:<6} → "
                         f"each published optimal window, and the change it would need:")
            for c in claims:
                # Only .correction is read: the journal claims state no absolute
                # range, so _assess_band's status would be meaningless for them.
                gap = _assess_band(b.metric, b.location_value, b.unit,
                                   {"opt_min": c.opt_min, "opt_max": c.opt_max}).correction
                change = "within optimal" if gap == 0 else (
                    f"needs {'+' if gap > 0 else ''}{gap} {b.unit}")
                band = f"{_num(c.opt_min)}–{_num(c.opt_max)} {b.unit}"
                condition = f"at {c.condition}" if c.condition else "no condition stated"
                lines.append(f"      {band:<9} → {change:<15} {c.source}  [{condition}]")
            continue
        note = "within optimal" if b.status == "optimal" else (
            f"needs {'+' if b.correction > 0 else ''}{b.correction} {b.unit} to reach optimal")
        lines.append(f"  {b.metric:<12} {b.location_value} {b.unit:<6} → {b.status.upper():<11} ({note})")
    if not a.warmest_month_reaches_opt:
        lines.append("  note: even the warmest month stays below FAO ECOCROP's optimal minimum.")
    lines.append("")
    lines.append(f"Verdict (on FAO ECOCROP's bands): {a.verdict}")
    value, unit = a.temperature.location_value, a.temperature.unit
    directions = divergence(value, claims)
    if {"below", "above"} <= directions:
        lines.append(_disagreement_line(value, unit, claims))
    lines.append(f"Conservative read: {value} {unit} is {conservative(directions)}.")
    lines.append("")
    lines.append(f"Data: NASA POWER (climate) · {crop['_source']['attribution']} (crop requirements).")
    lines.append("Temperature claims:")
    for c in claims:
        lines.append(f"  · {cite(c)}")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Assess a crop's suitability for a location.")
    ap.add_argument("--crop", required=True, help=f"crop slug (available: {', '.join(available_crops()) or 'none yet'})")
    ap.add_argument("--lat", type=float, required=True, help="latitude, e.g. 51.5")
    ap.add_argument("--lon", type=float, required=True, help="longitude, e.g. -0.13")
    ap.add_argument("--place", default=None, help="optional place label for the report")
    args = ap.parse_args(argv)

    crop = load_crop(args.crop)
    climate = fetch_climate(args.lat, args.lon)
    result = assess(crop, climate, place=args.place or f"{args.lat},{args.lon}")
    print(_format(result, crop, climate))


if __name__ == "__main__":
    main()
