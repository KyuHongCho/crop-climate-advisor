"""Command-line entry point for the Stage-0 slice.

Example:
    python3 -m crop_advisor.cli --crop basil --lat 51.5 --lon -0.13 --place London
"""
import argparse

from .climate import fetch_climate
from .ecocrop import available_crops, load_crop
from .suitability import Assessment, assess


def _format(a: Assessment, crop: dict, climate) -> str:
    lines = []
    lines.append(f"Crop-Climate Advisor — {a.crop} @ {a.place}")
    lines.append("=" * 56)
    lines.append(f"Location climate (NASA POWER): annual mean {climate.annual_mean_temp_c} °C, "
                 f"warmest month {climate.warmest_month_temp_c} °C, "
                 f"annual precip {climate.annual_precip_mm} mm")
    lines.append(f"Crop needs (FAO ECOCROP id {crop.get('ecocrop_id')}): "
                 f"temp opt {crop['temperature_c']['opt_min']}–{crop['temperature_c']['opt_max']} °C, "
                 f"rain opt {crop['rainfall_mm_yr']['opt_min']}–{crop['rainfall_mm_yr']['opt_max']} mm/yr")
    lines.append("")
    for b in (a.temperature, a.rainfall):
        note = "within optimal" if b.status == "optimal" else (
            f"needs {'+' if b.correction > 0 else ''}{b.correction} {b.unit} to reach optimal")
        lines.append(f"  {b.metric:<12} {b.location_value} {b.unit:<6} → {b.status.upper():<11} ({note})")
    if not a.warmest_month_reaches_opt:
        lines.append(f"  note: even the warmest month stays below the crop's optimal minimum.")
    lines.append("")
    lines.append(f"Verdict: {a.verdict}")
    lines.append("")
    lines.append(f"Data: NASA POWER (climate) · {crop['_source']['attribution']} (crop requirements).")
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
