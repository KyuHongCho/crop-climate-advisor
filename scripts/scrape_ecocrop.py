#!/usr/bin/env python3
"""One-time scraper for FAO ECOCROP numeric crop requirements.

Per decision ticket 05 (`.scratch/phase-1-spec/issues/05-*`): ECOCROP's numeric
fields are scraped ONCE into a bundled JSON lookup rather than fetched live per
request. Run this to (re)generate `data/ecocrop/<slug>.json`.

Data © FAO ECOCROP, used for non-commercial research with attribution
(see FAO Terms and Conditions: https://www.fao.org/contact-us/terms/en/).

Usage:
    python3 scripts/scrape_ecocrop.py --id 1547 --slug basil
"""
import argparse
import datetime as _dt
import html
import json
import re
import ssl
import urllib.request
from pathlib import Path

DATA_SHEET = "https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id={id}"
UA = "crop-climate-advisor/0.1 (portfolio project; non-commercial research)"

try:  # verify TLS against certifi's CA bundle when available (python.org macOS builds lack system certs)
    import certifi
    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:  # pragma: no cover
    _SSL_CTX = ssl.create_default_context()


def fetch(ecocrop_id: int) -> str:
    req = urllib.request.Request(DATA_SHEET.format(id=ecocrop_id), headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30, context=_SSL_CTX) as resp:
        return resp.read().decode("utf-8", "replace")


def _row_cells(html: str, label: str, n: int) -> list[str]:
    """Return the first `n` <td> cell texts following <th>label</th>."""
    m = re.search(re.escape(f"<th>{label}</th>") + r"(.*?)(?:<th>|</tr>)", html, re.S)
    if not m:
        return []
    cells = re.findall(r"<td>(.*?)</td>", m.group(1), re.S)
    return [c.strip() for c in cells][:n]


def _num(v: str):
    try:
        return float(v) if "." in v else int(v)
    except (ValueError, TypeError):
        return None


def parse(html_doc: str, ecocrop_id: int) -> dict:
    name = re.search(r"<h2>(.*?)</h2>", html_doc, re.S)
    t = [_num(x) for x in _row_cells(html_doc, "Temperat. requir.", 4)]
    r = [_num(x) for x in _row_cells(html_doc, "Rainfall (annual)", 4)]
    ph = [_num(x) for x in _row_cells(html_doc, "Soil PH", 4)]
    cyc = [_num(x) for x in _row_cells(html_doc, "Crop cycle", 2)]
    zones_raw = _row_cells(html_doc, "Climate zone", 1)
    zones = [html.unescape(z.strip()) for z in re.split(r",\s*", zones_raw[0])] if zones_raw else []

    def band(vals):
        keys = ["opt_min", "opt_max", "abs_min", "abs_max"]
        return {k: v for k, v in zip(keys, vals + [None] * 4)}

    return {
        "name": name.group(1).strip() if name else None,
        "ecocrop_id": ecocrop_id,
        "temperature_c": band(t),
        "rainfall_mm_yr": band(r),
        "soil_ph": band(ph),
        "crop_cycle_days": {"min": cyc[0] if cyc else None, "max": cyc[1] if len(cyc) > 1 else None},
        "climate_zones": zones,
        "_source": {
            "provider": "FAO ECOCROP",
            "url": DATA_SHEET.format(id=ecocrop_id),
            "accessed": _dt.date.today().isoformat(),
            "attribution": "© FAO ECOCROP",
            "license_note": "Non-commercial research use with attribution per FAO Terms and Conditions (https://www.fao.org/contact-us/terms/en/).",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--id", type=int, required=True, help="ECOCROP plant id (e.g. 1547 for basil)")
    ap.add_argument("--slug", required=True, help="output filename slug (e.g. basil)")
    ap.add_argument("--common-name", default=None, help="human-readable common name")
    args = ap.parse_args()

    record = parse(fetch(args.id), args.id)
    if args.common_name:
        record["common_name"] = args.common_name

    out = Path(__file__).resolve().parents[1] / "data" / "ecocrop" / f"{args.slug}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps(record, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
