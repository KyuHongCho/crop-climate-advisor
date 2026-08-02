"""Load bundled FAO ECOCROP crop requirements (scraped once; see decision 05).

The numeric fields are a one-time scrape into JSON under data/ecocrop/, not a
live per-request call. Regenerate with scripts/scrape_ecocrop.py.
"""
import json
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "ecocrop"


def load_crop(slug: str) -> dict:
    path = _DATA_DIR / f"{slug}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No bundled ECOCROP data for '{slug}' at {path}.\n"
            f"Generate it with:  python3 scripts/scrape_ecocrop.py --slug {slug} --id <ECOCROP id>"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def available_crops() -> list[str]:
    return sorted(p.stem for p in _DATA_DIR.glob("*.json"))
