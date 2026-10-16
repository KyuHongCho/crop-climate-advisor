"""The versioned eval set: one case per recorded site, each prompt carrying the site's coordinates."""
import hashlib
import json
from pathlib import Path

from .record_fixtures import SITES

DEFAULT_PATH = Path(__file__).resolve().parent / "cases.json"


def load(path=DEFAULT_PATH) -> dict:
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    for case in data["cases"]:
        lat, lon = SITES[case["site"]]
        # Replay looks recordings up by these exact floats, so a prompt drifting from SITES would miss them.
        if f"latitude {lat}, longitude {lon}" not in case["prompt"]:
            raise ValueError(f"case {case['id']!r}: prompt does not carry {case['site']}'s coordinates {lat}, {lon}")
        case["lat"], case["lon"] = lat, lon
    data["sha256"] = hashlib.sha256(raw).hexdigest()
    return data


def get(case_id: str, path=DEFAULT_PATH) -> tuple:
    data = load(path)
    for case in data["cases"]:
        if case["id"] == case_id:
            return data, case
    raise KeyError(f"no case {case_id!r} in {path}; choose from {', '.join(c['id'] for c in data['cases'])}")
