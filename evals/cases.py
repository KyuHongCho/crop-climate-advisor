"""The versioned eval set: one case per recorded site, each prompt carrying the site's coordinates."""
import hashlib
import json
from pathlib import Path

from .record_fixtures import SITES

DEFAULT_PATH = Path(__file__).resolve().parent / "cases.json"


def _check_shape(data, path) -> None:
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list) or not data["cases"]:
        raise ValueError(f"{path}: must be a JSON object with a non-empty 'cases' list")
    for i, case in enumerate(data["cases"]):
        if not isinstance(case, dict):
            raise ValueError(f"{path}: case {i} must be a JSON object")
        for field in ("id", "site", "prompt"):
            if not isinstance(case.get(field), str):
                raise ValueError(f"{path}: case {i}: {field!r} must be a string")
        if case["site"] not in SITES:
            raise ValueError(f"{path}: case {case['id']!r}: unknown site {case['site']!r}; "
                             f"choose from {', '.join(sorted(SITES))}")


def load(path=DEFAULT_PATH) -> dict:
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    _check_shape(data, path)
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
