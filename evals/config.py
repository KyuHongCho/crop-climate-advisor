"""The frozen eval config, and R3: the judge is never a subject's model."""
import json
import math
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent / "config.json"


class ConfigError(ValueError):
    pass


class ModelMismatch(RuntimeError):
    pass


def family(model_id: str) -> str:
    parts = model_id.split("-")
    if len(parts) < 3 or parts[0] != "claude":
        raise ConfigError(f"cannot read a model family from {model_id!r}; expected claude-<family>-<version>")
    return parts[1]


def check_judge_independent(judge: str, subjects) -> None:
    for subject in subjects:
        if judge == subject or family(judge) == family(subject):
            raise ConfigError(f"judge {judge!r} shares a model or family with subject {subject!r}")


def load(path=DEFAULT_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ConfigError(f"{path}: must be a JSON object")
    subjects, judge = config.get("subjects"), config.get("judge")
    if not subjects or not isinstance(subjects, list) or not isinstance(judge, str):
        raise ConfigError(f"{path}: needs a non-empty 'subjects' list and a 'judge' id")
    check_judge_independent(judge, subjects)
    for key in ("trial_budget_usd", "trial_timeout_s"):
        if not positive_number(config.get(key)):
            raise ConfigError(f"{path}: {key!r} must be a finite number greater than 0")
    return config


def positive_number(value) -> bool:
    # bool is an int subclass, so `true` would otherwise pass as 1.
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0
    except OverflowError:  # a JSON integer too large for a float
        return False


def check_canonical(configured: str, canonical_models, require_model: bool = False) -> None:
    """Record-time R3 for a subject: every model that ran must be the configured one."""
    if require_model and not canonical_models:
        raise ModelMismatch(f"configured {configured!r} but the CLI reported no model for an answered trial")
    others = sorted(set(canonical_models) - {configured})
    if others:
        raise ModelMismatch(f"configured {configured!r} but the CLI ran {', '.join(others)}")

