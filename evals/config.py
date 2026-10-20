"""The frozen eval config, and R3: the judge is never a subject's model."""
import json
import math
from pathlib import Path

from .adapters.claude_code import CONDITIONS

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
    if (not subjects or not isinstance(subjects, list) or not isinstance(judge, str)
            or not all(isinstance(s, str) for s in subjects)):
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


def _positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


_RUN_INTS = ("n_trials", "max_consecutive_excluded", "projection_min_trials", "bootstrap_resamples")


def run_settings(cfg: dict) -> dict:
    """The run loop's settings from a loaded config, each checked; `load` itself does not read them."""
    problems = [f"{k!r} must be a positive integer" for k in _RUN_INTS if not _positive_int(cfg.get(k))]
    if not positive_number(cfg.get("run_budget_usd")):
        problems.append("'run_budget_usd' must be a finite number greater than 0")
    seed = cfg.get("bootstrap_seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        problems.append("'bootstrap_seed' must be an integer")
    defaults = cfg.get("cost_defaults_usd")
    if not isinstance(defaults, dict):
        problems.append("'cost_defaults_usd' must be an object of {model: {condition: usd}}")
    else:
        for model in cfg["subjects"]:
            by_condition = defaults.get(model)
            for condition in CONDITIONS:
                if not isinstance(by_condition, dict) or not positive_number(by_condition.get(condition)):
                    problems.append(f"'cost_defaults_usd' needs a positive number for {model} / {condition}")
    if problems:
        raise ConfigError("; ".join(problems))
    keys = (*_RUN_INTS, "run_budget_usd", "bootstrap_seed", "cost_defaults_usd")
    return {k: cfg[k] for k in keys}


def check_canonical(configured: str, canonical_models, require_model: bool = False) -> None:
    """Record-time R3 for a subject: every model that ran must be the configured one."""
    if require_model and not canonical_models:
        raise ModelMismatch(f"configured {configured!r} but the CLI reported no model for an answered trial")
    others = sorted(set(canonical_models) - {configured})
    if others:
        raise ModelMismatch(f"configured {configured!r} but the CLI ran {', '.join(others)}")

