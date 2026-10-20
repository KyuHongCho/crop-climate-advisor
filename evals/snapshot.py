"""A run's snapshot and summary, derived from its trial records, and the cost projection read from earlier snapshots."""
import json
import math
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import config, report
from .adapters.claude_code import CONDITIONS

SNAPSHOT_FILE = "snapshot.json"
SUMMARY_FILE = "summary.md"
RUN_ID = re.compile(r"\d{8}T\d{6}Z-[0-9a-f]{8}")
_ACCESSED = re.compile(r"^- \*\*Accessed:\*\* (\d{4}-\d{2}-\d{2})", re.MULTILINE)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def new_run_id() -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"


def parse_fixture_date(text: str):
    m = _ACCESSED.search(text)
    return m.group(1) if m else None


def fixture_date(source_md_path):
    try:
        return parse_fixture_date(Path(source_md_path).read_text(encoding="utf-8"))
    except OSError:
        return None


def attempt_cost(rec):
    """The attempt's list-price cost, or None when it is unpriced (missing, negative or not a finite number)."""
    cost = (rec.get("usage") or {}).get("cost_usd")
    ok = isinstance(cost, (int, float)) and not isinstance(cost, bool) and math.isfinite(cost) and cost >= 0
    return cost if ok else None


def charged(records, trial_budget_usd) -> float:
    """What the run-cap check counts: each attempt's cost, an unpriced one at the per-trial cap."""
    return sum(trial_budget_usd if attempt_cost(r) is None else attempt_cost(r) for r in records)


def cost_by_key(records, subjects) -> dict:
    table = {m: {c: {"sum": 0.0, "priced": 0, "mean": None} for c in CONDITIONS} for m in subjects}
    for r in records:
        cost = attempt_cost(r)
        entry = table.get(r["subject"]["model"], {}).get(r["condition"]["condition"])
        if entry is not None and cost is not None:
            entry["sum"] += cost
            entry["priced"] += 1
    for by_condition in table.values():
        for entry in by_condition.values():
            entry["mean"] = round(entry["sum"] / entry["priced"], 6) if entry["priced"] else None
            entry["sum"] = round(entry["sum"], 6)
    return table


def _read(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _priced_mean(snap, key, min_trials):
    entry = snap.get("cost_by_key")
    for part in key:
        entry = entry.get(part) if isinstance(entry, dict) else None
    if not isinstance(entry, dict):
        return None
    priced, mean = entry.get("priced"), entry.get("mean")
    enough = isinstance(priced, int) and not isinstance(priced, bool) and priced >= min_trials
    return mean if enough and config.positive_number(mean) else None


def _run_dirs(results_dir):
    try:
        return sorted((p for p in Path(results_dir).iterdir() if p.is_dir() and RUN_ID.fullmatch(p.name)),
                      key=lambda p: p.name, reverse=True)
    except OSError:
        return []


def measured_means(results_dir, claude_path, min_trials, keys) -> dict:
    """{(model, condition): (mean, run id)} from the newest full, complete run made with this `claude_path`."""
    found = {}
    for run_dir in _run_dirs(results_dir):
        snap = _read(run_dir / SNAPSHOT_FILE)
        if (snap is None or snap.get("status") != "complete" or snap.get("overrides") != {}
                or snap.get("claude_path") != claude_path):
            continue
        for key in keys:
            mean = None if key in found else _priced_mean(snap, key, min_trials)
            if mean is not None:
                found[key] = (mean, run_dir.name)
    return found


def project(settings, subjects, n, n_cases, results_dir, claude_path) -> dict:
    keys = [(m, c) for m in subjects for c in CONDITIONS]
    measured = measured_means(results_dir, claude_path, settings["projection_min_trials"], keys)
    per_key = {m: {} for m in subjects}
    for model, condition in keys:
        mean, source = measured.get((model, condition)) or (settings["cost_defaults_usd"][model][condition], "defaults")
        slots = n * n_cases
        per_key[model][condition] = {"mean_usd": mean, "slots": slots, "usd": round(mean * slots, 6), "source": source}
    sources = {v["source"] for by_condition in per_key.values() for v in by_condition.values()}
    return {"total_usd": round(sum(v["usd"] for by_condition in per_key.values() for v in by_condition.values()), 6),
            "source": sources.pop() if len(sources) == 1 else "mixed", "per_key": per_key}


def build_snapshot(*, run_id, stop_reason, started, finished, overrides, approved_spend_usd, claude_path,
                   config_sha256, settings, eval_set, subjects, judge, fixture_accessed, planned_slots,
                   projection, records) -> dict:
    slots = {(r["case_id"], r["subject"]["model"], r["condition"]["condition"], r["rep"]) for r in records}
    costs = [attempt_cost(r) for r in records]
    snap = {
        "run_id": run_id,
        "status": "complete" if stop_reason is None else "partial",
        "stop_reason": stop_reason,
        "started_utc": started,
        "finished_utc": finished,
        "overrides": overrides,
        "approved_spend_usd": approved_spend_usd,
        "claude_path": claude_path,
        "config_sha256": config_sha256,
        "settings": settings,
        "eval_set_version": eval_set.get("version"),
        "eval_set_sha256": eval_set.get("sha256"),
        "subjects": list(subjects),
        "judge": judge,
        "fixture_accessed": fixture_accessed,
        "cli_versions": sorted({r["subject"]["cli_version"] for r in records if r["subject"].get("cli_version")}),
        "skill_sha256s": sorted({r["condition"]["skill_sha256"] for r in records
                                if r["condition"].get("skill_sha256")}),
        "counts": {"planned_slots": planned_slots, "slots_started": len(slots), "attempts": len(records),
                   "reruns": sum(r["attempt"] == 2 for r in records),
                   "by_status": dict(sorted(Counter(r["status"] for r in records).items()))},
        "spend_usd": round(sum(c for c in costs if c is not None), 6),
        "unpriced": sum(c is None for c in costs),
        "projection": projection,
        "cost_by_key": cost_by_key(records, subjects),
        "report": None,
    }
    if records:
        try:
            snap["report"] = report.build_report(records, eval_set, settings["bootstrap_seed"],
                                                 settings["bootstrap_resamples"])
        except Exception as e:  # noqa: BLE001 - a report fault must not cost the run its snapshot
            snap["report_error"] = f"{type(e).__name__}: {e}"
    return snap


def format_summary(snap) -> str:
    counts = snap["counts"]
    by_status = ", ".join(f"{s}: {n}" for s, n in counts["by_status"].items()) or "none"
    lines = [
        f"# Eval run {snap['run_id']}", "",
        f"- status: {snap['status']}",
        f"- stop reason: {snap['stop_reason'] or 'none'}",
        f"- slots started {counts['slots_started']} of {counts['planned_slots']} planned",
        f"- attempts: {counts['attempts']} (re-runs {counts['reruns']}); by status: {by_status}",
        f"- overrides: {json.dumps(snap['overrides'])}; approved spend: {snap['approved_spend_usd']}",
        f"- spend: ${snap['spend_usd']:.4f} at list price ({snap['unpriced']} unpriced attempts)",
        f"- projection: ${snap['projection']['total_usd']:.4f} (source: {snap['projection']['source']})",
        "- cost per attempt, by model and condition:",
    ]
    for model, by_condition in snap["cost_by_key"].items():
        for condition, e in by_condition.items():
            mean = "n/a" if e["mean"] is None else f"${e['mean']:.4f}"
            lines.append(f"  - {model} / {condition}: mean {mean} over {e['priced']} priced attempts "
                         f"(sum ${e['sum']:.4f})")
    lines.append("")
    if snap["report"] is None:
        lines.append("No trial records." if "report_error" not in snap else f"Report failed: {snap['report_error']}")
    else:
        lines += ["```", report.format_report(snap["report"]), "```"]
    return "\n".join(lines) + "\n"
