"""The per-cell report over a file of trial records. Reads only; it makes no model call and writes nothing.

    python -m evals.report <trials.jsonl | run dir> [--cases evals/cases.json]

A cell is case x model x condition. A slot is (case, model, condition, rep) and has one attempt, or two when the first
ended `infra-error`; the statistics read each slot's last attempt. The pass rate is graded by the deterministic
graders only.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from . import cases, graders, stats
from .adapters.claude_code import CONDITIONS

TRIALS_FILE = "trials.jsonl"
JUDGE_NOTE = "not run (E5)"
HEADLINE_NOTE = "deterministic checks only (D1-D4)"
P95_MAX_BELOW = 20


class ReportError(ValueError):
    pass


def load_records(source) -> list:
    path = Path(source)
    if path.is_dir():
        path = path / TRIALS_FILE
    records = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ReportError(f"cannot read {path}: {e}") from e
    # Not splitlines(): it also cuts at U+2028, U+0085 and others, which are legal inside a JSON string.
    for number, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            raise ReportError(f"{path}:{number}: not JSON: {e}") from e
        if not isinstance(record, dict):
            raise ReportError(f"{path}:{number}: a trial record must be a JSON object")
        records.append(record)
    if not records:
        raise ReportError(f"{path}: no trial records")
    return records


def _get(record, *keys):
    value = record
    for key in keys:
        value = value.get(key) if isinstance(value, dict) else None
    return value


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _check(record):
    who = record.get("trial_id") or "(no trial_id)"
    checks = (("case_id", isinstance(record.get("case_id"), str)),
              ("subject.model", isinstance(_get(record, "subject", "model"), str)),
              ("condition.condition", _get(record, "condition", "condition") in CONDITIONS),
              ("status", isinstance(record.get("status"), str)),
              ("rep (written by evals.run)", _is_int(record.get("rep"))),
              ("attempt (1 or 2)", record.get("attempt") in (1, 2) and _is_int(record.get("attempt"))))
    for name, ok in checks:
        if not ok:
            raise ReportError(f"record {who}: missing or invalid {name}")


def check_eval_set(records, eval_set_sha256):
    """The records' one eval-set hash, and the warnings for it; more than one hash is refused."""
    shas = {r.get("eval_set_sha256") for r in records}
    if len(shas) > 1:
        raise ReportError("records carry more than one eval_set_sha256: "
                          + ", ".join(sorted(str(s) for s in shas)) + "; report one eval set at a time")
    (sha,) = shas
    warnings = []
    if sha != eval_set_sha256:
        warnings.append(f"records carry eval_set_sha256 {sha}, but the cases file hashes to {eval_set_sha256}")
    return sha, warnings


def _slots(records):
    slots = {}
    for r in records:
        _check(r)
        key = (r["case_id"], r["subject"]["model"], r["condition"]["condition"], r["rep"])
        attempts = slots.setdefault(key, {})
        if r["attempt"] in attempts:
            raise ReportError(f"record {r.get('trial_id')}: slot {key} has two attempt {r['attempt']} records")
        attempts[r["attempt"]] = r
    return slots


def _number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _cell(case_id, model, condition, slots, case) -> dict:
    ordered = sorted(slots, key=lambda s: s[0])
    finals = [(max(attempts), attempts[max(attempts)]) for _, attempts in ordered]
    graded = [(r, graders.grade(r, case)) for _, r in finals]
    counted = [(r, g) for r, g in graded if r["status"] not in graders.EXCLUDED]
    excluded = {}
    for r, _ in graded:
        if r["status"] in graders.EXCLUDED:
            rule = str(r.get("status_rule"))
            by_rule = excluded.setdefault(r["status"], {})
            by_rule[rule] = by_rule.get(rule, 0) + 1
    reruns = [r for attempt, r in finals if attempt == 2]
    attempts = [r for _, a in ordered for r in a.values()]
    costs = [r["usage"]["cost_usd"] for r in attempts if _number(_get(r, "usage", "cost_usd"))]
    latencies = [r["usage"]["latency_ms"] for r, _ in counted if _number(_get(r, "usage", "latency_ms"))]
    d = len(counted)
    passes = [1 if g["passed"] is True else 0 for _, g in counted]
    c = sum(passes)
    n_excluded = sum(sum(rules.values()) for rules in excluded.values())
    interval = stats.wilson(c, d)
    return {
        "case": case_id, "model": model, "condition": condition,
        "slots": len(ordered), "attempts": len(attempts), "denominator": d,
        "passed": c, "rate": stats.rnd(c / d) if d else None,
        "ci": list(interval) if interval else None,
        "pass_at_k": {str(k): stats.pass_at_k(d, c, k) for k in stats.KS},
        "pass_pow_k": {str(k): stats.pass_pow_k(d, c, k) for k in stats.KS},
        "excluded": {s: dict(sorted(rules.items())) for s, rules in sorted(excluded.items())},
        "excluded_total": n_excluded,
        "excluded_share": stats.rnd(n_excluded / len(ordered)),
        "reruns": len(reruns),
        "infra_recovered": sum(r["status"] not in graders.EXCLUDED for r in reruns),
        "infra_recurred": sum(r["status"] == "infra-error" for r in reruns),
        "runaway": sum(r["status"] == "runaway" for r, _ in counted),
        "contract_violation": sum(r["status"] == "contract-violation" for r, _ in counted),
        "tags": dict(sorted(Counter(t for _, g in graded for t in g["tags"]).items(), key=lambda kv: (-kv[1], kv[0]))),
        "cost": {"sum": round(sum(costs), 6), "mean": round(sum(costs) / len(costs), 6) if costs else None,
                 "priced": len(costs), "unpriced": len(attempts) - len(costs)},
        "latency": {"n": len(latencies), "p50": stats.percentile(latencies, 50),
                    "p95": stats.percentile(latencies, 95),
                    "p95_is_max": 0 < len(latencies) < P95_MAX_BELOW},
        "pass_vector": passes,
    }


def build_report(records, eval_set, seed=stats.DEFAULT_SEED, resamples=stats.DEFAULT_RESAMPLES) -> dict:
    by_case = {c["id"]: c for c in eval_set["cases"]}
    grouped = {}
    for (case_id, model, condition, _rep), attempts in _slots(records).items():
        if case_id not in by_case:
            raise ReportError(f"a record names case {case_id!r}, which the cases file does not have")
        grouped.setdefault((case_id, model, condition), []).append((_rep, attempts))
    order = {c["id"]: i for i, c in enumerate(eval_set["cases"])}
    cells = [_cell(case_id, model, condition, slots, by_case[case_id])
             for (case_id, model, condition), slots in sorted(
                 grouped.items(), key=lambda kv: (order[kv[0][0]], kv[0][1], CONDITIONS.index(kv[0][2])))]
    index = {(c["case"], c["model"], c["condition"]): c for c in cells}
    deltas, overall = [], []
    for model in sorted({c["model"] for c in cells}):
        pairs = []
        for case in eval_set["cases"]:
            sk, ns = index.get((case["id"], model, "skill")), index.get((case["id"], model, "no-skill"))
            if not (sk or ns):
                continue
            pair = ((sk or {}).get("pass_vector", []), (ns or {}).get("pass_vector", []))
            pairs.append(pair)
            ci = stats.bootstrap_ci([pair], seed, resamples)
            deltas.append({"model": model, "case": case["id"], "skill_n": len(pair[0]), "noskill_n": len(pair[1]),
                           "delta": stats.delta(*pair), "ci": list(ci) if ci else None})
        value, used = stats.overall_delta(pairs)
        ci = stats.bootstrap_ci(pairs, seed, resamples)
        overall.append({"model": model, "delta": value, "ci": list(ci) if ci else None,
                        "cases_used": used, "cases_total": len(eval_set["cases"])})
    versions = sorted({str(_get(r, "subject", "cli_version")) for r in records if _get(r, "subject", "cli_version")})
    return {
        "cells": cells, "deltas": deltas, "overall": overall,
        "bootstrap": {"seed": seed, "resamples": resamples},
        "footer": {"eval_set_version": sorted({str(r.get("eval_set_version")) for r in records}),
                   "cli_versions": versions, "judge": JUDGE_NOTE, "records": len(records)},
    }


def _fmt(x, spec=".4f"):
    return "n/a" if x is None else format(x, spec)


def _interval(ci):
    return "n/a" if ci is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]"


def headline(cell) -> str:
    if not cell["denominator"]:
        return "n/a (no gradable trials)"
    c, d = cell["passed"], cell["denominator"]
    return f"{c}/{d} ({100 * c / d:.1f}%) {_interval(cell['ci'])}"


def _counts(counts) -> str:
    return ", ".join(f"{k}: {v}" for k, v in counts.items()) or "none"


def format_cell(cell) -> str:
    excluded = ", ".join(f"{s}/{rule}: {n}" for s, rules in cell["excluded"].items() for rule, n in rules.items())
    lat = cell["latency"]
    cost = cell["cost"]
    mean = "n/a" if cost["mean"] is None else f"${cost['mean']:.4f}"
    note = f" (the max: fewer than {P95_MAX_BELOW} trials)" if lat["p95_is_max"] else ""
    return "\n".join([
        f"{cell['case']} | {cell['model']} | {cell['condition']}",
        f"  slots {cell['slots']} (attempts {cell['attempts']}), denominator {cell['denominator']}",
        f"  headline    {headline(cell)}",
        "  pass@k      " + "  ".join(f"k={k}: {_fmt(v)}" for k, v in cell["pass_at_k"].items()),
        "  pass^k      " + "  ".join(f"k={k}: {_fmt(v)}" for k, v in cell["pass_pow_k"].items()),
        f"  excluded    {cell['excluded_total']} of {cell['slots']} slots ({100 * cell['excluded_share']:.1f}%): "
        f"{excluded or 'none'}; "
        f"infra recurred {cell['infra_recurred']}, infra recovered by re-run {cell['infra_recovered']}",
        f"  runaway {cell['runaway']}, contract-violation {cell['contract_violation']}",
        f"  tags        {_counts(cell['tags'])}",
        f"  cost        sum ${cost['sum']:.4f}, mean {mean} per priced attempt, unpriced {cost['unpriced']}",
        f"  latency     n={lat['n']} p50 {_fmt(lat['p50'], 'd')} ms, p95 {_fmt(lat['p95'], 'd')} ms{note}",
    ])


def format_report(report) -> str:
    boot = report["bootstrap"]
    lines = [f"Eval report. Pass rates are {HEADLINE_NOTE}.", ""]
    for cell in report["cells"]:
        lines += [format_cell(cell), ""]
    lines.append(f"Skill minus no-skill: pass-rate delta, 95% bootstrap CI ({boot['resamples']} resamples, "
                 f"seed {boot['seed']})")
    for model in dict.fromkeys(o["model"] for o in report["overall"]):
        lines.append(f"  {model}")
        for d in (d for d in report["deltas"] if d["model"] == model):
            value = ("n/a (a side has no gradable trials)" if d["delta"] is None
                     else f"{d['delta']:+.4f} {_interval(d['ci'])}")
            lines.append(f"    {d['case']:<10} {value}  (skill {d['skill_n']}, no-skill {d['noskill_n']} trials)")
        o = next(o for o in report["overall"] if o["model"] == model)
        value = "n/a" if o["delta"] is None else f"{o['delta']:+.4f} {_interval(o['ci'])}"
        lines.append(f"    {'overall':<10} {value}  over {o['cases_used']} of {o['cases_total']} cases")
    f = report["footer"]
    lines += ["", f"eval set version: {', '.join(f['eval_set_version'])}",
              f"CLI versions seen: {', '.join(f['cli_versions']) or 'none recorded'}",
              f"judge: {f['judge']}"]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Print the per-cell report for a file of trial records; writes nothing.")
    ap.add_argument("source", help="a trials.jsonl, or a run directory that holds one")
    ap.add_argument("--cases", default=str(cases.DEFAULT_PATH))
    args = ap.parse_args(argv)
    try:
        records = load_records(args.source)
        eval_set = cases.load(args.cases)
        _, warnings = check_eval_set(records, eval_set["sha256"])
        report = build_report(records, eval_set)
    except (ReportError, OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        ap.error(str(e))
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    print(format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
