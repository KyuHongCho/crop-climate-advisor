"""Judge one trial record with the Opus judge (J1 to J3, two runs each); this spends real money.
    python -m evals.judge <record.json> --approve-spend 0.5 [--out FILE]
`--approve-spend` gates the start against accidental spend; it is not consent and does not cap re-run calls."""
import argparse
import json
import math
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple

from crop_advisor import claims
from crop_advisor.climate import replay_path

from . import cases, config, graders, judgements, record, run, snapshot
from .adapters import claude_code
from .status import classify_status

FIXTURE_DIR = run.FIXTURE_DIR
RETRIES_SUBTYPE = "error_max_structured_output_retries"
REASONS = {"schema-failure": "judge output failed the schema", "judge-runaway": "judge hit the cost cap",
           "failed": "judge call failed"}


def site_facts(case, replay_dir=FIXTURE_DIR) -> dict:
    """The site's annual mean and each source's direction, from the recorded climate alone."""
    with open(replay_path(replay_dir, case["lat"], case["lon"]), encoding="utf-8") as f:
        mean = round(json.load(f)["properties"]["parameter"]["T2M"]["ANN"], 2)
    cs = graders.basil_claims()
    return {"place": case.get("place") or case["site"], "lat": case["lat"], "lon": case["lon"], "mean": mean,
            "sources": [{"source": c.source, "lo": c.opt_min, "hi": c.opt_max, "condition": c.condition,
                         "direction": claims.classify(mean, c)} for c in cs],
            "opposed": {"below", "above"} <= claims.divergence(mean, cs)}


def facts(case, replay_dir=FIXTURE_DIR) -> str:
    d = site_facts(case, replay_dir)
    lines = [judgements.SITE_LINE.format(**{k: d[k] for k in ("place", "lat", "lon", "mean")})]
    for s in d["sources"]:
        suffix = judgements.CONDITION_SUFFIX.format(condition=s["condition"]) if s["condition"] else ""
        lines.append(judgements.SOURCE_LINE.format(**{**s, "condition": suffix}))
    opposed = judgements.OPPOSED_YES if d["opposed"] else judgements.OPPOSED_NO
    lines.append(judgements.OPPOSED_LINE.format(opposed=opposed))
    return "\n".join(lines)


def fence_for(answer: str) -> str:
    return "`" * max(3, max(map(len, re.findall(r"`+", answer)), default=0) + 1)


def build_prompt(case, answer: str, replay_dir=FIXTURE_DIR) -> str:
    return judgements.PROMPT_TEMPLATE.format(rubric=judgements.rubric_text().strip(), facts=facts(case, replay_dir),
                                             fence=fence_for(answer), answer=answer)


def command(prompt, model, budget_usd, binary="claude"):
    return [binary, "-p", prompt, "--model", model, "--tools", "", "--json-schema", json.dumps(judgements.OUTPUT_SCHEMA),
            "--max-budget-usd", f"{budget_usd:.2f}", "--setting-sources", "project", "--strict-mcp-config",
            "--permission-mode", "default", "--no-session-persistence", "--output-format", "stream-json", "--verbose"]


def ineligible(rec):
    """Why a record is not judged, or None. Only an answered `ok` trial is."""
    if not isinstance(rec, dict):
        return "not a trial record"
    if rec.get("status") != "ok":
        return f"its status is {rec.get('status')!r}, and only ok trials are judged"
    if not isinstance(rec.get("trial_id"), str) or not rec["trial_id"]:
        return "it has no trial id"
    if not isinstance(rec.get("output"), dict) or not isinstance(rec["output"].get("answer"), str):
        return "it holds no answer"
    return None


class Mapped(NamedTuple):
    outcome: str
    status: str
    rule: str
    structured: object
    require_model: bool
    retry: bool
    systemic: bool


def map_call(run_result, parsed, failure=None) -> Mapped:
    """One judge call's outcome: classify_status first, then the structured-output check in place of its rule 11."""
    rr, event = run_result or {}, parsed.get("result_event")
    st = classify_status(record.status_inputs(
        condition="no-skill", exit_code=rr.get("exit_code"), signal=rr.get("signal"),
        timed_out=bool(rr.get("timed_out")), harness_exception=failure,
        stderr_tail=(rr.get("stderr") or "")[-record.STDERR_TAIL_CHARS:],
        stream_parsed=parsed.get("stream_parsed", False), result_event=event,
        permission_denials=(event or {}).get("permission_denials") or ()))

    def result(outcome, structured=None, require_model=False, retry=False, systemic=False):
        return Mapped(outcome, st.status, st.rule, structured, require_model, retry, systemic)

    if st.rule in run.SYSTEMIC:
        return result("failed", systemic=True)
    if st.status == "infra-error":
        return result("failed", retry=True)
    if st.rule == "3":
        return result("judge-runaway")
    if st.rule == "4c":
        return result("schema-failure" if (event or {}).get("subtype") == RETRIES_SUBTYPE else "failed")
    if st.rule in ("7", "8") or event is None:
        return result("failed")
    if event.get("is_error") or str(event.get("subtype") or "").startswith("error_"):
        return result("failed")
    structured = event.get("structured_output")
    outcome = "complete" if judgements.parse_output(structured) is not None else "schema-failure"
    return result(outcome, structured, require_model=True)


def _cost(event):
    cost = (event or {}).get("total_cost_usd")
    ok = isinstance(cost, (int, float)) and not isinstance(cost, bool) and math.isfinite(cost)
    return cost if ok else None


def judge_run(*, trial_id, run_index, args, work, env, timeout, model, scrub, hashes):
    """One run: the call (an infra-error re-run once), its mapping and record-time R3. Returns (line, stop reason)."""
    for attempt in (1, 2):
        started, failure, rr, parsed, version = snapshot.utc_now(), None, None, {}, None
        try:
            version = claude_code.cli_version(args[0], env, work)
            rr = claude_code.run(args, work, env, timeout)
            parsed = claude_code.parse(rr["lines"])
        except Exception as e:  # noqa: BLE001 - any harness fault becomes rule 1, never a lost call
            failure = f"{type(e).__name__}: {e}"
        mapped = map_call(rr, parsed, failure)
        if not (mapped.retry and attempt == 1):
            break
    event = parsed.get("result_event")
    models = claude_code.canonical_models(event)
    outcome, structured = mapped.outcome, mapped.structured
    stop = f"systemic:{mapped.rule}" if mapped.systemic else None
    try:
        config.check_canonical(model, models, require_model=mapped.require_model)
    except config.ModelMismatch as e:
        print(f"R3 stop: {e}", file=sys.stderr)
        # A verdict from another model must never be read back as one, so the line keeps only its cost.
        outcome, structured, stop = "failed", None, "r3"
    verdicts = judgements.parse_output(structured) if outcome == "complete" else None
    line = {"trial_id": trial_id, "run_index": run_index, "rubric_sha256": hashes[0], "judge_config_sha256": hashes[1],
            "judge_model": model, "canonical_models": models, "cli_version": version, "status": mapped.status,
            "outcome": outcome, "cost_usd": _cost(event), "latency_ms": (rr or {}).get("latency_ms"),
            "started_at": started, "ended_at": snapshot.utc_now(), "verdicts": verdicts,
            "raw": record.scrub_tree(structured, scrub)}
    return line, stop


# C0 and C1 controls except newline and tab, and the bidi overrides, which a terminal would act on.
_UNPRINTABLE = re.compile("[\x00-\x08\x0b-\x1f\x7f-\x9f\u202a-\u202e\u2066-\u2069]")


def _safe(text) -> str:
    return _UNPRINTABLE.sub("", str(text)).encode("utf-8", "backslashreplace").decode("utf-8")


def format_report(trial_id, case_id, lines) -> str:
    by_index = judgements.current_lines(lines)
    runs = [by_index.get((trial_id, i)) for i in range(1, judgements.RUNS + 1)]
    derived = judgements.derive(runs)
    out = [f"trial   {trial_id}  (case {case_id})"]
    for c in judgements.CRITERIA:
        each = [("not run" if r is None else r["verdicts"][c] if r["outcome"] == "complete" else r["outcome"])
                for r in runs]
        out.append(f"{c}      {derived[c] or 'pending'}  ({', '.join(f'run {i} {v}' for i, v in enumerate(each, 1))})")
        for i, r in enumerate(runs, 1):
            if r is None:
                continue
            why = _safe(r["raw"][c]["rationale"]) if r["outcome"] == "complete" else f"no rationale: {REASONS[r['outcome']]}"
            out.append(f"          run {i}: {why}")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Judge one trial record (J1-J3, two runs). This spends real money.")
    ap.add_argument("record", help="a trial record written by evals.trial")
    ap.add_argument("--approve-spend", type=run._spend, default=None, metavar="USD",
                    help="allow the judge calls to start if their projection (2 x judge_budget_usd) fits under this many dollars; "
                         "it does not cap an infra-error re-run, which can add calls")
    ap.add_argument("--out", default=None, help="append the judgement lines to this file")
    args = ap.parse_args(argv)
    path = Path(args.record)
    if path.is_dir():
        ap.error(f"{path} is a directory; judging a run directory is not built yet, give one record file")
    try:
        cfg = config.load()
        settings = {**config.run_settings(cfg), **config.judge_settings(cfg)}
        rec = json.loads(path.read_text(encoding="utf-8"))
        why = ineligible(rec)
        if why is None:
            _, case = cases.get(rec["case_id"])
            prompt = build_prompt(case, rec["output"]["answer"])
            if args.out:
                judgements.read_lines(args.out)
    except (config.ConfigError, OSError, ValueError, KeyError, TypeError) as e:
        ap.error(f"cannot judge {path}: {type(e).__name__}: {e}")
    if why:
        ap.error(f"{path}: {why}; nothing ran")
    which = shutil.which("claude")
    if which is None:
        ap.error("cannot start, missing: `claude` on PATH")

    runs = judgements.RUNS
    projection = runs * settings["judge_budget_usd"]
    if args.approve_spend is None:
        print(f"Projected spend for {runs} judge calls: ${projection:.2f} ({runs} x judge_budget_usd "
              f"{settings['judge_budget_usd']:g}, each call at its cap).\n"
              "Nothing ran. Add --approve-spend USD to start; it is a guard against accidents, not an approval.")
        return 2
    cap = min(settings["run_budget_usd"], args.approve_spend)
    if projection > cap:
        ap.error(f"projected spend ${projection:.2f} exceeds the cap ${cap:.2f}, the lower of run_budget_usd "
                 f"{settings['run_budget_usd']:g} and --approve-spend; nothing ran")

    claude_path = os.path.abspath(which)
    lines, stop = [], None
    with tempfile.TemporaryDirectory(prefix="crop-judge-") as work:
        env = claude_code.environment(work)
        leaks = claude_code.memory_leaks(work, env)
        if leaks:
            ap.error("instruction files would leak into the judge: " + ", ".join(map(str, leaks)))
        if args.out:
            try:
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                open(args.out, "ab").close()
            except OSError as e:
                ap.error(f"cannot write --out {args.out}: {type(e).__name__}: {e}; nothing ran")
        call = command(prompt, cfg["judge"], settings["judge_budget_usd"], claude_path)
        hashes, scrub = (judgements.rubric_sha256(), judgements.config_sha256()), record.scrubber(work, env)
        for index in range(1, runs + 1):
            line, stop = judge_run(trial_id=rec["trial_id"], run_index=index, args=call, work=work, env=env,
                                   timeout=cfg["trial_timeout_s"], model=cfg["judge"], scrub=scrub, hashes=hashes)
            lines.append(line)
            if args.out:
                try:
                    judgements.append_line(args.out, line)
                except (OSError, ValueError) as e:
                    print(f"cannot write --out {args.out}: {type(e).__name__}: {e}", file=sys.stderr)
                    stop = "write-failed"
            cost = "unpriced" if line["cost_usd"] is None else "$%.4f" % line["cost_usd"]
            print(f"run {index}: {line['outcome']} (status {line['status']}) cost {cost}", file=sys.stderr)
            if stop:
                print(f"stopped: {stop}", file=sys.stderr)
                break
    print(format_report(rec["trial_id"], rec["case_id"], lines))
    complete = all(v is not None for v in judgements.derive(
        [judgements.current_lines(lines).get((rec["trial_id"], i)) for i in range(1, runs + 1)]).values())
    return 0 if complete and stop is None else 1


if __name__ == "__main__":
    sys.exit(main())
