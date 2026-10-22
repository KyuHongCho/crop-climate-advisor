"""Run the eval grid (case x model x condition x rep) under the spend caps and the stop policy.

    python -m evals.run --n 3 --approve-spend 3

This spends real money. `--approve-spend` is a speed bump against accidental spend, not consent.
"""
import argparse
import hashlib
import itertools
import json
import os
import shutil
import signal
import sys
import threading
import uuid
from pathlib import Path

from crop_advisor.climate import replay_path

from . import cases, config, graders, record, report, snapshot
from .adapters import claude_code

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results"
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "nasa_power"
SYSTEMIC = frozenset({"1", "1b", "1c", "4b", "5"})


def systemic_rule(rec):
    """The rule that decided the trial, reading a rule-1d record's raw rule rather than "1d"."""
    rule = rec.get("status_rule")
    if rule == "1d":
        rule = ((rec.get("status_inputs") or {}).get("redaction_conflict") or {}).get("rule")
    return rule


def _positive_int(text):
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a positive integer") from None
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{text!r} is not a positive integer")
    return value


def _spend(text):
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not config.positive_number(value):
        raise argparse.ArgumentTypeError(f"{text!r} is not a finite number greater than 0")
    return value


def _preflight(eval_set) -> list:
    missing = [f"the recorded climate for case {c['id']!r} ({replay_path(FIXTURE_DIR, c['lat'], c['lon'])})"
               for c in eval_set["cases"] if not replay_path(FIXTURE_DIR, c["lat"], c["lon"]).is_file()]
    if not Path(claude_code.SKILL_DIR).is_dir():
        missing.append(f"the skill folder ({claude_code.SKILL_DIR})")
    if shutil.which("claude") is None:
        missing.append("`claude` on PATH")
    return missing


def _before_attempt(streaks, max_streak, spent, cap):
    for (model, condition), count in streaks.items():
        if count >= max_streak:
            return f"streak:{model}/{condition}"
    return "spend-cap" if spent >= cap else None


def _raise_interrupt(signum, frame):
    raise KeyboardInterrupt


def _install_handlers(installed):
    """Stop on SIGINT, SIGTERM or SIGHUP like a Ctrl-C, recording each replaced handler in `installed`."""
    handlers = {signal.SIGINT: signal.default_int_handler, signal.SIGTERM: _raise_interrupt}
    if signal.getsignal(signal.SIGHUP) != signal.SIG_IGN:
        handlers[signal.SIGHUP] = _raise_interrupt
    for signum, handler in handlers.items():
        installed[signum] = signal.signal(signum, handler)


def _restore_handlers(installed):
    for signum, previous in installed.items():
        if previous is not None:
            signal.signal(signum, previous)


def _after_record(rec, model, skill_hash):
    seen = rec["condition"].get("skill_sha256")
    if rec["condition"]["condition"] == "skill" and seen is not None and seen != skill_hash:
        return "skill-changed"
    try:
        config.check_canonical(model, rec["subject"]["canonical_models"], require_model=rec["status"] == "ok")
    except config.ModelMismatch as e:
        print(f"R3 stop: {e}", file=sys.stderr)
        return "r3"
    rule = systemic_rule(rec)
    return f"systemic:{rule}" if rule in SYSTEMIC else None


def _progress(records, rec):
    cost = snapshot.attempt_cost(rec)
    cost_text = "unpriced" if cost is None else "$%.4f" % cost
    spent = sum(c for c in map(snapshot.attempt_cost, records) if c is not None)
    print(f"[{len(records)}] {rec['case_id']} {rec['subject']['model']} {rec['condition']['condition']} "
          f"r{rec['rep']} a{rec['attempt']}: {rec['status']} (rule {rec['status_rule']}) "
          f"cost {cost_text}, spend ${spent:.4f}", file=sys.stderr)


def _loop(*, eval_set, subjects, n, max_streak, trial_budget_usd, timeout_s, cap, claude_path, run_id, records, trials,
          state, skill_hash):
    """Run every slot rep-major; return the first stop reason, or None when the grid finished."""
    streaks = {(m, c): 0 for m in subjects for c in claude_code.CONDITIONS}
    baseline = None
    for rep, case, model, condition in itertools.product(
            range(1, n + 1), eval_set["cases"], subjects, claude_code.CONDITIONS):
        attempt = 1
        while True:
            stop = _before_attempt(streaks, max_streak, snapshot.charged(records, trial_budget_usd), cap)
            if stop:
                return stop
            trial_id = f"{case['id']}-{condition}-{model}-r{rep}-a{attempt}-{uuid.uuid4().hex[:8]}"
            state["in_flight"] = {"case_id": case["id"], "model": model, "condition": condition, "rep": rep,
                                  "attempt": attempt, "trial_id": trial_id}
            try:
                rec = claude_code.run_trial(
                    case=case, eval_set=eval_set, model=model, condition=condition, budget_usd=trial_budget_usd,
                    timeout=timeout_s, binary=claude_path, trial_id=trial_id, expected_cli_version=baseline)
            except claude_code.VersionChanged as e:
                print(f"version changed: {e}", file=sys.stderr)
                state["in_flight"] = None
                return "version-changed"
            rec.update(run_id=run_id, rep=rep, attempt=attempt)
            # Held so a signal cannot split the file write from the count; it blocks this thread only, not others.
            held = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM, signal.SIGHUP})
            try:
                trials.write(json.dumps(rec) + "\n")
                trials.flush()
                records.append(rec)
                state["in_flight"] = None
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, held)
            baseline = baseline or rec["subject"]["cli_version"]
            _progress(records, rec)
            stop = _after_record(rec, model, skill_hash)
            if stop:
                return stop
            if attempt == 1 and rec["status"] == "infra-error":
                attempt = 2
                continue
            key = (model, condition)
            streaks[key] = streaks[key] + 1 if rec["status"] in graders.EXCLUDED else 0
            break
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run the eval grid under the spend caps. This spends real money.")
    ap.add_argument("--config", default=str(config.DEFAULT_PATH))
    ap.add_argument("--cases", default=str(cases.DEFAULT_PATH))
    ap.add_argument("--n", type=_positive_int, default=None, help="trials per cell; makes the run a pilot")
    ap.add_argument("--approve-spend", type=_spend, default=None, metavar="USD",
                    help="lower the run cap to this many dollars and allow the run to start")
    ap.add_argument("--results-dir", default=None, help=f"default: {DEFAULT_RESULTS_DIR}")
    args = ap.parse_args(argv)
    msg = graders.login_error()
    if msg:
        ap.error(msg)
    try:
        cfg = config.load(args.config)
        eval_set = cases.load(args.cases)
        settings = config.run_settings(cfg)
    except (config.ConfigError, OSError, ValueError) as e:
        ap.error(str(e))
    missing = _preflight(eval_set)
    if missing:
        ap.error("cannot start, missing: " + "; ".join(missing))

    try:
        skill_hash = record.skill_sha256(claude_code.SKILL_DIR)
    except OSError as e:
        ap.error(f"cannot read the skill folder {claude_code.SKILL_DIR}: {e}")

    results_dir = Path(args.results_dir) if args.results_dir else DEFAULT_RESULTS_DIR
    subjects = cfg["subjects"]
    n = args.n if args.n is not None else settings["n_trials"]
    claude_path = os.path.abspath(shutil.which("claude"))
    scrubbed_path = record.scrubber()(claude_path)
    projection = snapshot.project(settings, subjects, n, len(eval_set["cases"]), results_dir, scrubbed_path)
    planned = n * len(eval_set["cases"]) * len(subjects) * len(claude_code.CONDITIONS)
    if args.approve_spend is None:
        print(f"Projected spend for {planned} planned trials: ${projection['total_usd']:.2f} "
              f"(source: {projection['source']}).\n"
              f"Worst case if every trial reached its cap: {planned} x ${cfg['trial_budget_usd']:.2f} = "
              f"${planned * cfg['trial_budget_usd']:.2f}, plus up to one turn each.\n"
              "Nothing ran. Add --approve-spend USD to start; it is a guard against accidents, not an approval.")
        return 2
    cap = min(settings["run_budget_usd"], args.approve_spend)
    if projection["total_usd"] > cap:
        ap.error(f"projected spend ${projection['total_usd']:.2f} exceeds the cap ${cap:.2f}, the lower of "
                 f"run_budget_usd {settings['run_budget_usd']:g} and --approve-spend; nothing ran")
    run_id = snapshot.new_run_id()
    run_dir = results_dir / run_id
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except OSError as e:
        ap.error(f"cannot create {run_dir}: {e}")

    effective = {**settings, "n_trials": n, "trial_budget_usd": cfg["trial_budget_usd"],
                 "trial_timeout_s": cfg["trial_timeout_s"], "effective_cap_usd": cap}
    started, records, stop, failed = snapshot.utc_now(), [], None, False
    state, installed = {"in_flight": None}, {}
    print(f"run {run_id}: {planned} planned trials, projected ${projection['total_usd']:.2f}, "
          f"cap ${cap:.2f} -> {run_dir}", file=sys.stderr)
    try:
        if threading.current_thread() is threading.main_thread():
            _install_handlers(installed)
        with open(run_dir / report.TRIALS_FILE, "w", encoding="utf-8", newline="\n") as trials:
            stop = _loop(eval_set=eval_set, subjects=subjects, n=n, max_streak=settings["max_consecutive_excluded"],
                         trial_budget_usd=cfg["trial_budget_usd"], timeout_s=cfg["trial_timeout_s"], cap=cap,
                         claude_path=claude_path, run_id=run_id, records=records, trials=trials, state=state,
                         skill_hash=skill_hash)
    except KeyboardInterrupt:
        stop = "interrupted"
    except BaseException as e:
        stop, failed = f"error:{type(e).__name__}", True
        raise
    finally:
        for signum in installed:
            signal.signal(signum, signal.SIG_IGN)
        try:
            try:
                snap = snapshot.build_snapshot(
                    run_id=run_id, stop_reason=stop, started=started, finished=snapshot.utc_now(),
                    overrides={"n": args.n} if args.n is not None else {}, approved_spend_usd=args.approve_spend,
                    claude_path=scrubbed_path, config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
                    settings=effective, eval_set=eval_set, subjects=subjects, judge=cfg["judge"],
                    fixture_accessed=snapshot.fixture_date(FIXTURE_DIR / "SOURCE.md"), planned_slots=planned,
                    projection=projection, records=records, in_flight=state["in_flight"])
                (run_dir / snapshot.SNAPSHOT_FILE).write_text(json.dumps(snap, indent=2) + "\n", encoding="utf-8")
                try:
                    summary = snapshot.format_summary(snap)
                except Exception as e:  # noqa: BLE001 - must not replace the exception that stopped the run
                    failure = f"summary failed: {type(e).__name__}: {e}"
                    print(failure, file=sys.stderr)
                    summary = f"# Eval run {run_id}\n\n{failure}\n"
                (run_dir / snapshot.SUMMARY_FILE).write_text(summary, encoding="utf-8")
            except Exception as e:  # noqa: BLE001 - keep the exception that stopped the run
                if not failed:
                    raise
                print(f"snapshot failed: {type(e).__name__}: {e}", file=sys.stderr)
        finally:
            _restore_handlers(installed)
    print(f"{snap['status']} ({stop or 'all slots run'}): {run_dir}", file=sys.stderr)
    return 0 if stop is None else 1


if __name__ == "__main__":
    sys.exit(main())
