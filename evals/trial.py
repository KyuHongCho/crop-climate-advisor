"""Run one trial of one case and write its record.

    python -m evals.trial --adapter claude_code --case bangkok --model claude-sonnet-5-5 --condition skill
"""
import argparse
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import cases, config, graders, record
from .adapters import claude_code

ADAPTERS = {"claude_code": claude_code}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run one eval trial and write its record.")
    ap.add_argument("--adapter", required=True, choices=sorted(ADAPTERS))
    ap.add_argument("--case", required=True)
    ap.add_argument("--model", required=True, help="an exact subject id from the config")
    ap.add_argument("--condition", required=True, choices=claude_code.CONDITIONS)
    ap.add_argument("--config", default=str(config.DEFAULT_PATH))
    ap.add_argument("--cases", default=str(cases.DEFAULT_PATH))
    ap.add_argument("--timeout", type=float, default=None,
                    help="harness timeout in seconds, a finite number > 0 (default: config)")
    ap.add_argument("--out", default=None, help="record path (default: a new file under the system temp dir)")
    args = ap.parse_args(argv)
    msg = graders.login_error()
    if msg:
        ap.error(msg)
    if args.timeout is not None and not config.positive_number(args.timeout):
        ap.error(f"--timeout must be a finite number greater than 0, not {args.timeout:g}")

    try:
        cfg = config.load(args.config)
        eval_set, case = cases.get(args.case, args.cases)
    except (config.ConfigError, KeyError, ValueError, OSError) as e:
        ap.error(str(e))
    if args.model not in cfg["subjects"]:
        ap.error(f"--model {args.model!r} is not a configured subject ({', '.join(cfg['subjects'])})")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    trial_id = f"{case['id']}-{args.condition}-{args.model}-{stamp}-{uuid.uuid4().hex[:8]}"
    timeout = cfg["trial_timeout_s"] if args.timeout is None else args.timeout
    rec = ADAPTERS[args.adapter].run_trial(
        case=case, eval_set=eval_set, model=args.model, condition=args.condition,
        budget_usd=cfg["trial_budget_usd"], timeout=timeout, trial_id=trial_id)
    out = Path(args.out) if args.out else Path(tempfile.gettempdir()) / "crop-eval-trials" / f"{trial_id}.json"
    record.write(rec, out)
    print(f"{rec['status']} (rule {rec['status_rule']}: {rec['status_reason']}) -> {out}")
    if rec["status_rule"] in ("1b", "1c"):
        print(f"harness-error: {rec['status_reason']}", file=sys.stderr)
    try:
        config.check_canonical(args.model, rec["subject"]["canonical_models"], require_model=rec["status"] == "ok")
    except config.ModelMismatch as e:
        print(f"R3 stop: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
