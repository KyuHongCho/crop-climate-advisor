"""Score one trial record with the deterministic graders.

    python -m evals.score <record.json>
"""
import argparse
import json
import sys
from pathlib import Path

from . import cases, graders

SPAN_CHARS = 80


def _span(e: dict) -> str:
    text = json.dumps(e["text"] if len(e["text"]) <= SPAN_CHARS else e["text"][:SPAN_CHARS] + "...", ensure_ascii=False)
    where = "" if e["start"] is None else f"[{e['start']}:{e['end']}] "
    label = f"{e['source']}: " if e.get("source") else ""
    return f"{label}{where}{text}"


def format_report(result: dict) -> str:
    lines = [f"trial   {result['trial_id']}  (case {result['case_id']})",
             f"status  {result['status']} (rule {result['status_rule']}: {result['status_reason']})"]
    for g in result["criteria"]:
        lines.append(f"{g.criterion}      {g.verdict}  {g.reason}")
        lines += [f"          {_span(e)}" for e in g.evidence]
    passed = {True: "pass", False: "fail", None: "not graded"}[result["passed"]]
    lines += [f"tags    {', '.join(result['tags']) or '(none)'}", f"trial   {passed}"]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Print D1-D4, the status and the triage tags for one trial record.")
    ap.add_argument("record", help="a trial record written by evals.trial")
    ap.add_argument("--cases", default=str(cases.DEFAULT_PATH))
    args = ap.parse_args(argv)
    try:
        record = json.loads(Path(args.record).read_text(encoding="utf-8"))
        _, case = cases.get(record["case_id"], args.cases)
        result = graders.grade(record, case)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        ap.error(f"cannot score {args.record}: {type(e).__name__}: {e}")
    print(format_report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
