"""Deterministic graders D1-D4 for basil's three disagreeing temperature claims, each returning evidence spans."""
import contextlib
import io
import re
import shlex
from typing import NamedTuple

from crop_advisor import claims, ecocrop
from crop_advisor.cli import build_parser

from . import record
from .status import CLI_CALL

# Keyed by the claim's label before " (", so a claim added to claims.py without a name here fails loudly.
SOURCE_NAMES = {
    "FAO ECOCROP": r"\bECOCROP\b",
    "Chang, Alderson & Wright": r"\bChang\b",
    # "Walters" alone would also match the paper both journal figures were read in (Walters, Tarr & Lopez).
    "Walters & Currey": r"\bCurrey\b",
}
DLI_MARKER = re.compile(r"\bDLI\b|daily\s+light\s+integral|\bmol\b", re.IGNORECASE)
DLI_WINDOW = 80
EXCLUDED = ("infra-error", "harness-error")
TAGS = {"D1": "missing-source", "D2": "missing-band", "D3": "missing-condition", "D4": "no-tool-call"}
WRONG_CLI_ARGS = "wrong-cli-args"
# The words the graders read from the redacted record (D4 reaches python3 and crop_advisor.cli through
# status.CLI_CALL); words only the status rules read are left to rule 1d. Keep in step with the regexes above.
GRADED_WORDS = ("Bash", "skill", "python", "python3", "-m", "crop_advisor.cli", "--crop", "--lat", "--lon", "--place",
                "basil", "ECOCROP", "Chang", "Currey", "DLI", "daily light integral", "mol", "to", "between", "and",
                "degrees", "Celsius")


class Grade(NamedTuple):
    criterion: str
    verdict: str  # pass | fail | n/a | ungraded
    reason: str
    evidence: tuple  # dicts with the matched text and its [start, end) offsets in the answer
    tags: tuple


def basil_claims() -> tuple:
    return claims.temperature_claims(ecocrop.load_crop("basil"))


def _num(x) -> str:
    body = f"{float(x):g}"
    tail = r"(?:\.0+)?" if "." not in body else "0*"
    return rf"(?<![\d.]){re.escape(body)}{tail}(?!\d|\.\d)"


_UNIT = r"(?:\s*(?:[°º]\s*C?|degrees?(?:\s+C(?:elsius)?)?|C\b))?"
_DASH = r"\s*(?:[-\u2010-\u2015\u2212]|to)\s*"


def range_pattern(lo, hi):
    """lo-hi in the common spellings: "18-27", "18 to 27 °C", "between 18 and 27"."""
    a, b = _num(lo), _num(hi)
    return re.compile(rf"(?:{a}{_UNIT}{_DASH}{b}|between\s+{a}{_UNIT}\s+and\s+{b})", re.IGNORECASE)


def _span(m, source) -> dict:
    return {"source": source, "start": m.start(), "end": m.end(), "text": m.group(0)}


def _grade(criterion, hits, missing, what, ok) -> Grade:
    if missing:
        return Grade(criterion, "fail", f"{what}: {', '.join(missing)}", tuple(hits), (TAGS[criterion],))
    return Grade(criterion, "pass", ok, tuple(hits), ())


def d1_sources(answer: str, cs) -> Grade:
    hits, missing = [], []
    for c in cs:
        m = re.search(SOURCE_NAMES[c.source.split(" (")[0]], answer, re.IGNORECASE)
        if m:
            hits.append(_span(m, c.source))
        else:
            missing.append(c.source)
    return _grade("D1", hits, missing, "not named", "all three sources named")


def d2_bands(answer: str, cs) -> Grade:
    hits, missing = [], []
    for c in cs:
        m = range_pattern(c.opt_min, c.opt_max).search(answer)
        if m:
            hits.append(_span(m, c.source))
        else:
            missing.append(f"{c.source} {c.opt_min:g}-{c.opt_max:g}")
    return _grade("D2", hits, missing, "band not stated", "all three bands stated")


def _dli_figure(condition: str):
    m = re.search(r"DLI\s+(\d+(?:\.\d+)?)(?:\s*[–-]\s*(\d+(?:\.\d+)?))?", condition)
    if not m:
        raise ValueError(f"cannot read a DLI figure from claim condition {condition!r}")
    lo, hi = m.groups()
    return range_pattern(lo, hi) if hi else re.compile(_num(lo))


def _near_marker(answer: str, m) -> bool:
    return bool(DLI_MARKER.search(answer, max(0, m.start() - DLI_WINDOW), m.end() + DLI_WINDOW))


def d3_conditions(answer: str, cs) -> Grade:
    hits, missing = [], []
    for c in cs:
        if c.condition is None:
            continue
        m = next((m for m in _dli_figure(c.condition).finditer(answer) if _near_marker(answer, m)), None)
        if m:
            hits.append(_span(m, c.source))
        else:
            missing.append(f"{c.source} ({c.condition})")
    return _grade("D3", hits, missing, "condition not stated", "both DLI conditions stated")


def _cli_args(command: str):
    """The command's CLI arguments as the real CLI's own parser reads them, or None if it would reject them."""
    m = CLI_CALL.search(command)
    tail = re.split(r"[;&|<>\n]", command[m.end():], maxsplit=1)[0]
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return build_parser().parse_args(shlex.split(tail, comments=True))
    except (SystemExit, ValueError):
        return None


def d4_cli_call(record: dict, case: dict) -> Grade:
    if (record.get("condition") or {}).get("condition") != "skill":
        return Grade("D4", "n/a", "the no-skill condition has no tools", (), ())
    commands = [(c.get("input") or {}).get("command") for c in (record.get("output") or {}).get("tool_calls") or []
                if c.get("name") == "Bash"]
    cli = [cmd for cmd in commands if isinstance(cmd, str) and CLI_CALL.search(cmd)]
    crop = case.get("crop", "basil")
    for cmd in cli:
        args = _cli_args(cmd)
        if (args is not None and args.crop == crop and args.lat == float(case["lat"])
                and args.lon == float(case["lon"])):
            return Grade("D4", "pass", f"ran the CLI with --crop {crop} and the case coordinates",
                         ({"start": None, "end": None, "text": cmd},), ())
    why = (f"no CLI call with --crop {crop} --lat {case['lat']} --lon {case['lon']}" if cli
           else "no CLI call was made")
    tag = WRONG_CLI_ARGS if cli else TAGS["D4"]
    return Grade("D4", "fail", why, tuple({"start": None, "end": None, "text": cmd} for cmd in cli), (tag,))


def grade(record: dict, case: dict) -> dict:
    """Score one record: D1-D4, the status and the R5 tags; passes only if the status is ok and no criterion fails."""
    status, status_tags = record["status"], list(record.get("tags") or [])
    answer = (record.get("output") or {}).get("answer")
    skill = (record.get("condition") or {}).get("condition") == "skill"
    if status in EXCLUDED:
        grades = [Grade(d, "ungraded", f"{status}: the trial is excluded", (), ()) for d in ("D1", "D2", "D3", "D4")]
    elif status == "runaway":
        grades = [Grade(d, "fail", "runaway: no answer", (), ()) for d in ("D1", "D2", "D3")]
        grades.append(Grade("D4", "fail" if skill else "n/a", "runaway: no answer" if skill
                            else "the no-skill condition has no tools", (), ()))
    else:
        text, cs = answer if isinstance(answer, str) else "", basil_claims()
        grades = [grader(text, cs) for grader in (d1_sources, d2_bands, d3_conditions)]
        grades.append(d4_cli_call(record, case))
    # A runaway's grades carry no tags, so it keeps only its status tag.
    tags = list(dict.fromkeys(status_tags + [t for g in grades for t in g.tags]))
    applicable = [g for g in grades if g.verdict != "n/a"]
    passed = None if status in EXCLUDED else status == "ok" and all(g.verdict == "pass" for g in applicable)
    return {"trial_id": record.get("trial_id"), "case_id": record.get("case_id"), "status": status,
            "status_rule": record.get("status_rule"), "status_reason": record.get("status_reason"),
            "criteria": grades, "tags": tags, "passed": passed}


def login_error():
    """None, or the message that refuses a login the username rewrite would turn into a different grade."""
    rewrite = record.rewriter()
    changed = [w for w in GRADED_WORDS if rewrite(w) != w]
    if not changed:
        return None
    return (f"your login name is a word the graders read ({', '.join(repr(w) for w in changed)}): the stored record "
            "would show 'user' there and the grade would silently change. Run the same command under another "
            "name, e.g. LOGNAME=eval USER=eval python -m evals.trial ...; under that name your real login is not "
            "scrubbed where it appears as a bare word; it is itself a word the graders read, so a search cannot "
            "tell it from the legitimate word.")
