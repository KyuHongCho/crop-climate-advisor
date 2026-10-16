"""A trial's status, derived from its `status_inputs` alone, so the offline gate can recompute it."""
import posixpath
import re
from typing import NamedTuple

from crop_advisor.climate import REPLAY_MISSING

RUNAWAY_SUBTYPES = ("error_max_budget_usd", "error_max_turns")
_INFRA_HTTP = {408, 429}
_STDERR_API_STATUS = re.compile(r"API Error:?\s*(\d{3})")
_STDERR_API_ERROR = re.compile(r"API Error|rate_limit_error|overloaded_error|api_error", re.IGNORECASE)
_RATE_OR_OVERLOAD = re.compile(r"rate[ _-]?limit|overloaded", re.IGNORECASE)
_NETWORK_FAILURE = re.compile(r"urllib\.error\.URLError|URLError|socket\.timeout|TimeoutError|timed out")
CLI_CALL = re.compile(r"(?:^|[\s;&|(])python3?\s+-m\s+crop_advisor\.cli\b")
# Only a lone CLI invocation is on contract; a chained, piped or substituted command can do anything else too.
_CLI_ONLY = re.compile(r"[ \t]*python3?[ \t]+-m[ \t]+crop_advisor\.cli(?:[ \t]+(?:(?![;&|<>`\n\r]|\$\().)*)?[ \t]*")

SKILL_ROOTS = (".agents/skills/", ".claude/skills/")
SKILL_NAME = "crop-suitability"
# Still listed by Claude Code 2.1.301 with bundled skills off (built-in plugins); the adapter denies them.
DENIED_SKILLS = ("design", "doctor", "plugin-authoring")
OFF_CONTRACT = ("contract-violation", "off-contract-tool")


class Status(NamedTuple):
    status: str
    rule: str
    reason: str
    tags: tuple


def _api_error(si: dict):
    """Rule 4's match: (HTTP status or None, text to search for rate-limit/overload), or None."""
    event = si.get("result_event") or {}
    subtype = event.get("subtype") or ""
    http = event.get("api_error_status")
    stderr = si.get("stderr_tail") or ""
    stderr_hit = _STDERR_API_ERROR.search(stderr)
    if http is None and not subtype.startswith("error_") and not stderr_hit:
        return None
    if http is None:
        m = _STDERR_API_STATUS.search(stderr)
        http = int(m.group(1)) if m else None
    text = " ".join(str(x) for x in (event.get("result"), event.get("errors"), stderr) if x)
    return http, text, subtype


def non_cli_bash(commands) -> list:
    return [c for c in commands if not _CLI_ONLY.fullmatch(c or "")]


def off_skill_reads(paths) -> list:
    # Normalised again so a hand-edited record cannot climb out of a Skill folder with "..".
    return [p for p in paths if not posixpath.normpath(p or ".").startswith(SKILL_ROOTS)]


def unexpected_skills(listed) -> list:
    return sorted(set(listed or []) - {SKILL_NAME, *DENIED_SKILLS})


def classify_status(si: dict) -> Status:
    condition = si.get("condition")
    event = si.get("result_event")
    cli_results = si.get("cli_tool_results") or []

    if si.get("harness_exception"):
        return Status("harness-error", "1", f"harness exception: {si['harness_exception']}", ("harness-error",))
    extra = unexpected_skills(si.get("listed_skills")) if condition == "skill" else []
    if extra:
        return Status("harness-error", "1b", f"the subject was also offered {', '.join(extra)}; "
                      "add them to DENIED_SKILLS in evals/status.py", ("harness-error",))
    if condition == "skill" and si.get("listed_skills") is not None and SKILL_NAME not in si["listed_skills"]:
        return Status("harness-error", "1c", f"the subject was not offered {SKILL_NAME}", ("harness-error",))
    if si.get("timed_out") or si.get("signal") is not None:
        why = "timed out" if si.get("timed_out") else f"killed by signal {si['signal']}"
        return Status("infra-error", "2", why, ("infra-error",))
    if event and event.get("subtype") in RUNAWAY_SUBTYPES:
        return Status("runaway", "3", event["subtype"], ("runaway",))
    api = _api_error(si)
    if api is not None:
        http, text, subtype = api
        detail = f"api_error_status={http}, subtype={subtype or None}"
        if http is not None and (http in _INFRA_HTTP or 500 <= http <= 599) or _RATE_OR_OVERLOAD.search(text):
            return Status("infra-error", "4a", detail, ("infra-error",))
        if http is not None and 400 <= http <= 499:
            return Status("harness-error", "4b", detail, ("harness-error",))
        return Status("harness-error", "4c", detail, ("harness-error",))
    if any(REPLAY_MISSING in r for r in cli_results):
        return Status("harness-error", "5", f"the CLI reported {REPLAY_MISSING}", ("harness-error",))
    if any(_NETWORK_FAILURE.search(r) for r in cli_results):
        return Status("infra-error", "6", "the CLI hit a NASA POWER network failure", ("infra-error",))
    if si.get("exit_code") == 0 and (not si.get("stream_parsed") or event is None):
        why = "unparseable stream" if not si.get("stream_parsed") else "no result event"
        return Status("contract-violation", "7", f"exit 0 with {why}", ("contract-violation",))
    if si.get("exit_code") != 0 or event is None:
        return Status("harness-error", "8", f"exit {si.get('exit_code')} not matched by rules 1-7", ("harness-error",))
    if si.get("permission_denials"):
        tools = ", ".join(sorted({d.get("tool_name", "?") for d in si["permission_denials"]}))
        if condition == "skill":
            return Status("contract-violation", "9", f"tool use denied: {tools}", ("contract-violation", "denied-tool"))
        return Status("ok", "9", f"tool use denied: {tools}", ("denied-tool",))
    off_contract = non_cli_bash(si.get("bash_commands") or []) if condition == "skill" else []
    if off_contract:
        return Status("contract-violation", "9b", f"non-CLI Bash call: {off_contract[0][:200]}", OFF_CONTRACT)
    off_reads = off_skill_reads(si.get("read_paths") or []) if condition == "skill" else []
    if off_reads:
        return Status("contract-violation", "9c", f"Read outside the Skill: {off_reads[0][:200]}", OFF_CONTRACT)
    other_skills = [s for s in si.get("skill_calls") or [] if s != SKILL_NAME] if condition == "skill" else []
    if other_skills:
        return Status("contract-violation", "9d", f"Skill call other than {SKILL_NAME}: {other_skills[0][:200]}",
                      OFF_CONTRACT)
    if condition == "skill" and not cli_results:
        return Status("contract-violation", "10", "skill condition made no CLI call", ("contract-violation", "no-tool-call"))
    if event.get("result") is not None:
        return Status("ok", "11", "answered", ())
    return Status("harness-error", "11", "result event carries no answer", ("harness-error",))
