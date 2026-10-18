"""The trial record every adapter writes, and the redaction applied to everything copied from a subprocess."""
import getpass
import hashlib
import json
import os
import re
from pathlib import Path

from .status import classify_status

RECORD_VERSION = 1
STDERR_TAIL_CHARS = 4096
REDACTED = "[REDACTED]"
# A match must not swallow shell metacharacters (rule 9b reads them) or begin inside a word ("risk-free").
_SECRET_PATTERNS = (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"), re.compile(r"Bearer[ \t]+[^\s;&|<>`$()'\"\\]+"))
TOKEN_ENV = "CLAUDE_CODE_OAUTH_TOKEN"


def redact(text: str, env=None) -> str:
    env = os.environ if env is None else env
    token = env.get(TOKEN_ENV)
    if token:
        text = text.replace(token, REDACTED)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


def slug(path) -> str:
    """Claude Code's project-folder name for a path, as in `memory_paths`; pinned in tests/test_eval_trial.py."""
    return re.sub(r"[^A-Za-z0-9]", "-", str(path))


def rewriter(base=None, *, home=None, uid=None, user=None):
    """A function that replaces this machine's paths and username in a string, the longest match first."""
    pairs = {}

    def add(path, plain, slugged=None):
        for form in {str(path), os.path.realpath(path)}:
            if len(form) > 1:
                pairs[form] = plain
                if slugged:
                    pairs[slug(form)] = slugged

    if base is not None:
        add(base, "/TMP/trial", "-TMP-trial")
    try:
        add(Path.home() if home is None else home, "/HOME", "-HOME")
    except RuntimeError:
        pass
    uid = os.getuid() if uid is None and hasattr(os, "getuid") else uid
    if uid is not None:
        add(f"/tmp/claude-{uid}", "/TMP/claude")
    if user is None:
        try:
            user = getpass.getuser()
        except Exception:  # noqa: BLE001 - getuser raises OSError, KeyError or ImportError by platform
            user = None
    found = [(len(n), re.escape(n)) for n in pairs]
    if user:
        found.append((len(user), rf"(?<!\w){re.escape(user)}(?!\w)"))
    if not found:
        return lambda text: text
    pattern = re.compile("|".join(rx for _, rx in sorted(found, key=lambda f: -f[0])))
    return lambda text: pattern.sub(lambda m: pairs.get(m.group(0), "user"), text)


def scrubber(base=None, env=None, **runtime):
    rewrite = rewriter(base, **runtime)
    return lambda text: rewrite(redact(text, env))


def scrub_tree(value, scrub, keep=()):
    """Apply `scrub` to every string in a JSON-like value; `keep` names top-level keys to leave alone."""
    if isinstance(value, str):
        return scrub(value)
    if isinstance(value, dict):
        return {k: v if k in keep else scrub_tree(v, scrub) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub_tree(v, scrub) for v in value]
    return value


def stderr_tail(stderr: str, env=None, base=None) -> str:
    # Scrub before cutting, so the cut cannot split a secret or a path into an unrecognisable tail.
    return scrubber(base, env)(stderr)[-STDERR_TAIL_CHARS:]


def skill_sha256(skill_dir) -> str:
    root = Path(skill_dir)
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def status_inputs(*, condition, exit_code=None, signal=None, timed_out=False, harness_exception=None,
                  stderr_tail="", stream_parsed=False, result_event=None, cli_tool_results=(),
                  permission_denials=(), bash_commands=(), read_paths=(), skill_calls=(),
                  listed_skills=None, redaction_conflict=None, expected_replay_file=None) -> dict:
    return {
        "exit_code": exit_code,
        "signal": signal,
        "timed_out": timed_out,
        "harness_exception": harness_exception,
        "stderr_tail": stderr_tail,
        "stream_parsed": stream_parsed,
        "result_event": result_event,
        "cli_tool_results": list(cli_tool_results),
        "permission_denials": list(permission_denials),
        "bash_commands": list(bash_commands),
        "read_paths": list(read_paths),
        "skill_calls": list(skill_calls),
        "listed_skills": None if listed_skills is None else list(listed_skills),
        "condition": condition,
        "redaction_conflict": redaction_conflict,
        "expected_replay_file": expected_replay_file,
    }


def build(*, identity: dict, subject: dict, condition: dict, output: dict, usage: dict,
          inputs: dict, attempt: int = 1, base=None, env=None) -> dict:
    scrub = scrubber(base, env)
    # Exempt: the identity fields, `subject`, the skill hash, `listed_skills` and the condition label, which are harness-supplied.
    stored = scrub_tree(inputs, scrub, keep=("listed_skills", "condition"))
    raw, redacted = classify_status(inputs), classify_status(stored)
    if (raw.status, raw.rule) != (redacted.status, redacted.rule):
        # Only the verdict is recorded, never text from the inputs, so the conflict itself cannot leak.
        stored["redaction_conflict"] = {"status": raw.status, "rule": raw.rule, "redacted_rule": redacted.rule,
                                        "tags": list(raw.tags)}
    status = classify_status(stored)
    return {
        "record_version": RECORD_VERSION,
        **identity,
        "subject": subject,
        "condition": scrub_tree(condition, scrub, keep=("skill_sha256", "condition")),
        "output": scrub_tree(output, scrub),
        "usage": scrub_tree(usage, scrub),
        "status_inputs": stored,
        "status": status.status,
        "status_rule": status.rule,
        "status_reason": status.reason,
        "tags": list(status.tags),
        "attempt": attempt,
        "spans": [],
    }


def write(record: dict, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
