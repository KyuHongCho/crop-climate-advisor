"""The trial record every adapter writes, and the redaction applied to anything copied from a subprocess."""
import hashlib
import json
import os
import re
from pathlib import Path

from .status import classify_status

RECORD_VERSION = 1
STDERR_TAIL_CHARS = 4096
REDACTED = "[REDACTED]"
_SECRET_PATTERNS = (re.compile(r"sk-[A-Za-z0-9_\-]+"), re.compile(r"Bearer\s+\S+"))
TOKEN_ENV = "CLAUDE_CODE_OAUTH_TOKEN"


def redact(text: str, env=None) -> str:
    env = os.environ if env is None else env
    token = env.get(TOKEN_ENV)
    if token:
        text = text.replace(token, REDACTED)
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


def stderr_tail(stderr: str, env=None) -> str:
    # Redact before cutting, so the cut cannot split a secret into an unrecognisable tail.
    return redact(stderr, env)[-STDERR_TAIL_CHARS:]


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
                  listed_skills=None) -> dict:
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
    }


def build(*, identity: dict, subject: dict, condition: dict, output: dict, usage: dict,
          inputs: dict, attempt: int = 1) -> dict:
    status = classify_status(inputs)
    return {
        "record_version": RECORD_VERSION,
        **identity,
        "subject": subject,
        "condition": condition,
        "output": output,
        "usage": usage,
        "status_inputs": inputs,
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
