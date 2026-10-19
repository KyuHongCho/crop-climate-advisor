"""Stand-ins for the `claude` binary, so no test ever reaches the real one."""
import shlex
import stat
import sys
from pathlib import Path

SAMPLES_DIR = Path(__file__).resolve().parent / "fixtures" / "claude_code"


def sample_lines(name):
    return [(i * 10, line) for i, line in
            enumerate((SAMPLES_DIR / name).read_text(encoding="utf-8").splitlines(keepends=True))]


def fake_claude(directory, stream=None, exit_code=0, sleep=0, env_dump=None, emit=None, script=None):
    """A stand-in `claude`: answers --version, otherwise prints a stored stream."""
    path = Path(directory) / "claude"
    body = ["#!/bin/sh", 'if [ "$1" = "--version" ]; then echo "9.9.9 (Claude Code)"; exit 0; fi']
    if env_dump:
        body.append(f'env > "{env_dump}"; pwd >> "{env_dump}"; ls -a >> "{env_dump}"')
    if sleep:
        body.append(f"sleep {sleep}")
    if stream:
        body.append(f'cat "{SAMPLES_DIR / stream}"')
    if emit:
        body.append(f'"{sys.executable}" -c \'{emit}\'')
    if script:
        body.append(f'"{sys.executable}" "{script}"')
    body.append(f"exit {exit_code}")
    path.write_text("\n".join(body) + "\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def tripwire(directory):
    """Write a `claude` into `directory` that logs its argv and exits 1; the returned log exists once it has run."""
    directory = Path(directory)
    log = directory / "claude.log"
    path = directory / "claude"
    path.write_text(f'#!/bin/sh\necho "$@" >> {shlex.quote(str(log))}\nexit 1\n')
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return log
