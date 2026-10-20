"""Stand-ins for the `claude` binary, so no test ever reaches the real one.

    python -m tests.fake_claude <dir>    writes an `echo_model` stand-in named `claude` into <dir>
"""
import argparse
import shlex
import stat
import sys
from pathlib import Path

SAMPLES_DIR = Path(__file__).resolve().parent / "fixtures" / "claude_code"
STORED_MODEL = "claude-haiku-5-5"
ECHO_STREAMS = {"skill": "skill_ok.jsonl", "no-skill": "no_skill_ok.jsonl"}


def sample_lines(name):
    return [(i * 10, line) for i, line in
            enumerate((SAMPLES_DIR / name).read_text(encoding="utf-8").splitlines(keepends=True))]


def fake_claude(directory, stream=None, exit_code=0, sleep=0, env_dump=None, emit=None, script=None,
                echo_model=False, streams=None):
    """A stand-in `claude`: answers --version, otherwise prints a stored stream.

    With `echo_model` it prints the stream `streams` (default ECHO_STREAMS) names for the condition's --tools value,
    with the stored model id rewritten to --model, so record-time R3 passes for any configured subject.
    """
    path = Path(directory) / "claude"
    body = ["#!/bin/sh", 'if [ "$1" = "--version" ]; then echo "9.9.9 (Claude Code)"; exit 0; fi']
    if env_dump:
        body.append(f'env > "{env_dump}"; pwd >> "{env_dump}"; ls -a >> "{env_dump}"')
    if sleep:
        body.append(f"sleep {sleep}")
    if stream:
        body.append(f'cat "{SAMPLES_DIR / stream}"')
    if echo_model:
        picks = {**ECHO_STREAMS, **(streams or {})}
        body.append(f'"{sys.executable}" -I "{Path(__file__).resolve()}" --serve "{picks["skill"]}" '
                    f'"{picks["no-skill"]}" "$@"')
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


def _flag_value(argv, flag):
    return argv[argv.index(flag) + 1] if flag in argv[:-1] else None


def serve(skill_stream, no_skill_stream, argv):
    """The body of an `echo_model` stand-in: print the condition's stored stream under the model asked for."""
    stream = skill_stream if _flag_value(argv, "--tools") else no_skill_stream
    text = (SAMPLES_DIR / stream).read_text(encoding="utf-8")
    sys.stdout.write(text.replace(STORED_MODEL, _flag_value(argv, "--model") or STORED_MODEL))
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--serve"]:
        return serve(argv[1], argv[2], argv[3:])
    ap = argparse.ArgumentParser(description="Write an echo_model stand-in `claude` into a directory.")
    ap.add_argument("directory")
    args = ap.parse_args(argv)
    Path(args.directory).mkdir(parents=True, exist_ok=True)
    print(fake_claude(args.directory, echo_model=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
