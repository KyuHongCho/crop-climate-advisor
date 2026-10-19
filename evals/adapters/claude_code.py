"""Claude Code as an eval subject: an isolated workspace, the pinned argv, and the stream-json run."""
import json
import os
import shutil
import signal as signals
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from crop_advisor.climate import REPLAY_ENV, replay_path

from .. import record
from ..status import CLI_CALL, DENIED_SKILLS, SKILL_NAME

ADAPTER = "claude_code"
ROOT = Path(__file__).resolve().parents[2]
SKILL_DIR = ROOT / ".agents" / "skills" / SKILL_NAME
CONDITIONS = ("skill", "no-skill")
SKILL_TOOLS = "Bash,Read,Skill"
SKILL_ALLOWED = ("Bash(python -m crop_advisor.cli:*)", "Bash(python3 -m crop_advisor.cli:*)")
BUNDLED_SKILLS_ENV = "CLAUDE_CODE_DISABLE_BUNDLED_SKILLS"
# Per https://code.claude.com/docs/en/permissions a Skill(skill:<name>) deny "matches the skill under any of its names".
SKILL_DENIED = tuple(f"Skill(skill:{name})" for name in DENIED_SKILLS)
MEMORY_FILES = ("CLAUDE.md", "CLAUDE.local.md")
_KEEP_CLAUDE_VARS = {record.TOKEN_ENV, "CLAUDE_CONFIG_DIR"}


def argv(prompt, model, condition, budget_usd, binary="claude"):
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, not {condition!r}")
    # The prompt goes straight after -p: --allowedTools is variadic and would swallow a trailing positional.
    args = [binary, "-p", prompt, "--model", model,
            "--setting-sources", "project", "--strict-mcp-config",
            # Without it this account's headless default is "auto", which ran non-allowlisted Bash undenied.
            "--permission-mode", "default",
            "--max-budget-usd", f"{budget_usd:.2f}",
            "--no-session-persistence",
            "--output-format", "stream-json", "--verbose"]
    if condition == "skill":
        return args + ["--tools", SKILL_TOOLS, "--allowedTools", *SKILL_ALLOWED, "--disallowedTools", *SKILL_DENIED]
    return args + ["--tools", ""]


def build_workspace(base, condition, root=ROOT):
    """Return (workspace, replay_dir). The recordings sit beside the workspace, not in it."""
    base = Path(base)
    workspace, replay = base / "workspace", base / "replay"
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(root / "crop_advisor", workspace / "crop_advisor", ignore=ignore)
    shutil.copytree(root / "data", workspace / "data", ignore=ignore)
    shutil.copytree(root / "evals" / "fixtures" / "nasa_power", replay, ignore=shutil.ignore_patterns("*.md"))
    if condition == "skill":
        shutil.copytree(root / ".agents" / "skills" / SKILL_NAME, workspace / ".agents" / "skills" / SKILL_NAME)
        (workspace / ".claude" / "skills").mkdir(parents=True)
        (workspace / ".claude" / "skills" / SKILL_NAME).symlink_to(Path("..", "..", ".agents", "skills", SKILL_NAME))
    return workspace, replay


def _rules(folder):
    return sorted(p for p in folder.rglob("*.md") if p.is_file()) if folder.is_dir() else []


def memory_leaks(workspace, env):
    """The detect layer behind CLAUDE_CODE_DISABLE_CLAUDE_MDS, whose effect on rules/*.md is untested."""
    config_dir = Path(env.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")
    found = []
    for folder in Path(workspace).resolve().parents:
        found += [folder / n for n in (*MEMORY_FILES, ".claude/CLAUDE.md") if (folder / n).is_file()]
        found += _rules(folder / ".claude" / "rules")
    found += [config_dir / n for n in MEMORY_FILES if (config_dir / n).is_file()]
    return found + _rules(config_dir / "rules")


def environment(replay_dir, base_env=None, python_bin=None):
    base_env = os.environ if base_env is None else base_env
    # CLAUDE* vars set by a parent Claude Code session would reach the subject; CLAUDE_EFFORT changes its effort.
    env = {k: v for k, v in base_env.items()
           if k in _KEEP_CLAUDE_VARS or not (k.startswith("CLAUDE") or k == "AI_AGENT")}
    python_bin = Path(sys.executable).parent if python_bin is None else Path(python_bin)
    env["PATH"] = os.pathsep.join([str(python_bin), base_env.get("PATH", "")])
    env[REPLAY_ENV] = str(Path(replay_dir).resolve())
    env["DISABLE_AUTOUPDATER"] = "1"
    env[BUNDLED_SKILLS_ENV] = "1"
    env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] = "1"
    env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
    return env


def cli_version(binary, env, cwd) -> str:
    out = subprocess.run([binary, "--version"], capture_output=True, text=True, env=env, cwd=cwd,
                         stdin=subprocess.DEVNULL, timeout=60, check=True).stdout
    return out.split()[0]


def _drain(stream, sink, start):
    for line in stream:
        sink.append((round((time.monotonic() - start) * 1000), line))


def run(args, cwd, env, timeout):
    """Run to completion or timeout; every stdout line is kept with its arrival time in ms."""
    start = time.monotonic()
    proc = subprocess.Popen(args, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                            start_new_session=True)
    lines, errs = [], []
    readers = [threading.Thread(target=_drain, args=(proc.stdout, lines, start), daemon=True),
               threading.Thread(target=_drain, args=(proc.stderr, errs, start), daemon=True)]
    for t in readers:
        t.start()
    timed_out = False
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, signals.SIGKILL)
        proc.wait()
    for t in readers:
        t.join(timeout=10)
    proc.stdout.close()
    proc.stderr.close()
    rc = proc.returncode
    return {"lines": lines, "stderr": "".join(line for _, line in errs),
            "exit_code": rc if rc >= 0 else None, "signal": -rc if rc < 0 else None,
            "timed_out": timed_out, "latency_ms": round((time.monotonic() - start) * 1000)}


def _result_text(content):
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content or [] if isinstance(part, dict))


def parse(lines):
    """Read a stream-json capture: [(ms, line)] -> events, init, result event and tool calls."""
    parsed, init, result_event, calls = True, None, None, {}
    for ms, line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            parsed = False
            continue
        kind = event.get("type")
        if kind == "system" and event.get("subtype") == "init":
            init = event
        elif kind == "result":
            result_event = event
        elif kind in ("assistant", "user"):
            for part in (event.get("message") or {}).get("content") or []:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "tool_use":
                    calls[part.get("id")] = {"name": part.get("name"), "input": part.get("input"),
                                             "result": None, "is_error": None,
                                             "started_ms": ms, "ended_ms": None}
                elif part.get("type") == "tool_result" and part.get("tool_use_id") in calls:
                    call = calls[part["tool_use_id"]]
                    call.update(result=_result_text(part.get("content")), is_error=bool(part.get("is_error")),
                                ended_ms=ms)
    tool_calls = list(calls.values())
    cli = [c for c in tool_calls if is_cli_call(c)]
    return {"stream_parsed": parsed, "init": init, "result_event": result_event, "tool_calls": tool_calls,
            "cli_tool_results": [c["result"] or "" for c in cli]}


def _input_str(call, key) -> str:
    # A malformed tool input is kept as text, so the off-contract rules judge it instead of the harness crashing.
    value = call.get("input").get(key) if isinstance(call.get("input"), dict) else None
    return "" if value is None else value if isinstance(value, str) else str(value)


def _command(call) -> str:
    return _input_str(call, "command")


def is_cli_call(call) -> bool:
    return call.get("name") == "Bash" and bool(CLI_CALL.search(_command(call)))


def canonical_models(result_event):
    usage = (result_event or {}).get("modelUsage") or {}
    return sorted({(v or {}).get("canonicalModel") or k for k, v in usage.items()})


def _skill(call) -> str:
    return _input_str(call, "skill")


def _init_summary(init):
    if not init:
        return None
    keys = ("claude_code_version", "model", "permissionMode", "tools", "skills", "mcp_servers")
    return {k: init.get(k) for k in keys}


def read_paths(tool_calls, workspace):
    """Each Read's target, resolved while the workspace exists: workspace-relative inside it, absolute outside."""
    root = os.path.realpath(workspace) if workspace else None
    out = []
    for call in tool_calls:
        if call.get("name") != "Read":
            continue
        raw = _input_str(call, "file_path")
        if root is None:
            out.append(os.path.normpath(raw))
            continue
        resolved = os.path.realpath(os.path.join(root, raw))
        inside = resolved == root or resolved.startswith(root + os.sep)
        out.append(Path(os.path.relpath(resolved, root)).as_posix() if inside else resolved)
    return out


def to_record(*, case, eval_set, trial_id, model, condition, cli_version_, run_result, parsed,
              harness_exception=None, skill_sha=None, env=None, workspace=None, base=None) -> dict:
    run_result = run_result or {}
    event, init = parsed.get("result_event"), parsed.get("init")
    models = canonical_models(event)
    has_site = case.get("lat") is not None and case.get("lon") is not None
    inputs = record.status_inputs(
        condition=condition,
        exit_code=run_result.get("exit_code"),
        signal=run_result.get("signal"),
        timed_out=bool(run_result.get("timed_out")),
        harness_exception=harness_exception,
        stderr_tail=record.stderr_tail(run_result.get("stderr", ""), env, base),
        stream_parsed=parsed.get("stream_parsed", False),
        result_event=event,
        cli_tool_results=parsed.get("cli_tool_results", ()),
        permission_denials=(event or {}).get("permission_denials") or (),
        bash_commands=[_command(c) for c in parsed.get("tool_calls", []) if c.get("name") == "Bash"],
        skill_calls=[_skill(c) for c in parsed.get("tool_calls", []) if c.get("name") == "Skill"],
        read_paths=read_paths(parsed.get("tool_calls", []), workspace),
        listed_skills=list(init.get("skills") or []) if init else None,
        expected_replay_file=replay_path(".", case["lat"], case["lon"]).name if has_site else None,
    )
    return record.build(
        identity={"case_id": case["id"], "eval_set_version": eval_set.get("version"),
                  "eval_set_sha256": eval_set.get("sha256"), "trial_id": trial_id},
        subject={"adapter": ADAPTER, "cli_version": cli_version_, "model": model,
                 "canonical_model": models[0] if len(models) == 1 else None,
                 "canonical_models": models, "init": _init_summary(init)},
        condition={"condition": condition, "skill_sha256": skill_sha},
        output={"prompt": case["prompt"], "answer": (event or {}).get("result"),
                "tool_calls": parsed.get("tool_calls", [])},
        usage={"tokens": (event or {}).get("usage"), "cost_usd": (event or {}).get("total_cost_usd"),
               "latency_ms": run_result.get("latency_ms"), "duration_ms": (event or {}).get("duration_ms"),
               "turns": (event or {}).get("num_turns")},
        inputs=inputs, base=base, env=env,
    )


def run_trial(*, case, eval_set, model, condition, budget_usd, timeout, trial_id, binary="claude"):
    parsed, run_result, version, failure, workspace, skill_sha = {}, None, None, None, None, None
    with tempfile.TemporaryDirectory(prefix="crop-eval-") as base:
        env = os.environ
        try:
            skill_sha = record.skill_sha256(SKILL_DIR) if condition == "skill" else None
            workspace, replay = build_workspace(base, condition)
            env = environment(replay)
            leaks = memory_leaks(workspace, env)
            if leaks:
                raise RuntimeError(f"instruction files would leak into the subject: {', '.join(map(str, leaks))}")
            version = cli_version(binary, env, workspace)
            run_result = run(argv(case["prompt"], model, condition, budget_usd, binary), workspace, env, timeout)
            parsed = parse(run_result["lines"])
        except Exception as e:  # noqa: BLE001 - any harness fault becomes rule 1, never a lost trial
            failure = f"{type(e).__name__}: {e}"
        identity = dict(case=case, eval_set=eval_set, trial_id=trial_id, model=model, condition=condition,
                        cli_version_=version, skill_sha=skill_sha, env=env, base=base)
        try:
            return to_record(**identity, run_result=run_result, parsed=parsed, harness_exception=failure,
                             workspace=workspace)
        except Exception as e:  # noqa: BLE001 - same guarantee when building the record itself fails
            return to_record(**identity, run_result=None, parsed={},
                             harness_exception=f"record build failed: {type(e).__name__}: {e}")
