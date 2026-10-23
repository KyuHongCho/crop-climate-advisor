"""The run loop, offline: start-up order, projection, caps, the stop policy and the snapshot.

Loop tests replace `claude_code.run_trial` with a scripted function; the few that run the real one use a stand-in
`claude` first on PATH. A tripwire `claude` sits behind both, and the module fails if it was ever started.
"""
import contextlib
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from collections import namedtuple
from pathlib import Path
from unittest import mock

from evals import cases, config, record, record_fixtures, report, run, snapshot, stats, trial
from evals.adapters import claude_code
from tests.fake_claude import ECHO_STREAMS, SAMPLES_DIR, fake_claude, sample_lines, tripwire
from tests.test_eval_imports import _import_all

ROOT = Path(__file__).resolve().parent.parent
REAL_RESULTS_DIR = run.DEFAULT_RESULTS_DIR
SONNET, HAIKU = "claude-sonnet-5-5", "claude-haiku-5-5"
CONDITIONS = ("skill", "no-skill")
EVAL_SET = cases.load()
CASE_IDS = [c["id"] for c in EVAL_SET["cases"]]
SAMPLES = {s["name"]: s for s in json.loads((SAMPLES_DIR / "samples.json").read_text(encoding="utf-8"))["samples"]}
DEADLINE = 240
PID_WAIT, EXIT_WAIT, GONE_WAIT = 60, 20, 5
_state = {}


def setUpModule():
    tmp = tempfile.TemporaryDirectory()
    unittest.addModuleCleanup(tmp.cleanup)
    base = Path(tmp.name)
    for name in ("trip", "cfg", "results"):
        (base / name).mkdir()
    _state["log"] = tripwire(base / "trip")
    patcher = mock.patch.dict(os.environ, {
        "USER": "testuser", "LOGNAME": "testuser", "USERNAME": "testuser", "CLAUDE_CONFIG_DIR": str(base / "cfg"),
        "PATH": str(base / "trip") + os.pathsep + os.environ.get("PATH", "")})
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)
    results = mock.patch.object(run, "DEFAULT_RESULTS_DIR", base / "results")
    results.start()
    unittest.addModuleCleanup(results.stop)


def tearDownModule():
    if _state["log"].exists():
        raise AssertionError(f"a test reached `claude`: {_state['log'].read_text()}")


def alive(pid):
    """True while the process exists and is not a zombie, so no test depends on an init process reaping orphans."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, timeout=30).stdout
    return bool(state.strip()) and not state.strip().startswith("Z")


def group_members(pgid):
    found = subprocess.run(["pgrep", "-g", str(pgid)], capture_output=True, text=True, timeout=30).stdout.split()
    return [int(pid) for pid in found if alive(int(pid))]


def wait_until(condition, seconds, what):
    end = time.monotonic() + seconds
    while not condition():
        if time.monotonic() > end:
            raise AssertionError(f"{what} did not happen within {seconds}s")
        time.sleep(0.05)


def wait_pid(pid_file, seconds=PID_WAIT):
    def written():
        return Path(pid_file).is_file() and Path(pid_file).read_text().strip().isdigit()
    wait_until(written, seconds, f"a pid in {pid_file}")
    return int(Path(pid_file).read_text().strip())


def reap_group(pid, pgid):
    """SIGKILL the group `pid` was seen in, but only while `pid` is still in it, and never our own group."""
    if pgid <= 1 or pgid == os.getpgrp():
        return
    try:
        if os.getpgid(pid) == pgid:
            os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


class Tracking:
    def track(self, pid_file):
        """Wait for a stand-in's sleeper pid, arrange to reap its group, and return (pid, pgid)."""
        pid = wait_pid(pid_file)
        pgid = os.getpgid(pid)
        self.assertGreater(pgid, 1)
        self.assertNotEqual(pgid, os.getpgrp())
        self.addCleanup(reap_group, pid, pgid)
        return pid, pgid

    def assertGone(self, pid, pgid):
        wait_until(lambda: not alive(pid) and not group_members(pgid), GONE_WAIT,
                   f"process {pid} and group {pgid} gone")


# One name per scripted outcome, then the stored sample that produces it for each condition.
OUTCOMES = {
    "ok": {"skill": "skill_ok", "no-skill": "no_skill_ok"},
    "infra": {"skill": "nasa_network_failure", "no-skill": "api_429_rate_limit"},
    "sys": {"skill": "skill_not_offered", "no-skill": "api_404_bad_model"},
    "runaway": {"no-skill": "runaway_budget"},
    "cv": {"skill": "skill_no_cli_call"},
    "harness": {"no-skill": "error_during_execution"},
    "loggedout": {"no-skill": "not_logged_in"},
    "replay": {"skill": "replay_missing_model_answers"},
}


def sample_record(kw, sample, cost=0.05):
    """A record built by claude_code.to_record from a stored sample, under the slot `run_trial` was called for."""
    s = SAMPLES[sample]
    model, condition = kw["model"], kw["condition"]
    assert s["condition"] == condition, f"{sample} is a {s['condition']} sample"
    lines = ([(ms, line.replace("claude-haiku-5-5", model)) for ms, line in sample_lines(s["stream"])]
             if s["stream"] else [])
    rec = claude_code.to_record(
        case=kw["case"], eval_set=kw["eval_set"], trial_id=kw["trial_id"], model=model, condition=condition,
        cli_version_="9.9.9",
        run_result={"stderr": s["stderr"], "exit_code": s["exit_code"], "signal": s["signal"],
                    "timed_out": s["timed_out"], "latency_ms": 1500},
        parsed=claude_code.parse(lines) if lines else {}, harness_exception=s["harness_exception"],
        skill_sha=record.skill_sha256(claude_code.SKILL_DIR), env={})
    rec["usage"]["cost_usd"] = cost
    return rec


def outcome_record(kw, name, cost):
    if name == "excl":
        # Synthesised: a harness-error no systemic rule covers (rule 8), for a condition with no stored sample.
        rec = sample_record(kw, OUTCOMES["ok"][kw["condition"]], cost)
        rec.update(status="harness-error", status_rule="8", status_reason="synthesised: exit 1 matched no rule",
                   tags=["harness-error"])
        return rec
    return sample_record(kw, OUTCOMES[name][kw["condition"]], cost)


class Script:
    """Stands in for claude_code.run_trial: `plan` maps (rep, case, model, condition) to one outcome per attempt."""

    def __init__(self, plan=None, cost=0.05, error_at=None, tweak=None, default="ok", error=RuntimeError):
        self.plan, self.cost, self.error_at, self.tweak, self.default = plan or {}, cost, error_at, tweak, default
        self.error = error
        self.calls = []

    def __call__(self, **kw):
        i = len(self.calls)
        self.calls.append(kw)
        if i == self.error_at:
            raise self.error("boom")
        rep, attempt = map(int, re.search(r"-r(\d+)-a(\d)-", kw["trial_id"]).groups())
        outcomes = self.plan.get((rep, kw["case"]["id"], kw["model"], kw["condition"]), (self.default,))
        cost = self.cost(i, kw) if callable(self.cost) else self.cost
        rec = outcome_record(kw, outcomes[min(attempt, len(outcomes)) - 1], cost)
        if self.tweak:
            self.tweak(i, kw, rec)
        return rec


def forbidden(**kw):
    raise AssertionError(f"run_trial was reached: {kw.get('trial_id')}")


def refuse_spawn(*args, **kwargs):
    raise AssertionError(f"a process was started: {args!r}")


def slot_of(rec):
    return rec["rep"], rec["case_id"], rec["subject"]["model"], rec["condition"]["condition"]


Result = namedtuple("Result", "code out err run_dir")


class RunCase(Tracking, unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.tmp = Path(d.name)
        self.results = self.tmp / "results"

    def config_path(self, **over):
        cfg = json.loads(config.DEFAULT_PATH.read_text(encoding="utf-8"))
        cfg.update(bootstrap_resamples=200, bootstrap_seed=7, trial_timeout_s=60)
        cfg.update(over)
        path = self.tmp / f"config{len(list(self.tmp.glob('config*.json')))}.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        return str(path)

    def go(self, *args, script=forbidden, real=False, over=None, results=True):
        """Run `evals.run.main` in-process; with `real` the real run_trial runs, else `script` stands in for it."""
        argv = list(args)
        if "--config" not in argv:
            argv += ["--config", self.config_path(**(over or {}))]
        if results and "--results-dir" not in argv:
            argv += ["--results-dir", str(self.results)]
        out, err, code = io.StringIO(), io.StringIO(), 0
        self.last_err = err
        with contextlib.ExitStack() as stack:
            if not real:
                stack.enter_context(mock.patch.object(claude_code, "run_trial", script))
                stack.enter_context(mock.patch.object(subprocess, "Popen", side_effect=refuse_spawn))
                stack.enter_context(mock.patch.object(subprocess, "run", side_effect=refuse_spawn))
            stack.enter_context(contextlib.redirect_stdout(out))
            stack.enter_context(contextlib.redirect_stderr(err))
            try:
                code = run.main(argv)
            except SystemExit as e:
                code = e.code
        dirs = sorted(self.results.glob("*")) if self.results.is_dir() else []
        return Result(code, out.getvalue(), err.getvalue(), dirs[-1] if dirs else None)

    def snap(self, res):
        return json.loads((res.run_dir / "snapshot.json").read_text(encoding="utf-8"))

    def records(self, res):
        return report.load_records(res.run_dir)

    def stand_in(self, name="bin", **kw):
        d = self.tmp / name
        d.mkdir()
        fake_claude(d, echo_model=True, **kw)
        return d

    def on_path(self, directory):
        return mock.patch.dict(os.environ, {"PATH": f"{directory}{os.pathsep}{os.environ['PATH']}"})

    def assertNothingRan(self, res):
        self.assertEqual(res.code, 2, res.err)
        self.assertFalse(self.results.exists(), "a results directory was created")


class GridTest(RunCase):
    def test_grid_size_rep_major_order_record_fields_and_the_unscrubbed_binary(self):
        home = self.tmp / "home"
        (home / "bin").mkdir(parents=True)
        stand_in = fake_claude(home / "bin")
        script = Script()
        with mock.patch.dict(os.environ, {"HOME": str(home),
                                          "PATH": f"{home / 'bin'}{os.pathsep}{os.environ['PATH']}"}):
            expected_binary = os.path.abspath(stand_in)
            scrubbed = record.scrubber()(expected_binary)
            self.assertNotEqual(scrubbed, expected_binary)
            res = self.go("--n", "2", "--approve-spend", "2", script=script,
                          over={"trial_budget_usd": 0.25, "trial_timeout_s": 45})
        self.assertEqual(res.code, 0, res.err)
        self.assertEqual({c["binary"] for c in script.calls}, {expected_binary})
        self.assertEqual({c["budget_usd"] for c in script.calls}, {0.25})
        self.assertEqual({c["timeout"] for c in script.calls}, {45})
        settings = self.snap(res)["settings"]
        self.assertEqual((settings["trial_budget_usd"], settings["trial_timeout_s"]), (0.25, 45))
        stored = self.snap(res)["claude_path"]
        self.assertTrue(stored.startswith("/HOME/"), stored)
        self.assertEqual(stored, scrubbed)
        recs = self.records(res)
        expected = [(rep, case, model, cond) for rep in (1, 2) for case in CASE_IDS
                    for model in (SONNET, HAIKU) for cond in CONDITIONS]
        self.assertEqual(len(expected), 32)
        self.assertEqual([slot_of(r) for r in recs], expected)
        self.assertEqual([r["rep"] for r in recs], [1] * 16 + [2] * 16)
        for r in recs:
            case, cond, model = r["case_id"], r["condition"]["condition"], r["subject"]["model"]
            self.assertRegex(r["trial_id"], rf"^{case}-{cond}-{model}-r{r['rep']}-a1-[0-9a-f]{{8}}$")
            self.assertEqual((r["run_id"], r["attempt"]), (res.run_dir.name, 1))
        self.assertRegex(res.run_dir.name, r"^\d{8}T\d{6}Z-[0-9a-f]{8}$")
        self.assertEqual(len(re.findall(r"^\[\d+\] ", res.err, re.MULTILINE)), 32)


class LoginTest(RunCase):
    ENTRIES = {"basil": "basil", "skill": "skill", "Chang": "Chang", "crop": "--crop", "m": "-m", "to": "to",
               "light": "daily light integral"}

    def test_a_login_the_graders_read_is_refused_before_anything_starts(self):
        for login, entry in self.ENTRIES.items():
            with self.subTest(login=login), mock.patch.dict(
                    os.environ, {"USER": login, "LOGNAME": login, "USERNAME": login}):
                res = self.go("--n", "1", "--approve-spend", "1")
                self.assertNothingRan(res)
                self.assertIn(f"'{entry}'", res.err)
                self.assertIn("LOGNAME=eval USER=eval", res.err)

    def test_other_logins_proceed(self):
        for login in ("testuser", "eval"):
            with self.subTest(login=login), mock.patch.dict(
                    os.environ, {"USER": login, "LOGNAME": login, "USERNAME": login}):
                shutil.rmtree(self.results, ignore_errors=True)
                res = self.go("--n", "1", "--approve-spend", "1", script=Script())
                self.assertEqual(res.code, 0, res.err)


class PreflightTest(RunCase):
    def test_a_missing_fixture_skill_folder_or_claude_exits_2_and_names_it(self):
        empty = self.tmp / "empty"
        empty.mkdir()
        cases_ = {
            "fixture": (mock.patch.object(run, "FIXTURE_DIR", empty), "51.5_-0.13.json"),
            "skill folder": (mock.patch.object(claude_code, "SKILL_DIR", self.tmp / "no-skill-here"), "no-skill-here"),
            "claude": (mock.patch.dict(os.environ, {"PATH": str(empty)}), "`claude` on PATH"),
        }
        for name, (patch, named) in cases_.items():
            with self.subTest(name), patch:
                res = self.go("--n", "1", "--approve-spend", "1")
                self.assertNothingRan(res)
                self.assertIn(named, res.err)


class ApprovalTest(RunCase):
    def test_without_approve_spend_it_prints_the_projection_and_worst_case_and_stops(self):
        res = self.go("--n", "2")
        self.assertNothingRan(res)
        self.assertIn("Projected spend for 32 planned trials: $0.56 (source: defaults)", res.out)
        self.assertIn("32 x $0.50 = $16.00", res.out)

    def test_a_projection_over_the_cap_is_refused(self):
        res = self.go("--n", "3", "--approve-spend", "0.8")
        self.assertNothingRan(res)
        self.assertIn("exceeds the cap $0.80", res.err)

    def test_approve_spend_lowers_the_cap_and_never_raises_it(self):
        self.assertNothingRan(self.go("--n", "2", "--approve-spend", "0.5"))
        res = self.go("--n", "2", "--approve-spend", "5", over={"run_budget_usd": 0.5})
        self.assertNothingRan(res)
        self.assertIn("exceeds the cap $0.50", res.err)
        res = self.go("--n", "3", "--approve-spend", "3", script=Script())
        self.assertEqual((res.code, self.snap(res)["settings"]["effective_cap_usd"]), (0, 3.0))
        shutil.rmtree(self.results)
        res = self.go("--n", "1", "--approve-spend", "5", over={"run_budget_usd": 1}, script=Script())
        self.assertEqual((res.code, self.snap(res)["settings"]["effective_cap_usd"]), (0, 1))

    def test_a_projection_equal_to_the_cap_is_allowed(self):
        res = self.go("--n", "2", "--approve-spend", "0.56", script=Script(cost=0.001))
        self.assertEqual((res.code, self.snap(res)["status"]), (0, "complete"))

    def test_bad_arguments_exit_2_with_nothing_run(self):
        for flag, value in [("--n", "0"), ("--n", "-1"), ("--n", "1.5"), ("--n", "abc"), ("--approve-spend", "0"),
                            ("--approve-spend", "-1"), ("--approve-spend", "nan"), ("--approve-spend", "inf"),
                            ("--approve-spend", "-inf"), ("--approve-spend", "abc")]:
            args = ("--n", "1", flag, value) if flag != "--n" else ("--approve-spend", "1", flag, value)
            with self.subTest(flag=flag, value=value):
                self.assertNothingRan(self.go(*args))


class ProjectionSourceTest(RunCase):
    KEYS = [(m, c) for m in (SONNET, HAIKU) for c in CONDITIONS]

    def scrubbed_claude(self):
        return record.scrubber()(os.path.abspath(shutil.which("claude")))

    def write_snap(self, run_id, *, status="complete", overrides=None, path="default", priced=10, mean=0.1, raw=None,
                   models=(SONNET, HAIKU), cost_by_key=None):
        d = self.results / run_id
        d.mkdir(parents=True)
        cost = {m: {c: {"sum": mean * priced, "priced": priced, "mean": mean} for c in CONDITIONS} for m in models}
        if cost_by_key is not None:
            cost = cost_by_key
        snap = {"status": status, "overrides": {} if overrides is None else overrides, "approved_spend_usd": 2,
                "claude_path": self.scrubbed_claude() if path == "default" else path, "cost_by_key": cost}
        (d / "snapshot.json").write_text(raw if raw is not None else json.dumps(snap), encoding="utf-8")

    def source(self):
        res = self.go("--n", "2")
        self.assertEqual(res.code, 2, res.err)
        total, source = re.search(r"\$([\d.]+) \(source: ([^)]+)\)", res.out).groups()
        return total, source

    def test_snapshots_that_do_not_qualify_are_ignored(self):
        self.write_snap("20261001T000000Z-00000001", status="partial")
        self.write_snap("20261001T000000Z-00000002", overrides={"n": 3})
        self.write_snap("20261001T000000Z-00000003", priced=9)
        self.write_snap("20261001T000000Z-00000004", raw="{not json")
        self.write_snap("20261001T000000Z-00000005", raw="[1, 2]")
        self.write_snap("20261001T000000Z-00000006", path="/HOME/other/claude")
        self.write_snap("20261001T000000Z-00000007", path=None)
        self.write_snap("20261001T000000Z-00000008", mean=0)
        (self.results / "20261001T000000Z-00000009").mkdir()
        self.write_snap("20261001T000000Z-0000000a", cost_by_key="x")
        self.write_snap("20261001T000000Z-0000000b", cost_by_key={SONNET: "x", HAIKU: {"skill": "x", "no-skill": 3}})
        self.write_snap("20261001T000000Z-0000000c", cost_by_key={m: {c: {"priced": "10", "mean": 0.1}
                                                                       for c in CONDITIONS} for m in (SONNET, HAIKU)})
        self.write_snap("zzz-not-a-run-id")
        self.assertEqual(self.source(), ("0.56", "defaults"))

    def test_the_newest_qualifying_snapshot_wins_and_an_approved_one_without_n_qualifies(self):
        self.write_snap("20261001T000000Z-00000001", mean=0.2)
        self.write_snap("20261002T000000Z-00000002", mean=0.1)
        self.write_snap("20261003T000000Z-00000003", status="mixed-version", mean=0.9)
        self.assertEqual(self.source(), ("3.20", "20261002T000000Z-00000002"))

    def test_keys_priced_by_different_snapshots_give_a_mixed_source(self):
        self.write_snap("20261001T000000Z-00000001", mean=0.2)
        self.write_snap("20261002T000000Z-00000002", mean=0.1, models=(SONNET,))
        total, source = self.source()
        self.assertEqual((total, source), ("4.80", "mixed"))

    def test_a_results_dir_that_does_not_exist_yet_projects_from_the_defaults(self):
        self.assertEqual(self.source(), ("0.56", "defaults"))
        self.assertFalse(self.results.exists())

    def test_a_full_run_is_used_only_by_a_run_that_sees_the_same_claude(self):
        first = self.go("--approve-spend", "8", script=Script())
        self.assertEqual((first.code, self.snap(first)["overrides"]), (0, {}))
        self.assertEqual(self.source(), ("1.60", first.run_dir.name))
        other = self.tmp / "other"
        other.mkdir()
        fake_claude(other)
        with self.on_path(other):
            self.assertEqual(self.source(), ("0.56", "defaults"))


class CapTest(RunCase):
    CHEAP = {m: {c: 0.001 for c in CONDITIONS} for m in (SONNET, HAIKU)}

    def go_cheap(self, *args, **kw):
        return self.go(*args, over={"cost_defaults_usd": self.CHEAP}, **kw)

    def test_the_cap_stops_the_run_after_at_most_one_trial_of_overshoot(self):
        script = Script(cost=0.30)
        res = self.go_cheap("--n", "2", "--approve-spend", "1", script=script)
        snap = self.snap(res)
        self.assertEqual((res.code, snap["status"], snap["stop_reason"]), (1, "partial", "spend-cap"))
        self.assertEqual((len(script.calls), len(self.records(res))), (4, 4))
        self.assertAlmostEqual(snap["spend_usd"], 1.2)
        self.assertLess(snap["spend_usd"], 1 + 0.30)

    def test_an_unpriced_attempt_is_charged_at_the_trial_cap(self):
        script = Script(cost=None)
        res = self.go_cheap("--n", "2", "--approve-spend", "0.5", script=script)
        snap = self.snap(res)
        self.assertEqual((res.code, snap["stop_reason"], len(script.calls)), (1, "spend-cap", 1))
        self.assertEqual((snap["spend_usd"], snap["unpriced"]), (0.0, 1))

    def test_the_cap_is_checked_before_attempt_2(self):
        script = Script({(1, "london", SONNET, "skill"): ("infra", "ok")}, cost=lambda i, kw: 1.2 if i == 0 else 0.01)
        res = self.go_cheap("--n", "1", "--approve-spend", "1", script=script)
        snap = self.snap(res)
        self.assertEqual((res.code, snap["stop_reason"], len(script.calls)), (1, "spend-cap", 1))
        self.assertEqual(self.records(res)[0]["status"], "infra-error")

    def test_an_overshoot_on_the_final_slot_leaves_the_grid_complete(self):
        script = Script(cost=lambda i, kw: 5.0 if i == 15 else 0.01)
        res = self.go_cheap("--n", "1", "--approve-spend", "1", script=script)
        snap = self.snap(res)
        self.assertEqual((res.code, snap["status"], snap["stop_reason"], len(script.calls)), (0, "complete", None, 16))
        self.assertGreater(snap["spend_usd"], 1)

    def test_a_cap_reached_after_the_last_slots_infra_attempt_1_is_partial(self):
        script = Script({(1, "djibouti", HAIKU, "no-skill"): ("infra", "ok")},
                        cost=lambda i, kw: 5.0 if i == 15 else 0.01)
        res = self.go_cheap("--n", "1", "--approve-spend", "1", script=script)
        snap = self.snap(res)
        self.assertEqual((res.code, snap["status"], snap["stop_reason"], len(script.calls)),
                         (1, "partial", "spend-cap", 16))
        self.assertEqual((self.records(res)[-1]["status"], self.records(res)[-1]["attempt"]), ("infra-error", 1))


class RerunTest(RunCase):
    def test_only_an_infra_error_is_run_again_and_only_once(self):
        plan = {(1, "london", SONNET, "skill"): ("infra", "ok"),
                (1, "london", SONNET, "no-skill"): ("infra", "infra"),
                (1, "lagos", SONNET, "no-skill"): ("harness",),
                (1, "lagos", SONNET, "skill"): ("cv",),
                (1, "bangkok", SONNET, "no-skill"): ("runaway",)}
        script = Script(plan)
        res = self.go("--n", "1", "--approve-spend", "1", script=script)
        self.assertEqual((res.code, self.snap(res)["status"]), (0, "complete"))
        by_slot = {}
        for r in self.records(res):
            by_slot.setdefault(slot_of(r), []).append((r["attempt"], r["status"]))
        self.assertEqual(by_slot[(1, "london", SONNET, "skill")], [(1, "infra-error"), (2, "ok")])
        self.assertEqual(by_slot[(1, "london", SONNET, "no-skill")], [(1, "infra-error"), (2, "infra-error")])
        self.assertEqual(by_slot[(1, "lagos", SONNET, "no-skill")], [(1, "harness-error")])
        self.assertEqual(by_slot[(1, "lagos", SONNET, "skill")], [(1, "contract-violation")])
        self.assertEqual(by_slot[(1, "bangkok", SONNET, "no-skill")], [(1, "runaway")])
        self.assertEqual(len(script.calls), 16 + 2)
        counts = self.snap(res)["counts"]
        self.assertEqual((counts["reruns"], counts["slots_started"], counts["attempts"]), (2, 16, 18))
        cost = self.snap(res)["cost_by_key"][SONNET]
        self.assertEqual((cost["skill"]["priced"], cost["no-skill"]["priced"]), (5, 5))
        self.assertAlmostEqual(cost["skill"]["sum"], 0.25)


class SystemicTest(RunCase):
    def rec(self, sample, case="bangkok", condition=None):
        s = SAMPLES[sample]
        kw = {"case": cases.get(case)[1], "eval_set": EVAL_SET, "model": HAIKU,
              "condition": condition or s["condition"], "trial_id": "t"}
        return sample_record(kw, sample)

    def test_systemic_rule_unit_cases(self):
        self.assertEqual(run.SYSTEMIC, {"1", "1b", "1c", "4b", "5"})
        stops = {"harness_binary_missing": "1", "skill_ok_bundled_skills_on": "1b", "skill_not_offered": "1c",
                 "api_404_bad_model": "4b", "replay_missing_model_answers": "5"}
        for sample, rule in stops.items():
            with self.subTest(sample=sample):
                self.assertEqual(run.systemic_rule(self.rec(sample)), rule)
                self.assertIn(run.systemic_rule(self.rec(sample)), run.SYSTEMIC)
        for sample in ("api_429_rate_limit", "error_during_execution", "nasa_network_failure", "not_logged_in",
                       "exit0_null_result", "skill_ok", "runaway_budget", "skill_no_cli_call"):
            with self.subTest(sample=sample):
                self.assertNotIn(run.systemic_rule(self.rec(sample)), run.SYSTEMIC)

    def test_a_1d_record_is_read_by_its_raw_rule(self):
        def conflict(rule):
            return {"status_rule": "1d", "status_inputs": {"redaction_conflict": {"rule": rule}}}
        self.assertEqual(run.systemic_rule(conflict("4b")), "4b")
        self.assertIn(run.systemic_rule(conflict("4b")), run.SYSTEMIC)
        self.assertNotIn(run.systemic_rule(conflict("11")), run.SYSTEMIC)

    def test_a_subjects_mistyped_coordinates_do_not_stop_the_run(self):
        rec = self.rec("replay_missing_model_answers", case="lagos")
        self.assertEqual((rec["status"], rec["status_rule"]), ("ok", "11"))
        self.assertNotIn(run.systemic_rule(rec), run.SYSTEMIC)

    def test_end_to_end_a_systemic_record_stops_the_run_where_evals_trial_exits_0(self):
        d = self.stand_in(streams={"skill": "skill_not_offered.jsonl"})
        with self.on_path(d):
            res = self.go("--n", "1", "--approve-spend", "1", real=True)
            snap = self.snap(res)
            self.assertEqual((res.code, snap["status"], snap["stop_reason"]), (1, "partial", "systemic:1c"))
            self.assertEqual(len(self.records(res)), 1)
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = trial.main(["--adapter", "claude_code", "--case", "bangkok", "--model", SONNET,
                                   "--condition", "skill", "--timeout", "60", "--out", str(self.tmp / "t.json")])
            self.assertEqual(code, 0, err.getvalue())
            self.assertIn("rule 1c", out.getvalue())

    def test_a_rule_5_record_stops_the_run_through_the_loop(self):
        script = Script({(1, "bangkok", SONNET, "skill"): ("replay",)})
        res = self.go("--n", "1", "--approve-spend", "1", script=script)
        snap = self.snap(res)
        self.assertEqual((res.code, snap["status"], snap["stop_reason"], len(script.calls)),
                         (1, "partial", "systemic:5", 9))
        self.assertEqual(self.records(res)[-1]["status_rule"], "5")

    def test_r3_is_named_before_the_systemic_rule_when_one_record_trips_both(self):
        def also_wrong_model(i, kw, rec):
            rec["subject"]["canonical_models"] = ["claude-opus-5-5"]
        script = Script({(1, "london", SONNET, "skill"): ("sys",)}, tweak=also_wrong_model)
        res = self.go("--n", "1", "--approve-spend", "1", script=script)
        self.assertEqual((res.code, self.snap(res)["stop_reason"]), (1, "r3"))
        self.assertEqual(self.records(res)[0]["status_rule"], "1c")

    def test_a_systemic_rule_or_an_r3_mismatch_on_the_final_record_leaves_the_run_partial(self):
        res = self.go("--n", "1", "--approve-spend", "1", script=Script({(1, "djibouti", HAIKU, "no-skill"): ("sys",)}))
        snap = self.snap(res)
        self.assertEqual((res.code, snap["status"], snap["stop_reason"], snap["counts"]["attempts"]),
                         (1, "partial", "systemic:4b", 16))

        def wrong_model(i, kw, rec):
            if i == 15:
                rec["subject"]["canonical_models"] = ["claude-opus-5-5"]
        shutil.rmtree(self.results)
        res = self.go("--n", "1", "--approve-spend", "1", script=Script(tweak=wrong_model))
        snap = self.snap(res)
        self.assertEqual((res.code, snap["status"], snap["stop_reason"], snap["counts"]["attempts"]),
                         (1, "partial", "r3", 16))


class StreakTest(RunCase):
    HS = (HAIKU, "skill")

    def test_the_streak_is_named_before_the_cap_when_both_are_reached(self):
        plan = {(1, case, *self.HS): ("excl",) for case in ("london", "lagos", "bangkok")}
        script = Script(plan, cost=lambda i, kw: 5.0 if i == 10 else 0.01)
        res = self.go("--n", "1", "--approve-spend", "1", script=script)
        self.assertEqual((res.code, self.snap(res)["stop_reason"], len(script.calls)),
                         (1, "streak:claude-haiku-5-5/skill", 11))
        self.assertGreaterEqual(self.snap(res)["spend_usd"], 1)

    def stop(self, plan, n="1"):
        script = Script(plan)
        res = self.go("--n", n, "--approve-spend", "1" if n == "1" else "2", script=script)
        return res, script, self.snap(res)

    def test_a_streak_confined_to_one_key_stops_before_the_next_attempt_of_any_key(self):
        plan = {(1, case, HAIKU, "skill"): ("excl",) for case in ("london", "lagos", "bangkok")}
        res, script, snap = self.stop(plan, n="2")
        self.assertEqual((res.code, snap["status"], snap["stop_reason"]),
                         (1, "partial", "streak:claude-haiku-5-5/skill"))
        self.assertEqual(len(script.calls), 11)
        self.assertEqual(slot_of(self.records(res)[-1]), (1, "bangkok", HAIKU, "skill"))

    def test_a_runaway_or_a_contract_violation_resets_only_its_own_keys_streak(self):
        for key, outcome in (((HAIKU, "no-skill"), "runaway"), ((HAIKU, "skill"), "cv")):
            with self.subTest(outcome=outcome):
                shutil.rmtree(self.results, ignore_errors=True)
                plan = {(1, "london", *key): ("excl",), (1, "lagos", *key): ("excl",), (1, "bangkok", *key): (outcome,),
                        (1, "djibouti", *key): ("excl",), (2, "london", *key): ("excl",)}
                res, script, snap = self.stop(plan, n="2")
                self.assertEqual((res.code, snap["status"], len(script.calls)), (0, "complete", 32))
                self.assertEqual(snap["counts"]["by_status"].get("runaway" if outcome == "runaway"
                                                                  else "contract-violation"), 1)
        shutil.rmtree(self.results, ignore_errors=True)
        plan = {(1, "london", *self.HS): ("excl",), (1, "lagos", *self.HS): ("excl",),
                (1, "lagos", HAIKU, "no-skill"): ("runaway",), (1, "bangkok", *self.HS): ("excl",)}
        res, script, snap = self.stop(plan)
        self.assertEqual(snap["stop_reason"], "streak:claude-haiku-5-5/skill")

    def test_a_streak_stops_the_run_for_any_model_and_condition_even_while_other_keys_pass(self):
        for model in (SONNET, HAIKU):
            for condition in CONDITIONS:
                with self.subTest(model=model, condition=condition):
                    shutil.rmtree(self.results, ignore_errors=True)
                    plan = {(1, case, model, condition): ("excl",) for case in ("london", "lagos", "bangkok")}
                    res, script, snap = self.stop(plan)
                    self.assertEqual((res.code, snap["status"], snap["stop_reason"]),
                                     (1, "partial", f"streak:{model}/{condition}"))

    def test_a_logged_out_cli_in_the_no_skill_condition_stops_on_the_streak(self):
        res, script, snap = self.stop({(1, case, SONNET, "no-skill"): ("loggedout",) for case in CASE_IDS})
        self.assertEqual((res.code, snap["stop_reason"], len(script.calls)),
                         (1, "streak:claude-sonnet-5-5/no-skill", 10))
        self.assertEqual({r["status_rule"] for r in self.records(res) if r["status"] == "harness-error"}, {"8"})

    def test_four_keys_that_all_fail_non_systemically_stop_on_the_ninth_trial(self):
        script = Script(default="excl")
        res = self.go("--n", "1", "--approve-spend", "1", script=script)
        self.assertEqual((res.code, self.snap(res)["stop_reason"], len(script.calls)),
                         (1, "streak:claude-sonnet-5-5/skill", 9))

    def test_an_ok_of_the_same_key_resets_the_streak(self):
        plan = {(1, "london", *self.HS): ("excl",), (1, "lagos", *self.HS): ("excl",),
                (1, "djibouti", *self.HS): ("excl",), (2, "london", *self.HS): ("excl",)}
        res, script, snap = self.stop(plan, n="2")
        self.assertEqual((res.code, snap["status"], len(script.calls)), (0, "complete", 32))

    def test_a_recurred_infra_counts_as_excluded(self):
        plan = {(1, case, *self.HS): ("infra", "infra") for case in ("london", "lagos", "bangkok")}
        res, script, snap = self.stop(plan, n="2")
        self.assertEqual(snap["stop_reason"], "streak:claude-haiku-5-5/skill")
        last = self.records(res)[-1]
        self.assertEqual((slot_of(last), last["attempt"], last["status"]), ((1, "bangkok", *self.HS), 2, "infra-error"))

    def test_an_attempt_1_infra_followed_by_an_ok_leaves_the_counter_untouched(self):
        plan = {(1, "london", *self.HS): ("excl",), (1, "lagos", *self.HS): ("excl",),
                (1, "bangkok", *self.HS): ("infra", "ok"), (1, "djibouti", *self.HS): ("excl",)}
        res, script, snap = self.stop(plan)
        self.assertEqual((res.code, snap["status"], len(script.calls)), (0, "complete", 17))

    def test_three_excluded_slots_ending_on_the_final_slot_leave_the_grid_complete(self):
        plan = {(1, case, HAIKU, "no-skill"): ("excl",) for case in ("lagos", "bangkok", "djibouti")}
        res, script, snap = self.stop(plan)
        self.assertEqual((res.code, snap["status"], snap["stop_reason"], len(script.calls)), (0, "complete", None, 16))
        self.assertEqual(snap["counts"]["by_status"], {"harness-error": 3, "ok": 13})


class R3Test(RunCase):
    def test_haiku_streams_under_a_sonnet_first_config_stop_the_run(self):
        d = self.tmp / "plain"
        d.mkdir()
        fake_claude(d, stream="skill_ok.jsonl")
        with self.on_path(d):
            res = self.go("--n", "1", "--approve-spend", "1", real=True)
        snap = self.snap(res)
        self.assertEqual((res.code, snap["status"], snap["stop_reason"]), (1, "partial", "r3"))
        self.assertEqual(len(self.records(res)), 1)
        self.assertIn("R3 stop", res.err)

    def test_an_ok_record_that_reports_no_model_also_stops(self):
        def no_model(i, kw, rec):
            rec["subject"]["canonical_models"], rec["subject"]["canonical_model"] = [], None
        script = Script(tweak=no_model)
        res = self.go("--n", "1", "--approve-spend", "1", script=script)
        self.assertEqual((res.code, self.snap(res)["stop_reason"], len(script.calls)), (1, "r3", 1))


class SnapshotTest(RunCase):
    TOP = {"run_id", "status", "stop_reason", "in_flight", "started_utc", "finished_utc", "overrides",
           "approved_spend_usd", "claude_path", "config_sha256", "settings", "eval_set_version", "eval_set_sha256",
           "subjects", "judge", "fixture_accessed", "cli_versions", "skill_sha256s", "counts", "spend_usd", "unpriced",
           "projection", "cost_by_key", "report"}

    def test_fields_report_and_cost_by_key(self):
        cfg_path = self.config_path()
        costs = {(SONNET, "skill"): 0.05, (SONNET, "no-skill"): 0.02, (HAIKU, "skill"): 0.04, (HAIKU, "no-skill"): 0.03}

        def cost(i, kw):
            unpriced = kw["case"]["id"] == "london" and "-r1-" in kw["trial_id"] and kw["model"] == HAIKU
            return None if unpriced and kw["condition"] == "no-skill" else costs[(kw["model"], kw["condition"])]
        script = Script(cost=cost)
        res = self.go("--n", "2", "--approve-spend", "2", "--config", cfg_path, script=script)
        snap = self.snap(res)
        self.assertEqual(set(snap), self.TOP)
        self.assertEqual((snap["status"], snap["stop_reason"], snap["run_id"]), ("complete", None, res.run_dir.name))
        for key in ("started_utc", "finished_utc"):
            self.assertRegex(snap[key], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        self.assertEqual((snap["overrides"], snap["approved_spend_usd"]), ({"n": 2}, 2.0))
        self.assertEqual(snap["claude_path"], record.scrubber()(os.path.abspath(shutil.which("claude"))))
        self.assertEqual(snap["config_sha256"], hashlib.sha256(Path(cfg_path).read_bytes()).hexdigest())
        self.assertEqual((snap["eval_set_version"], snap["eval_set_sha256"]), ("1", EVAL_SET["sha256"]))
        self.assertEqual((snap["subjects"], snap["judge"]), ([SONNET, HAIKU], "claude-opus-5-5"))
        self.assertRegex(snap["fixture_accessed"], r"^\d{4}-\d\d-\d\d$")
        self.assertEqual((snap["cli_versions"], snap["skill_sha256s"]),
                         (["9.9.9"], [record.skill_sha256(claude_code.SKILL_DIR)]))
        self.assertEqual(snap["counts"], {"planned_slots": 32, "slots_started": 32, "attempts": 32, "reruns": 0,
                                          "by_status": {"ok": 32}})
        self.assertEqual((snap["unpriced"], round(snap["spend_usd"], 6)), (1, round(0.4 + 0.16 + 0.32 + 0.21, 6)))
        self.assertEqual(snap["settings"]["n_trials"], 2)
        self.assertEqual(snap["projection"]["total_usd"], 0.56)
        self.assertEqual(snap["projection"]["source"], "defaults")
        self.assertEqual(snap["cost_by_key"][SONNET]["skill"], {"sum": 0.4, "priced": 8, "mean": 0.05})
        self.assertEqual(snap["cost_by_key"][HAIKU]["skill"], {"sum": 0.32, "priced": 8, "mean": 0.04})
        self.assertEqual(snap["cost_by_key"][HAIKU]["no-skill"], {"sum": 0.21, "priced": 7, "mean": 0.03})
        recs = self.records(res)
        expected = report.build_report(recs, EVAL_SET, 7, 200)
        self.assertEqual(snap["report"], json.loads(json.dumps(expected)))
        text = (res.run_dir / "snapshot.json").read_text(encoding="utf-8")
        self.assertRegex(text, r'(?m)^  "status": "complete",$')
        self.assertRegex(text, r'(?m)^  "claude_path": ".*/claude",$')

    def test_cost_by_key_stores_the_mean_to_six_decimals(self):
        res = self.go("--n", "2", "--approve-spend", "2", script=Script(cost=0.012345))
        self.assertEqual(self.snap(res)["cost_by_key"][SONNET]["skill"],
                         {"sum": 0.09876, "priced": 8, "mean": 0.012345})

    def test_the_fixture_date_comes_from_the_line_source_md_writes(self):
        rows = [{"site": "london", "lat": 51.5, "lon": -0.13, "file": "51.5_-0.13.json", "service": "POWER x",
                 "version": "v1", "period": "p"}]
        self.assertEqual(snapshot.parse_fixture_date(record_fixtures.source_md(rows, "2031-02-03")), "2031-02-03")
        self.assertIsNone(snapshot.parse_fixture_date("# no accessed line\n"))
        self.assertIsNone(snapshot.fixture_date(self.tmp / "missing.md"))
        bare = self.tmp / "fixtures"
        shutil.copytree(run.FIXTURE_DIR, bare, ignore=shutil.ignore_patterns("*.md"))
        with mock.patch.object(run, "FIXTURE_DIR", bare):
            res = self.go("--n", "1", "--approve-spend", "1", script=Script())
        self.assertEqual((res.code, self.snap(res)["fixture_accessed"]), (0, None))

    def test_trials_jsonl_is_one_json_object_per_line_even_with_a_lone_surrogate_and_u2028(self):
        answer = "lone \ud800 and \u2028 and \u00e9"

        def odd_answer(i, kw, rec):
            rec["output"]["answer"] = answer
        res = self.go("--n", "1", "--approve-spend", "1", script=Script(tweak=odd_answer))
        raw = (res.run_dir / "trials.jsonl").read_bytes()
        text = raw.decode("utf-8")
        self.assertEqual(res.code, 0, res.err)
        self.assertNotIn("\u2028", text)
        self.assertTrue(text.isascii())
        self.assertEqual(len(text.splitlines()), 16)
        self.assertTrue(text.endswith("\n"))
        recs = self.records(res)
        self.assertEqual({r["output"]["answer"] for r in recs}, {answer})
        self.assertEqual(len(recs), 16)

    def test_each_record_is_on_disk_before_the_next_attempt_starts(self):
        written = []

        def lines_on_disk(i, kw, rec):
            (run_dir,) = self.results.iterdir()
            written.append((i, (run_dir / "trials.jsonl").read_text(encoding="utf-8").count("\n")))
        res = self.go("--n", "1", "--approve-spend", "1", script=Script(tweak=lines_on_disk))
        self.assertEqual(res.code, 0, res.err)
        self.assertEqual(written, [(i, i) for i in range(16)])

    def test_approve_spend_alone_is_not_an_override_but_n_is(self):
        res = self.go("--approve-spend", "8", script=Script())
        self.assertEqual((self.snap(res)["overrides"], self.snap(res)["settings"]["n_trials"]), ({}, 10))
        shutil.rmtree(self.results)
        res = self.go("--n", "1", "--approve-spend", "8", script=Script())
        self.assertEqual(self.snap(res)["overrides"], {"n": 1})

    def test_zero_records_give_a_null_report_and_a_summary_that_says_so(self):
        with self.assertRaises(RuntimeError):
            self.go("--n", "1", "--approve-spend", "1", script=Script(error_at=0))
        res = Result(None, "", "", sorted(self.results.glob("*"))[-1])
        snap = self.snap(res)
        self.assertIsNone(snap["report"])
        self.assertEqual((snap["status"], snap["stop_reason"], snap["counts"]["attempts"]),
                         ("partial", "error:RuntimeError", 0))
        self.assertIn("No trial records.", (res.run_dir / "summary.md").read_text(encoding="utf-8"))

    def test_an_unexpected_exception_writes_a_partial_snapshot_and_propagates(self):
        with self.assertRaises(RuntimeError):
            self.go("--n", "1", "--approve-spend", "1", script=Script(error_at=2))
        res = Result(None, "", "", sorted(self.results.glob("*"))[-1])
        snap = self.snap(res)
        self.assertEqual((snap["status"], snap["stop_reason"], snap["counts"]["attempts"]),
                         ("partial", "error:RuntimeError", 2))
        self.assertEqual(len(self.records(res)), 2)
        summary = (res.run_dir / "summary.md").read_text(encoding="utf-8")
        self.assertIn("slots started 2 of 16 planned", summary)

    def test_a_failing_summary_does_not_replace_the_exception_that_stopped_the_run(self):
        with mock.patch.object(report, "format_report", side_effect=ValueError("fmt")):
            with self.assertRaises(RuntimeError):
                self.go("--n", "1", "--approve-spend", "1", script=Script(error_at=2))
        res = Result(None, "", "", sorted(self.results.glob("*"))[-1])
        self.assertEqual(self.snap(res)["stop_reason"], "error:RuntimeError")
        self.assertIn("summary failed: ValueError: fmt", (res.run_dir / "summary.md").read_text(encoding="utf-8"))
        self.assertIn("summary failed: ValueError: fmt\n", self.last_err.getvalue())

    def test_a_failing_snapshot_does_not_replace_the_exception_that_stopped_the_run(self):
        with mock.patch.object(snapshot, "build_snapshot", side_effect=ValueError("snap")):
            with self.assertRaises(RuntimeError):
                self.go("--n", "1", "--approve-spend", "1", script=Script(error_at=2))
        self.assertEqual(self.last_err.getvalue().count("snapshot failed: ValueError: snap\n"), 1)

    def test_a_failing_snapshot_on_a_run_that_stopped_cleanly_still_raises(self):
        with mock.patch.object(snapshot, "build_snapshot", side_effect=ValueError("snap")):
            with self.assertRaises(ValueError):
                self.go("--n", "1", "--approve-spend", "1", script=Script())

    def test_a_failing_summary_is_visible_on_stderr_and_does_not_change_the_exit(self):
        with mock.patch.object(report, "format_report", side_effect=ValueError("fmt")):
            res = self.go("--n", "1", "--approve-spend", "1", script=Script())
        self.assertEqual(res.code, 0)
        self.assertIn("summary failed: ValueError: fmt\n", res.err)

    def test_a_failing_report_build_is_recorded_and_does_not_lose_the_snapshot(self):
        with mock.patch.object(report, "build_report", side_effect=ValueError("bad")):
            res = self.go("--n", "1", "--approve-spend", "1", script=Script())
        snap = self.snap(res)
        self.assertEqual((res.code, snap["status"], snap["report"], snap["report_error"]),
                         (0, "complete", None, "ValueError: bad"))
        self.assertIn("Report failed: ValueError: bad", (res.run_dir / "summary.md").read_text(encoding="utf-8"))

    def test_the_summary_holds_the_report_in_a_fenced_block(self):
        res = self.go("--n", "1", "--approve-spend", "1", script=Script())
        summary = (res.run_dir / "summary.md").read_text(encoding="utf-8")
        self.assertEqual(summary.count("```"), 2)
        self.assertRegex(summary, r"(?s)```\nEval report\..*judge: not run \(E5\)\n```")

    def test_a_cost_that_is_not_a_finite_non_negative_number_is_unpriced_and_charged_at_the_cap(self):
        for cost in (None, float("nan"), float("inf"), -0.01, True, "0.05"):
            with self.subTest(cost=cost):
                rec = {"usage": {"cost_usd": cost}}
                self.assertIsNone(snapshot.attempt_cost(rec))
                self.assertEqual(snapshot.charged([rec], 0.5), 0.5)
        self.assertEqual(snapshot.attempt_cost({"usage": {"cost_usd": 0}}), 0)


class BootstrapConfigTest(RunCase):
    def test_stats_defaults_equal_the_committed_configs_values(self):
        cfg = config.load()
        self.assertEqual((stats.DEFAULT_SEED, stats.DEFAULT_RESAMPLES),
                         (cfg["bootstrap_seed"], cfg["bootstrap_resamples"]))

    def test_the_snapshot_uses_the_configs_seed_and_resamples(self):
        plan = {(rep, case, model, "skill"): ("ok",) if (rep + len(case)) % 2 else ("excl",)
                for rep in (1, 2) for case in CASE_IDS for model in (SONNET, HAIKU)}
        res = self.go("--n", "2", "--approve-spend", "2", over={"bootstrap_seed": 7, "bootstrap_resamples": 200,
                                                                 "max_consecutive_excluded": 99}, script=Script(plan))
        snap = self.snap(res)
        self.assertEqual(snap["report"]["bootstrap"], {"seed": 7, "resamples": 200})
        recs = self.records(res)
        self.assertEqual(snap["report"], json.loads(json.dumps(report.build_report(recs, EVAL_SET, 7, 200))))


def version_changed(old="9.9.9", new="9.9.10"):
    return lambda message: claude_code.VersionChanged(f"{old} -> {new}")


class VersionTest(RunCase):
    def case(self):
        return cases.get("bangkok")[1]

    def trial(self, binary, **kw):
        return claude_code.run_trial(case=self.case(), eval_set=EVAL_SET, model=SONNET, condition="no-skill",
                                     budget_usd=0.5, timeout=60, trial_id="t", binary=binary, **kw)

    def test_a_version_change_stops_the_run_before_the_model_starts(self):
        d = self.stand_in(bump_version_after=5)
        with self.on_path(d):
            res = self.go("--n", "1", "--approve-spend", "1", real=True)
        snap = self.snap(res)
        self.assertEqual((res.code, snap["status"], snap["stop_reason"]), (1, "mixed-version", "version-changed"))
        self.assertEqual(len(self.records(res)), 5)
        self.assertEqual(len((d / "claude.starts").read_text().splitlines()), 5)
        self.assertEqual(len((d / "claude.count").read_text().splitlines()), 6)
        self.assertIsNone(snap["in_flight"])
        self.assertEqual(snap["cli_versions"], ["9.9.9"])
        self.assertIn("version changed: 9.9.9 -> 9.9.10", res.err)

    def test_run_trial_raises_on_a_different_version_and_runs_on_the_same_or_none(self):
        d = self.stand_in(bump_version_after=99)
        binary = str(d / "claude")
        with self.assertRaises(claude_code.VersionChanged) as caught:
            self.trial(binary, expected_cli_version="0.0.0")
        self.assertEqual(str(caught.exception), "0.0.0 -> 9.9.9")
        self.assertFalse((d / "claude.starts").exists())
        for expected in ("9.9.9", None):
            with self.subTest(expected=expected):
                rec = self.trial(binary, expected_cli_version=expected)
                self.assertEqual((rec["status"], rec["subject"]["cli_version"]), ("ok", "9.9.9"))
        self.assertEqual(len((d / "claude.starts").read_text().splitlines()), 2)

    def test_the_baseline_comes_from_the_first_record_and_a_scripted_change_stops_the_run(self):
        script = Script(error_at=3, error=version_changed())
        res = self.go("--n", "1", "--approve-spend", "1", script=script)
        snap = self.snap(res)
        self.assertEqual([c["expected_cli_version"] for c in script.calls], [None, "9.9.9", "9.9.9", "9.9.9"])
        self.assertEqual((res.code, snap["status"], snap["stop_reason"]), (1, "mixed-version", "version-changed"))
        self.assertEqual(len(self.records(res)), 3)
        self.assertIsNone(snap["in_flight"])
        self.assertIn("version changed: 9.9.9 -> 9.9.10", res.err)


class SkillHashTest(RunCase):
    def stop(self, tweak, **kw):
        script = Script(tweak=tweak, **kw)
        res = self.go("--n", "1", "--approve-spend", "1", script=script)
        return res, script, self.snap(res)

    def test_a_changed_skill_hash_stops_the_run_and_keeps_the_record(self):
        def changed(i, kw, rec):
            if i == 2:
                rec["condition"]["skill_sha256"] = "f" * 64

        res, script, snap = self.stop(changed)
        self.assertEqual((res.code, snap["status"], snap["stop_reason"]), (1, "mixed-skill", "skill-changed"))
        self.assertEqual((len(script.calls), len(self.records(res))), (3, 3))
        self.assertEqual(self.records(res)[-1]["condition"]["skill_sha256"], "f" * 64)
        self.assertIsNone(snap["in_flight"])

    def test_a_null_hash_and_a_no_skill_record_are_not_compared(self):
        def null_hash(i, kw, rec):
            if rec["condition"]["condition"] == "skill":
                rec["condition"]["skill_sha256"] = None

        def hash_on_no_skill(i, kw, rec):
            if rec["condition"]["condition"] == "no-skill":
                rec["condition"]["skill_sha256"] = "f" * 64

        for name, tweak in (("null", null_hash), ("no-skill", hash_on_no_skill)):
            with self.subTest(name):
                shutil.rmtree(self.results, ignore_errors=True)
                res, script, snap = self.stop(tweak)
                self.assertEqual((res.code, snap["status"], len(script.calls)), (0, "complete", 16))

    def test_a_record_that_trips_both_the_skill_guard_and_r3_reports_the_skill_change(self):
        def both(i, kw, rec):
            if i == 2:
                rec["condition"]["skill_sha256"] = "f" * 64
                rec["subject"]["canonical_models"] = ["claude-opus-5-5"]

        res, script, snap = self.stop(both)
        self.assertEqual((res.code, snap["stop_reason"]), (1, "skill-changed"))


class RunStopsTheChildTest(RunCase):
    """`claude_code.run` on a stand-in that sleeps, with Popen, killpg or Thread.start rigged to interrupt it."""

    def stand_in_path(self):
        return fake_claude(tempfile.mkdtemp(dir=self.tmp), sleep=25)

    def run_interrupted(self, patch_in, path=None, group_dies=True):
        path = path or self.stand_in_path()
        pid_file = path + ".pid"
        state = {}

        def interrupt():
            state["pid"], state["pgid"] = self.track(pid_file)
            state["at"] = time.monotonic()
            raise KeyboardInterrupt

        with patch_in(interrupt), self.assertRaises(KeyboardInterrupt):
            claude_code.run([path, "-p", "x"], self.tmp, dict(os.environ), 60)
        self.assertLess(time.monotonic() - state["at"], 5)
        if group_dies:
            self.assertGone(state["pid"], state["pgid"])

    @staticmethod
    def popen_that_interrupts(interrupt, made=None):
        class Popen(subprocess.Popen):
            armed = True

            def wait(self, timeout=None):
                if Popen.armed:
                    Popen.armed = False
                    if made is not None:
                        made.append(self)
                    interrupt()
                return super().wait(timeout)
        return mock.patch.object(subprocess, "Popen", Popen)

    def test_an_interrupt_in_the_first_wait_kills_the_group_and_propagates(self):
        self.run_interrupted(self.popen_that_interrupts)

    def test_a_group_that_is_already_gone_does_not_replace_the_interrupt(self):
        real_killpg = os.killpg
        for error in (ProcessLookupError, PermissionError):
            with self.subTest(error=error.__name__):
                def killpg_then_fail(pgid, sig):
                    real_killpg(pgid, sig)
                    raise error

                def patch_in(interrupt):
                    stack = contextlib.ExitStack()
                    stack.enter_context(self.popen_that_interrupts(interrupt))
                    stack.enter_context(mock.patch.object(claude_code.os, "killpg", killpg_then_fail))
                    return stack
                self.run_interrupted(patch_in)

    def test_a_survivor_outside_the_group_does_not_stall_the_interrupt(self):
        stand_in = Path(tempfile.mkdtemp(dir=self.tmp)) / "claude"
        code = "import os,sys,time; os.setsid(); open(sys.argv[1], 'w').write(str(os.getpid())); time.sleep(25)"
        stand_in.write_text(f'#!/bin/sh\n{shlex.quote(sys.executable)} -I -c {shlex.quote(code)} "$0.pid" &\nwait\n')
        stand_in.chmod(stand_in.stat().st_mode | stat.S_IXUSR)
        made = []

        def close_leaked_streams():
            for proc in made:
                for stream in (proc.stdout, proc.stderr):
                    try:
                        stream.close()
                    except OSError:
                        pass
        # Registered before the survivor is tracked, so it runs after the survivor is reaped and closing cannot block.
        self.addCleanup(close_leaked_streams)
        self.run_interrupted(lambda interrupt: self.popen_that_interrupts(interrupt, made), path=str(stand_in),
                             group_dies=False)

    def test_an_interrupt_while_the_readers_start_kills_the_group(self):
        def patch_in(interrupt):
            def start(thread):
                interrupt()
            return mock.patch.object(threading.Thread, "start", start)
        self.run_interrupted(patch_in)


class InterruptTest(RunCase):
    def interrupt(self, signum):
        d = self.stand_in(f"bin{int(signum)}", sleep=25)
        env = {**os.environ, "PATH": f"{d}{os.pathsep}{os.environ['PATH']}"}

        def child_signals():
            # The signal under test starts ignored, as a background job's SIGINT does; SIGHUP never starts ignored.
            signal.signal(signal.SIGHUP, signal.SIG_DFL)
            if signum != signal.SIGHUP:
                signal.signal(signum, signal.SIG_IGN)
        proc = subprocess.Popen([sys.executable, "-m", "evals.run", "--n", "2", "--approve-spend", "2",
                                 "--results-dir", str(self.results)], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, preexec_fn=child_signals)
        self.addCleanup(lambda: proc.poll() is None and (proc.kill(), proc.wait()))
        pid, pgid = self.track(d / "claude.pid")
        proc.send_signal(signum)
        try:
            _, err = proc.communicate(timeout=EXIT_WAIT)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            self.fail(f"evals.run did not stop within {EXIT_WAIT}s of signal {int(signum)}")
        return proc.returncode, err, pid, pgid

    def test_sigint_sigterm_and_sighup_stop_the_run_with_no_child_left(self):
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            with self.subTest(signal=signum.name):
                shutil.rmtree(self.results, ignore_errors=True)
                code, err, pid, pgid = self.interrupt(signum)
                (run_dir,) = self.results.iterdir()
                snap = self.snap(Result(None, "", "", run_dir))
                self.assertEqual((code, snap["status"], snap["stop_reason"]), (1, "partial", "interrupted"), err)
                self.assertNotIn("Traceback", err)
                slot = snap["in_flight"]
                self.assertEqual(set(slot), {"case_id", "model", "condition", "rep", "attempt", "trial_id"})
                self.assertEqual((slot["case_id"], slot["model"], slot["condition"], slot["rep"], slot["attempt"]),
                                 (CASE_IDS[0], SONNET, "skill", 1, 1))
                self.assertEqual(snap["counts"]["attempts"], 0)
                self.assertGone(pid, pgid)

    def test_a_scripted_interrupt_is_a_clean_stop_with_the_attempt_in_flight(self):
        script = Script(error_at=2, error=KeyboardInterrupt)
        res = self.go("--n", "1", "--approve-spend", "1", script=script)
        snap = self.snap(res)
        self.assertEqual((res.code, snap["status"], snap["stop_reason"]), (1, "partial", "interrupted"))
        self.assertEqual(len(self.records(res)), 2)
        call = script.calls[2]
        self.assertEqual(snap["in_flight"], {"case_id": call["case"]["id"], "model": call["model"],
                                             "condition": call["condition"], "rep": 1, "attempt": 1,
                                             "trial_id": call["trial_id"]})
        self.assertIn("partial (interrupted)", res.err)

    def test_an_interrupt_before_any_record_gives_a_null_report(self):
        res = self.go("--n", "1", "--approve-spend", "1", script=Script(error_at=0, error=KeyboardInterrupt))
        snap = self.snap(res)
        self.assertEqual((res.code, snap["stop_reason"], snap["report"]), (1, "interrupted", None))
        self.assertIn("No trial records.", (res.run_dir / "summary.md").read_text(encoding="utf-8"))

    def test_a_signal_during_the_record_write_leaves_the_file_and_the_snapshot_in_agreement(self):
        flushes, masks, real_open, sigint = [], [], open, signal.getsignal(signal.SIGINT)
        mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        script = Script()

        def scripted(**kw):
            if script.calls:
                masks.append(signal.pthread_sigmask(signal.SIG_BLOCK, set()))
            return script(**kw)

        class Wrapper:
            def __init__(self, f):
                self.f = f

            def write(self, text):
                return self.f.write(text)

            def flush(self):
                self.f.flush()
                flushes.append(1)
                if len(flushes) == 2:
                    os.kill(os.getpid(), signal.SIGINT)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.f.close()
                return False

        def fake_open(path, *args, **kwargs):
            f = real_open(path, *args, **kwargs)
            return Wrapper(f) if str(path).endswith("trials.jsonl") else f
        with mock.patch.object(run, "open", fake_open, create=True):
            res = self.go("--n", "1", "--approve-spend", "1", script=scripted)
        snap = self.snap(res)
        on_disk = self.records(res)
        self.assertTrue(masks)
        for held in masks:
            self.assertFalse(held & {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}, held)
        self.assertEqual(len(masks), 1)
        self.assertEqual((res.code, snap["stop_reason"]), (1, "interrupted"))
        self.assertEqual((len(on_disk), snap["counts"]["attempts"]), (2, 2))
        self.assertTrue(snap["in_flight"] is None or snap["in_flight"]["trial_id"] not in
                        {r["trial_id"] for r in on_disk})
        self.assertIs(signal.getsignal(signal.SIGINT), sigint)
        self.assertEqual(signal.pthread_sigmask(signal.SIG_BLOCK, set()), mask)

    def test_a_failing_snapshot_after_an_interrupt_raises(self):
        with mock.patch.object(snapshot, "build_snapshot", side_effect=ValueError("snap")):
            with self.assertRaises(ValueError):
                self.go("--n", "1", "--approve-spend", "1", script=Script(error_at=1, error=KeyboardInterrupt))


class HandlerTest(RunCase):
    SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)

    def setUp(self):
        super().setUp()
        self.addCleanup(lambda saved=self.current(): [signal.signal(s, h) for s, h in saved.items()])
        signal.signal(signal.SIGHUP, signal.SIG_DFL)
        self.original = self.current()

    def current(self):
        return {s: signal.getsignal(s) for s in self.SIGNALS}

    def watched(self, seen, error_at=None):
        """A scripted run_trial and a build_snapshot patch that note the handlers in force during each."""
        script = Script(error_at=error_at)

        def call(**kw):
            seen.setdefault("trial", self.current())
            return script(**kw)
        build = snapshot.build_snapshot

        def spy(**kw):
            seen["snapshot"] = self.current()
            return build(**kw)
        return call, mock.patch.object(snapshot, "build_snapshot", spy)

    def test_the_handlers_are_installed_ignored_while_writing_and_restored(self):
        seen = {}
        script, spy = self.watched(seen)
        with spy:
            res = self.go("--n", "1", "--approve-spend", "1", script=script)
        self.assertEqual(res.code, 0)
        self.assertIs(seen["trial"][signal.SIGINT], signal.default_int_handler)
        for signum in (signal.SIGTERM, signal.SIGHUP):
            with self.assertRaises(KeyboardInterrupt):
                seen["trial"][signum](signum, None)
        self.assertEqual(seen["snapshot"], {s: signal.SIG_IGN for s in self.SIGNALS})
        self.assertEqual(self.current(), self.original)

    def test_the_handlers_are_restored_after_an_unexpected_exception(self):
        seen = {}
        script, spy = self.watched(seen, error_at=1)
        with spy, self.assertRaises(RuntimeError):
            self.go("--n", "1", "--approve-spend", "1", script=script)
        self.assertEqual(seen["snapshot"], {s: signal.SIG_IGN for s in self.SIGNALS})
        self.assertEqual(self.current(), self.original)

    def test_a_sighup_that_is_already_ignored_is_left_alone(self):
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        seen = {}
        script, spy = self.watched(seen)
        with spy:
            self.go("--n", "1", "--approve-spend", "1", script=script)
        self.assertIs(seen["trial"][signal.SIGHUP], signal.SIG_IGN)
        self.assertIs(seen["trial"][signal.SIGINT], signal.default_int_handler)
        self.assertIsNot(seen["trial"][signal.SIGTERM], self.original[signal.SIGTERM])
        self.assertIs(self.current()[signal.SIGHUP], signal.SIG_IGN)

    def test_main_on_another_thread_starts_and_leaves_the_handlers_alone(self):
        seen, out = {}, {}
        script, spy = self.watched(seen)

        def target():
            try:
                out["res"] = self.go("--n", "1", "--approve-spend", "1", script=script)
            except BaseException as e:  # noqa: BLE001 - reported to the test thread
                out["error"] = e
        with spy:
            thread = threading.Thread(target=target)
            thread.start()
            thread.join(DEADLINE)
        self.assertFalse(thread.is_alive())
        self.assertNotIn("error", out)
        self.assertEqual(out["res"].code, 0)
        self.assertEqual(seen["trial"], self.original)
        self.assertEqual(self.current(), self.original)


class StatusMapTest(unittest.TestCase):
    def build(self, stop_reason, **kw):
        return snapshot.build_snapshot(
            run_id="r", stop_reason=stop_reason, started="s", finished="f", overrides={}, approved_spend_usd=None,
            claude_path="c", config_sha256="x", settings={}, eval_set={}, subjects=[SONNET], judge="j",
            fixture_accessed=None, planned_slots=0, projection={}, records=[], **kw)

    def test_the_stop_reason_decides_the_status_and_in_flight_defaults_to_null(self):
        table = {None: "complete", "version-changed": "mixed-version", "skill-changed": "mixed-skill",
                 "interrupted": "partial", "spend-cap": "partial", "error:RuntimeError": "partial",
                 "r3": "partial", "systemic:1c": "partial"}
        for stop, status in table.items():
            with self.subTest(stop=stop):
                snap = self.build(stop)
                self.assertEqual((snap["status"], snap["stop_reason"], snap["in_flight"]), (status, stop, None))
        self.assertEqual(self.build("interrupted", in_flight={"trial_id": "t"})["in_flight"], {"trial_id": "t"})


class InFlightTest(RunCase):
    CHEAP = {m: {c: 0.001 for c in CONDITIONS} for m in (SONNET, HAIKU)}

    def test_an_unexpected_exception_names_the_attempt_in_flight(self):
        script = Script(error_at=2)
        with self.assertRaises(RuntimeError):
            self.go("--n", "1", "--approve-spend", "1", script=script)
        snap = self.snap(Result(None, "", "", sorted(self.results.glob("*"))[-1]))
        self.assertEqual(snap["stop_reason"], "error:RuntimeError")
        self.assertEqual(snap["in_flight"]["trial_id"], script.calls[2]["trial_id"])

    def test_other_stops_leave_it_null(self):
        stops = {"spend-cap": (dict(cost=0.30), {"cost_defaults_usd": self.CHEAP}),
                 "version-changed": (dict(error_at=3, error=version_changed()), {}),
                 "r3": (dict(tweak=lambda i, kw, rec: rec["subject"].update(canonical_models=["claude-opus-5-5"])), {}),
                 "systemic:1c": (dict(plan={(1, "london", SONNET, "skill"): ("sys",)}), {})}
        for stop, (kw, over) in stops.items():
            with self.subTest(stop=stop):
                shutil.rmtree(self.results, ignore_errors=True)
                res = self.go("--n", "1", "--approve-spend", "1", script=Script(**kw), over=over)
                snap = self.snap(res)
                self.assertEqual((res.code, snap["stop_reason"], snap["in_flight"]), (1, stop, None))

    def test_a_record_that_could_not_be_written_is_not_counted(self):
        real_dumps, seen = json.dumps, []

        def dumps(obj, *args, **kwargs):
            if isinstance(obj, dict) and "trial_id" in obj:
                seen.append(obj["trial_id"])
                if len(seen) == 2:
                    raise KeyboardInterrupt
            return real_dumps(obj, *args, **kwargs)
        script = Script()
        with mock.patch.object(run.json, "dumps", dumps):
            res = self.go("--n", "1", "--approve-spend", "1", script=script)
        snap = self.snap(res)
        self.assertEqual((res.code, snap["stop_reason"], snap["counts"]["attempts"]), (1, "interrupted", 1))
        self.assertEqual(len(self.records(res)), 1)
        self.assertEqual(snap["in_flight"]["trial_id"], script.calls[1]["trial_id"])
        self.assertEqual(snap["in_flight"]["trial_id"], seen[1])


class SkillFolderTest(RunCase):
    def test_an_unreadable_skill_folder_exits_2_with_nothing_created(self):
        with mock.patch.object(record, "skill_sha256", side_effect=OSError("denied")):
            res = self.go("--n", "1", "--approve-spend", "1")
        self.assertNothingRan(res)
        self.assertIn(str(claude_code.SKILL_DIR), res.err)
        self.assertIn("denied", res.err)


class ModuleTest(unittest.TestCase):
    def test_gitignore_and_the_default_results_dir(self):
        self.assertIn("evals/results/", (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines())
        self.assertEqual(REAL_RESULTS_DIR, ROOT / "evals" / "results")
        with tempfile.TemporaryDirectory() as elsewhere:
            proc = subprocess.run([sys.executable, "-c", "from evals import run; print(run.DEFAULT_RESULTS_DIR)"],
                                  cwd=elsewhere, env={**os.environ, "PYTHONPATH": str(ROOT)},
                                  capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.stdout.strip(), str(ROOT / "evals" / "results"), proc.stderr)

    def test_the_new_modules_are_covered_by_the_import_check(self):
        proc, imported = _import_all(ROOT)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(imported["calls"], [])
        self.assertTrue({"evals.run", "evals.snapshot"} <= set(imported["modules"]))


class StandInTest(Tracking, unittest.TestCase):
    def run_stand_in(self, directory, *args):
        return subprocess.run([str(Path(directory) / "claude"), *args], capture_output=True, text=True, timeout=60)

    def test_echo_model_rewrites_the_model_and_picks_the_stream_by_the_tools_value(self):
        with tempfile.TemporaryDirectory() as d:
            fake_claude(d, echo_model=True)
            skill = self.run_stand_in(d, "-p", "x", "--model", "claude-sonnet-5-5", "--tools", "Bash,Read,Skill")
            plain = self.run_stand_in(d, "-p", "x", "--model", "claude-sonnet-5-5", "--tools", "")
            self.assertEqual(self.run_stand_in(d, "--version").stdout.split()[0], "9.9.9")
        self.assertIn("claude-sonnet-5-5", skill.stdout)
        self.assertNotIn("claude-haiku-5-5", skill.stdout + plain.stdout)
        self.assertEqual(plain.stdout, (SAMPLES_DIR / ECHO_STREAMS["no-skill"]).read_text(encoding="utf-8")
                         .replace("claude-haiku-5-5", "claude-sonnet-5-5"))
        self.assertIn("Skill", skill.stdout)

    def test_streams_can_be_overridden_per_test(self):
        with tempfile.TemporaryDirectory() as d:
            fake_claude(d, echo_model=True, streams={"skill": "skill_not_offered.jsonl"})
            out = self.run_stand_in(d, "-p", "x", "--model", HAIKU, "--tools", "Bash,Read,Skill").stdout
        self.assertEqual(out, (SAMPLES_DIR / "skill_not_offered.jsonl").read_text(encoding="utf-8"))

    def test_the_module_entry_writes_an_executable_stand_in(self):
        with tempfile.TemporaryDirectory() as d:
            proc = subprocess.run([sys.executable, "-m", "tests.fake_claude", d], cwd=ROOT, capture_output=True,
                                  text=True, timeout=60)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stdout.strip(), str(Path(d) / "claude"))
            self.assertTrue(os.access(Path(d) / "claude", os.X_OK))

    def make(self, *flags):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        proc = subprocess.run([sys.executable, "-m", "tests.fake_claude", d.name, *flags], cwd=ROOT,
                              capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return Path(d.name)

    def test_the_bump_flag_answers_the_new_version_after_k_calls_and_the_default_never_does(self):
        d = self.make("--bump-version-after", "2")
        versions = [self.run_stand_in(d, "--version").stdout.split()[0] for _ in range(3)]
        self.assertEqual(versions, ["9.9.9", "9.9.9", "9.9.10"])
        self.assertEqual(len((d / "claude.count").read_text().splitlines()), 3)
        self.assertFalse((d / "claude.starts").exists())
        self.run_stand_in(d, "-p", "x", "--model", HAIKU, "--tools", "")
        self.assertEqual(len((d / "claude.starts").read_text().splitlines()), 1)
        plain = self.make()
        self.assertEqual([self.run_stand_in(plain, "--version").stdout.split()[0] for _ in range(3)], ["9.9.9"] * 3)
        self.assertEqual(sorted(p.name for p in plain.iterdir()), ["claude"])

    def test_the_sleep_flag_starts_a_sleeper_that_carries_the_stand_ins_path(self):
        d = self.make("--sleep", "25")
        stand_in = str(d / "claude")
        proc = subprocess.Popen([stand_in, "-p", "x", "--model", HAIKU, "--tools", ""], start_new_session=True,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(lambda: proc.wait(timeout=30))
        pid, pgid = self.track(d / "claude.pid")
        self.assertTrue(alive(pid))
        command = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True,
                                 timeout=30).stdout
        self.assertIn(stand_in, command)
        self.assertEqual(pgid, proc.pid)
        reap_group(pid, pgid)
        self.assertGone(pid, pgid)


class SubprocessRunTest(unittest.TestCase):
    def test_a_full_run_against_the_stand_in_writes_all_three_files(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "bin").mkdir()
            fake_claude(d / "bin", echo_model=True)
            env = {**os.environ, "PATH": f"{d / 'bin'}{os.pathsep}{os.environ['PATH']}"}
            try:
                proc = subprocess.run([sys.executable, "-m", "evals.run", "--n", "2", "--approve-spend", "2",
                                       "--results-dir", str(d / "results")], cwd=ROOT, env=env,
                                      capture_output=True, text=True, timeout=DEADLINE)
            except subprocess.TimeoutExpired:
                self.fail(f"evals.run did not finish within {DEADLINE}s")
            self.assertEqual(proc.returncode, 0, proc.stderr)
            (run_dir,) = (d / "results").iterdir()
            self.assertEqual(sorted(p.name for p in run_dir.iterdir()), ["snapshot.json", "summary.md", "trials.jsonl"])
            self.assertEqual(len(report.load_records(run_dir)), 32)
            self.assertEqual(len(re.findall(r"^\[\d+\] ", proc.stderr, re.MULTILINE)), 32)
            for line in re.findall(r"^\[\d+\] .*$", proc.stderr, re.MULTILINE):
                self.assertRegex(line, r"^\[\d+\] \w+ claude-\w+-5-5 [\w-]+ r\d a\d: ok \(rule 11\) "
                                       r"cost \$[\d.]+, spend \$[\d.]+$")
            header = (run_dir / "summary.md").read_text(encoding="utf-8").split("```")[0]
            for needle in ("complete", "slots started 32 of 32 planned", "(source: defaults)"):
                self.assertIn(needle, header)
            snap = json.loads((run_dir / "snapshot.json").read_text(encoding="utf-8"))
            self.assertEqual((snap["status"], snap["counts"]["slots_started"]), ("complete", 32))
            self.assertTrue(snap["claude_path"].endswith("/claude"))
            self.assertFalse(Path(_state["log"]).exists())


if __name__ == "__main__":
    unittest.main()
