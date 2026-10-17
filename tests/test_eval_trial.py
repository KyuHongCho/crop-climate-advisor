"""One Claude Code trial, offline: argv, workspace, environment, stream parsing and the record."""
import contextlib
import io
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from crop_advisor.climate import REPLAY_ENV
from evals import cases, record, status, trial
from evals.adapters import claude_code

SAMPLES_DIR = Path(__file__).resolve().parent / "fixtures" / "claude_code"
PROMPT = "Is basil suited to Bangkok (latitude 13.75, longitude 100.5)?"
COMMON = ["claude", "-p", PROMPT, "--model", "claude-sonnet-5-5",
          "--setting-sources", "project", "--strict-mcp-config", "--permission-mode", "default",
          "--max-budget-usd", "0.50", "--no-session-persistence", "--output-format", "stream-json", "--verbose"]


def _sample_lines(name):
    return [(i * 10, line) for i, line in
            enumerate((SAMPLES_DIR / name).read_text(encoding="utf-8").splitlines(keepends=True))]


def _fake_claude(directory, stream=None, exit_code=0, sleep=0, env_dump=None, emit=None):
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
    body.append(f"exit {exit_code}")
    path.write_text("\n".join(body) + "\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


class ArgvTest(unittest.TestCase):
    def test_skill_condition(self):
        self.assertEqual(claude_code.argv(PROMPT, "claude-sonnet-5-5", "skill", 0.5),
                         COMMON + ["--tools", "Bash,Read,Skill", "--allowedTools",
                                   "Bash(python -m crop_advisor.cli:*)", "Bash(python3 -m crop_advisor.cli:*)",
                                   "--disallowedTools", "Skill(skill:design)", "Skill(skill:doctor)",
                                   "Skill(skill:plugin-authoring)"])

    def test_the_denied_skills_are_the_ones_2_1_301_still_lists(self):
        listed = claude_code.parse(_sample_lines("skill_denied_other_skill.jsonl"))["init"]["skills"]
        self.assertEqual(sorted(listed), sorted([claude_code.SKILL_NAME, *claude_code.DENIED_SKILLS]))

    def test_no_skill_condition(self):
        self.assertEqual(claude_code.argv(PROMPT, "claude-sonnet-5-5", "no-skill", 0.5),
                         COMMON + ["--tools", ""])

    def test_unknown_condition_is_refused(self):
        with self.assertRaises(ValueError):
            claude_code.argv(PROMPT, "claude-sonnet-5-5", "skills", 0.5)


class WorkspaceTest(unittest.TestCase):
    def _build(self, condition):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        return Path(d.name), *claude_code.build_workspace(d.name, condition)

    def test_skill_workspace(self):
        base, ws, replay = self._build("skill")
        link = ws / ".claude" / "skills" / "crop-suitability"
        self.assertTrue(link.is_symlink())
        self.assertEqual(os.readlink(link), "../../.agents/skills/crop-suitability")
        self.assertTrue((link / "SKILL.md").is_file())
        self.assertTrue((link / "reference" / "temperature-claims.md").is_file())
        self.assertEqual(sorted(p.name for p in ws.iterdir()), [".agents", ".claude", "crop_advisor", "data"])
        self.assertEqual(sorted(p.name for p in (ws / ".claude").iterdir()), ["skills"])

    def test_no_skill_workspace(self):
        _, ws, _ = self._build("no-skill")
        self.assertEqual(sorted(p.name for p in ws.iterdir()), ["crop_advisor", "data"])

    def test_no_instruction_files_hooks_or_bytecode(self):
        _, ws, _ = self._build("skill")
        names = {p.name for p in ws.rglob("*")}
        for forbidden in ("CLAUDE.md", "CLAUDE.local.md", "settings.json", "hooks", "__pycache__"):
            self.assertNotIn(forbidden, names)
        self.assertTrue((ws / "crop_advisor" / "cli.py").is_file())
        self.assertTrue((ws / "data" / "ecocrop" / "basil.json").is_file())

    def test_recordings_sit_outside_the_workspace(self):
        _, ws, replay = self._build("skill")
        self.assertNotIn(ws.resolve(), replay.resolve().parents)
        self.assertEqual({p.name for p in replay.iterdir()},
                         {"51.5_-0.13.json", "6.52_3.38.json", "13.75_100.5.json", "11.59_43.15.json"})
        self.assertEqual(list(ws.rglob("*.json*_*")), [])


class MemoryLeakTest(unittest.TestCase):
    def test_ancestor_and_config_dir_instruction_files_are_found(self):
        with tempfile.TemporaryDirectory() as d:
            ws, config_dir = Path(d) / "a" / "workspace", Path(d) / "config"
            ws.mkdir(parents=True)
            config_dir.mkdir()
            env = {"CLAUDE_CONFIG_DIR": str(config_dir)}
            self.assertEqual(claude_code.memory_leaks(ws, env), [])
            (Path(d) / "a" / "CLAUDE.md").write_text("x")
            (config_dir / "CLAUDE.md").write_text("x")
            self.assertEqual({p.name for p in claude_code.memory_leaks(ws, env)}, {"CLAUDE.md"})
            self.assertEqual(len(claude_code.memory_leaks(ws, env)), 2)

    def test_dot_claude_instruction_files_above_the_workspace_are_found(self):
        files = ("CLAUDE.md", "CLAUDE.local.md", ".claude/CLAUDE.md", ".claude/rules/canary.md",
                 ".claude/rules/nested/canary.md")
        # A real workspace is <temp base>/workspace, so a real leak sits two or more levels up.
        for relative, depth in [(f, n) for f in files for n in (1, 3)]:
            with self.subTest(file=relative, levels_up=depth), tempfile.TemporaryDirectory() as d:
                ws = Path(d, "a", *["b"] * (depth - 1), "workspace")
                config_dir = Path(d) / "config"
                ws.mkdir(parents=True)
                config_dir.mkdir()
                leak = Path(d) / "a" / relative
                leak.parent.mkdir(parents=True, exist_ok=True)
                leak.write_text("x")
                found = claude_code.memory_leaks(ws, {"CLAUDE_CONFIG_DIR": str(config_dir)})
                self.assertEqual(found, [leak.resolve()])

    def test_user_rules_are_found(self):
        with tempfile.TemporaryDirectory() as d:
            ws, config_dir = Path(d) / "workspace", Path(d) / "config"
            ws.mkdir()
            (config_dir / "rules").mkdir(parents=True)
            (config_dir / "rules" / "r.md").write_text("x")
            self.assertEqual(claude_code.memory_leaks(ws, {"CLAUDE_CONFIG_DIR": str(config_dir)}),
                             [config_dir / "rules" / "r.md"])

    def test_a_leak_stops_the_trial_as_a_harness_error(self):
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(claude_code, "memory_leaks", return_value=[Path(d) / "CLAUDE.md"]):
            _, case = cases.get("bangkok")
            rec = claude_code.run_trial(case=case, eval_set={"version": "1"}, model="claude-sonnet-5-5",
                                        condition="no-skill", budget_usd=0.5, timeout=10, trial_id="t",
                                        binary=_fake_claude(d, "no_skill_ok.jsonl"))
        self.assertEqual((rec["status"], rec["status_rule"]), ("harness-error", "1"))
        self.assertIn("would leak", rec["status_inputs"]["harness_exception"])


class EnvironmentTest(unittest.TestCase):
    def test_parent_session_vars_are_dropped_and_the_trial_vars_set(self):
        parent = {"PATH": "/usr/bin", "HOME": "/h", "CLAUDECODE": "1", "CLAUDE_EFFORT": "medium",
                  "CLAUDE_CODE_SESSION_ID": "s", "AI_AGENT": "a", "CLAUDE_CODE_OAUTH_TOKEN": "tok",
                  "CLAUDE_CONFIG_DIR": "/c"}
        env = claude_code.environment("/r", base_env=parent, python_bin="/venv/bin")
        self.assertEqual(env["PATH"], os.pathsep.join(["/venv/bin", "/usr/bin"]))
        self.assertEqual(env[REPLAY_ENV], str(Path("/r").resolve()))
        self.assertTrue(Path(env[REPLAY_ENV]).is_absolute())
        self.assertEqual(env["DISABLE_AUTOUPDATER"], "1")
        self.assertEqual(env["CLAUDE_CODE_DISABLE_BUNDLED_SKILLS"], "1")
        self.assertEqual(env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"], "1")
        self.assertEqual(env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"], "1")
        for dropped in ("CLAUDECODE", "CLAUDE_EFFORT", "CLAUDE_CODE_SESSION_ID", "AI_AGENT"):
            self.assertNotIn(dropped, env)
        self.assertEqual((env["CLAUDE_CODE_OAUTH_TOKEN"], env["CLAUDE_CONFIG_DIR"], env["HOME"]), ("tok", "/c", "/h"))

    def test_default_python_is_the_running_interpreters_directory(self):
        env = claude_code.environment("/r", base_env={"PATH": "/usr/bin"})
        self.assertEqual(env["PATH"].split(os.pathsep)[0], str(Path(sys.executable).parent))


class ParseTest(unittest.TestCase):
    def test_skill_stream(self):
        parsed = claude_code.parse(_sample_lines("skill_ok.jsonl"))
        self.assertTrue(parsed["stream_parsed"])
        self.assertEqual([c["name"] for c in parsed["tool_calls"]], ["Skill", "Bash"])
        skill, cli = parsed["tool_calls"]
        self.assertEqual(skill["input"]["skill"], "crop-suitability")
        self.assertIn("--crop basil --lat 13.75 --lon 100.5", cli["input"]["command"])
        self.assertLess(cli["started_ms"], cli["ended_ms"])
        self.assertEqual(len(parsed["cli_tool_results"]), 1)
        self.assertIn("Sources disagree:", parsed["cli_tool_results"][0])
        self.assertIn("crop-suitability", parsed["init"]["skills"])
        self.assertEqual(claude_code.canonical_models(parsed["result_event"]), ["claude-haiku-5-5"])

    def test_canonical_models_lists_every_model_by_its_canonical_id(self):
        event = {"modelUsage": {"claude-sonnet-5-5": {"canonicalModel": "claude-sonnet-5-5"},
                                "alias-x": {"canonicalModel": "claude-haiku-5-5"}}}
        self.assertEqual(claude_code.canonical_models(event), ["claude-haiku-5-5", "claude-sonnet-5-5"])
        self.assertEqual(claude_code.canonical_models({"modelUsage": {"claude-x-1": {}}}), ["claude-x-1"])
        self.assertEqual(claude_code.canonical_models(None), [])

    def test_a_bad_line_marks_the_stream_unparsed_but_keeps_the_rest(self):
        parsed = claude_code.parse(_sample_lines("exit0_unparseable.jsonl"))
        self.assertFalse(parsed["stream_parsed"])
        self.assertIsNotNone(parsed["result_event"])

    def test_cli_call_detection(self):
        for command, expected in [("python -m crop_advisor.cli --crop basil", True),
                                  ("cd /w && python3 -m crop_advisor.cli --crop basil", True),
                                  ("python -c 'import crop_advisor.cli'", False),
                                  ("cat crop_advisor/cli.py", False)]:
            with self.subTest(command=command):
                self.assertEqual(claude_code.is_cli_call({"name": "Bash", "input": {"command": command}}), expected)
        self.assertFalse(claude_code.is_cli_call({"name": "Read", "input": {"command": "python -m crop_advisor.cli"}}))

    def test_record_captures_status_inputs(self):
        parsed = claude_code.parse(_sample_lines("skill_ok.jsonl"))
        rec = claude_code.to_record(case={"id": "bangkok", "prompt": PROMPT}, eval_set={"version": "1", "sha256": "h"},
                                    trial_id="t", model="claude-haiku-5-5", condition="skill", cli_version_="2.1.301",
                                    run_result={"stderr": "warn\n", "exit_code": 0, "signal": None, "timed_out": False,
                                                "latency_ms": 5},
                                    parsed=parsed, skill_sha="abc", env={})
        si = rec["status_inputs"]
        self.assertEqual(set(si), {"exit_code", "signal", "timed_out", "harness_exception", "stderr_tail",
                                   "stream_parsed", "result_event", "cli_tool_results", "permission_denials",
                                   "bash_commands", "read_paths", "skill_calls", "listed_skills", "condition"})
        self.assertEqual(si["skill_calls"], ["crop-suitability"])
        self.assertEqual(si["listed_skills"], parsed["init"]["skills"])
        self.assertEqual(si["bash_commands"], [parsed["tool_calls"][1]["input"]["command"]])
        self.assertEqual((si["exit_code"], si["stderr_tail"], si["condition"]), (0, "warn\n", "skill"))
        self.assertEqual(si["result_event"], parsed["result_event"])
        self.assertEqual((rec["status"], rec["status_rule"]), ("ok", "11"))
        self.assertEqual(rec["subject"]["canonical_model"], "claude-haiku-5-5")
        self.assertEqual(rec["condition"], {"condition": "skill", "skill_sha256": "abc"})
        self.assertEqual(rec["output"]["answer"], parsed["result_event"]["result"])
        self.assertEqual(rec["usage"]["cost_usd"], parsed["result_event"]["total_cost_usd"])
        self.assertEqual(rec["usage"]["turns"], parsed["result_event"]["num_turns"])


class ReadPathsTest(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.ws, _ = claude_code.build_workspace(d.name, "skill")
        self.outside = str(Path(d.name).resolve() / "replay" / "13.75_100.5.json")

    def _resolve(self, *paths):
        return claude_code.read_paths([{"name": "Read", "input": {"file_path": p}} for p in paths], self.ws)

    def test_the_reference_file_is_allowed_through_either_skill_path(self):
        ref = "crop-suitability/reference/temperature-claims.md"
        paths = [f".agents/skills/{ref}", f".claude/skills/{ref}", str(self.ws / ".claude" / "skills" / ref),
                 str(self.ws / ".agents" / "skills" / ref)]
        resolved = self._resolve(*paths)
        self.assertEqual(set(resolved), {f".agents/skills/{ref}"})
        self.assertEqual(status.off_skill_reads(resolved), [])

    def test_reads_outside_the_skill_are_caught_however_they_are_spelled(self):
        escape = self.ws / ".agents" / "skills" / "crop-suitability" / "escape"
        escape.symlink_to(Path("..", "..", "..", "crop_advisor"))
        cases = {"crop_advisor/claims.py": "crop_advisor/claims.py",
                 str(self.ws / "data" / "ecocrop" / "basil.json"): "data/ecocrop/basil.json",
                 ".claude/skills/crop-suitability/../../../crop_advisor/claims.py": "crop_advisor/claims.py",
                 ".agents/skills/crop-suitability/escape/claims.py": "crop_advisor/claims.py",
                 ".agents/skills/crop-suitability/../../../../replay/13.75_100.5.json": self.outside,
                 self.outside: self.outside}
        for raw, expected in cases.items():
            with self.subTest(path=raw):
                resolved = self._resolve(raw)
                self.assertEqual(resolved, [expected])
                self.assertEqual(status.off_skill_reads(resolved), [expected])

    def test_only_read_calls_are_collected(self):
        calls = [{"name": "Bash", "input": {"command": "cat x"}}, {"name": "Read", "input": {"file_path": "x"}}]
        self.assertEqual(claude_code.read_paths(calls, self.ws), ["x"])

    def test_a_skill_call_without_a_skill_name_is_kept_and_off_contract(self):
        lines = _sample_lines("skill_ok.jsonl")
        odd = {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "s2", "name": "Skill", "input": {"command": "code-review"}}]}}
        parsed = claude_code.parse(lines[:-1] + [(0, json.dumps(odd) + "\n"), lines[-1]])
        rec = claude_code.to_record(case={"id": "bangkok", "prompt": PROMPT}, eval_set={}, trial_id="t",
                                    model="claude-haiku-5-5", condition="skill", cli_version_="2.1.301",
                                    run_result={"exit_code": 0}, parsed=parsed, env={}, workspace=self.ws)
        self.assertEqual(rec["status_inputs"]["skill_calls"], ["crop-suitability", ""])
        self.assertEqual((rec["status"], rec["status_rule"]), ("contract-violation", "9d"))

    def test_a_skill_trial_reading_crop_advisor_is_a_contract_violation(self):
        lines = _sample_lines("skill_ok.jsonl")
        read = {"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": "r1", "name": "Read",
             "input": {"file_path": str(self.ws / "crop_advisor" / "claims.py")}}]}}
        parsed = claude_code.parse(lines[:-1] + [(0, json.dumps(read) + "\n"), lines[-1]])
        rec = claude_code.to_record(case={"id": "bangkok", "prompt": PROMPT}, eval_set={}, trial_id="t",
                                    model="claude-haiku-5-5", condition="skill", cli_version_="2.1.301",
                                    run_result={"exit_code": 0}, parsed=parsed, env={}, workspace=self.ws)
        self.assertEqual(rec["status_inputs"]["read_paths"], ["crop_advisor/claims.py"])
        self.assertEqual((rec["status"], rec["status_rule"], rec["tags"]),
                         ("contract-violation", "9c", ["contract-violation", "off-contract-tool"]))


class RedactionTest(unittest.TestCase):
    def test_token_api_keys_and_bearer_values_are_redacted(self):
        text = "tok=SECRETVALUE key sk-ant-oat01-abc_DEF-123 hdr Authorization: Bearer abc.def"
        out = record.redact(text, {record.TOKEN_ENV: "SECRETVALUE"})
        for secret in ("SECRETVALUE", "sk-ant", "abc_DEF", "abc.def"):
            self.assertNotIn(secret, out)
        self.assertEqual(out.count(record.REDACTED), 3)

    def test_tail_is_capped_and_redacted_before_the_cut(self):
        secret = "sk-" + "a" * 100
        tail = record.stderr_tail("x" * 10000 + secret + "y" * (record.STDERR_TAIL_CHARS - 20), env={})
        self.assertLessEqual(len(tail), record.STDERR_TAIL_CHARS)
        self.assertNotIn("aaaa", tail)

    def test_the_record_stores_only_the_redacted_tail(self):
        rec = claude_code.to_record(case={"id": "c", "prompt": "p"}, eval_set={}, trial_id="t", model="m",
                                    condition="no-skill", cli_version_=None,
                                    run_result={"stderr": "Bearer zzz " + "e" * 5000, "exit_code": 1},
                                    parsed={}, env={})
        self.assertNotIn("zzz", rec["status_inputs"]["stderr_tail"])
        self.assertEqual(len(rec["status_inputs"]["stderr_tail"]), record.STDERR_TAIL_CHARS)


class SkillHashTest(unittest.TestCase):
    def test_hash_is_stable_and_changes_with_content(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "SKILL.md").write_text("a")
            first = record.skill_sha256(d)
            self.assertEqual(first, record.skill_sha256(d))
            (Path(d) / "SKILL.md").write_text("b")
            self.assertNotEqual(first, record.skill_sha256(d))

    def test_hash_changes_with_a_file_in_a_subfolder(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "SKILL.md").write_text("a")
            (Path(d) / "reference").mkdir()
            (Path(d) / "reference" / "claims.md").write_text("a")
            first = record.skill_sha256(d)
            (Path(d) / "reference" / "claims.md").write_text("b")
            self.assertNotEqual(first, record.skill_sha256(d))


class RunTrialTest(unittest.TestCase):
    def _trial(self, d, condition, **fake):
        _, case = cases.get("bangkok")
        with mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": d}):
            return claude_code.run_trial(case=case, eval_set={"version": "1"}, model="claude-haiku-5-5",
                                         condition=condition, budget_usd=0.5, timeout=fake.pop("timeout", 30),
                                         trial_id="t", binary=_fake_claude(d, **fake))

    def test_skill_trial_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            dump = Path(d) / "env.txt"
            rec = self._trial(d, "skill", stream="skill_ok.jsonl", env_dump=dump)
            seen = dump.read_text()
        self.assertEqual((rec["status"], rec["subject"]["cli_version"]), ("ok", "9.9.9"))
        self.assertEqual(rec["condition"]["skill_sha256"], record.skill_sha256(claude_code.SKILL_DIR))
        self.assertIn("DISABLE_AUTOUPDATER=1", seen)
        self.assertIn("CLAUDE_CODE_DISABLE_BUNDLED_SKILLS=1", seen)
        self.assertIn("CLAUDE_CODE_DISABLE_CLAUDE_MDS=1", seen)
        self.assertIn("CLAUDE_CODE_DISABLE_AUTO_MEMORY=1", seen)
        self.assertIn(f"{REPLAY_ENV}=/", seen)
        self.assertNotIn("CLAUDECODE=", seen)
        self.assertIn(".agents", seen)
        self.assertEqual(len(rec["status_inputs"]["cli_tool_results"]), 1)

    def test_read_paths_are_resolved_against_the_live_workspace(self):
        # The stand-in builds absolute Read paths from its own working directory, as Claude Code does.
        emit = ("import json, os; w = os.getcwd(); p = lambda e: print(json.dumps(e)); "
                "cli = \"python -m crop_advisor.cli --crop basil --lat 13.75 --lon 100.5\"; "
                "p({\"type\": \"assistant\", \"message\": {\"content\": ["
                "{\"type\": \"tool_use\", \"id\": \"b\", \"name\": \"Bash\", \"input\": {\"command\": cli}}, "
                "{\"type\": \"tool_use\", \"id\": \"r\", \"name\": \"Read\", "
                "\"input\": {\"file_path\": os.path.join(w, os.environ[\"READ_TARGET\"])}}]}}); "
                "p({\"type\": \"user\", \"message\": {\"content\": ["
                "{\"type\": \"tool_result\", \"tool_use_id\": \"b\", \"content\": \"Crop-Climate Advisor\"}]}}); "
                "p({\"type\": \"result\", \"subtype\": \"success\", \"result\": \"answer\", "
                "\"permission_denials\": []})")
        ref = ".claude/skills/crop-suitability/reference/temperature-claims.md"
        for target, expected in [(ref, ("ok", "11")), ("crop_advisor/claims.py", ("contract-violation", "9c"))]:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as d, \
                    mock.patch.dict(os.environ, {"READ_TARGET": target}):
                rec = self._trial(d, "skill", emit=emit)
                self.assertTrue(rec["output"]["tool_calls"][1]["input"]["file_path"].startswith("/"))
                self.assertEqual((rec["status"], rec["status_rule"]), expected, rec["status_reason"])
                self.assertEqual(rec["status_inputs"]["read_paths"],
                                 [".agents/skills/crop-suitability/reference/temperature-claims.md"
                                  if target == ref else target])

    def test_malformed_tool_inputs_are_recorded_not_raised(self):
        emit = ("import json; p = lambda e: print(json.dumps(e)); "
                "p({\"type\": \"system\", \"subtype\": \"init\", \"skills\": [\"crop-suitability\"]}); "
                "p({\"type\": \"assistant\", \"message\": {\"content\": ["
                "{\"type\": \"tool_use\", \"id\": \"t\", \"name\": \"%s\", \"input\": %s}]}}); "
                "p({\"type\": \"result\", \"subtype\": \"success\", \"result\": \"answer\", "
                "\"permission_denials\": []})")
        cases_ = [("Read", '{\"file_path\": 5}', "read_paths", "9c", "5"),
                  ("Skill", '{\"skill\": {\"name\": \"x\"}}', "skill_calls", "9d", "{'name': 'x'}")]
        for tool, payload, field, rule, stored in cases_:
            with self.subTest(tool=tool), tempfile.TemporaryDirectory() as d:
                rec = self._trial(d, "skill", emit=emit % (tool, payload))
                self.assertIsNone(rec["status_inputs"]["harness_exception"])
                self.assertEqual(rec["status_inputs"][field], [stored])
                self.assertEqual((rec["status"], rec["status_rule"]), ("contract-violation", rule))
                self.assertTrue(record.write(rec, Path(d) / "r.json").exists())

    def test_a_failure_while_building_the_record_still_returns_a_rule_1_record(self):
        real, calls = claude_code.to_record, []

        def first_call_fails(**kw):
            calls.append(kw)
            if len(calls) == 1:
                raise RuntimeError("boom")
            return real(**kw)

        with tempfile.TemporaryDirectory() as d, mock.patch.object(claude_code, "to_record", first_call_fails):
            rec = self._trial(d, "skill", stream="skill_ok.jsonl")
        self.assertEqual(len(calls), 2)
        self.assertIsNotNone(calls[0]["run_result"])
        self.assertEqual((calls[1]["parsed"], calls[1]["run_result"]), ({}, None))
        self.assertEqual((rec["status"], rec["status_rule"]), ("harness-error", "1"))
        self.assertIn("record build failed: RuntimeError: boom", rec["status_inputs"]["harness_exception"])
        self.assertEqual(rec["trial_id"], "t")

    def test_a_data_dependent_build_failure_is_not_repeated_by_the_fallback(self):
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(claude_code, "_skill", side_effect=ValueError("bad skill call")):
            rec = self._trial(d, "skill", stream="skill_ok.jsonl")
        self.assertEqual((rec["status"], rec["status_rule"]), ("harness-error", "1"))
        self.assertIn("record build failed: ValueError: bad skill call", rec["status_inputs"]["harness_exception"])
        self.assertEqual(rec["output"]["tool_calls"], [])

    def test_a_skill_hash_failure_is_a_rule_1_record(self):
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(record, "skill_sha256", side_effect=OSError("unreadable")):
            rec = self._trial(d, "skill", stream="skill_ok.jsonl")
        self.assertEqual((rec["status"], rec["status_rule"]), ("harness-error", "1"))
        self.assertIn("OSError: unreadable", rec["status_inputs"]["harness_exception"])

    def test_no_skill_trial_has_no_skill_hash(self):
        with tempfile.TemporaryDirectory() as d:
            rec = self._trial(d, "no-skill", stream="no_skill_ok.jsonl")
        self.assertEqual((rec["status"], rec["condition"]["skill_sha256"]), ("ok", None))

    def test_nonzero_exit_is_captured(self):
        with tempfile.TemporaryDirectory() as d:
            rec = self._trial(d, "no-skill", stream="api_404_bad_model.jsonl", exit_code=1)
        self.assertEqual((rec["status_inputs"]["exit_code"], rec["status_rule"]), (1, "4b"))

    def test_timeout_kills_the_subject_and_is_infra(self):
        with tempfile.TemporaryDirectory() as d:
            rec = self._trial(d, "no-skill", sleep=30, timeout=1)
        si = rec["status_inputs"]
        self.assertTrue(si["timed_out"])
        self.assertIsNone(si["exit_code"])
        self.assertIsNotNone(si["signal"])
        self.assertEqual((rec["status"], rec["status_rule"]), ("infra-error", "2"))
        self.assertLess(rec["usage"]["latency_ms"], 10000)

    def test_missing_binary_is_a_harness_error(self):
        _, case = cases.get("bangkok")
        rec = claude_code.run_trial(case=case, eval_set={"version": "1"}, model="claude-haiku-5-5",
                                    condition="no-skill", budget_usd=0.5, timeout=5, trial_id="t",
                                    binary="/nonexistent/claude")
        self.assertEqual((rec["status"], rec["status_rule"]), ("harness-error", "1"))


class TrialCommandTest(unittest.TestCase):
    def _main(self, args, rec=None):
        out, err = io.StringIO(), io.StringIO()
        fake = mock.patch.object(claude_code, "run_trial", return_value=rec) if rec else contextlib.nullcontext()
        with fake, contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = trial.main(args)
            except SystemExit as e:
                code = e.code
        return code, out.getvalue(), err.getvalue()

    def _record(self, canonical):
        return {"status": "ok", "status_rule": "11", "status_reason": "answered",
                "subject": {"canonical_models": canonical}}

    def test_an_unexpected_listed_skill_is_a_harness_error_named_on_stderr(self):
        parsed = claude_code.parse(_sample_lines("skill_ok_bundled_skills_on.jsonl"))
        rec = claude_code.to_record(case={"id": "bangkok", "prompt": PROMPT}, eval_set={}, trial_id="t",
                                    model="claude-haiku-5-5", condition="skill", cli_version_="2.1.301",
                                    run_result={"exit_code": 0}, parsed=parsed, env={})
        self.assertEqual((rec["status"], rec["status_rule"]), ("harness-error", "1b"))
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "r.json"
            code, stdout, err = self._main(["--adapter", "claude_code", "--case", "bangkok", "--model",
                                            "claude-haiku-5-5", "--condition", "skill", "--out", str(out)], rec)
            self.assertTrue(out.exists())
        self.assertEqual(code, 0)
        self.assertIn("harness-error (rule 1b", stdout)
        self.assertIn("code-review", err)
        self.assertIn("add them to DENIED_SKILLS in evals/status.py", err)

    def test_writes_the_record_and_exits_0(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "r.json"
            code, stdout, _ = self._main(["--adapter", "claude_code", "--case", "bangkok", "--model",
                                          "claude-sonnet-5-5", "--condition", "skill", "--out", str(out)],
                                         self._record(["claude-sonnet-5-5"]))
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(out.read_text())["status"], "ok")
            self.assertIn(str(out), stdout)

    def test_record_time_r3_stops_on_a_different_canonical_model(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "r.json"
            code, _, err = self._main(["--adapter", "claude_code", "--case", "bangkok", "--model",
                                       "claude-sonnet-5-5", "--condition", "no-skill", "--out", str(out)],
                                      self._record(["claude-opus-5-5"]))
            self.assertEqual(code, 1)
            self.assertIn("R3 stop", err)
            self.assertTrue(out.exists())

    def test_a_second_model_in_the_stream_stops_the_trial(self):
        lines = _sample_lines("no_skill_ok.jsonl")
        parsed = claude_code.parse(lines)
        parsed["result_event"]["modelUsage"]["alias-x"] = {"canonicalModel": "claude-opus-5-5"}
        rec = claude_code.to_record(case={"id": "bangkok", "prompt": PROMPT}, eval_set={}, trial_id="t",
                                    model="claude-haiku-5-5", condition="no-skill", cli_version_="2.1.301",
                                    run_result={"exit_code": 0}, parsed=parsed, env={})
        self.assertEqual(rec["subject"]["canonical_models"], ["claude-haiku-5-5", "claude-opus-5-5"])
        with tempfile.TemporaryDirectory() as d:
            code, _, err = self._main(["--adapter", "claude_code", "--case", "bangkok", "--model",
                                       "claude-haiku-5-5", "--condition", "no-skill",
                                       "--out", str(Path(d) / "r.json")], rec)
        self.assertEqual(code, 1)
        self.assertIn("R3 stop", err)
        self.assertIn("claude-opus-5-5", err)

    def test_an_ok_record_with_no_reported_model_stops_the_trial(self):
        parsed = claude_code.parse(_sample_lines("skill_ok.jsonl"))
        parsed["result_event"]["modelUsage"] = {}
        rec = claude_code.to_record(case={"id": "bangkok", "prompt": PROMPT}, eval_set={}, trial_id="t",
                                    model="claude-haiku-5-5", condition="skill", cli_version_="2.1.301",
                                    run_result={"exit_code": 0}, parsed=parsed, env={})
        self.assertEqual((rec["status"], rec["subject"]["canonical_models"]), ("ok", []))
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "r.json"
            code, _, err = self._main(["--adapter", "claude_code", "--case", "bangkok", "--model",
                                       "claude-haiku-5-5", "--condition", "skill", "--out", str(out)], rec)
            self.assertTrue(out.exists())
        self.assertEqual(code, 1)
        self.assertIn("R3 stop", err)
        self.assertIn("no model", err)

    def test_a_non_ok_record_with_no_reported_model_does_not_stop(self):
        rec = {"status": "harness-error", "status_rule": "4b", "status_reason": "x",
               "subject": {"canonical_models": []}}
        with tempfile.TemporaryDirectory() as d:
            code, _, err = self._main(["--adapter", "claude_code", "--case", "bangkok", "--model",
                                       "claude-sonnet-5-5", "--condition", "skill",
                                       "--out", str(Path(d) / "r.json")], rec)
        self.assertEqual(code, 0)
        self.assertNotIn("R3 stop", err)

    def test_the_judge_model_is_refused_as_a_subject(self):
        code, _, err = self._main(["--adapter", "claude_code", "--case", "bangkok", "--model",
                                   "claude-opus-5-5", "--condition", "skill"])
        self.assertEqual(code, 2)
        self.assertIn("not a configured subject", err)

    def test_timeout_must_be_positive(self):
        for value in ("0", "-1", "nan", "inf"):
            with self.subTest(timeout=value), tempfile.TemporaryDirectory() as d:
                # A stand-in run_trial, so a broken check can never start a live trial.
                code, _, err = self._main(["--adapter", "claude_code", "--case", "bangkok", "--model",
                                           "claude-sonnet-5-5", "--condition", "skill", "--timeout", value,
                                           "--out", str(Path(d) / "r.json")], self._record(["claude-sonnet-5-5"]))
                self.assertEqual(code, 2)
                self.assertIn("--timeout must be a finite number greater than 0", err)

    def test_timeout_is_passed_through_or_taken_from_config(self):
        for given, expected in ((None, 300), ("0.5", 0.5), ("42", 42.0)):
            with self.subTest(timeout=given), tempfile.TemporaryDirectory() as d, \
                    mock.patch.object(claude_code, "run_trial", return_value=self._record(["claude-sonnet-5-5"])) as run:
                args = ["--adapter", "claude_code", "--case", "bangkok", "--model", "claude-sonnet-5-5",
                        "--condition", "skill", "--out", str(Path(d) / "r.json")]
                with contextlib.redirect_stdout(io.StringIO()):
                    code = trial.main(args + (["--timeout", given] if given else []))
            self.assertEqual(code, 0)
            self.assertEqual(run.call_args.kwargs["timeout"], expected)

    def test_rule_1c_is_named_on_stderr(self):
        rec = {"status": "harness-error", "status_rule": "1c",
               "status_reason": "the no-skill subject was offered crop-suitability", "subject": {"canonical_models": []}}
        with tempfile.TemporaryDirectory() as d:
            code, _, err = self._main(["--adapter", "claude_code", "--case", "bangkok", "--model",
                                       "claude-sonnet-5-5", "--condition", "no-skill",
                                       "--out", str(Path(d) / "r.json")], rec)
        self.assertEqual(code, 0)
        self.assertIn("harness-error: the no-skill subject was offered crop-suitability", err)

    def test_unknown_case_is_refused(self):
        code, _, err = self._main(["--adapter", "claude_code", "--case", "paris", "--model",
                                   "claude-sonnet-5-5", "--condition", "skill"])
        self.assertEqual(code, 2)
        self.assertIn("no case 'paris'", err)


if __name__ == "__main__":
    unittest.main()
