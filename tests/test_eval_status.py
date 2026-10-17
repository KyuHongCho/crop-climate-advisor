"""classify_status: the ordered rules, the pinned overlaps, and every stored stream-json sample."""
import json
import unittest
from pathlib import Path

from evals.adapters import claude_code
from evals.record import status_inputs
from evals.status import classify_status

SAMPLES_DIR = Path(__file__).resolve().parent / "fixtures" / "claude_code"
SAMPLES = json.loads((SAMPLES_DIR / "samples.json").read_text(encoding="utf-8"))["samples"]

PINNED_OVERLAPS = {
    "skill_ok_bundled_skills_on": "1b",
    "skill_not_offered": "1c",
    "timeout_truncated": "2",
    "runaway_budget": "3",
    "runaway_turns": "3",
    "api_404_bad_model": "4b",
    "api_429_rate_limit": "4a",
    "error_during_execution": "4c",
    "replay_missing_model_answers": "5",
    "nonzero_no_result_unrecognised_stderr": "8",
    "skill_denied_no_cli_call": "9",
    "skill_read_only_bash_no_denial": "9b",
    "skill_read_outside_skill": "9c",
    "skill_denied_other_skill": "9",
    "skill_other_skill_ran": "9d",
    "exit0_null_result": "11",
}


def sample_record(sample):
    stream = sample["stream"]
    lines = (SAMPLES_DIR / stream).read_text(encoding="utf-8").splitlines(keepends=True) if stream else []
    parsed = claude_code.parse([(0, line) for line in lines]) if stream else {}
    run_result = {"stderr": sample["stderr"], "exit_code": sample["exit_code"], "signal": sample["signal"],
                  "timed_out": sample["timed_out"], "latency_ms": 0}
    case = {"id": "bangkok", "prompt": "p"}
    return claude_code.to_record(case=case, eval_set={"version": "1"}, trial_id=sample["name"],
                                 model="claude-haiku-5-5", condition=sample["condition"], cli_version_="2.1.301",
                                 run_result=run_result, parsed=parsed, harness_exception=sample["harness_exception"],
                                 env={})


def _si(**overrides):
    base = dict(condition="no-skill", exit_code=0, stream_parsed=True,
                result_event={"type": "result", "subtype": "success", "result": "an answer"})
    base.update(overrides)
    return status_inputs(**base)


class StoredSamplesTest(unittest.TestCase):
    def test_each_sample_gets_its_recorded_status_and_rule(self):
        for sample in SAMPLES:
            with self.subTest(sample=sample["name"]):
                rec = sample_record(sample)
                self.assertEqual((rec["status"], rec["status_rule"]),
                                 (sample["expected_status"], sample["expected_rule"]), rec["status_reason"])

    def test_every_record_status_is_recomputable_from_status_inputs_alone(self):
        for sample in SAMPLES:
            with self.subTest(sample=sample["name"]):
                rec = json.loads(json.dumps(sample_record(sample)))
                self.assertEqual(rec["status"], classify_status(rec["status_inputs"]).status)

    def test_the_pinned_overlaps_are_stored(self):
        by_name = {s["name"]: s for s in SAMPLES}
        for name, rule in PINNED_OVERLAPS.items():
            with self.subTest(sample=name):
                self.assertEqual(by_name[name]["expected_rule"], rule)
        self.assertTrue(by_name["exit0_null_result"]["synthesised"])

    def test_every_rule_has_a_sample(self):
        self.assertEqual({s["expected_rule"] for s in SAMPLES},
                         {"1", "1b", "1c", "2", "3", "4a", "4b", "4c", "5", "6", "7", "8", "9", "9b", "9c", "9d", "10", "11"})

    def test_synthesised_samples_are_labelled(self):
        for sample in SAMPLES:
            with self.subTest(sample=sample["name"]):
                self.assertEqual(sample["synthesised"], sample["provenance"].startswith("synthesised"))


class RuleOrderTest(unittest.TestCase):
    def test_timeout_wins_over_a_runaway_subtype(self):
        si = _si(timed_out=True, exit_code=None, signal=9,
                 result_event={"subtype": "error_max_budget_usd", "result": None})
        self.assertEqual(classify_status(si).rule, "2")

    def test_runaway_wins_over_its_nonzero_exit(self):
        si = _si(exit_code=1, result_event={"subtype": "error_max_turns", "result": None})
        self.assertEqual(classify_status(si)[:2], ("runaway", "3"))

    def test_api_status_classes(self):
        cases = {408: ("infra-error", "4a"), 429: ("infra-error", "4a"), 503: ("infra-error", "4a"),
                 529: ("infra-error", "4a"), 401: ("harness-error", "4b"), 403: ("harness-error", "4b"),
                 404: ("harness-error", "4b")}
        for http, expected in cases.items():
            with self.subTest(http=http):
                si = _si(exit_code=1, result_event={"subtype": "success", "is_error": True,
                                                    "api_error_status": http, "result": "x"})
                self.assertEqual(classify_status(si)[:2], expected)

    def test_overload_text_without_a_status_is_infra(self):
        si = _si(exit_code=1, result_event={"subtype": "error_during_execution",
                                            "errors": ["overloaded_error: Overloaded"]})
        self.assertEqual(classify_status(si)[:2], ("infra-error", "4a"))

    def test_api_error_in_stderr_alone_matches_rule_4(self):
        si = _si(exit_code=1, result_event=None, stderr_tail="API Error: 529 overloaded")
        self.assertEqual(classify_status(si)[:2], ("infra-error", "4a"))
        si = _si(exit_code=1, result_event=None, stderr_tail="API Error: 401 invalid x-api-key")
        self.assertEqual(classify_status(si)[:2], ("harness-error", "4b"))

    def test_api_error_text_in_stderr_does_not_void_a_successful_answer(self):
        success = {"subtype": "success", "is_error": False, "result": "x"}
        for stderr in ("API Error: 529 overloaded, retrying", "rate_limit_error: slow down"):
            with self.subTest(stderr=stderr):
                self.assertEqual(classify_status(_si(result_event=success, stderr_tail=stderr))[:2], ("ok", "11"))
                skill = _si(condition="skill", cli_tool_results=["out"], result_event=success, stderr_tail=stderr)
                self.assertEqual(classify_status(skill)[:2], ("ok", "11"))
        failed = _si(exit_code=1, result_event={"subtype": "success", "is_error": True, "result": None},
                     stderr_tail="API Error: 401 invalid x-api-key")
        self.assertEqual(classify_status(failed)[:2], ("harness-error", "4b"))
        no_result = _si(exit_code=1, result_event=None, stderr_tail="API Error: 529 overloaded")
        self.assertEqual(classify_status(no_result)[:2], ("infra-error", "4a"))
        reported = _si(exit_code=1, result_event={"subtype": "success", "is_error": False, "api_error_status": 529,
                                                  "result": "x"})
        self.assertEqual(classify_status(reported)[:2], ("infra-error", "4a"))

    def test_an_unknown_error_subtype_is_a_harness_error(self):
        si = _si(exit_code=1, result_event={"subtype": "error_something_new", "result": None})
        self.assertEqual(classify_status(si)[:2], ("harness-error", "4c"))

    def test_replay_missing_wins_over_a_network_failure_text(self):
        si = _si(condition="skill", cli_tool_results=["CROP_ADVISOR_REPLAY_MISSING: ... URLError"])
        self.assertEqual(classify_status(si).rule, "5")

    def test_socket_timeout_in_cli_output_is_infra(self):
        si = _si(condition="skill", cli_tool_results=["socket.timeout: The read operation timed out"])
        self.assertEqual(classify_status(si)[:2], ("infra-error", "6"))

    def test_exit_0_without_a_parsed_stream_is_a_contract_violation(self):
        self.assertEqual(classify_status(_si(stream_parsed=False)).rule, "7")
        self.assertEqual(classify_status(_si(result_event=None)).rule, "7")

    def test_nonzero_exit_with_a_plain_result_is_the_catch_all(self):
        si = _si(exit_code=1, result_event={"subtype": "success", "is_error": True, "result": "Not logged in"})
        self.assertEqual(classify_status(si)[:2], ("harness-error", "8"))

    def test_denials_by_condition(self):
        denial = [{"tool_name": "Bash", "tool_input": {"command": "ls"}}]
        skill = classify_status(_si(condition="skill", permission_denials=denial, cli_tool_results=["ok"]))
        self.assertEqual((skill.status, skill.rule, skill.tags), ("contract-violation", "9", ("contract-violation", "denied-tool")))
        base = classify_status(_si(permission_denials=denial))
        self.assertEqual((base.status, base.tags), ("ok", ("denied-tool",)))

    def test_skill_condition_needs_a_cli_call_and_no_skill_does_not(self):
        missed = classify_status(_si(condition="skill"))
        self.assertEqual((missed.status, missed.rule, missed.tags), ("contract-violation", "10", ("contract-violation", "no-tool-call")))
        self.assertEqual(classify_status(_si(condition="skill", cli_tool_results=["Crop-Climate Advisor"])).status, "ok")
        self.assertEqual(classify_status(_si()).status, "ok")

    def test_skill_condition_non_cli_bash_is_a_contract_violation(self):
        cli = "python -m crop_advisor.cli --crop basil --lat 13.75 --lon 100.5 --place \"Bangkok\""
        for command in ("ls -la", "cat data/ecocrop/basil.json", f"cd /w && {cli}", f"{cli} | head",
                        f"{cli} 2>&1", 'python -m crop_advisor.cli --place "$(id)"',
                        "python -m crop_advisor.cli\ncat crop_advisor/claims.py",
                        "python -m crop_advisor.cli \n cat /etc/hosts",
                        "python -m crop_advisor.cli\r\ncat x", "python -m crop_advisor.cli --crop basil\rcat x",
                        f"{cli}\nls", "ls\n" + cli, f"{cli}; ls", f"{cli} & ls", f"{cli} > out.txt",
                        f"{cli} < in.txt",
                        "python -m crop_advisor.cli --place `id`"):
            with self.subTest(command=command):
                got = classify_status(_si(condition="skill", cli_tool_results=["out"], bash_commands=[cli, command]))
                self.assertEqual((got.status, got.rule, got.tags),
                                 ("contract-violation", "9b", ("contract-violation", "off-contract-tool")))
        for command in (cli, "python3 -m crop_advisor.cli --crop basil --lat 6.52 --lon 3.38"):
            with self.subTest(command=command):
                self.assertEqual(classify_status(_si(condition="skill", cli_tool_results=["out"],
                                                     bash_commands=[command])).status, "ok")

    def test_non_cli_bash_rule_order(self):
        denial = [{"tool_name": "Bash", "tool_input": {"command": "touch x"}}]
        self.assertEqual(classify_status(_si(condition="skill", bash_commands=["touch x"],
                                             permission_denials=denial)).rule, "9")
        self.assertEqual(classify_status(_si(condition="skill", bash_commands=["ls"])).rule, "9b")
        self.assertEqual(classify_status(_si(bash_commands=["ls"])).status, "ok")
        si = _si(condition="skill", cli_tool_results=["out"])
        del si["bash_commands"]
        self.assertEqual(classify_status(si).status, "ok")

    def test_skill_condition_read_outside_the_skill_is_a_contract_violation(self):
        allowed = [".agents/skills/crop-suitability/reference/temperature-claims.md",
                   ".claude/skills/crop-suitability/reference/temperature-claims.md",
                   ".agents/skills/crop-suitability/SKILL.md"]
        self.assertEqual(classify_status(_si(condition="skill", cli_tool_results=["out"],
                                             read_paths=allowed)).status, "ok")
        for path in ("crop_advisor/claims.py", "data/ecocrop/basil.json", "/etc/hosts",
                     ".agents/skills/crop-suitability/../../../crop_advisor/claims.py", ".agents/skillsX/a", "."):
            with self.subTest(path=path):
                got = classify_status(_si(condition="skill", cli_tool_results=["out"], read_paths=allowed + [path]))
                self.assertEqual((got.status, got.rule, got.tags),
                                 ("contract-violation", "9c", ("contract-violation", "off-contract-tool")))

    def test_read_rule_order(self):
        denial = [{"tool_name": "Read", "tool_input": {"file_path": "/etc/hosts"}}]
        self.assertEqual(classify_status(_si(condition="skill", read_paths=["/etc/hosts"],
                                             permission_denials=denial)).rule, "9")
        self.assertEqual(classify_status(_si(condition="skill", bash_commands=["ls"], read_paths=["x"])).rule, "9b")
        self.assertEqual(classify_status(_si(condition="skill", read_paths=["x"])).rule, "9c")
        self.assertEqual(classify_status(_si(read_paths=["x"])).status, "ok")
        si = _si(condition="skill", cli_tool_results=["out"])
        del si["read_paths"]
        self.assertEqual(classify_status(si).status, "ok")

    def test_skill_condition_other_skill_calls_are_a_contract_violation(self):
        ok = _si(condition="skill", cli_tool_results=["out"], skill_calls=["crop-suitability"])
        self.assertEqual(classify_status(ok).status, "ok")
        for other in ("plugin-authoring", "doctor", "code-review", "cc-plugin-plugin-authoring:plugin-authoring", "",
                      "x:crop-suitability", "crop-suitability-x", " crop-suitability"):
            with self.subTest(skill=other):
                got = classify_status(_si(condition="skill", cli_tool_results=["out"],
                                          skill_calls=["crop-suitability", other]))
                self.assertEqual((got.status, got.rule, got.tags),
                                 ("contract-violation", "9d", ("contract-violation", "off-contract-tool")))

    def test_skill_rule_order(self):
        denial = [{"tool_name": "Skill", "tool_input": {"skill": "plugin-authoring"}}]
        denied = classify_status(_si(condition="skill", skill_calls=["plugin-authoring"], permission_denials=denial))
        self.assertEqual((denied.rule, denied.tags), ("9", ("contract-violation", "denied-tool")))
        self.assertEqual(classify_status(_si(condition="skill", read_paths=["x"], skill_calls=["doctor"])).rule, "9c")
        self.assertEqual(classify_status(_si(condition="skill", skill_calls=["doctor"])).rule, "9d")
        self.assertEqual(classify_status(_si(skill_calls=["doctor"])).status, "ok")
        si = _si(condition="skill", cli_tool_results=["out"])
        del si["skill_calls"]
        self.assertEqual(classify_status(si).status, "ok")

    def test_an_extra_listed_skill_makes_the_skill_trial_a_harness_error(self):
        listed = ["crop-suitability", "design", "doctor", "plugin-authoring"]
        ok = _si(condition="skill", cli_tool_results=["out"], listed_skills=listed)
        self.assertEqual(classify_status(ok).status, "ok")
        got = classify_status(_si(condition="skill", cli_tool_results=["out"], listed_skills=listed + ["code-review"]))
        self.assertEqual((got.status, got.rule, got.tags), ("harness-error", "1b", ("harness-error",)))
        self.assertIn("code-review", got.reason)
        self.assertIn("add them to DENIED_SKILLS in evals/status.py", got.reason)
        # The no-skill arm ignores extra skills (1b); offering it crop-suitability is rule 1c, tested below.
        self.assertEqual(classify_status(_si(listed_skills=["design", "code-review"])).status, "ok")

    def test_no_init_event_does_not_trip_the_listed_skills_rules(self):
        self.assertEqual(classify_status(_si(condition="skill", cli_tool_results=["out"], listed_skills=None)).status,
                         "ok")
        # An empty list is an init event that offered nothing, so the Skill was not offered (rule 1c).
        got = classify_status(_si(condition="skill", cli_tool_results=["out"], listed_skills=[]))
        self.assertEqual((got.status, got.rule), ("harness-error", "1c"))
        si = _si(condition="skill", cli_tool_results=["out"])
        del si["listed_skills"]
        self.assertEqual(classify_status(si).status, "ok")

    def test_listed_skills_rule_order(self):
        extra = ["crop-suitability", "code-review"]
        self.assertEqual(classify_status(_si(condition="skill", harness_exception="x", listed_skills=extra)).rule, "1")
        for overrides in ({"timed_out": True, "exit_code": None},
                          {"exit_code": 1, "result_event": {"subtype": "error_max_turns", "result": None}},
                          {"bash_commands": ["ls"]}, {"skill_calls": ["doctor"]}):
            with self.subTest(**{k: str(v) for k, v in overrides.items()}):
                self.assertEqual(classify_status(_si(condition="skill", listed_skills=extra, **overrides)).rule, "1b")

    def test_skill_not_offered_is_a_harness_error(self):
        got = classify_status(_si(condition="skill", cli_tool_results=["out"],
                                  listed_skills=["design", "doctor", "plugin-authoring"]))
        self.assertEqual((got.status, got.rule, got.reason, got.tags),
                         ("harness-error", "1c", "the subject was not offered crop-suitability", ("harness-error",)))
        self.assertEqual(classify_status(_si(listed_skills=["design"])).status, "ok")
        offered = _si(condition="skill", listed_skills=["crop-suitability", "design"])
        self.assertEqual(classify_status(offered).rule, "10")

    def test_no_skill_subject_offered_the_skill_is_a_harness_error(self):
        got = classify_status(_si(listed_skills=["crop-suitability", "design"]))
        self.assertEqual((got.status, got.rule, got.reason, got.tags),
                         ("harness-error", "1c", "the no-skill subject was offered crop-suitability", ("harness-error",)))
        for listed in (None, [], ["design", "doctor", "plugin-authoring"]):
            with self.subTest(listed=listed):
                self.assertEqual(classify_status(_si(listed_skills=listed)).status, "ok")
        si = _si()
        del si["listed_skills"]
        self.assertEqual(classify_status(si).status, "ok")

    def test_no_skill_1c_rule_order(self):
        offered = ["crop-suitability"]
        self.assertEqual(classify_status(_si(harness_exception="x", listed_skills=offered)).rule, "1")
        for overrides in ({"timed_out": True, "exit_code": None},
                          {"exit_code": 1, "result_event": {"subtype": "error_max_turns", "result": None}}):
            with self.subTest(**{k: str(v) for k, v in overrides.items()}):
                self.assertEqual(classify_status(_si(listed_skills=offered, **overrides)).rule, "1c")

    def test_an_error_result_event_is_not_an_answer(self):
        for condition, extra in (("no-skill", {}), ("skill", {"cli_tool_results": ["out"]})):
            with self.subTest(condition=condition):
                got = classify_status(_si(condition=condition, result_event={
                    "subtype": "success", "is_error": True, "result": "API Error: rate limit"}, **extra))
                self.assertEqual((got.status, got.rule, got.reason), ("harness-error", "11", "result event is an error"))
                ok = classify_status(_si(condition=condition, result_event={
                    "subtype": "success", "is_error": False, "result": "answer"}, **extra))
                self.assertEqual(ok[:2], ("ok", "11"))

    def test_skill_not_offered_rule_order(self):
        missing = ["design"]
        self.assertEqual(classify_status(_si(condition="skill", harness_exception="x", listed_skills=missing)).rule, "1")
        self.assertEqual(classify_status(_si(condition="skill", listed_skills=["code-review"])).rule, "1b")
        for overrides in ({"timed_out": True, "exit_code": None},
                          {"exit_code": 1, "result_event": {"subtype": "error_max_turns", "result": None}}):
            with self.subTest(**{k: str(v) for k, v in overrides.items()}):
                self.assertEqual(classify_status(_si(condition="skill", listed_skills=missing, **overrides)).rule, "1c")

    def test_a_harness_exception_wins_over_everything(self):
        si = _si(harness_exception="RuntimeError: x", timed_out=True)
        self.assertEqual(classify_status(si)[:2], ("harness-error", "1"))


if __name__ == "__main__":
    unittest.main()
