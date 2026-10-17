"""Deterministic graders D1-D4: a passing and a failing answer for each, the triage tags, and `evals.score`."""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evals import cases, graders, score
from evals.adapters import claude_code


def setUpModule():
    # The username rewrite is machine-dependent, so pin a name no fixture word can collide with.
    patcher = mock.patch.dict(os.environ, {"USER": "testuser", "LOGNAME": "testuser", "USERNAME": "testuser"})
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "tests" / "fixtures" / "claude_code"
BANGKOK = cases.get("bangkok")[1]
CLI = 'python -m crop_advisor.cli --crop basil --lat 13.75 --lon 100.5 --place "Bangkok"'

GOOD = """Basil is marginal outdoors in Bangkok (27.85 °C annual mean).

| Source | Optimal window | Condition |
|---|---|---|
| FAO ECOCROP (id 1547) | 18–27 °C | none stated |
| Chang, Alderson & Wright (2005) | 25–30 °C | at a daily light integral (DLI) of 20–22 mol·m⁻²·d⁻¹ |
| Walters & Currey (2019) | 29–35 °C | at a DLI of 19.5 mol·m⁻²·d⁻¹ |

The sources point in opposite directions here, so I cannot call Bangkok optimal."""

# Names every source and condition, but 25-35 is Bangkok's climate range, not any source's optimal window.
CLIMATE_RANGE = """Bangkok's monthly means run 25–35 °C through the year. FAO ECOCROP, Chang et al. and Walters & Currey
disagree on the optimum; the studies hold a DLI of 20–22 and 19.5 mol·m⁻²·d⁻¹ respectively."""

# Remembered figures, as a no-tool answer gives them: no source, no source band, no condition.
FROM_MEMORY = "Basil grows very well in Bangkok. The ideal growing range is about 20-30 °C, best at 24-30 °C."


def tool_call(command):
    return {"name": "Bash", "input": {"command": command}, "result": "out"}


def make_record(answer, condition="skill", commands=(CLI,), status="ok", rule="11", tags=()):
    return {"trial_id": "t", "case_id": "bangkok", "status": status, "status_rule": rule, "status_reason": "r",
            "tags": list(tags), "condition": {"condition": condition},
            "output": {"answer": answer, "tool_calls": [tool_call(c) for c in commands]}}


def verdicts(result):
    return {g.criterion: g.verdict for g in result["criteria"]}


def grade_answer(answer, **kw):
    return graders.grade(make_record(answer, **kw), BANGKOK)


class ClaimsTest(unittest.TestCase):
    def test_every_basil_claim_has_a_source_name_and_a_readable_condition(self):
        found = graders.basil_claims()
        self.assertEqual([c.source.split(" (")[0] for c in found], list(graders.SOURCE_NAMES))
        self.assertEqual([c.condition for c in found if c.condition],
                         ["DLI 20–22 mol·m⁻²·d⁻¹", "DLI 19.5 mol·m⁻²·d⁻¹"])
        for c in found:
            if c.condition:
                graders._dli_figure(c.condition)


class D1SourcesTest(unittest.TestCase):
    def test_all_three_sources_named_passes_with_evidence_spans(self):
        g = grade_answer(GOOD)["criteria"][0]
        self.assertEqual((g.criterion, g.verdict, g.tags), ("D1", "pass", ()))
        self.assertEqual([GOOD[e["start"]:e["end"]] for e in g.evidence], ["ECOCROP", "Chang", "Currey"])

    def test_a_missing_source_fails_and_is_tagged(self):
        g = grade_answer("FAO EcoCrop and Chang et al. (2005) give 18-27 and 25-30.")["criteria"][0]
        self.assertEqual((g.verdict, g.tags), ("fail", ("missing-source",)))
        self.assertIn("Walters & Currey (2019)", g.reason)
        self.assertEqual(len(g.evidence), 2)

    def test_the_paper_both_figures_were_read_in_is_not_a_source(self):
        via = "ECOCROP says 18–27. Chang's figure was read via Walters, Tarr & Lopez (2023); see Walters et al."
        g = grade_answer(via)["criteria"][0]
        self.assertEqual(g.verdict, "fail")
        self.assertEqual(g.reason, "not named: Walters & Currey (2019)")
        self.assertEqual(grade_answer(via + " Currey's window is hotter.")["criteria"][0].verdict, "pass")

    def test_names_inside_other_words_do_not_count(self):
        for text in ("Changeable weather", "ECOCROPS", "Curreys", "Interchang", "METECOCROP", "Mcurrey"):
            with self.subTest(text=text):
                g = grade_answer(text)["criteria"][0]
                self.assertEqual((g.verdict, g.evidence), ("fail", ()))
                self.assertEqual(g.reason, "not named: " + ", ".join(c.source for c in graders.basil_claims()))


class D2BandsTest(unittest.TestCase):
    def test_all_three_bands_stated_passes(self):
        g = grade_answer(GOOD)["criteria"][1]
        self.assertEqual((g.verdict, g.tags), ("pass", ()))
        self.assertEqual([e["text"] for e in g.evidence], ["18–27", "25–30", "29–35"])

    def test_common_spellings_of_a_band_are_accepted(self):
        for text in ("18-27", "18 – 27 °C", "18°C to 27°C", "18—27 C", "between 18 and 27 degrees", "18−27",
                     "18.0 to 27.0", "18 to 27 degrees Celsius", "(18–27)"):
            with self.subTest(text=text):
                self.assertTrue(graders.range_pattern(18, 27).search(text))

    def test_a_climate_range_is_not_an_optimal_window(self):
        result = grade_answer(CLIMATE_RANGE)
        g = result["criteria"][1]
        self.assertEqual((g.verdict, g.tags), ("fail", ("missing-band",)))
        self.assertEqual(g.evidence, ())
        self.assertEqual(verdicts(result)["D1"], "pass")
        self.assertEqual(verdicts(result)["D3"], "pass")

    def test_near_misses_are_not_bands(self):
        for text in ("118–270", "118–27", "3.18–27", "18.5–27", "18–27.5", "18 27", "18 and 27", "18–2", "18, 27",
                     "5–35"):
            with self.subTest(text=text):
                self.assertFalse(graders.range_pattern(18, 27).search(text))
        self.assertFalse(graders.range_pattern(25, 30).search("25–35 °C"))
        self.assertFalse(graders.range_pattern(29, 35).search("25–35 °C"))

    def test_one_missing_band_fails_and_names_it(self):
        g = grade_answer("ECOCROP 18–27, Chang 25–30 and Currey 30–35.")["criteria"][1]
        self.assertEqual(g.verdict, "fail")
        self.assertIn("Walters & Currey (2019) 29-35", g.reason)
        self.assertNotIn("FAO", g.reason)


class D3ConditionsTest(unittest.TestCase):
    def test_both_dli_conditions_stated_passes(self):
        g = grade_answer(GOOD)["criteria"][2]
        self.assertEqual((g.verdict, g.tags), ("pass", ()))
        self.assertEqual([e["text"] for e in g.evidence], ["20–22", "19.5"])

    def test_one_condition_fails(self):
        g = grade_answer("Chang's window holds at a DLI of 20-22 mol per square metre.")["criteria"][2]
        self.assertEqual((g.verdict, g.tags), ("fail", ("missing-condition",)))
        self.assertIn("Walters & Currey", g.reason)
        self.assertNotIn("Chang", g.reason)

    def test_a_figure_with_no_light_integral_beside_it_is_not_a_condition(self):
        g = grade_answer("Nights of 20-22 °C and a 19.5 percent chance of rain; " + "padding. " * 30 + "DLI.")
        self.assertEqual(g["criteria"][2].verdict, "fail")

    def test_each_light_integral_marker_counts_on_its_own(self):
        for text in ("Held at 20–22 mol·m⁻²·d⁻¹ for Chang; Currey used 19.5 mol·m⁻²·d⁻¹.",
                     "At a daily light integral of 20–22 for Chang and of 19.5 for Currey.",
                     "At a DLI of 20–22 for Chang and of 19.5 for Currey."):
            with self.subTest(text=text):
                self.assertEqual(grade_answer(text)["criteria"][2].verdict, "pass")

    def test_the_marker_may_follow_the_figure(self):
        text = "Held at 20–22 mol·m⁻²·d⁻¹ for Chang; Currey used 19.5 DLI."
        self.assertEqual(grade_answer(text)["criteria"][2].verdict, "pass")


class D4CliCallTest(unittest.TestCase):
    def test_the_cli_run_with_the_case_coordinates_passes(self):
        g = grade_answer(GOOD)["criteria"][3]
        self.assertEqual((g.verdict, g.tags, g.evidence[0]["text"]), ("pass", (), CLI))

    def test_argument_spellings_and_extra_calls(self):
        for command in ("cd /w && python3 -m crop_advisor.cli --crop basil --lat 13.750 --lon 100.50",
                        "python -m crop_advisor.cli --crop=basil --lat=13.75 --lon=100.5 --place 'Bangkok'"):
            with self.subTest(command=command):
                self.assertEqual(grade_answer(GOOD, commands=["ls", command])["criteria"][3].verdict, "pass")

    def test_no_cli_call_fails_with_the_no_tool_call_tag(self):
        for commands in ((), ("ls -la",)):
            with self.subTest(commands=commands):
                g = grade_answer(GOOD, commands=commands)["criteria"][3]
                self.assertEqual((g.verdict, g.tags, g.reason), ("fail", ("no-tool-call",), "no CLI call was made"))

    def test_a_cli_call_with_the_wrong_arguments_gets_the_wrong_cli_args_tag(self):
        for command in ("python -m crop_advisor.cli --crop basil --lat 6.52 --lon 3.38",
                        "python -m crop_advisor.cli --crop lettuce --lat 13.75 --lon 100.5",
                        "python -m crop_advisor.cli --crop basil --lat 13.75",
                        "python -m crop_advisor.cli --crop basil --lon 100.5",
                        "python -m crop_advisor.cli --crop basil",
                        "python -m crop_advisor.cli --crop basil --lat 13.7 --lon 100.5",
                        "python -m crop_advisor.cli --crop basil --lat x --lon 100.5"):
            with self.subTest(command=command):
                g = grade_answer(GOOD, commands=[command])["criteria"][3]
                self.assertEqual((g.verdict, g.tags), ("fail", ("wrong-cli-args",)))
                self.assertIn("--lat 13.75 --lon 100.5", g.reason)
                self.assertEqual(g.evidence[0]["text"], command)

    def test_no_cli_call_and_a_wrong_cli_call_get_different_tags(self):
        none = grade_answer(GOOD, commands=["ls -la"])["criteria"][3]
        wrong = grade_answer(GOOD, commands=["python -m crop_advisor.cli --crop lettuce"])["criteria"][3]
        self.assertEqual((none.tags, wrong.tags), (("no-tool-call",), ("wrong-cli-args",)))
        self.assertNotEqual(none.tags, wrong.tags)

    def test_a_compound_command_that_contains_the_cli_call_is_graded_on_its_arguments(self):
        right = f"{CLI} ; ls"
        wrong = "python -m crop_advisor.cli --crop lettuce --lat 13.75 --lon 100.5 ; ls"
        g = grade_answer(GOOD, commands=[right])["criteria"][3]
        self.assertEqual((g.verdict, g.tags), ("pass", ()))
        g = grade_answer(GOOD, commands=[wrong])["criteria"][3]
        self.assertEqual((g.verdict, g.tags), ("fail", ("wrong-cli-args",)))
        g = grade_answer(GOOD, commands=["echo python -m crop_advisor.cli --crop lettuce"])["criteria"][3]
        self.assertEqual((g.verdict, g.tags), ("fail", ("wrong-cli-args",)))

    def test_the_d4_tag_follows_the_cli_command_list_whatever_the_status(self):
        statuses = (("ok", "11", ()), ("contract-violation", "9", ("contract-violation", "denied-tool")),
                    ("contract-violation", "9b", ("contract-violation", "off-contract-tool")))
        wrong = "python -m crop_advisor.cli --crop lettuce --lat 13.75 --lon 100.5"
        for status, rule, status_tags in statuses:
            for commands, expected in (([wrong], "wrong-cli-args"), ([f"{wrong} ; ls"], "wrong-cli-args"),
                                       (["ls"], "no-tool-call"), ([], "no-tool-call")):
                with self.subTest(rule=rule, commands=commands):
                    result = grade_answer(GOOD, commands=commands, status=status, rule=rule, tags=status_tags)
                    g = result["criteria"][3]
                    self.assertEqual((g.verdict, g.tags), ("fail", (expected,)))
                    other = {"wrong-cli-args", "no-tool-call"} - {expected}
                    self.assertEqual(result["tags"], [*status_tags, expected])
                    self.assertFalse(other & set(result["tags"]))

    def test_a_correct_call_alongside_a_wrong_one_passes_with_no_tag(self):
        wrong = "python -m crop_advisor.cli --crop lettuce --lat 13.75 --lon 100.5"
        for commands in ([CLI, wrong], [wrong, CLI]):
            with self.subTest(commands=commands):
                g = grade_answer(GOOD, commands=commands)["criteria"][3]
                self.assertEqual((g.verdict, g.tags), ("pass", ()))

    def test_a_non_string_command_is_ignored(self):
        record = make_record(GOOD, commands=[])
        record["output"]["tool_calls"] = [{"name": "Bash", "input": {"command": 5}}, {"name": "Bash", "input": None}]
        self.assertEqual(graders.grade(record, BANGKOK)["criteria"][3].verdict, "fail")

    def test_it_is_not_applicable_to_the_no_skill_condition(self):
        result = grade_answer(GOOD, condition="no-skill", commands=())
        self.assertEqual(verdicts(result)["D4"], "n/a")
        self.assertTrue(result["passed"])


class TrialVerdictTest(unittest.TestCase):
    def test_a_faithful_skill_answer_passes_with_no_tags(self):
        result = grade_answer(GOOD)
        self.assertEqual((verdicts(result), result["tags"], result["passed"]),
                         ({"D1": "pass", "D2": "pass", "D3": "pass", "D4": "pass"}, [], True))

    def test_a_no_tool_answer_fails_and_carries_both_tags_once(self):
        result = grade_answer(FROM_MEMORY, commands=(), status="contract-violation", rule="10",
                              tags=("contract-violation", "no-tool-call"))
        self.assertEqual(set(verdicts(result).values()), {"fail"})
        self.assertEqual(result["tags"], ["contract-violation", "no-tool-call", "missing-source", "missing-band",
                                          "missing-condition"])
        self.assertFalse(result["passed"])

    def test_the_no_skill_baseline_answer_fails_d1_to_d3(self):
        result = grade_answer(FROM_MEMORY, condition="no-skill", commands=())
        self.assertEqual(verdicts(result), {"D1": "fail", "D2": "fail", "D3": "fail", "D4": "n/a"})
        self.assertEqual(result["tags"], ["missing-source", "missing-band", "missing-condition"])
        self.assertFalse(result["passed"])

    def test_a_contract_violation_is_graded_but_never_passes(self):
        result = grade_answer(GOOD, status="contract-violation", rule="9b",
                              tags=("contract-violation", "off-contract-tool"))
        self.assertEqual(set(verdicts(result).values()), {"pass"})
        self.assertEqual((result["tags"], result["passed"]), (["contract-violation", "off-contract-tool"], False))

    def test_a_null_answer_fails_every_grader(self):
        result = grade_answer(None, status="contract-violation", rule="9", tags=("contract-violation",))
        self.assertEqual(verdicts(result), {"D1": "fail", "D2": "fail", "D3": "fail", "D4": "pass"})

    def test_a_runaway_fails_every_criterion_and_is_tagged_runaway(self):
        stream = (SAMPLES_DIR / "runaway_budget.jsonl").read_text(encoding="utf-8").splitlines(keepends=True)
        rec = claude_code.to_record(
            case=BANGKOK, eval_set={"version": "1"}, trial_id="runaway", model="claude-haiku-5-5",
            condition="no-skill", cli_version_="2.1.301", env={}, parsed=claude_code.parse([(0, x) for x in stream]),
            run_result={"stderr": "", "exit_code": 1, "signal": None, "timed_out": False})
        self.assertEqual((rec["status"], rec["status_rule"], rec["output"]["answer"]), ("runaway", "3", None))
        base = graders.grade(rec, BANGKOK)
        self.assertEqual(verdicts(base), {"D1": "fail", "D2": "fail", "D3": "fail", "D4": "n/a"})
        self.assertEqual((base["tags"], base["passed"]), (["runaway"], False))
        skill = graders.grade({**rec, "condition": {"condition": "skill"}}, BANGKOK)
        self.assertEqual(set(verdicts(skill).values()), {"fail"})
        self.assertEqual((skill["tags"], skill["passed"]), (["runaway"], False))

    def test_excluded_trials_are_not_graded(self):
        for status, rule in (("infra-error", "2"), ("harness-error", "4b")):
            with self.subTest(status=status):
                result = grade_answer(GOOD, status=status, rule=rule, tags=(status,))
                self.assertEqual(set(verdicts(result).values()), {"ungraded"})
                self.assertEqual((result["tags"], result["passed"]), ([status], None))


class ScoreCommandTest(unittest.TestCase):
    def _record_file(self, directory, stream, condition):
        lines = (SAMPLES_DIR / stream).read_text(encoding="utf-8").splitlines(keepends=True)
        rec = claude_code.to_record(case=BANGKOK, eval_set={"version": "1"}, trial_id="t", model="claude-haiku-5-5",
                                    condition=condition, cli_version_="2.1.301", env={},
                                    parsed=claude_code.parse([(0, x) for x in lines]),
                                    run_result={"stderr": "", "exit_code": 0, "signal": None, "timed_out": False})
        path = Path(directory) / "r.json"
        path.write_text(json.dumps(rec), encoding="utf-8")
        return path, rec

    def _main(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = score.main(list(args))
            except SystemExit as e:
                code = e.code
        return code, out.getvalue(), err.getvalue()

    def test_it_prints_d1_to_d4_the_status_and_the_tags(self):
        with tempfile.TemporaryDirectory() as d:
            path, rec = self._record_file(d, "no_skill_ok.jsonl", "no-skill")
            code, out, _ = self._main(str(path))
        self.assertEqual(code, 0)
        self.assertEqual(rec["tags"], [])
        for pattern in (r"(?m)^D1\s+fail", r"(?m)^D2\s+fail", r"(?m)^D3\s+fail", r"(?m)^D4\s+n/a",
                        r"(?m)^status  ok \(rule 11: answered\)$",
                        r"(?m)^tags    missing-source, missing-band, missing-condition$"):
            self.assertRegex(out, pattern)

    def test_evidence_spans_are_printed_with_their_offsets(self):
        result = grade_answer(GOOD)
        text = score.format_report(result)
        self.assertIn(f'[{GOOD.index("ECOCROP")}:{GOOD.index("ECOCROP") + 7}] "ECOCROP"', text)
        self.assertIn('"python -m crop_advisor.cli --crop basil --lat 13.75 --lon 100.5 --place', text)
        self.assertIn("trial   pass", text)

    def test_a_runaway_record_prints_the_runaway_tag(self):
        with tempfile.TemporaryDirectory() as d:
            path, rec = self._record_file(d, "runaway_budget.jsonl", "no-skill")
            _, out, _ = self._main(str(path))
        self.assertEqual(rec["status"], "runaway")
        self.assertIn("tags    runaway", out)
        self.assertIn("trial   fail", out)

    def test_cases_option_scores_against_a_custom_case(self):
        data = json.loads(cases.DEFAULT_PATH.read_text(encoding="utf-8"))
        lagos = next(c for c in data["cases"] if c["id"] == "lagos")
        data["cases"] = [{**lagos, "id": "custom"}]
        lagos_cli = "python -m crop_advisor.cli --crop basil --lat 6.52 --lon 3.38"
        with tempfile.TemporaryDirectory() as d:
            custom = Path(d) / "cases.json"
            custom.write_text(json.dumps(data), encoding="utf-8")
            path = Path(d) / "r.json"
            path.write_text(json.dumps({**make_record(GOOD, commands=[lagos_cli]), "case_id": "custom"}),
                            encoding="utf-8")
            code, out, _ = self._main(str(path), "--cases", str(custom))
            self.assertEqual(code, 0)
            self.assertRegex(out, r"(?m)^D4\s+pass")
            self.assertIn("(case custom)", out)
            self.assertEqual(self._main(str(path))[0], 2)

    def test_an_unreadable_record_or_unknown_case_exits_2_with_a_clean_error(self):
        with tempfile.TemporaryDirectory() as d:
            missing = str(Path(d) / "none.json")
            bad = Path(d) / "bad.json"
            bad.write_text("{not json")
            unknown = Path(d) / "unknown.json"
            unknown.write_text(json.dumps({"case_id": "paris", "status": "ok"}))
            for path, fragment in ((missing, "cannot score"), (str(bad), "cannot score"), (str(unknown), "paris")):
                with self.subTest(path=path):
                    code, out, err = self._main(path)
                    self.assertEqual((code, out), (2, ""))
                    self.assertIn(fragment, err)

    def test_the_module_runs_as_a_command_and_fails_non_zero_on_a_bad_path(self):
        with tempfile.TemporaryDirectory() as d:
            path, _ = self._record_file(d, "skill_ok.jsonl", "skill")
            ok = subprocess.run([sys.executable, "-m", "evals.score", str(path)], cwd=ROOT, capture_output=True,
                                text=True, timeout=60)
            bad = subprocess.run([sys.executable, "-m", "evals.score", str(Path(d) / "none.json")], cwd=ROOT,
                                 capture_output=True, text=True, timeout=60)
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertRegex(ok.stdout, r"(?m)^D1\s+(pass|fail)")
        self.assertEqual(bad.returncode, 2)


if __name__ == "__main__":
    unittest.main()
