"""The per-cell report, offline: accounting, statistics, refusals, exit codes and the committed demo fixture.

    python -m tests.test_eval_report --write-fixture    rewrites tests/fixtures/eval_report/trials.jsonl
"""
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

from evals import cases, report, stats
from evals.adapters import claude_code
from tests.fake_claude import SAMPLES_DIR, sample_lines, tripwire

_trip = {}


def setUpModule():
    # The username rewrite is machine-dependent, so pin a name no fixture word can collide with.
    tmp = tempfile.TemporaryDirectory()
    unittest.addModuleCleanup(tmp.cleanup)
    _trip["log"] = tripwire(tmp.name)
    patcher = mock.patch.dict(os.environ, {"USER": "testuser", "LOGNAME": "testuser", "USERNAME": "testuser",
                                           "PATH": tmp.name + os.pathsep + os.environ.get("PATH", "")})
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


def tearDownModule():
    if _trip["log"].exists():
        raise AssertionError(f"a test reached `claude`: {_trip['log'].read_text()}")


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "eval_report" / "trials.jsonl"
HAIKU, SONNET = "claude-haiku-5-5", "claude-sonnet-5-5"
RUN_ID = "20261001T120000Z-5eed0001"
SAMPLES = {s["name"]: s for s in json.loads((SAMPLES_DIR / "samples.json").read_text(encoding="utf-8"))["samples"]}
EVAL_SET = cases.load()


def make(sample, *, case="bangkok", model=HAIKU, condition=None, rep=1, attempt=1, latency=1500, serial=0,
         eval_set=EVAL_SET):
    """A record built by claude_code.to_record from a stored sample, then given the run loop's `rep` and `attempt`."""
    s = SAMPLES[sample]
    condition = condition or s["condition"]
    lines = [(ms, line.replace("claude-haiku-5-5", model)) for ms, line in sample_lines(s["stream"])] if s["stream"] else []
    parsed = claude_code.parse(lines) if lines else {}
    with mock.patch.dict(os.environ, {"USER": "testuser", "LOGNAME": "testuser", "USERNAME": "testuser"}):
        rec = claude_code.to_record(
            case=cases.get(case)[1], eval_set=eval_set,
            trial_id=f"{case}-{condition}-{model}-r{rep}-a{attempt}-{serial:08x}",
            model=model, condition=condition, cli_version_="2.1.303",
            run_result={"stderr": s["stderr"], "exit_code": s["exit_code"], "signal": s["signal"],
                        "timed_out": s["timed_out"], "latency_ms": latency},
            parsed=parsed, harness_exception=s["harness_exception"], skill_sha="0" * 64, env={})
    rec.update(run_id=RUN_ID, rep=rep, attempt=attempt)
    return rec


# (case, model, condition, one entry per rep: the stored samples for its attempts)
GRID = [
    ("bangkok", HAIKU, "skill",
     [["skill_ok"], ["skill_ok"], ["skill_no_cli_call"], ["nasa_network_failure", "skill_ok"]]),
    ("bangkok", HAIKU, "no-skill", [["no_skill_ok"], ["runaway_budget"], ["api_404_bad_model"]]),
    ("lagos", HAIKU, "skill", [["skill_ok"], ["timeout_truncated", "timeout_truncated"]]),
    ("lagos", HAIKU, "no-skill", [["no_skill_ok"], ["no_skill_ok"]]),
    ("bangkok", SONNET, "skill", [["skill_ok"], ["skill_ok"]]),
    ("bangkok", SONNET, "no-skill", [["no_skill_ok"], ["no_skill_ok"]]),
    ("lagos", SONNET, "no-skill", [["no_skill_ok"], ["no_skill_ok"]]),
]
LABEL = ("synthesised: assembled by claude_code.to_record from the stored samples in tests/fixtures/claude_code, "
         "not a real run; a stream may be paired with another case's coordinates, its model id is rewritten to the "
         "record's model, and its cost is the stored Haiku list price whatever the model")


def fixture_records():
    records, serial = [], 0
    for case, model, condition, reps in GRID:
        for rep, samples in enumerate(reps, 1):
            for attempt, sample in enumerate(samples, 1):
                serial += 1
                rec = make(sample, case=case, model=model, condition=condition, rep=rep, attempt=attempt,
                           latency=2000 + 137 * (serial * 7 % 23), serial=serial)
                records.append({**rec, "synthesised": True, "provenance": f"{LABEL}; stream {sample}"})
    return records


def fixture_text():
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in fixture_records())


def write_jsonl(directory, records, name="trials.jsonl"):
    path = Path(directory) / name
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def run_main(*args):
    out, err = io.StringIO(), io.StringIO()
    code = 0
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = report.main([str(a) for a in args])
        except SystemExit as e:
            code = e.code
    return code, out.getvalue(), err.getvalue()


def build(records):
    return report.build_report(records, EVAL_SET, seed=1, resamples=200)


def cell_of(rep, case="bangkok", model=HAIKU, condition="skill"):
    return next(c for c in rep["cells"] if (c["case"], c["model"], c["condition"]) == (case, model, condition))


class FixtureTest(unittest.TestCase):
    def test_the_committed_fixture_is_what_the_test_code_regenerates(self):
        self.assertEqual(FIXTURE.read_text(encoding="utf-8"), fixture_text())

    def test_every_fixture_record_is_labelled_synthesised(self):
        for rec in report.load_records(FIXTURE):
            self.assertTrue(rec["synthesised"])
            self.assertIn("synthesised", rec["provenance"])

    def test_the_fixture_reports_with_exit_0_and_no_warning(self):
        code, out, err = run_main(FIXTURE)
        self.assertEqual((code, err), (0, ""))
        self.assertIn("judge: not run (E5)", out)
        self.assertIn("deterministic checks only (D1-D4)", out)
        self.assertIn("CLI versions seen: 2.1.303", out)
        self.assertIn("eval set version: 1", out)

    def test_the_fixture_exercises_the_accounting(self):
        rep = report.build_report(report.load_records(FIXTURE), EVAL_SET, seed=1, resamples=200)
        skill = cell_of(rep)
        self.assertEqual((skill["slots"], skill["attempts"], skill["denominator"], skill["passed"]), (4, 5, 4, 3))
        self.assertEqual((skill["reruns"], skill["infra_recovered"], skill["infra_recurred"]), (1, 1, 0))
        lagos = cell_of(rep, "lagos")
        self.assertEqual((lagos["denominator"], lagos["infra_recurred"], lagos["excluded"]),
                         (1, 1, {"infra-error": {"2": 1}}))
        no_skill = cell_of(rep, condition="no-skill")
        self.assertEqual((no_skill["denominator"], no_skill["runaway"], no_skill["excluded"]),
                         (2, 1, {"harness-error": {"4b": 1}}))


class CellTest(unittest.TestCase):
    def cell(self, samples, **kw):
        records = [make(s, rep=i, serial=i, **kw) for i, s in enumerate(samples, 1)]
        return cell_of(build(records), **{k: v for k, v in kw.items() if k in ("case", "model", "condition")})

    def test_cell_keys_are_stable(self):
        self.assertEqual(sorted(self.cell(["skill_ok"])), sorted([
            "case", "model", "condition", "slots", "attempts", "denominator", "passed", "rate", "ci", "pass_at_k",
            "pass_pow_k", "excluded", "excluded_total", "excluded_share", "reruns", "infra_recovered",
            "infra_recurred", "runaway",
            "contract_violation", "tags", "cost", "latency", "pass_vector"]))

    def test_excluded_trials_are_out_of_the_denominator(self):
        cell = self.cell(["skill_ok", "api_404_bad_model", "timeout_truncated", "skill_no_cli_call"], condition="skill")
        self.assertEqual((cell["slots"], cell["denominator"], cell["passed"], cell["excluded_total"]), (4, 2, 1, 2))
        self.assertEqual(cell["ci"], [0.0945, 0.9055])

    def test_contract_violation_and_runaway_stay_in_as_fails(self):
        cell = self.cell(["skill_ok", "skill_no_cli_call"])
        self.assertEqual((cell["denominator"], cell["passed"], cell["contract_violation"]), (2, 1, 1))
        cell = self.cell(["runaway_budget", "runaway_turns", "no_skill_ok"], condition="no-skill")
        self.assertEqual((cell["denominator"], cell["passed"], cell["runaway"]), (3, 0, 2))
        self.assertEqual(cell["tags"]["runaway"], 2)

    def test_excluded_counts_by_status_and_rule(self):
        cell = self.cell(["api_404_bad_model", "error_during_execution", "not_logged_in",
                          "nonzero_no_result_unrecognised_stderr", "api_429_rate_limit"], condition="no-skill")
        self.assertEqual(cell["excluded"], {"harness-error": {"4b": 1, "4c": 1, "8": 2}, "infra-error": {"4a": 1}})
        self.assertEqual((cell["excluded_total"], cell["denominator"]), (5, 0))

    def test_infra_then_ok_is_one_ok_slot_with_one_recovered_rerun(self):
        cell = cell_of(build([make("nasa_network_failure", rep=1, attempt=1, serial=1),
                              make("skill_ok", rep=1, attempt=2, serial=2)]))
        self.assertEqual((cell["slots"], cell["attempts"], cell["denominator"], cell["passed"]), (1, 2, 1, 1))
        self.assertEqual((cell["reruns"], cell["infra_recovered"], cell["infra_recurred"], cell["excluded_total"]),
                         (1, 1, 0, 0))

    def test_cost_counts_the_infra_attempt_that_was_re_run(self):
        first = make("nasa_network_failure", rep=1, attempt=1, serial=1)
        second = make("skill_ok", rep=1, attempt=2, serial=2)
        total = first["usage"]["cost_usd"] + second["usage"]["cost_usd"]
        self.assertEqual(cell_of(build([first, second]))["cost"],
                         {"sum": round(total, 6), "mean": round(total / 2, 6), "priced": 2, "unpriced": 0})

    def test_infra_twice_is_one_excluded_recurred_slot(self):
        cell = cell_of(build([make("nasa_network_failure", rep=1, attempt=1, serial=1),
                              make("nasa_network_failure", rep=1, attempt=2, serial=2)]))
        self.assertEqual((cell["slots"], cell["attempts"], cell["denominator"]), (1, 2, 0))
        self.assertEqual((cell["reruns"], cell["infra_recovered"], cell["infra_recurred"], cell["excluded_total"]),
                         (1, 0, 1, 1))
        self.assertEqual(cell["excluded"], {"infra-error": {"6": 1}})
        self.assertEqual(cell["excluded_share"], 1.0)

    def test_only_the_final_attempt_is_read(self):
        cell = cell_of(build([make("timeout_truncated", rep=1, attempt=1, serial=1),
                              make("skill_no_cli_call", rep=1, attempt=2, serial=2)]))
        self.assertEqual((cell["denominator"], cell["passed"], cell["contract_violation"]), (1, 0, 1))
        self.assertEqual(cell["tags"].get("infra-error"), None)

    def test_an_unrerun_infra_error_is_excluded_but_not_recurred(self):
        cell = cell_of(build([make("timeout_truncated", rep=1, attempt=1, serial=1)]))
        self.assertEqual((cell["denominator"], cell["excluded_total"], cell["reruns"], cell["infra_recurred"]),
                         (0, 1, 0, 0))

    def test_excluded_share_is_excluded_slots_over_slots(self):
        cell = self.cell(["skill_ok", "api_404_bad_model", "timeout_truncated", "skill_no_cli_call"], condition="skill")
        self.assertEqual((cell["excluded_total"], cell["slots"], cell["excluded_share"]), (2, 4, 0.5))
        self.assertIn("excluded    2 of 4 slots (50.0%): harness-error/", report.format_cell(cell))
        self.assertEqual(self.cell(["skill_ok"])["excluded_share"], 0.0)

    def test_tags_include_the_status_tags_of_excluded_trials(self):
        cell = self.cell(["skill_ok", "api_404_bad_model", "timeout_truncated"], condition="skill")
        self.assertEqual((cell["tags"]["harness-error"], cell["tags"]["infra-error"]), (1, 1))

    def test_a_re_run_that_ends_in_a_graded_non_ok_trial_is_recovered(self):
        for second, kw in [("skill_no_cli_call", {}), ("runaway_budget", {"condition": "no-skill"})]:
            first = "nasa_network_failure" if not kw else "api_429_rate_limit"
            with self.subTest(second=second):
                cell = cell_of(build([make(first, rep=1, attempt=1, serial=1, **kw),
                                      make(second, rep=1, attempt=2, serial=2, **kw)]), **kw)
                self.assertEqual((cell["denominator"], cell["infra_recovered"], cell["infra_recurred"]), (1, 1, 0))

    def test_a_re_run_that_ends_in_a_harness_error_is_neither_recovered_nor_recurred(self):
        cell = cell_of(build([make("nasa_network_failure", case="bangkok", rep=1, attempt=1, serial=1),
                              make("replay_missing_model_answers", rep=1, attempt=2, serial=2)]))
        self.assertEqual((cell["reruns"], cell["infra_recovered"], cell["infra_recurred"], cell["denominator"]),
                         (1, 0, 0, 0))
        self.assertEqual(cell["excluded"], {"harness-error": {"5": 1}})

    def test_latency_of_a_cell_with_no_gradable_trial_has_no_percentiles_and_no_max_note(self):
        cell = self.cell(["api_404_bad_model"], condition="no-skill")
        self.assertEqual(cell["latency"], {"n": 0, "p50": None, "p95": None, "p95_is_max": False})
        self.assertNotIn("the max", report.format_cell(cell))
        self.assertIn("p50 n/a ms, p95 n/a ms", report.format_cell(cell))

    def test_a_bool_is_not_a_cost_or_a_latency(self):
        record = make("skill_ok", rep=1, serial=1)
        record["usage"]["cost_usd"], record["usage"]["latency_ms"] = True, True
        cell = cell_of(build([record]))
        self.assertEqual(cell["cost"], {"sum": 0.0, "mean": None, "priced": 0, "unpriced": 1})
        self.assertEqual(cell["latency"]["n"], 0)

    def test_wilson_bounds_never_print_a_negative_zero_or_exceed_one(self):
        for n in (7, 9, 14, 17, 18):
            with self.subTest(n=n):
                low, high = stats.wilson(0, n)
                self.assertEqual(repr(low), "0.0")
                self.assertEqual(repr(stats.wilson(n, n)[1]), "1.0")
        cell = self.cell(["no_skill_ok"] * 7, condition="no-skill")
        self.assertEqual(report.headline(cell), "0/7 (0.0%) [0.0000, 0.3543]")
        cell = self.cell(["skill_ok"] * 7)
        self.assertEqual(report.headline(cell), "7/7 (100.0%) [0.6457, 1.0000]")

    def test_slots_are_read_in_rep_order_whatever_the_file_order(self):
        names = ["skill_ok", "skill_ok", "skill_no_cli_call", "skill_ok", "skill_ok", "skill_ok", "skill_no_cli_call",
                 "skill_ok", "skill_ok", "skill_no_cli_call"]
        records = [make(n, rep=i, serial=i) for i, n in enumerate(names, 1)]
        records += [make("no_skill_ok", rep=i, serial=100 + i) for i in range(1, 5)]
        rep = report.build_report(records, EVAL_SET, seed=1, resamples=300)
        self.assertEqual(cell_of(rep)["pass_vector"], [1, 1, 0, 1, 1, 1, 0, 1, 1, 0])
        self.assertEqual(rep, report.build_report(list(reversed(records)), EVAL_SET, seed=1, resamples=300))

    def test_cells_are_ordered_by_cases_file_then_model_then_skill_first(self):
        records = [make(s, case=c, model=m, condition=cond, rep=1, serial=i) for i, (s, c, m, cond) in enumerate([
            ("no_skill_ok", "bangkok", SONNET, "no-skill"), ("skill_ok", "bangkok", SONNET, "skill"),
            ("no_skill_ok", "bangkok", HAIKU, "no-skill"), ("skill_ok", "bangkok", HAIKU, "skill"),
            ("no_skill_ok", "lagos", HAIKU, "no-skill"), ("no_skill_ok", "london", HAIKU, "no-skill")], 1)]
        order = [(c["case"], c["model"], c["condition"]) for c in build(records)["cells"]]
        self.assertEqual(order, [("london", HAIKU, "no-skill"), ("lagos", HAIKU, "no-skill"),
                                 ("bangkok", HAIKU, "skill"), ("bangkok", HAIKU, "no-skill"),
                                 ("bangkok", SONNET, "skill"), ("bangkok", SONNET, "no-skill")])

    def test_no_gradable_trials_and_k_beyond_the_denominator_are_na(self):
        cell = self.cell(["api_404_bad_model"], condition="no-skill")
        self.assertEqual((cell["denominator"], cell["rate"], cell["ci"]), (0, None, None))
        self.assertEqual(cell["pass_at_k"], {"1": None, "3": None, "5": None})
        self.assertEqual(report.headline(cell), "n/a (no gradable trials)")
        cell = self.cell(["skill_ok", "skill_ok", "skill_no_cli_call"])
        self.assertEqual(cell["pass_at_k"], {"1": 0.6667, "3": 1.0, "5": None})
        self.assertEqual(cell["pass_pow_k"], {"1": 0.6667, "3": 0.0, "5": None})
        self.assertIn("k=5: n/a", report.format_cell(cell))

    def test_headline_and_statistics_for_8_of_10(self):
        cell = self.cell(["skill_ok"] * 8 + ["skill_no_cli_call"] * 2)
        self.assertEqual(report.headline(cell), "8/10 (80.0%) [0.4902, 0.9433]")
        self.assertEqual(cell["pass_at_k"], {"1": 0.8, "3": 1.0, "5": 1.0})
        self.assertEqual(cell["pass_pow_k"], {"1": 0.8, "3": 0.4667, "5": 0.2222})

    def test_cost_includes_excluded_attempts_and_counts_unpriced(self):
        records = [make("skill_ok", rep=1, serial=1), make("api_404_bad_model", rep=2, serial=2, condition="skill"),
                   make("timeout_truncated", rep=3, serial=3), make("nasa_network_failure", rep=4, serial=4)]
        records[1]["usage"]["cost_usd"] = 0.25
        cell = cell_of(build(records))
        total = records[0]["usage"]["cost_usd"] + 0.25 + records[3]["usage"]["cost_usd"]
        self.assertEqual(cell["cost"],
                         {"sum": round(total, 6), "mean": round(total / 3, 6), "priced": 3, "unpriced": 1})
        self.assertEqual(cell["excluded_total"], 3)

    def test_a_null_cost_is_unpriced_and_a_cell_with_no_priced_attempt_has_no_mean(self):
        cell = cell_of(build([make("timeout_truncated", rep=1, serial=1)]))
        self.assertEqual(cell["cost"], {"sum": 0.0, "mean": None, "priced": 0, "unpriced": 1})

    def test_latency_reads_final_non_excluded_attempts_only(self):
        records = [make("skill_ok", rep=i, serial=i, latency=100 * i) for i in range(1, 11)]
        records.append(make("api_404_bad_model", rep=11, serial=11, latency=99999, condition="skill"))
        records.append(make("skill_ok", rep=12, serial=12, latency=5))
        records[-1]["usage"]["latency_ms"] = None
        lat = cell_of(build(records))["latency"]
        self.assertEqual(lat, {"n": 10, "p50": 500, "p95": 1000, "p95_is_max": True})

    def test_p95_is_not_flagged_as_the_max_from_20_trials(self):
        records = [make("skill_ok", rep=i, serial=i, latency=i) for i in range(1, 21)]
        lat = cell_of(build(records))["latency"]
        self.assertEqual((lat["p95"], lat["p95_is_max"]), (19, False))

    def test_tag_counts_cover_final_attempts_once_each(self):
        cell = self.cell(["skill_no_cli_call", "skill_no_cli_call", "skill_ok"])
        self.assertEqual(cell["tags"]["no-tool-call"], 2)
        self.assertEqual(cell["tags"]["contract-violation"], 2)
        self.assertNotIn("wrong-cli-args", cell["tags"])


class DeltaTest(unittest.TestCase):
    def rows(self, records):
        rep = build(records)
        return rep["deltas"], rep["overall"]

    def test_delta_per_case_and_overall_over_k_of_4_cases(self):
        records = [make("skill_ok", rep=1, serial=1), make("skill_ok", rep=2, serial=2),
                   make("no_skill_ok", rep=1, serial=3), make("no_skill_ok", rep=2, serial=4),
                   make("no_skill_ok", case="lagos", rep=1, serial=5)]
        deltas, overall = self.rows(records)
        self.assertEqual([(d["case"], d["delta"], d["skill_n"], d["noskill_n"]) for d in deltas],
                         [("lagos", None, 0, 1), ("bangkok", 1.0, 2, 2)])
        self.assertEqual(deltas[0]["ci"], None)
        self.assertEqual(deltas[1]["ci"], [1.0, 1.0])
        self.assertEqual(overall, [{"model": HAIKU, "delta": 1.0, "ci": [1.0, 1.0], "cases_used": 1, "cases_total": 4}])
        text = report.format_report(build(records))
        self.assertIn("over 1 of 4 cases", text)
        self.assertIn("n/a (a side has no gradable trials)", text)

    def test_overall_is_na_when_every_case_is_dropped(self):
        records = [make("no_skill_ok", rep=1, serial=1), make("skill_ok", case="lagos", rep=1, serial=2)]
        _, overall = self.rows(records)
        self.assertEqual(overall, [{"model": HAIKU, "delta": None, "ci": None, "cases_used": 0, "cases_total": 4}])
        self.assertIn("overall    n/a  over 0 of 4 cases", report.format_report(build(records)))

    def test_deltas_are_kept_per_model(self):
        records = [make("skill_ok", rep=1, serial=1), make("no_skill_ok", rep=1, serial=2),
                   make("skill_no_cli_call", model=SONNET, rep=1, serial=3),
                   make("no_skill_ok", model=SONNET, rep=1, serial=4)]
        _, overall = self.rows(records)
        self.assertEqual([(o["model"], o["delta"]) for o in overall], [(HAIKU, 1.0), (SONNET, 0.0)])

    def test_the_seed_and_resamples_reach_every_bootstrap_call(self):
        records = [make("skill_ok", rep=i, serial=i) for i in (1, 2, 3)] + [make("skill_no_cli_call", rep=4, serial=4)]
        records += [make("no_skill_ok", rep=i, serial=10 + i) for i in (1, 2, 3)]
        records += [make("skill_ok", case="lagos", rep=1, serial=20), make("no_skill_ok", case="lagos", rep=1, serial=21)]
        with mock.patch.object(report.stats, "bootstrap_ci", wraps=stats.bootstrap_ci) as spy:
            rep = report.build_report(records, EVAL_SET, seed=5, resamples=300)
        self.assertEqual(rep["bootstrap"], {"seed": 5, "resamples": 300})
        self.assertEqual(spy.call_count, len(rep["deltas"]) + len(rep["overall"]))
        self.assertEqual(spy.call_count, 3)
        for call in spy.call_args_list:
            self.assertEqual(call.args[1:], (5, 300))
        pair = ([1, 1, 1, 0], [0, 0, 0])
        self.assertEqual(rep["deltas"][1]["ci"], list(stats.bootstrap_ci([pair], 5, 300)))
        self.assertEqual(rep, report.build_report(records, EVAL_SET, seed=5, resamples=300))


class CommandTest(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.dir = Path(d.name)

    def test_a_trials_file_and_a_run_directory_report_the_same(self):
        records = [make("skill_ok", rep=1, serial=1), make("no_skill_ok", rep=1, serial=2)]
        path = write_jsonl(self.dir, records)
        by_file, by_dir = run_main(path), run_main(self.dir)
        self.assertEqual(by_file[0], 0)
        self.assertEqual(by_file, by_dir)

    def test_it_writes_nothing_and_starts_no_process(self):
        write_jsonl(self.dir, [make("skill_ok", rep=1, serial=1)])
        before = {p.name: p.stat().st_mtime_ns for p in self.dir.iterdir()}
        boom = AssertionError("the report started a process")
        with mock.patch.multiple(subprocess, Popen=mock.Mock(side_effect=boom), run=mock.Mock(side_effect=boom)):
            code, out, _ = run_main(self.dir)
        self.assertEqual(code, 0)
        self.assertIn("bangkok | claude-haiku-5-5 | skill", out)
        self.assertEqual({p.name: p.stat().st_mtime_ns for p in self.dir.iterdir()}, before)

    def test_mixed_eval_set_hashes_are_refused_with_exit_2(self):
        other = {**EVAL_SET, "sha256": "f" * 64}
        path = write_jsonl(self.dir, [make("skill_ok", rep=1, serial=1),
                                      make("skill_ok", rep=2, serial=2, eval_set=other)])
        code, out, err = run_main(path)
        self.assertEqual((code, out), (2, ""))
        self.assertIn("more than one eval_set_sha256", err)

    def test_a_hash_that_differs_from_the_cases_file_warns_and_still_reports(self):
        other = {**EVAL_SET, "sha256": "f" * 64}
        path = write_jsonl(self.dir, [make("skill_ok", rep=1, serial=1, eval_set=other)])
        code, out, err = run_main(path)
        self.assertEqual(code, 0)
        self.assertIn("warning: records carry eval_set_sha256 " + "f" * 64, err)
        self.assertIn(EVAL_SET["sha256"], err)
        self.assertIn("headline", out)

    def test_a_matching_hash_does_not_warn(self):
        path = write_jsonl(self.dir, [make("skill_ok", rep=1, serial=1)])
        self.assertEqual(run_main(path)[2], "")

    def test_unreadable_input_exits_2(self):
        for label, arrange in {
            "missing file": lambda: self.dir / "nope.jsonl",
            "empty file": lambda: write_jsonl(self.dir, [], "empty.jsonl"),
            "not json": lambda: self._text("{oops\n"),
            "not an object": lambda: self._text("[1]\n"),
        }.items():
            with self.subTest(label):
                code, out, err = run_main(arrange())
                self.assertEqual((code, out), (2, ""))
                self.assertTrue(err)

    def _text(self, text):
        path = self.dir / "bad.jsonl"
        path.write_text(text, encoding="utf-8")
        return path

    def test_a_record_that_cannot_be_placed_in_a_slot_exits_2(self):
        good = make("skill_ok", rep=1, serial=1)
        cases_ = {
            "no rep": {k: v for k, v in good.items() if k != "rep"},
            "no attempt": {k: v for k, v in good.items() if k != "attempt"},
            "attempt 3": {**good, "attempt": 3},
            "string rep": {**good, "rep": "1"},
            "no status": {k: v for k, v in good.items() if k != "status"},
            "no model": {**good, "subject": {}},
            "bad condition": {**good, "condition": {"condition": "maybe"}},
            "bool rep": {**good, "rep": True},
        }
        for label, bad in cases_.items():
            with self.subTest(label):
                code, out, err = run_main(write_jsonl(self.dir, [bad]))
                self.assertEqual((code, out), (2, ""))
                self.assertTrue(err)

    def test_an_unknown_case_exits_2_and_names_it(self):
        code, out, err = run_main(write_jsonl(self.dir, [{**make("skill_ok", rep=1, serial=1), "case_id": "atlantis"}]))
        self.assertEqual((code, out), (2, ""))
        self.assertIn("'atlantis'", err)
        self.assertIn("cases file", err)

    def test_a_line_separator_inside_a_json_string_does_not_split_the_record(self):
        record = make("skill_ok", rep=1, serial=1)
        record["output"]["answer"] += "a\u2028b\u2029c\u0085d\x0be\x0cf\x1cg\x1dh\x1ei"
        path = self.dir / "trials.jsonl"
        path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
        self.assertIn("\u2028", path.read_text(encoding="utf-8"))
        self.assertEqual(report.load_records(path), [record])
        self.assertEqual(run_main(path)[0], 0)

    def test_two_records_for_one_slot_and_attempt_exit_2(self):
        twice = [make("skill_ok", rep=1, serial=1), make("skill_ok", rep=1, serial=2)]
        code, _, err = run_main(write_jsonl(self.dir, twice))
        self.assertEqual(code, 2)
        self.assertIn("two attempt 1 records", err)

    def test_a_missing_cases_file_exits_2(self):
        path = write_jsonl(self.dir, [make("skill_ok", rep=1, serial=1)])
        self.assertEqual(run_main(path, "--cases", self.dir / "nope.json")[0], 2)

    def test_the_cases_option_is_read_and_its_hash_compared(self):
        path = write_jsonl(self.dir, [make("skill_ok", rep=1, serial=1)])
        text = (Path(cases.DEFAULT_PATH)).read_text(encoding="utf-8").replace('"version": "1"', '"version": "9"')
        cases_path = self.dir / "cases.json"
        cases_path.write_text(text, encoding="utf-8")
        code, _, err = run_main(path, "--cases", cases_path)
        self.assertEqual(code, 0)
        self.assertIn("warning", err)


if __name__ == "__main__":
    if sys.argv[1:] == ["--write-fixture"]:
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        with mock.patch.dict(os.environ, {"USER": "testuser", "LOGNAME": "testuser", "USERNAME": "testuser"}):
            FIXTURE.write_text(fixture_text(), encoding="utf-8")
        print(f"wrote {FIXTURE}")
    else:
        unittest.main()
