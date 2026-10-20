"""The frozen config and eval set, and R3 at config time and at record time."""
import json
import tempfile
import unittest
from pathlib import Path

from evals import cases, config
from evals.record_fixtures import SITES


def _write(d, payload, name="c.json"):
    path = Path(d) / name
    path.write_text(json.dumps(payload))
    return path


LIMITS = {"trial_budget_usd": 0.5, "trial_timeout_s": 300}


class ConfigTest(unittest.TestCase):
    def test_the_committed_config_pins_exact_ids(self):
        cfg = config.load()
        self.assertEqual(cfg["subjects"], ["claude-sonnet-5-5", "claude-haiku-5-5"])
        self.assertEqual(cfg["judge"], "claude-opus-5-5")
        self.assertEqual((cfg["trial_budget_usd"], cfg["trial_timeout_s"]), (0.5, 300))

    def test_config_time_r3_refuses_a_judge_sharing_an_id_or_family(self):
        for judge in ("claude-sonnet-5-5", "claude-sonnet-4-6", "claude-haiku-9"):
            with self.subTest(judge=judge), tempfile.TemporaryDirectory() as d:
                with self.assertRaises(config.ConfigError):
                    config.load(_write(d, {"subjects": ["claude-sonnet-5-5", "claude-haiku-5-5"], "judge": judge,
                                           **LIMITS}))

    def test_an_unreadable_model_id_is_refused(self):
        with self.assertRaises(config.ConfigError):
            config.family("gpt-5")

    def test_missing_subjects_or_judge_is_refused(self):
        for payload in ({"judge": "claude-opus-5-5", **LIMITS}, {"subjects": [], "judge": "claude-opus-5-5", **LIMITS},
                        {"subjects": ["claude-sonnet-5-5"], **LIMITS}):
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as d:
                with self.assertRaises(config.ConfigError):
                    config.load(_write(d, payload))

    def test_non_string_subjects_are_refused(self):
        for subjects in ([1], [None], [{}], ["claude-sonnet-5-5", 7]):
            with self.subTest(subjects=subjects), tempfile.TemporaryDirectory() as d:
                with self.assertRaises(config.ConfigError):
                    config.load(_write(d, {"subjects": subjects, "judge": "claude-opus-5-5", **LIMITS}))

    def test_a_bad_config_shape_budget_or_timeout_is_refused(self):
        good = {"subjects": ["claude-sonnet-5-5"], "judge": "claude-opus-5-5", **LIMITS}
        bad = [[], "x"] + [{**good, key: value} for key in LIMITS
                           for value in (None, 0, -1, True, "0.5", float("inf"), 10 ** 400)]
        bad += [{k: v for k, v in good.items() if k != key} for key in LIMITS]
        for payload in bad:
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as d:
                with self.assertRaises(config.ConfigError):
                    config.load(_write(d, payload))
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(config.load(_write(d, {**good, "trial_budget_usd": 1}))["trial_budget_usd"], 1)
        with tempfile.TemporaryDirectory() as d, self.assertRaisesRegex(
                config.ConfigError, "'trial_timeout_s' must be a finite number greater than 0"):
            config.load(_write(d, {**good, "trial_timeout_s": float("inf")}))

    def test_record_time_r3(self):
        config.check_canonical("claude-sonnet-5-5", ["claude-sonnet-5-5"])
        # No reported model passes only when the trial has no answer to attribute; an answered one must name it.
        config.check_canonical("claude-sonnet-5-5", [])
        with self.assertRaises(config.ModelMismatch):
            config.check_canonical("claude-sonnet-5-5", [], require_model=True)
        config.check_canonical("claude-sonnet-5-5", ["claude-sonnet-5-5"], require_model=True)
        for ran in (["claude-opus-5-5"], ["claude-sonnet-5-5", "claude-haiku-5-5"]):
            with self.subTest(ran=ran), self.assertRaises(config.ModelMismatch):
                config.check_canonical("claude-sonnet-5-5", ran)


RUN_KEYS = {"n_trials": 10, "run_budget_usd": 30, "max_consecutive_excluded": 3, "projection_min_trials": 10,
            "bootstrap_seed": 20260101, "bootstrap_resamples": 10000}


def _run_cfg(**over):
    cfg = {"subjects": ["claude-sonnet-5-5", "claude-haiku-5-5"], "judge": "claude-opus-5-5", **LIMITS,
           **RUN_KEYS, "cost_defaults_usd": {m: {"skill": 0.06, "no-skill": 0.03}
                                              for m in ("claude-sonnet-5-5", "claude-haiku-5-5")}}
    cfg.update(over)
    return cfg


class RunSettingsTest(unittest.TestCase):
    def test_the_committed_config_pins_the_run_settings_and_the_default_cost_structure(self):
        cfg = config.load()
        settings = config.run_settings(cfg)
        for key, value in RUN_KEYS.items():
            self.assertEqual(settings[key], value, key)
        self.assertEqual({m: sorted(c) for m, c in settings["cost_defaults_usd"].items()},
                         {m: ["no-skill", "skill"] for m in cfg["subjects"]})

    def test_a_valid_config_passes_and_load_ignores_the_run_keys(self):
        self.assertEqual(config.run_settings(_run_cfg())["n_trials"], 10)
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(config.load(_write(d, {k: v for k, v in _run_cfg().items() if k not in RUN_KEYS})
                                         )["judge"], "claude-opus-5-5")

    def test_a_missing_or_bad_run_key_is_refused(self):
        bad = [{k: v for k, v in _run_cfg().items() if k != key} for key in (*RUN_KEYS, "cost_defaults_usd")]
        for key in ("n_trials", "max_consecutive_excluded", "projection_min_trials", "bootstrap_resamples"):
            bad += [_run_cfg(**{key: v}) for v in (0, -1, 1.5, True, "3", None)]
        bad += [_run_cfg(run_budget_usd=v) for v in (0, -1, True, "30", float("inf"), float("nan"))]
        bad += [_run_cfg(bootstrap_seed=v) for v in (1.5, True, "7", None)]
        bad.append(_run_cfg(cost_defaults_usd=[]))
        for cfg in bad:
            with self.subTest(cfg={k: cfg.get(k) for k in RUN_KEYS}), self.assertRaises(config.ConfigError):
                config.run_settings(cfg)

    def test_a_missing_or_bad_cost_default_for_any_subject_or_condition_is_refused(self):
        full = _run_cfg()["cost_defaults_usd"]
        bad = [{"claude-haiku-5-5": full["claude-haiku-5-5"]}, {"claude-sonnet-5-5": full["claude-sonnet-5-5"]}]
        for model in full:
            for condition in ("skill", "no-skill"):
                without = {m: dict(c) for m, c in full.items()}
                del without[model][condition]
                bad.append(without)
                for value in (0, -0.01, True, "0.06", None, float("nan")):
                    bad.append({m: {**c, **({condition: value} if m == model else {})} for m, c in full.items()})
        for defaults in bad:
            with self.subTest(defaults=defaults), self.assertRaises(config.ConfigError):
                config.run_settings(_run_cfg(cost_defaults_usd=defaults))


class CasesTest(unittest.TestCase):
    def test_every_case_carries_its_recorded_sites_exact_coordinates(self):
        data = cases.load()
        self.assertEqual(data["version"], "1")
        self.assertEqual({c["id"] for c in data["cases"]}, set(SITES))
        for case in data["cases"]:
            with self.subTest(case=case["id"]):
                self.assertEqual((case["lat"], case["lon"]), SITES[case["site"]])
                self.assertEqual(case["crop"], "basil")
        self.assertEqual(len(data["sha256"]), 64)

    def test_a_prompt_without_the_sites_coordinates_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            path = _write(d, {"version": "x", "cases": [{"id": "bangkok", "site": "bangkok", "crop": "basil",
                                                         "place": "Bangkok", "prompt": "basil at 13.7, 100.5?"}]})
            with self.assertRaises(ValueError):
                cases.load(path)

    def test_a_malformed_cases_file_is_refused(self):
        good = {"id": "bangkok", "site": "bangkok", "crop": "basil", "place": "Bangkok",
                "prompt": "Basil in Bangkok (latitude 13.75, longitude 100.5)?"}
        bad = {"a JSON list": [], "no cases": {"version": "1"}, "cases not a list": {"cases": {"bangkok": good}},
               "empty cases": {"cases": []}, "a case not an object": {"cases": ["bangkok"]},
               "numeric prompt": {"cases": [{**good, "prompt": 5}]}, "numeric id": {"cases": [{**good, "id": 7}]},
               "unknown site": {"cases": [{**good, "site": "paris"}]},
               "non-string site": {"cases": [{**good, "site": ["bangkok"]}]}}
        for name, payload in bad.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as d:
                with self.assertRaises(ValueError):
                    cases.load(_write(d, payload))
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(cases.load(_write(d, {"cases": [good]}))["cases"][0]["lat"], 13.75)

    def test_get(self):
        _, case = cases.get("bangkok")
        self.assertIn("latitude 13.75, longitude 100.5", case["prompt"])
        with self.assertRaises(KeyError):
            cases.get("paris")


if __name__ == "__main__":
    unittest.main()
