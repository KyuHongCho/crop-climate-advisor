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

    def test_get(self):
        _, case = cases.get("bangkok")
        self.assertIn("latitude 13.75, longitude 100.5", case["prompt"])
        with self.assertRaises(KeyError):
            cases.get("paris")


if __name__ == "__main__":
    unittest.main()
