"""Offline unit tests for the suitability logic (no network)."""
import json
import unittest
from pathlib import Path

from crop_advisor.climate import ClimateSummary
from crop_advisor.suitability import _assess_band, assess

BASIL = json.loads((Path(__file__).resolve().parents[1] / "data" / "ecocrop" / "basil.json").read_text())


class TestAssessBand(unittest.TestCase):
    def test_optimal(self):
        b = _assess_band("temperature", 22.0, "°C", BASIL["temperature_c"])
        self.assertEqual(b.status, "optimal")
        self.assertEqual(b.correction, 0.0)

    def test_survivable_below_optimal(self):
        # London annual mean ~10.4°C: below opt_min 18 but within absolute 7–36
        b = _assess_band("temperature", 10.39, "°C", BASIL["temperature_c"])
        self.assertEqual(b.status, "survivable")
        self.assertAlmostEqual(b.correction, 7.6, places=1)  # +7.6°C to reach opt_min 18

    def test_unsuitable_below_absolute(self):
        b = _assess_band("temperature", 5.0, "°C", BASIL["temperature_c"])  # below abs_min 7
        self.assertEqual(b.status, "unsuitable")

    def test_rainfall_deficit(self):
        b = _assess_band("rainfall", 708.0, "mm/yr", BASIL["rainfall_mm_yr"])
        self.assertEqual(b.status, "survivable")  # within abs 600–4300, below opt 1000
        self.assertAlmostEqual(b.correction, 292.0, places=1)


class TestAssess(unittest.TestCase):
    def test_london_basil_is_marginal(self):
        climate = ClimateSummary(
            latitude=51.5, longitude=-0.13,
            annual_mean_temp_c=10.39,
            monthly_mean_temp_c={"JUL": 17.63, "JAN": 4.14},
            annual_precip_mm=708.0,
        )
        a = assess(BASIL, climate, place="London")
        self.assertEqual(a.temperature.status, "survivable")
        self.assertFalse(a.warmest_month_reaches_opt)  # 17.63 < opt_min 18
        self.assertIn("marginal", a.verdict.lower())


if __name__ == "__main__":
    unittest.main()
