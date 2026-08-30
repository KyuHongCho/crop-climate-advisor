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

    def test_survivable_above_optimal(self):
        # 30°C: above opt_max 27 but within absolute 7–36. The correction is
        # negative because the value must come *down* to reach the optimal band.
        b = _assess_band("temperature", 30.0, "°C", BASIL["temperature_c"])
        self.assertEqual(b.status, "survivable")
        self.assertAlmostEqual(b.correction, -3.0, places=1)  # -3.0°C to reach opt_max 27

    def test_unsuitable_above_absolute(self):
        # 40°C: above abs_max 36, so no longer survivable. Without this case the
        # abs_max lookup is never exercised — every other test sits below the
        # optimal band, so abs_max could be dropped entirely and all tests pass.
        b = _assess_band("temperature", 40.0, "°C", BASIL["temperature_c"])
        self.assertEqual(b.status, "unsuitable")
        self.assertAlmostEqual(b.correction, -13.0, places=1)  # -13.0°C to reach opt_max 27

    def test_exactly_at_absolute_max_is_still_survivable(self):
        # abs_max is inclusive: 36°C is the last survivable value, 36.1 is not.
        # Pins the boundary so `value <= abs_max` cannot be weakened to `<`.
        b = _assess_band("temperature", 36.0, "°C", BASIL["temperature_c"])
        self.assertEqual(b.status, "survivable")
        self.assertEqual(b.opt_min, 18)   # band bounds are reported back unchanged
        self.assertEqual(b.opt_max, 27)

    def test_exactly_at_optimal_max_is_optimal(self):
        # opt_max is inclusive: 27°C is still optimal, 27.1 is not.
        b = _assess_band("temperature", 27.0, "°C", BASIL["temperature_c"])
        self.assertEqual(b.status, "optimal")
        self.assertEqual(b.correction, 0.0)

    def test_exactly_at_absolute_min_is_survivable(self):
        # abs_min is inclusive: 7°C is the coldest survivable value, 6.9 is not.
        b = _assess_band("temperature", 7.0, "°C", BASIL["temperature_c"])
        self.assertEqual(b.status, "survivable")

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


class TestAssessVerdicts(unittest.TestCase):
    """assess() picks one of three verdicts; only the marginal one was covered."""

    @staticmethod
    def _climate(temp_c: float, precip_mm: float) -> ClimateSummary:
        return ClimateSummary(
            latitude=0.0, longitude=0.0,
            annual_mean_temp_c=temp_c,
            monthly_mean_temp_c={"JUL": temp_c},
            annual_precip_mm=precip_mm,
        )

    def test_good_match_when_both_bands_optimal(self):
        # 22°C inside opt 18–27, and 1200 mm inside opt 1000–1600.
        a = assess(BASIL, self._climate(22.0, 1200.0), place="Somewhere")
        self.assertEqual(a.temperature.status, "optimal")
        self.assertEqual(a.rainfall.status, "optimal")
        self.assertIn("good outdoor match", a.verdict)

    def test_unsuitable_when_a_band_is_outside_absolute(self):
        # 40°C is above abs_max 36, so outdoor growing is ruled out entirely.
        a = assess(BASIL, self._climate(40.0, 1200.0), place="Somewhere")
        self.assertEqual(a.temperature.status, "unsuitable")
        self.assertIn("cannot be grown outdoors", a.verdict)

    def test_warmest_month_exactly_at_opt_min_counts_as_reaching_it(self):
        # The comparison is >=, so a warmest month of exactly 18°C reaches opt_min.
        climate = ClimateSummary(
            latitude=0.0, longitude=0.0, annual_mean_temp_c=10.0,
            monthly_mean_temp_c={"JUL": 18.0, "JAN": 2.0}, annual_precip_mm=1200.0,
        )
        self.assertTrue(assess(BASIL, climate, place="X").warmest_month_reaches_opt)

    def test_name_falls_back_to_scientific_name_then_generic(self):
        # assess() reads crop["common_name"] or crop["name"] or "crop". basil.json
        # always has common_name, so neither fallback was ever exercised. Note the
        # name reaches Assessment.crop un-capitalised (suitability.py:57), which is
        # why both the raw field and the rendered verdict are asserted here.
        climate = self._climate(22.0, 1200.0)
        no_common = {k: v for k, v in BASIL.items() if k != "common_name"}
        a1 = assess(no_common, climate, "X")
        self.assertEqual(a1.crop, "Ocimum basilicum")
        self.assertTrue(a1.verdict.startswith("Ocimum basilicum"))
        neither = {k: v for k, v in no_common.items() if k != "name"}
        a2 = assess(neither, climate, "X")
        self.assertEqual(a2.crop, "crop")
        self.assertTrue(a2.verdict.startswith("Crop"))


if __name__ == "__main__":
    unittest.main()
