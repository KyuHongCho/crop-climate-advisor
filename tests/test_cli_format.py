"""Offline tests pinning the CLI's rendered report (no network).

The climate fetch is stubbed on **`crop_advisor.cli.fetch_climate`**, not on
`crop_advisor.climate.fetch_climate`: `cli.py` does `from .climate import
fetch_climate`, which binds the name into the `cli` module at import time, so
patching it in `climate` leaves the CLI calling the live NASA POWER API. Every
test here asserts the stub was actually used, which is what makes "offline"
true rather than merely intended.
"""
import io
import re
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest import mock

from crop_advisor import claims as claims_module
from crop_advisor import cli
from crop_advisor.claims import CHANG_2005, WALTERS_CURREY_2019
from crop_advisor.climate import ClimateSummary

REPO_ROOT = Path(__file__).resolve().parents[1]

# London, from a real NASA POWER climatology response. Built field by field:
# dataclasses.asdict() would silently drop warmest_month_temp_c, which is a
# @property rather than a field.
LONDON = ClimateSummary(
    latitude=51.5,
    longitude=-0.13,
    annual_mean_temp_c=10.39,
    monthly_mean_temp_c={"JAN": 4.14, "JUL": 17.63},
    annual_precip_mm=708.6,
)

ARGV = ["--crop", "basil", "--lat", "51.5", "--lon", "-0.13", "--place", "London"]

EXPECTED_LONDON = """\
Crop-Climate Advisor — basil @ London
========================================================
Location climate (NASA POWER): annual mean 10.39 °C, warmest month 17.63 °C, annual precip 708.6 mm
Crop needs — temperature: no single window satisfies all 3 published sources; every claim is listed below.
Crop needs — rainfall (FAO ECOCROP id 1547): opt 1000–1600 mm/yr

  temperature  10.39 °C     → each published optimal window, and the change it would need:
      18–27 °C  → needs +7.6 °C   FAO ECOCROP (id 1547)  [no condition stated]
      25–30 °C  → needs +14.6 °C  Chang, Alderson & Wright (2005)  [at DLI 20–22 mol·m⁻²·d⁻¹]
      29–35 °C  → needs +18.6 °C  Walters & Currey (2019)  [at DLI 19.5 mol·m⁻²·d⁻¹]
  rainfall     708.6 mm/yr  → SURVIVABLE  (needs +291.4 mm/yr to reach optimal)
  note: even the warmest month stays below FAO ECOCROP's optimal minimum.

Verdict (on FAO ECOCROP's bands): Basil is marginal outdoors at London; a controlled-environment chamber would need to close the gaps below.

Data: NASA POWER (climate) · © FAO ECOCROP (crop requirements).
Temperature claims:
  · FAO ECOCROP data sheet, id 1547 (read directly): https://ecocrop.apps.fao.org/ecocrop/srv/en/dataSheet?id=1547
  · Chang, Alderson & Wright (2005), J. Hortic. Sci. Biotechnol. 80:593–598 (via Barickman et al. (2021), Plants 10(6):1072): https://pmc.ncbi.nlm.nih.gov/articles/PMC8226578/
  · Walters & Currey (2019), HortScience 54(11):1915 (via Walters, Tarr & Lopez (2023), PLoS One 18(11):e0294905): https://pmc.ncbi.nlm.nih.gov/articles/PMC10688745/
"""

# Frozen by this slice: these two lines must render exactly as they did before
# the provenance work, byte for byte.
CLIMATE_LINE = ("Location climate (NASA POWER): annual mean 10.39 °C, "
                "warmest month 17.63 °C, annual precip 708.6 mm")
RAINFALL_ROW = "  rainfall     708.6 mm/yr  → SURVIVABLE  (needs +291.4 mm/yr to reach optimal)"


def run_cli(argv=ARGV, climate=LONDON):
    """Run the CLI with the climate fetch stubbed out, returning its stdout."""
    buf = io.StringIO()
    with mock.patch.object(cli, "fetch_climate", return_value=climate) as stub:
        with redirect_stdout(buf):
            cli.main(argv)
    if stub.call_count != 1:
        raise AssertionError(
            "the climate stub was never called: the CLI would have hit the live "
            "NASA POWER API, so this test is not offline"
        )
    return buf.getvalue()


def readme_example_fence() -> str:
    """The fenced example output block from README.md, located by its prose anchor."""
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(
        r"^Example output \(basil @ London\):\n\n```\n(.*?)^```\n",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise AssertionError("could not find the example-output fence in README.md")
    return match.group(1)


class TestRenderedReport(unittest.TestCase):
    def test_output_is_byte_exact(self):
        self.assertEqual(run_cli(), EXPECTED_LONDON)

    def test_no_line_states_an_optimal_band_without_naming_its_source(self):
        out = run_cli()
        # The line this slice exists to kill: FAO ECOCROP's 18 °C floor presented
        # as *the* optimal, unattributed.
        self.assertNotIn("needs +7.6 °C to reach optimal", out)
        self.assertNotIn("the crop's optimal minimum", out)
        for line in out.splitlines():
            if "18–27" in line or "25–30" in line or "29–35" in line:
                with self.subTest(line=line):
                    self.assertTrue(
                        "ECOCROP" in line or "Chang" in line or "Walters" in line,
                        "a published optimal band appears with no source named",
                    )

    def test_all_three_sources_appear_with_their_stated_conditions(self):
        out = run_cli()
        self.assertIn("18–27 °C", out)
        self.assertIn("25–30 °C", out)
        self.assertIn("29–35 °C", out)
        self.assertIn("[no condition stated]", out)
        self.assertIn("[at DLI 20–22 mol·m⁻²·d⁻¹]", out)
        self.assertIn("[at DLI 19.5 mol·m⁻²·d⁻¹]", out)

    def test_indirect_sources_are_cited_via_the_paper_actually_read(self):
        out = run_cli()
        self.assertIn("(via Barickman et al. (2021), Plants 10(6):1072)", out)
        self.assertIn("(via Walters, Tarr & Lopez (2023), PLoS One 18(11):e0294905)", out)


class TestFrozenLines(unittest.TestCase):
    """The rows this slice promised not to touch."""

    def test_climate_line_is_unchanged(self):
        self.assertIn(CLIMATE_LINE, run_cli())

    def test_rainfall_row_is_unchanged(self):
        # Rainfall genuinely has one source, so it keeps its original row.
        self.assertIn(RAINFALL_ROW, run_cli())


class TestHeaderIsDerivedFromConsensus(unittest.TestCase):
    def test_header_reports_no_common_window_for_the_real_three(self):
        self.assertIn(
            "Crop needs — temperature: no single window satisfies all 3 published sources;",
            run_cli(),
        )

    def test_header_names_the_agreed_band_when_the_sources_do_overlap(self):
        # Narrow Walters & Currey to 26–28 °C: 18–27 ∩ 25–30 ∩ 26–28 = 26–27.
        # The header must follow consensus() rather than restate a fixed sentence.
        narrowed = replace(WALTERS_CURREY_2019, opt_min=26, opt_max=28)
        with mock.patch.object(claims_module, "JOURNAL_TEMPERATURE_CLAIMS",
                               (CHANG_2005, narrowed)):
            out = run_cli()
        self.assertIn(
            "Crop needs — temperature: all 3 published sources are satisfied by 26–27 °C;",
            out,
        )
        self.assertNotIn("no single window satisfies", out)


class TestReadmeExampleIsReal(unittest.TestCase):
    def test_readme_example_output_matches_what_the_cli_actually_prints(self):
        # A README example that is never executed is untested code; this one
        # had already drifted from real output before this test existed.
        self.assertEqual(readme_example_fence(), run_cli())


if __name__ == "__main__":
    unittest.main()
