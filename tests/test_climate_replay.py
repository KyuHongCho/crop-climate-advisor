"""CROP_ADVISOR_CLIMATE_REPLAY: fetch_climate answers from recorded NASA POWER responses.

Every replay test patches urllib's urlopen to raise, so a test that passes made no live call.
"""
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from crop_advisor import cli
from crop_advisor.climate import REPLAY_ENV, REPLAY_MISSING, fetch_climate, power_url
from evals import record_fixtures

FIXTURES = Path(__file__).resolve().parent.parent / "evals" / "fixtures" / "nasa_power"


def _offline():
    return mock.patch("urllib.request.urlopen", side_effect=AssertionError("network used"))


def _response(t2m_ann, precip_ann):
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    return {
        "type": "Feature",
        "header": {"api": {"name": "POWER Climatology API", "version": "v9.9.9"},
                   "range": "test period"},
        "messages": [],
        "properties": {"parameter": {
            "T2M": {**{m: t2m_ann + i / 10 for i, m in enumerate(months)}, "ANN": t2m_ann},
            "PRECTOTCORR": {**{m: precip_ann for m in months}, "ANN": precip_ann},
            "WS2M": {"ANN": 1.0},
        }},
    }


class ReplayTest(unittest.TestCase):
    def test_committed_fixtures_replay_offline_for_every_site(self):
        # Annual means the live CLI printed for these sites.
        expected = {"london": 10.39, "lagos": 26.38, "bangkok": 27.85, "djibouti": 29.27}
        self.assertEqual(set(record_fixtures.SITES), set(expected))
        with mock.patch.dict(os.environ, {REPLAY_ENV: str(FIXTURES)}), _offline():
            for site, (lat, lon) in record_fixtures.SITES.items():
                with self.subTest(site=site):
                    self.assertEqual(fetch_climate(lat, lon).annual_mean_temp_c, expected[site])

    def test_cli_answers_bangkok_from_the_fixture(self):
        out = io.StringIO()
        with mock.patch.dict(os.environ, {REPLAY_ENV: str(FIXTURES)}), _offline(), \
                contextlib.redirect_stdout(out):
            cli.main(["--crop", "basil", "--lat", "13.75", "--lon", "100.5", "--place", "Bangkok"])
        self.assertIn("Sources disagree:", out.getvalue())

    def test_integer_coordinates_name_the_same_file(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "51.0_0.0.json").write_text(json.dumps(_response(10.0, 2.0)))
            with mock.patch.dict(os.environ, {REPLAY_ENV: d}), _offline():
                self.assertEqual(fetch_climate(51, 0).annual_mean_temp_c, 10.0)

    def test_missing_fixture_raises_marker_and_never_goes_live(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {REPLAY_ENV: d}), \
                _offline(), mock.patch("crop_advisor.climate.get_json") as live:
            with self.assertRaises(FileNotFoundError) as caught:
                fetch_climate(1.0, 2.0)
        self.assertTrue(str(caught.exception).startswith(REPLAY_MISSING), str(caught.exception))
        self.assertIn("1.0_2.0.json", str(caught.exception))
        live.assert_not_called()

    def test_unset_or_empty_variable_fetches_live(self):
        for value in (None, ""):
            with self.subTest(value=value), mock.patch.dict(os.environ), \
                    mock.patch("crop_advisor.climate.get_json",
                               return_value=_response(12.0, 1.0)) as live:
                os.environ.pop(REPLAY_ENV, None)
                if value is not None:
                    os.environ[REPLAY_ENV] = value
                self.assertEqual(fetch_climate(51.5, -0.13).annual_mean_temp_c, 12.0)
                live.assert_called_once_with(power_url(51.5, -0.13))


class RecordFixturesTest(unittest.TestCase):
    def test_records_trimmed_responses_and_source_that_replay_reads(self):
        responses = {power_url(*record_fixtures.SITES["bangkok"]): _response(27.0, 4.0),
                     power_url(*record_fixtures.SITES["london"]): _response(10.0, 2.0)}
        with tempfile.TemporaryDirectory() as d, \
                mock.patch("crop_advisor._http.get_json", side_effect=responses.__getitem__) as live:
            written = record_fixtures.record(["bangkok", "london"], d, accessed="2026-01-02")
            self.assertEqual(live.call_count, 2)
            self.assertEqual([p.name for p in written],
                             ["13.75_100.5.json", "51.5_-0.13.json", "SOURCE.md"])

            stored = json.loads((Path(d) / "13.75_100.5.json").read_text())
            self.assertEqual(list(stored), ["properties"])
            self.assertEqual(set(stored["properties"]["parameter"]), {"T2M", "PRECTOTCORR"})

            source = (Path(d) / "SOURCE.md").read_text()
            for text in ("POWER Climatology API v9.9.9", "2026-01-02", "test period",
                         "POWER Project's Climatology v9.9.9 version on 2026/01/02.",
                         "Langley Research Center's Prediction Of Worldwide Energy Resources",
                         "There are no restrictions on the use", "Creative Commons BY 4.0",
                         "https://creativecommons.org/licenses/by/4.0/",
                         "| bangkok | 13.75 | 100.5 | `13.75_100.5.json` |"):
                self.assertIn(text, source)

            with mock.patch.dict(os.environ, {REPLAY_ENV: d}), _offline():
                climate = fetch_climate(13.75, 100.5)
        self.assertEqual(climate.annual_mean_temp_c, 27.0)
        self.assertEqual(climate.annual_precip_mm, round(4.0 * 365.25, 1))

    def test_partial_rerecord_is_refused_before_any_call(self):
        responses = {power_url(*record_fixtures.SITES["bangkok"]): _response(27.0, 4.0),
                     power_url(*record_fixtures.SITES["london"]): _response(10.0, 2.0)}
        with tempfile.TemporaryDirectory() as d, \
                mock.patch("crop_advisor._http.get_json", side_effect=responses.__getitem__) as live:
            record_fixtures.record(["bangkok", "london"], d, accessed="2026-01-02")
            source = (Path(d) / "SOURCE.md").read_text()
            live.reset_mock()

            with self.assertRaises(FileExistsError) as caught:
                record_fixtures.record(["bangkok"], d, accessed="2026-02-03")
            self.assertIn("51.5_-0.13.json", str(caught.exception))
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exited:
                record_fixtures.main(["--sites", "bangkok", "--out", d])
            self.assertEqual(exited.exception.code, 2)
            live.assert_not_called()
            self.assertEqual((Path(d) / "SOURCE.md").read_text(), source)

            record_fixtures.record(["london", "bangkok"], d, accessed="2026-02-03")
            self.assertIn("2026-02-03", (Path(d) / "SOURCE.md").read_text())

    def test_a_failed_fetch_writes_nothing(self):
        first = _response(27.0, 4.0)
        with tempfile.TemporaryDirectory() as d:
            with mock.patch("crop_advisor._http.get_json", return_value=first):
                record_fixtures.record(["bangkok", "london"], d, accessed="2026-01-02")
            before = {p.name: p.read_text() for p in Path(d).iterdir()}

            with mock.patch("crop_advisor._http.get_json",
                            side_effect=[_response(99.0, 9.0), OSError("second call fails")]) as live:
                with self.assertRaises(OSError):
                    record_fixtures.record(["bangkok", "london"], d, accessed="2026-02-03")
            self.assertEqual(live.call_count, 2)
            self.assertEqual({p.name: p.read_text() for p in Path(d).iterdir()}, before)

        with tempfile.TemporaryDirectory() as d:
            with mock.patch("crop_advisor._http.get_json",
                            side_effect=[_response(99.0, 9.0), OSError("second call fails")]), \
                    self.assertRaises(OSError):
                record_fixtures.record(["bangkok", "london"], d)
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_sites_accept_comma_or_space_separated_ids(self):
        for argv in (["bangkok,lagos"], ["bangkok", "lagos"], ["bangkok,", "lagos,bangkok"]):
            with self.subTest(argv=argv), \
                    mock.patch.object(record_fixtures, "record", return_value=[]) as rec:
                record_fixtures.main(["--sites", *argv, "--out", "unused"])
            rec.assert_called_once_with(["bangkok", "lagos"], "unused")

    def test_cli_rejects_unknown_site(self):
        for argv in (["atlantis"], ["bangkok,atlantis"], [","]):
            with self.subTest(argv=argv), mock.patch.object(record_fixtures, "record") as rec, \
                    contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                record_fixtures.main(["--sites", *argv, "--out", "unused"])
            rec.assert_not_called()


if __name__ == "__main__":
    unittest.main()
