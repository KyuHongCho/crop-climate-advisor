"""Offline tests for direction, divergence and the conservative read (no network).

`classify` answers a different question from the report header. The header asks
whether the published sources share *any* window at all (they do not, for
basil). These functions ask where one particular location falls relative to each
source's own band — which can be unanimous even when the header says the sources
never agree. 26.0 °C is such a place: no single window satisfies all three, yet
nobody there calls the site too hot.

Every literal below is either a published band edge (18/27, 25/30, 29/35, all
visible in `crop_advisor/claims.py` and `data/ecocrop/basil.json`) or an
observation derived from those edges and pinned here on purpose. Nothing is a
tuned threshold.
"""
import itertools
import json
import unittest
from pathlib import Path
from typing import get_args

from crop_advisor.claims import (
    ConservativeRead,
    classify,
    conservative,
    divergence,
    temperature_claims,
)
from crop_advisor.suitability import _assess_band

BASIL = json.loads((Path(__file__).resolve().parents[1] / "data" / "ecocrop" / "basil.json").read_text())

#: FAO ECOCROP 18–27, Chang 25–30, Walters & Currey 29–35, in that order.
CLAIMS = temperature_claims(BASIL)
ECOCROP, CHANG, WALTERS = CLAIMS

ALLOWED = set(get_args(ConservativeRead))
DISAGREE = "not optimal (sources disagree in direction)"

#: -10.0 to 50.0 on a 0.1 grid, built as i / 10 so no float accumulation creeps
#: in. The band edges are integers, so no epsilon is needed anywhere.
SWEEP = [i / 10 for i in range(-100, 501)]


class TestClassify(unittest.TestCase):
    def test_below_optimal_above_for_one_claim(self):
        # ECOCROP's band is 18–27; the edges are inclusive.
        cases = [
            (10.39, "below"),
            (17.9, "below"),
            (18, "optimal"),      # lower edge, inclusive
            (22.0, "optimal"),
            (27, "optimal"),      # upper edge, inclusive
            (27.1, "above"),
            (28.0, "above"),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(classify(value, ECOCROP), expected)

    def test_the_three_real_claims_at_28c(self):
        # In claim order: 28 is past ECOCROP's 27, inside Chang's 25–30, and
        # short of Walters & Currey's 29.
        self.assertEqual([classify(28.0, c) for c in CLAIMS],
                         ["above", "optimal", "below"])


class TestClassifyAgreesWithAssessBand(unittest.TestCase):
    """classify() must not drift from the suitability module's optimal test.

    classify() reads the band directly rather than routing through
    `_assess_band`, because `_assess_band` also derives a status from
    abs_min/abs_max, which the journal claims do not state. This test is what
    keeps the two definitions of "optimal" welded together anyway.
    """

    def test_the_sign_of_the_correction_matches_the_direction(self):
        mismatches = []
        for value in SWEEP:
            for claim in CLAIMS:
                gap = _assess_band("temperature", value, "°C",
                                   {"opt_min": claim.opt_min, "opt_max": claim.opt_max}).correction
                expected = "optimal" if gap == 0 else ("below" if gap > 0 else "above")
                got = classify(value, claim)
                if got != expected:
                    mismatches.append((value, claim.source, got, expected))
        self.assertEqual(mismatches, [])


class TestConservativeAtTheFourReferenceTemperatures(unittest.TestCase):
    """The four locations the README's eval case turns on."""

    def test_the_conservative_read_at_each(self):
        cases = [
            # London's annual mean: every source says too cold.
            (10.39, "below optimal"),
            # No single window satisfies all three here either, yet no source
            # calls the site too hot — so there is nothing to flag.
            (26.0, "below optimal"),
            # ECOCROP says too hot, Walters & Currey says too cold.
            (28.0, DISAGREE),
            (31.0, "above optimal"),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(conservative(divergence(value, CLAIMS)), expected)


class TestConservativeIsTotal(unittest.TestCase):
    def test_every_subset_of_directions_has_a_defined_result(self):
        expected = {
            # Unreachable from the CLI: temperature_claims() always returns at
            # least the ECOCROP claim. Pinned so the default cannot drift to
            # "optimal", which is the one answer it must never give.
            (): DISAGREE,
            ("below",): "below optimal",
            ("optimal",): "optimal",
            ("above",): "above optimal",
            ("below", "optimal"): "below optimal",
            ("below", "above"): DISAGREE,
            ("optimal", "above"): "above optimal",
            ("below", "optimal", "above"): DISAGREE,
        }
        directions = ("below", "optimal", "above")
        subsets = [s for n in range(len(directions) + 1)
                   for s in itertools.combinations(directions, n)]
        self.assertEqual(len(subsets), 8)
        for subset in subsets:
            with self.subTest(subset=subset):
                got = conservative(subset)
                self.assertIn(got, ALLOWED)
                self.assertEqual(got, expected[subset])

    def test_order_and_repetition_do_not_matter(self):
        self.assertEqual(conservative(["below", "optimal"]),
                         conservative(["optimal", "below", "below", "optimal"]))
        self.assertEqual(conservative(("above", "below")),
                         conservative(("below", "above")))

    def test_realistic_temperature_sweep(self):
        regions = {}
        for value in SWEEP:
            got = conservative(divergence(value, CLAIMS))
            self.assertIn(got, ALLOWED)
            regions.setdefault(got, []).append(value)
        counts = {k: len(v) for k, v in regions.items()}
        self.assertEqual(len(SWEEP), 601)
        self.assertEqual(counts, {"below optimal": 371, DISAGREE: 19, "above optimal": 211})
        # "optimal" never occurs anywhere: that is consensus() being None
        # showing up end to end. There is no temperature at which all three
        # published sources call the site optimal.
        self.assertNotIn("optimal", regions)
        self.assertEqual((min(regions["below optimal"]), max(regions["below optimal"])), (-10.0, 27.0))
        self.assertEqual((min(regions[DISAGREE]), max(regions[DISAGREE])), (27.1, 28.9))
        self.assertEqual((min(regions["above optimal"]), max(regions["above optimal"])), (29.0, 50.0))


class TestTheFlagConditionMatchesConservative(unittest.TestCase):
    """The CLI's one-line flag test must mean exactly what conservative() means."""

    def test_the_flag_condition_matches_conservatives_disagreement_branch(self):
        directions = ("below", "optimal", "above")
        subsets = [s for n in range(1, len(directions) + 1)
                   for s in itertools.combinations(directions, n)]
        for subset in subsets:
            with self.subTest(subset=subset):
                flagged = {"below", "above"} <= set(subset)
                self.assertEqual(flagged, conservative(subset) == DISAGREE)


class TestDisagreementWindowEdges(unittest.TestCase):
    """The window is the open interval (27, 29).

    Above FAO ECOCROP's optimal maximum of 27 but below Walters & Currey's
    optimal minimum of 29 — so on a 0.1 grid it is exactly 27.1 to 28.9.
    """

    def test_every_tenth_between_27_and_29_diverges(self):
        window = [i / 10 for i in range(271, 290)]
        self.assertEqual(len(window), 19)
        for value in window:
            with self.subTest(value=value):
                self.assertEqual(sorted(divergence(value, CLAIMS)),
                                 ["above", "below", "optimal"])
                self.assertEqual(conservative(divergence(value, CLAIMS)), DISAGREE)

    def test_the_lower_tail_of_the_window_diverges(self):
        for value in (27.1, 27.2, 27.3, 27.4):
            with self.subTest(value=value):
                self.assertEqual(conservative(divergence(value, CLAIMS)), DISAGREE)

    def test_the_upper_tail_of_the_window_diverges(self):
        for value in (28.6, 28.7, 28.8, 28.9):
            with self.subTest(value=value):
                self.assertEqual(conservative(divergence(value, CLAIMS)), DISAGREE)

    def test_no_disagreement_at_the_lower_edge_27_0(self):
        # 27.0 is ECOCROP's inclusive optimal maximum, so nothing is "above".
        self.assertEqual(sorted(divergence(27.0, CLAIMS)), ["below", "optimal"])
        self.assertEqual(conservative(divergence(27.0, CLAIMS)), "below optimal")

    def test_no_disagreement_at_the_upper_edge_29_0(self):
        # 29.0 is Walters & Currey's inclusive optimal minimum, so nothing is
        # "below".
        self.assertEqual(sorted(divergence(29.0, CLAIMS)), ["above", "optimal"])
        self.assertEqual(conservative(divergence(29.0, CLAIMS)), "above optimal")

    def test_just_outside_the_window_on_either_side(self):
        self.assertEqual(conservative(divergence(26.9, CLAIMS)), "below optimal")
        self.assertEqual(conservative(divergence(29.1, CLAIMS)), "above optimal")


if __name__ == "__main__":
    unittest.main()
