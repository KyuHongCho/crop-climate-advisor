"""Offline unit tests for the provenance-carrying temperature claims (no network)."""
import dataclasses
import json
import unittest
from pathlib import Path

from crop_advisor.claims import (
    CHANG_2005,
    JOURNAL_TEMPERATURE_CLAIMS,
    WALTERS_CURREY_2019,
    Claim,
    cite,
    consensus,
    temperature_claims,
)

BASIL = json.loads((Path(__file__).resolve().parents[1] / "data" / "ecocrop" / "basil.json").read_text())


class TestCite(unittest.TestCase):
    def test_direct_claim_is_marked_read_directly_and_never_says_via(self):
        rendered = cite(temperature_claims(BASIL)[0])
        self.assertIn("(read directly)", rendered)
        self.assertNotIn("(via ", rendered)

    def test_indirect_claim_renders_a_via_form_naming_the_paper_actually_read(self):
        rendered = cite(CHANG_2005)
        self.assertIn("(via Barickman et al. (2021), Plants 10(6):1072)", rendered)
        self.assertIn("Chang, Alderson & Wright (2005)", rendered)
        self.assertIn("https://pmc.ncbi.nlm.nih.gov/articles/PMC8226578/", rendered)

    def test_indirect_claim_with_no_via_raises_rather_than_using_a_placeholder(self):
        # A fallback string here would make every other integrity check vacuous:
        # a claim with no provenance would render, and the suite would stay green.
        broken = dataclasses.replace(CHANG_2005, via=None)
        with self.assertRaises(ValueError):
            cite(broken)

    def test_indirect_claim_with_a_blank_via_raises(self):
        broken = dataclasses.replace(CHANG_2005, via="   ")
        with self.assertRaises(ValueError):
            cite(broken)

    def test_every_real_claim_renders(self):
        for claim in temperature_claims(BASIL):
            with self.subTest(source=claim.source):
                rendered = cite(claim)
                self.assertIn(claim.url, rendered)
                if not claim.read_directly:
                    self.assertIn("(via ", rendered)


class TestClaimIsFrozen(unittest.TestCase):
    def test_a_published_figure_cannot_be_reassigned(self):
        with self.assertRaises(dataclasses.FrozenInstanceError):
            WALTERS_CURREY_2019.opt_min = 18  # type: ignore[misc]

    def test_the_journal_claim_collection_is_a_tuple(self):
        self.assertIsInstance(JOURNAL_TEMPERATURE_CLAIMS, tuple)


class TestTemperatureClaims(unittest.TestCase):
    def test_the_three_published_bands(self):
        # These numbers are the README's "The eval case" table; the ECOCROP row
        # comes from the bundled data file, the other two from this module.
        self.assertEqual(
            [(c.source, c.band) for c in temperature_claims(BASIL)],
            [
                ("FAO ECOCROP (id 1547)", (18, 27)),
                ("Chang, Alderson & Wright (2005)", (25, 30)),
                ("Walters & Currey (2019)", (29, 35)),
            ],
        )

    def test_stated_conditions(self):
        ecocrop, chang, walters = temperature_claims(BASIL)
        self.assertIsNone(ecocrop.condition)  # ECOCROP states no condition
        self.assertEqual(chang.condition, "DLI 20–22 mol·m⁻²·d⁻¹")
        self.assertEqual(walters.condition, "DLI 19.5 mol·m⁻²·d⁻¹")

    def test_ecocrop_claim_is_built_from_the_crop_dict_not_hardcoded(self):
        fake = {"ecocrop_id": 99, "temperature_c": {"opt_min": 1, "opt_max": 2},
                "_source": {"url": "https://example.invalid/99"}}
        ecocrop = temperature_claims(fake)[0]
        self.assertEqual(ecocrop.band, (1, 2))
        self.assertEqual(ecocrop.source, "FAO ECOCROP (id 99)")
        self.assertEqual(ecocrop.url, "https://example.invalid/99")


class TestConsensus(unittest.TestCase):
    def test_the_real_three_have_no_common_window(self):
        # max(18, 25, 29) = 29 > min(27, 30, 35) = 27
        self.assertIsNone(consensus(temperature_claims(BASIL)))

    def test_intersection_when_one_exists(self):
        narrowed = dataclasses.replace(WALTERS_CURREY_2019, opt_min=26, opt_max=28)
        claims = temperature_claims(BASIL)[:2] + (narrowed,)
        self.assertEqual(consensus(claims), (26.0, 27.0))

    def test_touching_bands_intersect_at_a_point(self):
        a = dataclasses.replace(CHANG_2005, opt_min=10, opt_max=20)
        b = dataclasses.replace(CHANG_2005, opt_min=20, opt_max=30)
        self.assertEqual(consensus((a, b)), (20.0, 20.0))

    def test_a_single_claim_is_its_own_consensus(self):
        self.assertEqual(consensus((CHANG_2005,)), (25.0, 30.0))

    def test_no_claims_means_no_consensus(self):
        self.assertIsNone(consensus(()))

    def test_accepts_any_iterable(self):
        self.assertIsNone(consensus(c for c in temperature_claims(BASIL)))


class TestNoUnattributedConstants(unittest.TestCase):
    def test_claim_is_the_only_way_the_journal_numbers_are_carried(self):
        for claim in JOURNAL_TEMPERATURE_CLAIMS:
            with self.subTest(source=claim.source):
                self.assertIsInstance(claim, Claim)
                self.assertFalse(claim.read_directly)
                self.assertTrue((claim.via or "").strip())
                self.assertTrue(claim.url.startswith("https://"))


class TestJournalClaimsAreBoundToTheCropTheyStudy(unittest.TestCase):
    """The two journal claims are studies of basil; they must not travel."""

    def test_a_different_crop_gets_only_its_own_ecocrop_claim(self):
        fake = {"ecocrop_id": 99, "temperature_c": {"opt_min": 1, "opt_max": 2},
                "_source": {"url": "https://example.invalid/99"}}
        got = temperature_claims(fake)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0].source, "FAO ECOCROP (id 99)")
        sources = [c.source for c in got]
        for journal in JOURNAL_TEMPERATURE_CLAIMS:
            self.assertNotIn(journal.source, sources)

    def test_basil_still_gets_all_three(self):
        self.assertEqual(
            [c.source for c in temperature_claims(BASIL)],
            ["FAO ECOCROP (id 1547)",
             "Chang, Alderson & Wright (2005)",
             "Walters & Currey (2019)"],
        )

    def test_a_crop_always_gets_at_least_the_ecocrop_claim(self):
        # Nothing downstream ever sees an empty claim set, whatever the crop.
        fake = {"ecocrop_id": 99, "temperature_c": {"opt_min": 1, "opt_max": 2},
                "_source": {"url": "https://example.invalid/99"}}
        for crop, label in ((BASIL, "basil"), (fake, "a crop with no journal claims")):
            with self.subTest(crop=label):
                got = temperature_claims(crop)
                self.assertGreaterEqual(len(got), 1)
                self.assertTrue(got[0].source.startswith("FAO ECOCROP"))


class TestClaimRejectsFalseProvenance(unittest.TestCase):
    def test_a_direct_claim_that_also_names_a_via_cannot_be_constructed(self):
        # cite() renders a read_directly claim as first-hand and never reads
        # `via`, so this state would credit the wrong URL and drop the chain.
        direct = temperature_claims(BASIL)[0]
        self.assertTrue(direct.read_directly)
        with self.assertRaises(ValueError):
            dataclasses.replace(direct, via="Some paper")


if __name__ == "__main__":
    unittest.main()
